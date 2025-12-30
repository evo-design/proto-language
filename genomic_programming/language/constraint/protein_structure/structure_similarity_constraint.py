"""
structure_similarity_constraint.py

Contains implementation of generic structure similarity constraints (RMSD, TM-score)
supporting multiple structure prediction tools (ESMFold, AlphaFold3, Boltz, Chai).
"""

from __future__ import annotations

import os
from pydantic import model_validator
import shutil
import subprocess
import re
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from logging import getLogger

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
)
from proto_language.tools.structure_prediction.schemas import (
    StructurePredictionComplex,
)
from proto_language.utils import MAX_ENERGY, sigmoid_score
from .structure_prediction_dispatcher import predict_structures

logger = getLogger(__name__)


# ============================================================================
# Metrics and scoring utils
# ============================================================================

def _compute_ce_aligned_rmsd(pdb_text1: str, pdb_text2: str) -> Dict[str, Any]:
    """
    Compute CE-aligned RMSD using PyMOL's cealign.
    Text strings are the full PDB file contents.
    """
    try:
        import pymol
        from pymol import cmd
    except ImportError as e:
        raise ImportError(
            "PyMOL is required for RMSD constraints but was not found. "
            "Please install the open-source version via Conda:\n\n"
            "  conda install -c conda-forge pymol-open-source\n"
        ) from e

    # Initialize PyMOL in quiet mode without GUI.
    pymol.finish_launching(['pymol', '-qc'])
    cmd.reinitialize()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1:
        f1.write(pdb_text1)
        tmp1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
        f2.write(pdb_text2)
        tmp2 = f2.name

    try:
        cmd.load(tmp1, "ref")
        cmd.load(tmp2, "mobile")

        # cealign aligns 'mobile' to 'ref'.
        # For multimers, this aligns the whole complex if chain identifiers match
        # or performs a global alignment.
        result = cmd.cealign("ref", "mobile")

        return {
            'rmsd': result['RMSD'],
            'aligned_length': result['alignment_length'],
            'alignment_score': result.get('raw_score', None)
        }
    except Exception as e:
        logger.warning(f"PyMOL alignment failed: {e}, returning very bad RMSD value")
        # Return bad values on failure.
        return {'rmsd': 999.0, 'aligned_length': 0}
    finally:
        if os.path.exists(tmp1):
            os.unlink(tmp1)
        if os.path.exists(tmp2):
            os.unlink(tmp2)
        cmd.delete("all")
        cmd.reinitialize()


def _filter_pdb_by_plddt(pdb_text: str, threshold: float) -> str:
    """
    Filters PDB text, keeping only residues with B-factor (pLDDT) >= threshold.
    """
    if threshold is None or threshold <= 0:
        return pdb_text

    filtered_lines = []
    for line in pdb_text.splitlines():
        # PDB ATOM records: B-factor is columns 61-66 (index 60:66)
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                b_factor = float(line[60:66])
                if b_factor >= threshold:
                    filtered_lines.append(line)
            except ValueError:
                # Keep line if B-factor parsing fails to be safe
                filtered_lines.append(line)
        else:
            # Keep header/footer lines
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _compute_tm_score_from_pdb(
    target_pdb_text: str,
    candidate_pdb_text: str,
    plddt_threshold: Optional[float] = None,
) -> float:
    """
    Compute TM-score using the 'TMalign' binary.
    Returns the TM-score normalized by the length of the target (reference) structure.
    """
    tmalign_path = shutil.which("TMalign")
    if not tmalign_path:
        raise ImportError(
            "The 'TMalign' (or 'USalign') binary is required for TM-score constraints. "
            "Please install it (e.g., via Conda):\n\n"
            "  conda install -c bioconda tmalign\n"
        )

    if plddt_threshold is not None:
        candidate_pdb_text = _filter_pdb_by_plddt(candidate_pdb_text, plddt_threshold)
        # If filtering removed everything, return worst score immediately
        if not any(line.startswith("ATOM") for line in candidate_pdb_text.splitlines()):
            return 0.0

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f_target:
        f_target.write(target_pdb_text)
        target_path = f_target.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f_candidate:
        f_candidate.write(candidate_pdb_text)
        candidate_path = f_candidate.name

    try:
        cmd = [tmalign_path, candidate_path, target_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        # TMalign outputs two scores. We want the one normalized by the target (Chain_2).
        # Example output line: "TM-score= 0.5678 (if normalized by length of Chain_2 ...)"
        match = re.search(r"TM-score=\s*([0-9.]+)\s+\(if normalized by length of Chain_2", output)

        if match:
            return float(match.group(1))

        # If Chain_2 pattern not found, try to get both and take the second one (usually target).
        matches = re.findall(r"TM-score=\s*([0-9.]+)", output)
        if len(matches) >= 2:
            return float(matches[1])
        elif matches:
            return float(matches[0])

        logger.warning(f"Could not find TMscore in TMalign output, returning worst value")
        return 0.0

    except subprocess.CalledProcessError as e:
        logger.warning(f"TMalign execution failed: {e}, returning worst value")
        return 0.0
    finally:
        if os.path.exists(target_path):
            os.unlink(target_path)
        if os.path.exists(candidate_path):
            os.unlink(candidate_path)

# ============================================================================
# Configuration
# ============================================================================

class StructureConstraintBaseConfig(BaseConfig):
    """Base configuration for structure similarity constraints.

    This configuration manages the setup for predicting protein structures from
    candidate sequences and defining the target structure against which candidates
    are compared. It supports defining targets via direct sequence folding or
    by providing an existing PDB structure.

    The user should provide a target structure as **one** of:
    - `target_chains`: A list of protein sequences to dynamically fold.
    - `target_pdb_file`: A path to a PDB file.
    - `target_pdb_content`: The raw string content of a PDB file.

    Attributes:

        target_chains (Optional[Tuple[str]]):
            The amino acid sequences of the target protein chains. If provided,
            these sequences will be folded using the specified `structure_tool`
            to generate the reference structure. This is mutually exclusive
            with `target_pdb_file` and `target_pdb_content`.

        target_pdb_file (Optional[str]):
            The local file path to a PDB file serving as the reference structure.
            This is mutually exclusive with `target_chains` and `target_pdb_content`.

        target_pdb_content (Optional[str]):
            The raw string content of a PDB file serving as the reference structure.
            Useful when the PDB data is loaded in memory or passed via API.
            This is mutually exclusive with `target_chains` and `target_pdb_file`.

        structure_tool (str):
            The structure prediction tool to use for folding both the target (if provided
            as a sequence) and the candidate sequences. Supported options include:
            - "esmfold": ESMFold (Meta AI)
            - "alphafold3" or "af3": AlphaFold 3 (Google DeepMind)
            - "boltz": Boltz-1 (MIT)
            - "chai": Chai-1 (Chai Discovery)
            Default is "esmfold".

        tool_config (Dict[str, Any]):
            A dictionary of configuration parameters to pass directly to the underlying
            structure prediction tool runner.
            Defaults to an empty dictionary.

        min_target_plddt (float):
            Only used if the target structure is provided via `target_chains`. This is
            the minimum average pLDDT confidence score required for the folded target
            structure. If the target is provided as a sequence and its predicted
            structure has a confidence below this threshold, the constraint may return
            a default/penalty score or log a warning. Default is 0.6.
    """

    # Target specification (mutually exclusive):
    target_chains: Optional[Tuple[str, ...]] = ConfigField(
        title="Target Chains",
        default=None,
        description="Amino acid sequences of the target. Will be folded using the selected tool.",
    )
    target_pdb_file: Optional[str] = ConfigField(
        title="Target PDB File",
        default=None,
        description="Path to a local PDB file serving as the reference structure.",
    )
    target_pdb_content: Optional[str] = ConfigField(
        title="Target PDB Content",
        default=None,
        description="Raw string content of the target PDB.",
        advanced=True,
    )

    # Tool selection:
    structure_tool: str = ConfigField(
        title="Structure Prediction Tool",
        default="esmfold",
        description="Tool to use: 'esmfold', 'alphafold3', 'boltz', or 'chai'.",
    )
    tool_config: Dict[str, Any] = ConfigField(
        title="Tool Config",
        default_factory=dict,
        description="Dictionary of configuration parameters passed to the specific tool.",
        advanced=True,
    )

    min_target_plddt: float = ConfigField(
        title="Min Target pLDDT",
        default=0.6,
        description="Minimum confidence for the target if it is folded from sequence.",
    )

    @model_validator(mode="after")
    def validate_target(self) -> StructureConstraintBaseConfig:
        """Ensure exactly one target source is provided."""
        sources = [self.target_chains, self.target_pdb_file, self.target_pdb_content]
        provided = sum(s is not None for s in sources)
        if provided != 1:
            raise ValueError(
                "Exactly one of 'target_chains', 'target_pdb_file', or 'target_pdb_content' "
                "must be provided."
            )
        return self


class StructureRMSDConfig(StructureConstraintBaseConfig):
    """
    Configuration for RMSD-based structure similarity.

    This configuration extends `StructureConstraintBaseConfig` to specific parameters
    for calculating the Root Mean Square Deviation (RMSD) between the target and
    candidate structures. The raw RMSD value is transformed into a 0-1 constraint
    score using a sigmoid function, where 0 represents a perfect match (low RMSD)
    and 1 represents a poor match (high RMSD).

    Attributes:
        inflection_point_angstroms (float):
            The RMSD value (in Angstroms) at which the sigmoid scoring function
            returns 0.5. RMSD values significantly lower than this point will yield
            scores close to 0 (good), while values higher will yield scores close to 1 (bad).
            A value < 2.0 is generally considered a good structural match. Default is 2.0.

        sigmoid_slope (float):
            The steepness of the sigmoid penalty curve. A higher slope results in a
            sharper transition from good to bad scores around the inflection point.
            Default is 3.0.

        target_chains (Optional[Tuple[str]]):
            Inherited from `StructureConstraintBaseConfig`.

        target_pdb_file (Optional[str]):
            Inherited from `StructureConstraintBaseConfig`.

        target_pdb_content (Optional[str]):
            Inherited from `StructureConstraintBaseConfig`.

        structure_tool (str):
            Inherited from `StructureConstraintBaseConfig`.

        tool_config (Dict[str, Any]):
            Inherited from `StructureConstraintBaseConfig`.

        min_target_plddt (float):
            Inherited from `StructureConstraintBaseConfig`.
    """

    inflection_point_angstroms: float = ConfigField(
        title="RMSD Inflection Point",
        default=2.0,
        description="RMSD (Angstroms) where score is 0.5. < 2.0 is good.",
    )
    sigmoid_slope: float = ConfigField(
        title="Sigmoid Slope",
        default=3.0,
        description="Steepness of the penalty curve.",
    )


class StructureTMScoreConfig(StructureConstraintBaseConfig):
    """
    Configuration for TM-score based structure similarity.

    This configuration extends `StructureConstraintBaseConfig` for calculating the
    Template Modeling score (TM-score) between the target and candidate structures.
    TM-score is a metric for assessing the topological similarity of protein structures
    and is less sensitive to local variations than RMSD.

    The constraint returns a score calculated as (1.0 - TM_score), where 0.0 indicates
    a perfect match (TM-score = 1.0) and values closer to 1.0 indicate poor structural
    similarity.

    Attributes:
        plddt_threshold (Optional[float]):
            If provided, this will first filter out atoms in the predicted structure
            with pLDDT less than this threshold. Defaults to ``None``.

        target_chains (Optional[Tuple[str]]):
            Inherited from `StructureConstraintBaseConfig`.

        target_pdb_file (Optional[str]):
            Inherited from `StructureConstraintBaseConfig`.

        target_pdb_content (Optional[str]):
            Inherited from `StructureConstraintBaseConfig`.

        structure_tool (str):
            Inherited from `StructureConstraintBaseConfig`.

        tool_config (Dict[str, Any]):
            Inherited from `StructureConstraintBaseConfig`.

        min_target_plddt (float):
            Inherited from `StructureConstraintBaseConfig`.
    """
    plddt_threshold: Optional[float] = ConfigField(
        title="pLDDT Threshold",
        default=None,
        description="Ignore residues in the candidate with pLDDT < threshold (e.g. 70).",
    )


# ============================================================================
# Constraints
# ============================================================================

def _prepare_target_structure(config: StructureConstraintBaseConfig) -> Optional[str]:
    """
    Resolve the target structure to a PDB string.
    If target is a sequence, it folds it (as a monomer).
    """
    # The user just provides the full PDB content.
    if config.target_pdb_content:
        return config.target_pdb_content

    # The user provided a path to a PDB file.
    if config.target_pdb_file:
        with open(config.target_pdb_file, 'r') as f:
            return f.read()

    # The user provided a list of protein sequences.
    if config.target_chains:
        complexes = [
            StructurePredictionComplex(
                chains=config.target_chains,
                entity_types=["protein"] * len(config.target_chains),
            )
        ]

        try:
            output = predict_structures(complexes, config.structure_tool, config.tool_config)
        except Exception as e:
            logger.error(f"Failed to fold target sequence: {e}")
            return None

        # Check confidence.
        if output.structures[0].avg_plddt < config.min_target_plddt:
            logger.warning(
                f"Target fold confidence ({output.structures[0].avg_plddt:.2f}) "
                f"below threshold ({config.min_target_plddt})."
            )
            return None

        return output.structures[0].structure_pdb

    return None


@ConstraintRegistry.register(
    key="structure-rmsd",
    label="Structural RMSD Similarity",
    config=StructureRMSDConfig,
    description="Compare structure RMSD against a target (PDB or Sequence) using generic predictors.",
    batched=True,
    concatenate=False,  # Input is List[Tuple[Sequence, ...]]
    gpu_required=True,
    tools_called=["esmfold", "alphafold3", "boltz", "chai", "pymol"],
    category="protein_structure",
)
def structure_rmsd_constraint(
    candidates: List[Tuple[Sequence, ...]], config: StructureRMSDConfig
) -> List[float]:
    """
    Predicts structure of input candidates and compares RMSD against a target.
    Returns a score 0-1 (0 is perfect match).
    """

    # Prepare target.
    target_pdb = _prepare_target_structure(config)
    if not target_pdb:
        logger.warning("Target preparation failed, returning worst score.")
        return [1.0] * len(candidates)

    # Prepare candidates.
    complexes = []
    for candidate_tuple in candidates:
        # Extract sequences and types
        chain_seqs = [s.sequence for s in candidate_tuple]
        chain_types = [s.sequence_type for s in candidate_tuple]

        complexes.append(
            StructurePredictionComplex(chains=chain_seqs, entity_types=chain_types)
        )

    # Run prediction on candidates.
    try:
        results = predict_structures(complexes, config.structure_tool, config.tool_config)
    except Exception as e:
        logger.error(f"Structure prediction failed: {e}")
        return [MAX_ENERGY] * len(candidates)

    # Compute RMSD scores.
    scores = []
    for candidate_structure, candidate_tuple in zip(results.structures, candidates):
        rmsd_data = _compute_ce_aligned_rmsd(target_pdb, candidate_structure.structure_pdb)
        rmsd_val = rmsd_data['rmsd']

        score = sigmoid_score(
            rmsd_val, config.inflection_point_angstroms, config.sigmoid_slope
        )

        # Metadata storage (attach to the first sequence in the tuple to ensure visibility)
        if candidate_tuple:
            candidate_tuple[0]._metadata.update({
                "rmsd_val": rmsd_val,
                "rmsd_score": score,
                "pdb_output": candidate_structure.structure_pdb
            })

        scores.append(score)

    return scores


@ConstraintRegistry.register(
    key="structure-tmscore",
    label="Structural TM-score Similarity",
    config=StructureTMScoreConfig,
    description="Compare structure TM-score against a target. Returns 1 - TMscore.",
    batched=True,
    concatenate=False,  # Input is List[Tuple[Sequence, ...]]
    gpu_required=True,
    tools_called=["esmfold", "alphafold3", "boltz", "chai", "tmalign"],
    category="protein_structure",
)
def structure_tmscore_constraint(
    candidates: List[Tuple[Sequence, ...]], config: StructureTMScoreConfig
) -> List[float]:
    """
    Predicts structure and compares TM-score. Returns (1.0 - TMscore).
    """

    # Prepare target.
    target_pdb = _prepare_target_structure(config)
    if not target_pdb:
        logger.warning("Target preparation failed, returning worst score.")
        return [1.0] * len(candidates)

    # Prepare candidates.
    complexes = []
    for candidate_tuple in candidates:
        chain_seqs = [s.sequence for s in candidate_tuple]
        chain_types = [s.sequence_type for s in candidate_tuple]

        complexes.append(
            StructurePredictionComplex(chains=chain_seqs, entity_types=chain_types)
        )

    # Run prediction on candidates.
    try:
        results = predict_structures(complexes, config.structure_tool, config.tool_config)
    except Exception as e:
        logger.error(f"Structure prediction failed: {e}")
        return [MAX_ENERGY] * len(candidates)

    # Compute TMscores.
    scores = []
    for candidate_structure, candidate_tuple in zip(results.structures, candidates):
        tm_val = _compute_tm_score_from_pdb(
            target_pdb,
            candidate_structure.structure_pdb,
            plddt_threshold=config.plddt_threshold,
        )
        score = 1.0 - tm_val

        if candidate_tuple:
            candidate_tuple[0]._metadata.update({
                "tm_score_raw": tm_val,
                "tm_score_inverted": score,
                "pdb_output": candidate_structure.structure_pdb
            })

        scores.append(score)

    return scores
