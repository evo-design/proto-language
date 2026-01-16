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
from typing import Optional, List, Dict, Any, Tuple, Literal
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
from proto_language.utils.helpers import predict_structures

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


def _compute_tmalign_score_from_pdb(
    target_pdb_text: str,
    candidate_pdb_text: str,
    plddt_threshold: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute TM-score using the 'TMalign' binary.

    Returns a tuple of two floats with, respectively:
        - 'tm_score_1': TM-score normalized by length of structure 1 (candidate)
        - 'tm_score_2': TM-score normalized by length of structure 2 (target)
    """
    tmalign_path = shutil.which("TMalign")
    if not tmalign_path:
        raise ImportError(
            "The 'TMalign' binary is required for TM-score constraints. "
            "Please install it (e.g., via Conda):\n\n"
            "  conda install -c bioconda tmalign\n"
        )

    if plddt_threshold is not None:
        candidate_pdb_text = _filter_pdb_by_plddt(
            candidate_pdb_text,
            plddt_threshold
        )
        if not any(
            line.startswith("ATOM") for line in candidate_pdb_text.splitlines()
        ):
            return (0., 0.)

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

        # Parse both TM-scores from output
        # TMalign outputs:
        #   TM-score= X.XXXX (if normalized by length of Chain_1, i.e., the first structure)
        #   TM-score= X.XXXX (if normalized by length of Chain_2, i.e., the second structure)

        tm_score_1 = None
        tm_score_2 = None

        match_chain1 = re.search(
            r"TM-score=\s*([0-9.]+)\s+\(if normalized by length of Chain_1", output
        )
        match_chain2 = re.search(
            r"TM-score=\s*([0-9.]+)\s+\(if normalized by length of Chain_2", output
        )

        if match_chain1:
            tm_score_1 = float(match_chain1.group(1))
        if match_chain2:
            tm_score_2 = float(match_chain2.group(1))

        # Fallback: parse all TM-scores in order if specific patterns not found
        if tm_score_1 is None or tm_score_2 is None:
            matches = re.findall(r"TM-score=\s*([0-9.]+)", output)
            if len(matches) >= 2:
                tm_score_1 = tm_score_1 \
                    if tm_score_1 is not None else float(matches[0])
                tm_score_2 = tm_score_2 \
                    if tm_score_2 is not None else float(matches[1])
            elif len(matches) == 1:
                # Only one score found, use it for both
                tm_score_1 = tm_score_2 = float(matches[0])
            else:
                logger.warning(
                    "Could not find TMscore in TMalign output, "
                    "returning worst value"
                )
                tm_score_1 = tm_score_2 = 0.0

        return (tm_score_1, tm_score_2)

    except subprocess.CalledProcessError as e:
        logger.warning(f"TMalign execution failed: {e}, returning worst value")
        return (0., 0.)
    finally:
        if os.path.exists(target_path):
            os.unlink(target_path)
        if os.path.exists(candidate_path):
            os.unlink(candidate_path)


def _compute_usalign_score_from_pdb(
    target_pdb_text: str,
    candidate_pdb_text: str,
    plddt_threshold: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute TM-score using 'USalign' for multimers.

    Returns a tuple of two floats with, respectively:
        - 'tm_score_1': TM-score normalized by length of structure 1 (candidate)
        - 'tm_score_2': TM-score normalized by length of structure 2 (target)
    """
    usalign_path = shutil.which("USalign")
    if not usalign_path:
        raise ImportError(
            "The 'USalign' binary is required for multimer structural alignment. "
            "Please install it (e.g., via Conda):\n\n"
            "  conda install -c bioconda usalign\n"
        )

    if plddt_threshold is not None:
        candidate_pdb_text = _filter_pdb_by_plddt(
            candidate_pdb_text,
            plddt_threshold
        )
        if not any(
            line.startswith("ATOM") for line in candidate_pdb_text.splitlines()
        ):
            return (0., 0.)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f_target:
        f_target.write(target_pdb_text)
        target_path = f_target.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f_candidate:
        f_candidate.write(candidate_pdb_text)
        candidate_path = f_candidate.name

    try:
        cmd = [usalign_path, candidate_path, target_path, "-mm", "1", "-ter", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse both TM-scores
        # USalign outputs:
        #   TM-score= X.XXXX (normalized by length of Structure_1: ...)
        #   TM-score= X.XXXX (normalized by length of Structure_2: ...)

        tm_score_1 = None
        tm_score_2 = None

        match_struct1 = re.search(
            r"TM-score=\s*([0-9.]+)\s+\(normalized by length of Structure_1", output
        )
        match_struct2 = re.search(
            r"TM-score=\s*([0-9.]+)\s+\(normalized by length of Structure_2", output
        )

        if match_struct1:
            tm_score_1 = float(match_struct1.group(1))
        if match_struct2:
            tm_score_2 = float(match_struct2.group(1))

        # Fallback
        if tm_score_1 is None or tm_score_2 is None:
            matches = re.findall(r"TM-score=\s*([0-9.]+)", output)
            if len(matches) >= 2:
                tm_score_1 = tm_score_1 \
                    if tm_score_1 is not None else float(matches[0])
                tm_score_2 = tm_score_2 \
                    if tm_score_2 is not None else float(matches[1])
            elif len(matches) == 1:
                tm_score_1 = tm_score_2 = float(matches[0])
            else:
                logger.warning("Could not find TM-score in USalign output")
                tm_score_1 = tm_score_2 = 0.0

        return (tm_score_1, tm_score_2)

    except subprocess.CalledProcessError as e:
        logger.warning(f"USalign execution failed: {e}, returning worst value")
        return (0., 0.)
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

        structure_tool (Literal["esmfold", "alphafold3", "boltz", "chai"]):
            The structure prediction tool to use for folding both the target (if provided
            as a sequence) and the candidate sequences. Supported options:
            - "esmfold": ESMFold (Meta AI)
            - "alphafold3": AlphaFold 3 (Google DeepMind)
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
    structure_tool: Literal["esmfold", "alphafold3", "boltz", "chai"] = ConfigField(
        title="Structure Prediction Tool",
        default="esmfold",
        description="Tool to use for structure prediction.",
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

        structure_tool (Literal["esmfold", "alphafold3", "boltz", "chai"]):
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

        tm_score_normalization (Literal["structure1", "structure2", "max", "min", "mean"]):
            How to select or combine the two TM-scores (normalized by different structure
            lengths). Importantly, the ``target_chains`` are passed as the second structure
            to the alignment programs. Options:
            - "structure1": Use TM-score normalized by candidate structure length.
            - "structure2": Use TM-score normalized by target structure length.
            - "max": Take the maximum of both TM-scores (most lenient).
            - "min": Take the minimum of both TM-scores (most strict).
            - "mean": Take the arithmetic mean of both TM-scores (default).
            Default is "mean".

        target_chains (Optional[Tuple[str]]):
            Inherited from `StructureConstraintBaseConfig`.

        target_pdb_file (Optional[str]):
            Inherited from `StructureConstraintBaseConfig`.

        target_pdb_content (Optional[str]):
            Inherited from `StructureConstraintBaseConfig`.

        structure_tool (Literal["esmfold", "alphafold3", "boltz", "chai"]):
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
    tm_score_normalization: Literal[
        "structure1", "structure2", "max", "min", "mean"
    ] = ConfigField(
        title="TM-score Normalization",
        default="mean",
        description=(
            "How to handle the two TM-scores returned by TMalign/USalign."
        ),
    )

    @model_validator(mode="after")
    def validate_normalization(self) -> StructureTMScoreConfig:
        """Validate TMscore normalization field."""
        valid = ["structure1", "structure2", "max", "min", "mean"]
        if self.tm_score_normalization not in valid:
            raise ValueError(
                f"Invalid TMscore normalization mode {self.tm_score_normalization}, "
                f"valid options are: {', '.join(valid)}"
            )
        return self


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
    supported_sequence_types=["protein"],
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


def _count_pdb_chains(pdb_text: str) -> int:
    """
    Counts unique chain identifiers in PDB text to determine oligomer state.
    """
    chains = set()
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Chain ID is in column 22 (index 21)
            if len(line) > 21:
                chains.add(line[21])
    return len(chains) if chains else 1


@ConstraintRegistry.register(
    key="structure-tmscore",
    label="Structural TM-score Similarity",
    config=StructureTMScoreConfig,
    description="Compare structure TM-score against a target. Returns 1 - TMscore.",
    batched=True,
    concatenate=False,  # Input is List[Tuple[Sequence, ...]]
    gpu_required=True,
    tools_called=["esmfold", "alphafold3", "boltz", "chai", "tmalign", "usalign"],
    category="protein_structure",
    supported_sequence_types=["protein"],
)
def structure_tmscore_constraint(
    candidates: List[Tuple[Sequence, ...]], config: StructureTMScoreConfig
) -> List[float]:
    """
    Predicts structure and compares TM-score. Returns (1.0 - TMscore).

    This constraint automatically selects the appropriate alignment tool based on
    the oligomer state of the inputs:
    - Monomer vs monomer comparisons use standard `TMalign`.
    - Comparisons involving multiple chains use `USalign` with `-mm 1` and default
      values for all other parameters.

    Note:
    All TM-scores are normalized by the length of the **target** structure. This
    can help ensure consistent scoring magnitude across evaluations.
    """

    # Prepare target.
    target_pdb = _prepare_target_structure(config)
    if not target_pdb:
        logger.warning("Target preparation failed, returning worst score.")
        return [1.0] * len(candidates)

    n_target_chains = _count_pdb_chains(target_pdb)

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
        n_cand_chains = len(candidate_tuple)

        if n_target_chains == 1 and n_cand_chains == 1:
            # Monomer vs monomer uses standard TMalign.
            s1, s2 = _compute_tmalign_score_from_pdb(
                target_pdb,
                candidate_structure.structure_pdb,
                plddt_threshold=config.plddt_threshold,
            )
        else:
            # USalign is needed for multimer comparison.
            s1, s2 = _compute_usalign_score_from_pdb(
                target_pdb,
                candidate_structure.structure_pdb,
                plddt_threshold=config.plddt_threshold
            )

        if config.tm_score_normalization == "structure1":
            tm_val = s1
        elif config.tm_score_normalization == "structure2":
            tm_val = s2
        elif config.tm_score_normalization == "max":
            tm_val = max(s1, s2)
        elif config.tm_score_normalization == "min":
            tm_val = min(s1, s2)
        elif config.tm_score_normalization == "mean":
            tm_val = (s1 + s2) / 2.0
        else:
            raise ValueError(
                f"Invalid TMscore normalization: {config.tm_score_normalization}"
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
