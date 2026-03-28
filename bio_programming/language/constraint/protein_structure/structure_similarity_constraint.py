"""
proto_language/language/constraint/protein_structure/structure_similarity_constraint.py

Contains implementation of generic structure similarity constraints (RMSD, TM-score)
supporting multiple structure prediction tools (ESMFold, AlphaFold3, Boltz, Chai1).
"""

from __future__ import annotations

import os
import tempfile
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Tuple

from proto_tools import (
    Structure,
    StructurePredictionComplex,
    TMalignConfig,
    TMalignInput,
    USalignConfig,
    USalignInput,
    predict_structures,
    run_tmalign,
    run_usalign,
)
from pydantic import model_validator

from proto_language.base_config import ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.language.core import Sequence
from proto_language.storage import FileType, store_file
from proto_language.utils import MAX_ENERGY, sigmoid_score

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


# ============================================================================
# Configuration
# ============================================================================

class StructureSimilarityConfig(StructureBasedConstraintConfig):
    """Base configuration for structure similarity constraints.

    This configuration manages the setup for predicting protein structures from
    proposal sequences and defining the target structure against which proposals
    are compared. It supports defining targets via direct sequence folding or
    by providing an existing structure.

    The user should provide a target as **one** of:
    - `target_chains`: Sequences to dynamically fold (tuple of strings or a
      StructurePredictionComplex).
    - `target_structure`: A Structure object, a file path to a PDB/CIF file,
      or raw PDB/CIF content as a string.

    Inherits tool selection and configuration from StructureBasedConstraintConfig:

        structure_tool (Literal['esmfold', 'alphafold3', 'boltz2', 'chai1']):
            The structure prediction tool to use for folding both the target (if provided
            as a sequence) and the proposal sequences. Supported options:
            - "esmfold": ESMFold (Meta AI)
            - "alphafold3": AlphaFold 3 (Google DeepMind)
            - "boltz2": Boltz2 (MIT)
            - "chai1": Chai-1 (Chai Discovery)
            Default is "esmfold".

        tool_config (dict[str, Any] | ESMFoldConfig | AlphaFold3Config | Boltz2Config | Chai1Config | None):
            A dictionary of configuration parameters to pass directly to the underlying
            structure prediction tool runner. Can be a typed config object or a dictionary.
            Automatically validated and converted to the appropriate config type based on
            structure_tool. Defaults to an empty dictionary.

    Attributes:

        target_chains (tuple[str, ...] | StructurePredictionComplex | None):
            The sequences of the target chains. Accepts either a tuple of sequence
            strings (entity types are auto-detected) or a StructurePredictionComplex.
            If provided, these will be folded using the specified `structure_tool`
            to generate the reference structure. Mutually exclusive with
            `target_structure`.

        target_structure (Structure | str | None):
            The target structure. Accepts a Structure object, a file path to a
            PDB/CIF file (identified by .pdb/.cif/.mmcif extension), or raw
            PDB/CIF content as a string. Mutually exclusive with `target_chains`.

        min_target_plddt (float):
            Only used if the target structure is provided via `target_chains`. This is
            the minimum average pLDDT confidence score required for the folded target
            structure. If the target is provided as a sequence and its predicted
            structure has a confidence below this threshold, the constraint may return
            a default/penalty score or log a warning. Default is 0.6.
        structure_tool (Literal['esmfold', 'alphafold3', 'boltz2', 'chai1']): Structure prediction tool key to use.
        tool_config (dict[str, Any] | ESMFoldConfig | AlphaFold3Config | Boltz2Config | Chai1Config | None): Configuration dict passed to the structure prediction tool.
    """

    # Target specification (mutually exclusive):
    target_chains: Optional[Tuple[str, ...] | StructurePredictionComplex] = ConfigField(
        title="Target Chains",
        default=None,
        description="Target chains: a tuple of sequence strings (entity types auto-detected).",
    )
    target_structure: Optional[Structure | str] = ConfigField(
        title="Target Structure",
        default=None,
        description="Target structure: a Structure object, file path (.pdb/.cif), or raw PDB/CIF content string.",
    )

    min_target_plddt: float = ConfigField(
        title="Min Target pLDDT",
        default=0.6,
        description="Minimum confidence for the target if it is folded from sequence.",
        depends_on={"field": "target_chains", "not_null": True},
    )

    @model_validator(mode="after")
    def validate_target(self) -> StructureSimilarityConfig:
        """Ensure exactly one target source is provided."""
        sources = [self.target_chains, self.target_structure]
        provided = sum(s is not None for s in sources)
        if provided != 1:
            raise ValueError("Exactly one of 'target_chains' or 'target_structure' must be provided.")
        return self


class StructureRMSDConfig(StructureSimilarityConfig):
    """
    Configuration for RMSD-based structure similarity.

    This configuration extends `StructureSimilarityConfig` with specific parameters
    for calculating the Root Mean Square Deviation (RMSD) between the target and
    proposal structures. The raw RMSD value is transformed into a 0-1 constraint
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

        target_chains (tuple[str, ...] | StructurePredictionComplex | None):
            The sequences of the target chains. Accepts a tuple of sequence strings
            (entity types auto-detected) or a StructurePredictionComplex. If provided,
            these will be folded using the specified `structure_tool` to generate the
            reference structure. Mutually exclusive with `target_structure`.

        target_structure (Structure | str | None):
            The target structure. Accepts a Structure object, a file path to a
            PDB/CIF file, or raw PDB/CIF content as a string. Mutually exclusive
            with `target_chains`.

        structure_tool:
            The structure prediction tool to use for folding both the target (if provided
            as a sequence) and the proposal sequences. Supported options:
            - "esmfold": ESMFold (Meta AI)
            - "alphafold3": AlphaFold 3 (Google DeepMind)
            - "boltz2": Boltz2 (MIT)
            - "chai1": Chai-1 (Chai Discovery)
            Default is "esmfold".

        tool_config:
            A dictionary of configuration parameters to pass directly to the underlying
            structure prediction tool runner. Can be a typed config object or a dictionary.
            Automatically validated and converted to the appropriate config type based on
            structure_tool. Defaults to an empty dictionary.

        min_target_plddt (float):
            Only used if the target structure is provided via `target_chains`. This is
            the minimum average pLDDT confidence score required for the folded target
            structure. If the target is provided as a sequence and its predicted
            structure has a confidence below this threshold, the constraint may return
            a default/penalty score or log a warning. Default is 0.6.
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


class StructureTMScoreConfig(StructureSimilarityConfig):
    """
    Configuration for TM-score based structure similarity.

    This configuration extends `StructureSimilarityConfig` for calculating the
    Template Modeling score (TM-score) between the target and proposal structures.
    TM-score is a metric for assessing the topological similarity of protein structures
    and is less sensitive to local variations than RMSD.

    The constraint returns a score calculated as (1.0 - TM_score), where 0.0 indicates
    a perfect match (TM-score = 1.0) and values closer to 1.0 indicate poor structural
    similarity.

    Attributes:
        plddt_threshold (float | None):
            If provided, this will first filter out atoms in the predicted structure
            with pLDDT less than this threshold. Defaults to ``None``.

        tm_score_normalization (Literal['structure1', 'structure2', 'max', 'min', 'mean']):
            How to select or combine the two TM-scores (normalized by different structure
            lengths). Importantly, the ``target_chains`` are passed as the second structure
            to the alignment programs. Options:
            - "structure1": Use TM-score normalized by proposal structure length.
            - "structure2": Use TM-score normalized by target structure length.
            - "max": Take the maximum of both TM-scores (most lenient).
            - "min": Take the minimum of both TM-scores (most strict).
            - "mean": Take the arithmetic mean of both TM-scores (default).
            Default is "mean".

        target_chains (tuple[str, ...] | StructurePredictionComplex | None):
            The sequences of the target chains. Accepts a tuple of sequence strings
            (entity types auto-detected) or a StructurePredictionComplex. If provided,
            these will be folded using the specified `structure_tool` to generate the
            reference structure. Mutually exclusive with `target_structure`.

        target_structure (Structure | str | None):
            The target structure. Accepts a Structure object, a file path to a
            PDB/CIF file, or raw PDB/CIF content as a string. Mutually exclusive
            with `target_chains`.

        structure_tool:
            The structure prediction tool to use for folding both the target (if provided
            as a sequence) and the proposal sequences. Supported options:
            - "esmfold": ESMFold (Meta AI)
            - "alphafold3": AlphaFold 3 (Google DeepMind)
            - "boltz2": Boltz2 (MIT)
            - "chai1": Chai-1 (Chai Discovery)
            Default is "esmfold".

        tool_config:
            A dictionary of configuration parameters to pass directly to the underlying
            structure prediction tool runner. Can be a typed config object or a dictionary.
            Automatically validated and converted to the appropriate config type based on
            structure_tool. Defaults to an empty dictionary.

        min_target_plddt (float):
            Only used if the target structure is provided via `target_chains`. This is
            the minimum average pLDDT confidence score required for the folded target
            structure. If the target is provided as a sequence and its predicted
            structure has a confidence below this threshold, the constraint may return
            a default/penalty score or log a warning. Default is 0.6.
    """
    plddt_threshold: Optional[float] = ConfigField(
        title="pLDDT Threshold",
        default=None,
        description="Ignore residues in the proposal with pLDDT < threshold (e.g. 70).",
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

def _prepare_target_structure(config: StructureSimilarityConfig) -> Optional[str]:
    """
    Resolve the target structure to a PDB string.
    If target is a sequence, it folds it.
    """
    if config.target_structure is not None:
        if isinstance(config.target_structure, Structure):
            return config.target_structure.structure_pdb
        return Structure(config.target_structure).structure_pdb

    if config.target_chains is not None:
        if isinstance(config.target_chains, StructurePredictionComplex):
            complexes = [config.target_chains]
        else:
            from proto_language.language.core import detect_sequence_type

            chains = [
                {"sequence": seq, "entity_type": detect_sequence_type(seq)}
                for seq in config.target_chains
            ]
            complexes = [StructurePredictionComplex(chains=chains)]

        output = predict_structures(complexes, config.structure_tool, config.tool_config)

        if output.structures[0].avg_plddt < config.min_target_plddt:
            logger.warning(
                f"Target fold confidence ({output.structures[0].avg_plddt:.2f}) "
                f"below threshold ({config.min_target_plddt})."
            )
            return None

        return output.structures[0].structure_pdb

    return None


@constraint(
    key="structure-rmsd",
    label="Structural RMSD Similarity",
    config=StructureRMSDConfig,
    description="Compare structure RMSD against a target (PDB or Sequence) using generic predictors.",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "alphafold3-prediction", "boltz2-prediction", "chai1-prediction", "pymol"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    num_input_sequences_per_tuple=None,
)
def structure_rmsd_constraint(
    input_sequences: List[Tuple[Sequence, ...]], config: StructureRMSDConfig
) -> List[float]:
    """
    Predicts structure of input proposals and compares RMSD against a target.
    Returns a score 0-1 (0 is perfect match).
    """

    # Prepare target.
    target_pdb = _prepare_target_structure(config)
    if not target_pdb:
        logger.warning("Target preparation failed, returning worst score.")
        return [1.0] * len(input_sequences)

    # Prepare proposals.
    structure_complexes = []
    for proposal_tuple in input_sequences:
        # Extract sequences and types
        chains = [
            {"sequence": s.sequence, "entity_type": s.sequence_type}
            for s in proposal_tuple
        ]
        structure_complexes.append(StructurePredictionComplex(chains=chains))

    # Run prediction on proposals.
    try:
        results = predict_structures(structure_complexes, config.structure_tool, config.tool_config)
    except Exception as e:
        logger.error(f"Structure prediction failed: {e}")
        return [MAX_ENERGY] * len(input_sequences)

    # Compute RMSD scores.
    scores = []
    for proposal_structure, proposal_tuple in zip(results.structures, input_sequences):
        rmsd_data = _compute_ce_aligned_rmsd(target_pdb, proposal_structure.structure_pdb)
        rmsd_val = rmsd_data['rmsd']

        score = sigmoid_score(
            rmsd_val, config.inflection_point_angstroms, config.sigmoid_slope
        )

        # Metadata storage (attach to the first sequence in the tuple to ensure visibility)
        if proposal_tuple:
            proposal_tuple[0]._metadata.update({
                "rmsd_val": rmsd_val,
                "rmsd_score": score,
                "pdb_output": store_file(proposal_structure.structure_pdb, FileType.PDB),
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


@constraint(
    key="structure-tmscore",
    label="Structural TM-score Similarity",
    config=StructureTMScoreConfig,
    description="Compare structure TM-score against a target. Returns 1 - TMscore.",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "alphafold3-prediction", "boltz2-prediction", "chai1-prediction", "tmalign-alignment", "usalign-alignment"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    num_input_sequences_per_tuple=None,
)
def structure_tmscore_constraint(
    input_sequences: List[Tuple[Sequence, ...]], config: StructureTMScoreConfig
) -> List[float]:
    """
    Predicts structure and compares TM-score. Returns (1.0 - TMscore).

    This constraint automatically selects the appropriate alignment tool based on
    the oligomer state of the inputs:
    - Monomer vs monomer comparisons use standard `TMalign`.
    - Comparisons involving multiple chains use `USalign` with `-mm 1` and default
      values for all other parameters.

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (StructureTMScoreConfig): Constraint configuration controlling evaluation parameters.

    Note:
    All TM-scores are normalized by the length of the **target** structure. This
    can help ensure consistent scoring magnitude across evaluations.
    """

    # Prepare target.
    target_pdb = _prepare_target_structure(config)
    if not target_pdb:
        logger.warning("Target preparation failed, returning worst score.")
        return [1.0] * len(input_sequences)

    n_target_chains = _count_pdb_chains(target_pdb)

    # Prepare proposals.
    structure_complexes = []
    for proposal_tuple in input_sequences:
        chains = [
            {"sequence": s.sequence, "entity_type": s.sequence_type}
            for s in proposal_tuple
        ]
        structure_complexes.append(StructurePredictionComplex(chains=chains))

    # Run prediction on proposals.
    try:
        results = predict_structures(structure_complexes, config.structure_tool, config.tool_config)
    except Exception as e:
        logger.error(f"Structure prediction failed: {e}")
        return [MAX_ENERGY] * len(input_sequences)

    # Compute TMscores.
    scores = []
    for proposal_structure, proposal_tuple in zip(results.structures, input_sequences):
        n_cand_chains = len(proposal_tuple)

        # Apply pLDDT filtering at the constraint level before alignment.
        proposal_pdb = proposal_structure.structure_pdb
        if config.plddt_threshold is not None:
            proposal_pdb = _filter_pdb_by_plddt(
                proposal_pdb, config.plddt_threshold
            )
            if not any(
                line.startswith("ATOM") for line in proposal_pdb.splitlines()
            ):
                scores.append(1.0)
                continue

        if n_target_chains == 1 and n_cand_chains == 1:
            _tmalign_out = run_tmalign(
                TMalignInput(
                    pdb_text_1=proposal_pdb,
                    pdb_text_2=target_pdb,
                ),
                TMalignConfig(),
            )
            if _tmalign_out.success is False:
                logger.warning(f"TMalign failed: {_tmalign_out.errors}")
                scores.append(1.0)
                continue
            s1, s2 = _tmalign_out.tm_score_chain_1, _tmalign_out.tm_score_chain_2
        else:
            _usalign_out = run_usalign(
                USalignInput(
                    pdb_text_1=proposal_pdb,
                    pdb_text_2=target_pdb,
                ),
                USalignConfig(),
            )
            if _usalign_out.success is False:
                logger.warning(f"USalign failed: {_usalign_out.errors}")
                scores.append(1.0)
                continue
            s1, s2 = _usalign_out.tm_score_structure_1, _usalign_out.tm_score_structure_2

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

        if proposal_tuple:
            proposal_tuple[0]._metadata.update({
                "tm_score_raw": tm_val,
                "tm_score_inverted": score,
                "pdb_output": store_file(proposal_structure.structure_pdb, FileType.PDB),
            })

        scores.append(score)

    return scores
