"""Contains implementation of generic structure similarity constraints (RMSD, TM-score).

supporting multiple structure prediction tools (ESMFold, ESMFold2, AlphaFold3, Boltz2, Chai1, Protenix, AlphaFold2).
"""

from logging import getLogger
from typing import Literal

from proto_tools import (
    Complex,
    PyMOLRMSDConfig,
    PyMOLRMSDInput,
    Structure,
    TMalignConfig,
    TMalignInput,
    USalignConfig,
    USalignInput,
    predict_structures,
    run_pymol_rmsd_alignment,
    run_tmalign,
    run_usalign,
)
from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
    resolve_metric,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, sigmoid_score
from proto_language.utils.base import ConfigField

logger = getLogger(__name__)


def _filter_pdb_by_plddt(pdb_text: str, threshold: float) -> str:
    """Filters PDB text, keeping only residues with B-factor (pLDDT) >= threshold."""
    if threshold <= 0:
        return pdb_text

    filtered_lines = []
    for line in pdb_text.splitlines():
        # PDB ATOM records: B-factor is columns 61-66 (index 60:66)
        if line.startswith(("ATOM", "HETATM")):
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
    - ``target_chains``: Sequences to dynamically fold (tuple of strings or a
      Complex).
    - ``target_structure``: A Structure object, a file path to a PDB/CIF file,
      or raw PDB/CIF content as a string.

    Inherits tool selection and per-tool configuration from
    ``StructureBasedConstraintConfig`` (``structure_tool``, ``esmfold_config``,
    ``esmfold2_config``, ``alphafold3_config``, ``boltz2_config``, ``chai1_config``,
    ``protenix_config``, ``alphafold2_config``, ``alphafold2_binder_config``).

    Attributes:
        target_chains (tuple[str, ...] | Complex | None):
            The sequences of the target chains. Accepts either a tuple of sequence
            strings (entity types are auto-detected) or a Complex.
            If provided, these will be folded using the specified ``structure_tool``
            to generate the reference structure. Mutually exclusive with
            ``target_structure``.

        target_structure (Structure | str | None):
            The target structure. Accepts a Structure object, a file path to a
            PDB/CIF file (identified by .pdb/.cif/.mmcif extension), or raw
            PDB/CIF content as a string. Mutually exclusive with ``target_chains``.

        min_target_plddt (float):
            Only used if the target structure is provided via ``target_chains``. This is
            the minimum average pLDDT confidence score required for the folded target
            structure. If the target is provided as a sequence and its predicted
            structure has a confidence below this threshold, the constraint may return
            a default/penalty score or log a warning. Default is 0.6.
    """

    # Target specification (mutually exclusive):
    target_chains: tuple[str, ...] | Complex | None = ConfigField(
        title="Target Chains",
        default=None,
        description="Target chains: a tuple of sequence strings (entity types auto-detected).",
    )
    target_structure: Structure | str | None = ConfigField(
        title="Target Structure",
        default=None,
        description="Target structure: a Structure object, file path (.pdb/.cif), or raw PDB/CIF content string.",
    )

    @field_validator("target_structure", mode="before")
    @classmethod
    def _coerce_empty_structure_stub(cls, v: object) -> object:
        """Coerce an empty structure widget stub to None.

        The client emits a dict with only optional Structure fields
        (e.g. ``{"b_factor_type": "unspecified", "metrics": {}}``) when a
        structure input widget is toggled on but left unfilled. That dict fails
        validation as Structure (missing required ``structure`` field), str, and
        None. Treat any dict lacking a non-empty ``structure`` key as None so
        the field behaves as if it was never set.
        """
        if isinstance(v, dict) and not v.get("structure"):
            return None
        return v

    min_target_plddt: float = ConfigField(
        title="Min Target pLDDT",
        default=0.6,
        description="Min mean pLDDT (0-1 scale) for a target folded from sequence; ignored when target_structure is set.",
    )

    @model_validator(mode="after")
    def validate_target(self) -> "StructureSimilarityConfig":
        """Ensure exactly one target source is provided."""
        sources = [self.target_chains, self.target_structure]
        provided = sum(s is not None for s in sources)
        if provided != 1:
            raise ValueError("Exactly one of 'target_chains' or 'target_structure' must be provided.")
        return self


class StructureRMSDConfig(StructureSimilarityConfig):
    """Configuration for RMSD-based structure similarity.

    This configuration extends ``StructureSimilarityConfig`` with specific parameters
    for calculating the Root Mean Square Deviation (RMSD) between the target and
    proposal structures. The raw RMSD value is transformed into a 0-1 constraint
    score using a sigmoid function, where 0 represents a perfect match (low RMSD)
    and 1 represents a poor match (high RMSD).

    Inherits target specification (``target_chains``, ``target_structure``,
    ``min_target_plddt``) from ``StructureSimilarityConfig`` and tool selection
    (``structure_tool``, ``esmfold_config``, ``esmfold2_config``, ``alphafold3_config``,
    ``boltz2_config``, ``chai1_config``, ``protenix_config``, ``alphafold2_config``,
    ``alphafold2_binder_config``) from ``StructureBasedConstraintConfig``.

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

        pymol_alignment_method (Literal["cealign", "align"]):
            PyMOL alignment routine to use for RMSD calculation. Default is
            "cealign".
    """

    inflection_point_angstroms: float = ConfigField(
        title="RMSD Inflection (Å)",
        default=2.0,
        description="RMSD in Ångströms where the sigmoid score equals 0.5; values below 2 Å are generally a good match.",
    )
    sigmoid_slope: float = ConfigField(
        title="Sigmoid Slope",
        default=3.0,
        description="Steepness of the penalty curve.",
    )
    pymol_alignment_method: Literal["cealign", "align"] = ConfigField(
        title="PyMOL Alignment Method",
        default="cealign",
        description="PyMOL alignment routine for RMSD calculation.",
    )


class StructureTMScoreConfig(StructureSimilarityConfig):
    """Configuration for TM-score based structure similarity.

    This configuration extends ``StructureSimilarityConfig`` for calculating the
    Template Modeling score (TM-score) between the target and proposal structures.
    TM-score is a metric for assessing the topological similarity of protein structures
    and is less sensitive to local variations than RMSD.

    The constraint returns a score calculated as (1.0 - TM_score), where 0.0 indicates
    a perfect match (TM-score = 1.0) and values closer to 1.0 indicate poor structural
    similarity.

    Inherits target specification (``target_chains``, ``target_structure``,
    ``min_target_plddt``) from ``StructureSimilarityConfig`` and tool selection
    (``structure_tool``, ``esmfold_config``, ``esmfold2_config``, ``alphafold3_config``,
    ``boltz2_config``, ``chai1_config``, ``protenix_config``, ``alphafold2_config``,
    ``alphafold2_binder_config``) from ``StructureBasedConstraintConfig``.

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
    """

    plddt_threshold: float | None = ConfigField(
        title="pLDDT Threshold",
        default=None,
        description="Drop residues with pLDDT (0-100 scale, in B-factor) below this before alignment; None keeps all.",
    )
    tm_score_normalization: Literal["structure1", "structure2", "max", "min", "mean"] = ConfigField(
        title="TM-score Normalization",
        default="mean",
        description=(
            "How to combine the two TM-scores from TM-align/US-align: structure1, structure2, max, min, or mean."
        ),
    )


# ============================================================================
# Constraints
# ============================================================================


def _prepare_target_structure(config: StructureSimilarityConfig) -> str | None:
    """Resolve the target structure to a PDB string.

    If target is a sequence, it folds it.
    """
    if config.target_structure is not None:
        if isinstance(config.target_structure, Structure):
            return config.target_structure.structure_pdb  # type: ignore[no-any-return]
        return Structure(structure=config.target_structure).structure_pdb  # type: ignore[no-any-return]

    if config.target_chains is not None:
        if isinstance(config.target_chains, Complex):
            complexes = [config.target_chains]
        else:
            from proto_language.core import detect_sequence_type

            chains = [{"sequence": seq, "entity_type": detect_sequence_type(seq)} for seq in config.target_chains]
            complexes = [Complex(chains=chains)]

        output = predict_structures(complexes, config.structure_tool, config.tool_config)

        metrics = dict(output.structures[0].metrics.items())
        target_plddt = resolve_metric(metrics, "avg_plddt")
        if target_plddt is None:
            logger.warning("Target fold lacks pLDDT metric; cannot apply min_target_plddt threshold.")
            return None
        # Normalize 0-100 pLDDT (e.g. AlphaFold3) to the 0-1 scale used by min_target_plddt.
        plddt_norm = target_plddt / 100.0 if target_plddt > 1.0 else target_plddt
        if plddt_norm < config.min_target_plddt:
            logger.warning("Target fold confidence (%.2f) below threshold (%s).", plddt_norm, config.min_target_plddt)
            return None

        return output.structures[0].structure_pdb  # type: ignore[no-any-return]

    return None


@constraint(
    key="structure-rmsd",
    label="Structural RMSD Similarity",
    config=StructureRMSDConfig,
    description="Compare structure RMSD against a target (PDB or Sequence) using generic predictors.",
    uses_gpu=True,
    tools_called=[
        "esmfold-prediction",
        "esmfold2-prediction",
        "alphafold3-prediction",
        "boltz2-prediction",
        "chai1-prediction",
        "protenix-prediction",
        "alphafold2-prediction",
        "pymol-rmsd-alignment",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_rmsd_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureRMSDConfig
) -> list[ConstraintOutput]:
    """Predicts structure of input proposals and compares RMSD against a target.

    Returns a score 0-1 (0 is perfect match). Metadata describes the predicted
    full input tuple/complex, not an individual chain.
    """
    # Prepare target.
    target_pdb = _prepare_target_structure(config)
    if not target_pdb:
        return [ConstraintOutput(score=MAX_ENERGY, metadata={"reason": "unconfident_target"}) for _ in input_sequences]

    # Prepare proposals.
    structure_complexes = []
    for proposal_tuple in input_sequences:
        chains = [{"sequence": s.sequence, "entity_type": s.sequence_type} for s in proposal_tuple]
        structure_complexes.append(Complex(chains=chains))

    try:
        prediction = predict_structures(structure_complexes, config.structure_tool, config.tool_config)
    except Exception as e:
        raise RuntimeError(
            f"structure-rmsd: {config.structure_tool} prediction failed for {len(structure_complexes)} complexes: {e}"
        ) from e

    results: list[ConstraintOutput] = []
    for proposal_structure, proposal_tuple in zip(prediction.structures, input_sequences, strict=True):
        rmsd_output = run_pymol_rmsd_alignment(
            PyMOLRMSDInput(
                target_structure=Structure(structure=target_pdb),
                mobile_structure=Structure(structure=proposal_structure.structure_pdb),
            ),
            PyMOLRMSDConfig(method=config.pymol_alignment_method),
        )
        rmsd_val = rmsd_output.rmsd

        score = sigmoid_score(rmsd_val, config.inflection_point_angstroms, config.sigmoid_slope)

        n = len(proposal_tuple)
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "rmsd_val": rmsd_val,
                    "rmsd_score": score,
                    "rmsd_alignment_method": config.pymol_alignment_method,
                    "pdb_output": proposal_structure.structure_pdb,
                },
                structures=(proposal_structure,) + (None,) * (n - 1),
            )
        )

    return results


def _count_pdb_chains(pdb_text: str) -> int:
    """Counts unique chain identifiers in PDB text to determine oligomer state."""
    chains = set()
    for line in pdb_text.splitlines():
        # Chain ID is in column 22 (index 21)
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            chains.add(line[21])
    return len(chains) if chains else 1


@constraint(
    key="structure-tmscore",
    label="Structural TM-score Similarity",
    config=StructureTMScoreConfig,
    description="Compare structure TM-score against a target. Returns 1 - TMscore.",
    uses_gpu=True,
    tools_called=[
        "esmfold-prediction",
        "esmfold2-prediction",
        "alphafold3-prediction",
        "boltz2-prediction",
        "chai1-prediction",
        "protenix-prediction",
        "alphafold2-prediction",
        "tmalign-alignment",
        "usalign-alignment",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_tmscore_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureTMScoreConfig
) -> list[ConstraintOutput]:
    """Predicts structure and compares TM-score. Returns (1.0 - TMscore).

    This constraint automatically selects the appropriate alignment tool based on
    the oligomer state of the inputs:
    - Monomer vs monomer comparisons use standard `TMalign`.
    - Comparisons involving multiple chains use `USalign` with `-mm 1` and default
      values for all other parameters.

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (StructureTMScoreConfig): Constraint configuration controlling evaluation parameters.

    Note:
        The two raw TM-scores are combined per ``config.tm_score_normalization``
        (default: arithmetic mean of the proposal- and target-normalized scores);
        the target is the reference structure. Metadata describes the predicted
        full input tuple/complex, not an individual chain.
    """
    # Prepare target.
    target_pdb = _prepare_target_structure(config)
    if not target_pdb:
        return [ConstraintOutput(score=MAX_ENERGY, metadata={"reason": "unconfident_target"}) for _ in input_sequences]

    n_target_chains = _count_pdb_chains(target_pdb)

    structure_complexes = []
    for proposal_tuple in input_sequences:
        chains = [{"sequence": s.sequence, "entity_type": s.sequence_type} for s in proposal_tuple]
        structure_complexes.append(Complex(chains=chains))

    try:
        prediction = predict_structures(structure_complexes, config.structure_tool, config.tool_config)
    except Exception as e:
        raise RuntimeError(
            f"structure-tmscore: {config.structure_tool} prediction failed for {len(structure_complexes)} complexes: {e}"
        ) from e

    results: list[ConstraintOutput] = []
    for proposal_structure, proposal_tuple in zip(prediction.structures, input_sequences, strict=True):
        n_cand_chains = len(proposal_tuple)

        # Apply pLDDT filtering at the constraint level before alignment.
        proposal_pdb = proposal_structure.structure_pdb
        if config.plddt_threshold is not None:
            proposal_pdb = _filter_pdb_by_plddt(proposal_pdb, config.plddt_threshold)
            if not any(line.startswith("ATOM") for line in proposal_pdb.splitlines()):
                results.append(
                    ConstraintOutput(
                        score=MAX_ENERGY,
                        metadata={
                            "structure_tmscore_error": f"all atoms filtered out by plddt_threshold={config.plddt_threshold}"
                        },
                    )
                )
                continue

        if n_target_chains == 1 and n_cand_chains == 1:
            try:
                _tmalign_out = run_tmalign(
                    TMalignInput(
                        query_structure=proposal_pdb,
                        reference_structure=target_pdb,
                    ),
                    TMalignConfig(),
                )
            except Exception as e:
                logger.warning("structure-tmscore: TMalign failed: %s", e)
                results.append(
                    ConstraintOutput(
                        score=MAX_ENERGY,
                        metadata={"structure_tmscore_error": f"tmalign failed: {e}"},
                    )
                )
                continue
            s1, s2 = _tmalign_out.tm_score_chain_1, _tmalign_out.tm_score_chain_2
        else:
            try:
                _usalign_out = run_usalign(
                    USalignInput(
                        query_structure=proposal_pdb,
                        reference_structure=target_pdb,
                    ),
                    USalignConfig(),
                )
            except Exception as e:
                logger.warning("structure-tmscore: USalign failed: %s", e)
                results.append(
                    ConstraintOutput(
                        score=MAX_ENERGY,
                        metadata={"structure_tmscore_error": f"usalign failed: {e}"},
                    )
                )
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
            raise ValueError(f"Invalid TMscore normalization: {config.tm_score_normalization}")

        score = 1.0 - tm_val

        n = len(proposal_tuple)
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "tm_score_raw": tm_val,
                    "tm_score_inverted": score,
                    "pdb_output": proposal_structure.structure_pdb,
                },
                structures=(proposal_structure,) + (None,) * (n - 1),
            )
        )

    return results
