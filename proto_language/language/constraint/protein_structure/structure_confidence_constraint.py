"""Generic structure prediction confidence constraints for ESMFold, AlphaFold3, Boltz2, and Chai1.

Normalizes confidence metrics to be between 0 and 1, inclusive, where lower is
better (more confident).

Constraints:
- structure-plddt: Average predicted LDDT score
- structure-ptm: Predicted TM-score
- structure-iptm: Interface predicted TM-score (multimer)
- structure-pae: Average predicted aligned error
- structure-composite: Composite of all four above from a single prediction call.
"""

from logging import getLogger

from proto_tools import StructurePredictionComplex, predict_structures

from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.language.core import Sequence
from proto_language.storage import FileType, store_file

logger = getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

TOOL_AVAILABLE_METRICS: dict[str, set[str]] = {
    "esmfold": {"avg_plddt", "ptm", "avg_pae"},
    "alphafold3": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "boltz2": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "chai1": {"avg_plddt", "ptm", "iptm", "avg_pae"},
}
PAE_MAXIMUM: float = 31.75  # Angstroms.
COMPOSITE_REQUIRED_METRICS: frozenset[str] = frozenset({"avg_plddt", "iptm", "ptm", "avg_pae"})


# ============================================================================
# Constraints
# ============================================================================


def _structure_confidence(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    target_metric: str,
) -> list[float | None]:
    """Core helper for structure confidence constraints.

    Args:
        proposals (list[tuple[Sequence, ...]]): List of sequence tuples, where each tuple represents a
            complex (monomer = 1-tuple, dimer = 2-tuple, etc.).
        config (StructureBasedConstraintConfig): Configuration specifying tool and tool-specific parameters.
        target_metric (str): Metric to extract from structure predictions.

    Returns:
        list[float | None]: List of raw metrics requested by `target_metric`. Invalid
            raw metrics are returned as None and should be checked by the caller.

    Raises:
        ValueError: If target_metric is not available for the specified tool.
    """
    available = TOOL_AVAILABLE_METRICS.get(config.structure_tool, set())
    if target_metric not in available:
        raise ValueError(
            f"Metric '{target_metric}' is not available for tool '{config.structure_tool}'. "
            f"Available metrics: {', '.join(sorted(available))}"
        )

    # Build complexes from proposal tuples.
    complexes = []
    for proposal_tuple in proposals:
        chains = [{"sequence": seq.sequence, "entity_type": seq.sequence_type} for seq in proposal_tuple]
        complexes.append(StructurePredictionComplex(chains=chains))

    # Run structure prediction.
    output = predict_structures(complexes, config.structure_tool, config.tool_config)

    # Extract and return raw requested metric.
    raw_metrics: list[float | None] = []
    for structure, proposal_tuple in zip(output.structures, proposals, strict=False):
        metric_value = structure.metrics.get(target_metric)
        if metric_value is None:
            alt = {"avg_plddt": "complex_plddt", "avg_pae": "complex_pde"}.get(target_metric)
            if alt:
                metric_value = structure.metrics.get(alt)

        if metric_value is None:
            logger.warning(f"Metric '{target_metric}' not found in structure output, returning worst score.")
            raw_metrics.append(None)
            continue

        # Attach structure and metadata to first sequence in tuple for visibility.
        if proposal_tuple:
            proposal_tuple[0].structure = structure
            proposal_tuple[0]._metadata.update(
                {
                    target_metric: metric_value,
                    "pdb_output": store_file(structure.structure_pdb, FileType.PDB),
                    "structure_tool": config.structure_tool,
                }
            )

        raw_metrics.append(metric_value)

    return raw_metrics


@constraint(
    key="structure-plddt",
    label="Structure pLDDT Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate structure quality using predicted LDDT score",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "alphafold3-prediction", "boltz2-prediction", "chai1-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_plddt_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[float]:
    """Evaluate structure quality using predicted LDDT (pLDDT) score.

    pLDDT (predicted Local Distance Difference Test) measures per-residue
    confidence in the predicted structure. Values range from 0.0 to 100.0
    (sometimes, these are normalized from 0.0 to 1.0) where higher values
    indicate more reliable predictions.

    This constraint returns 1.0 - **normalized** pLDDT, so lower scores
    indicate better predicted structure quality.

    Note that for Boltz2, this is based on the ``"complex_plddt"`` score
    returned natively by the package.

    **Supported tools**: ESMFold, AlphaFold3, Boltz2, Chai1

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Example:
        Programming a homo-trimer with ESMFold:

        >>> from proto_language.language.core import Segment
        >>> protomer = Segment(length=10, sequence_type="protein")
        >>> esmfold_plddt = Constraint(
        ...     inputs=[protomer, protomer, protomer],
        ...     function=structure_plddt_constraint,
        ...     function_config={"structure_tool": "esmfold"},
        ... )
    """
    raw_metrics = _structure_confidence(input_sequences, config, "avg_plddt")
    scores = []
    for metric in raw_metrics:
        if metric is None:
            scores.append(1.0)
            continue
        # Each structure predictor returns differently normalized pLDDTs.
        normalized = metric / 100.0 if config.structure_tool == "alphafold3" else metric
        scores.append(1.0 - normalized)
    return scores


@constraint(
    key="structure-ptm",
    label="Structure pTM Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate structure quality using predicted TM score",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "alphafold3-prediction", "boltz2-prediction", "chai1-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_ptm_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[float]:
    """Evaluate structure quality using predicted TM-score (pTM).

    pTM (predicted Template Modeling score) measures overall structural
    accuracy of the predicted model. Values range from 0.0 to 1.0, where
    higher values indicate better global fold quality.

    This constraint returns ``1.0 - ptm``, so lower scores indicate
    better predicted structure quality.

    **Supported tools**: ESMFold, AlphaFold3, Boltz2, Chai1

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Example:
        Programming a homo-dimer with ESMFold:

        >>> from proto_language.language.core import Segment
        >>> protomer = Segment(length=20, sequence_type="protein")
        >>> esmfold_plddt = Constraint(
        ...     inputs=[protomer, protomer],
        ...     function=structure_ptm_constraint,
        ...     function_config={"structure_tool": "esmfold"},
        ... )
    """
    raw_metrics = _structure_confidence(input_sequences, config, "ptm")
    # pTM is pretty standard, just return 1 minus the raw metric.
    return [1.0 - metric if metric is not None else 1.0 for metric in raw_metrics]


@constraint(
    key="structure-iptm",
    label="Structure ipTM Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate interface quality using predicted interface TM score",
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "boltz2-prediction", "chai1-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_iptm_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[float]:
    """Evaluate interface quality using predicted interface TM-score (ipTM).

    ipTM (interface predicted TM-score) specifically measures the quality
    of predicted inter-chain interfaces in multimeric complexes. Values
    range from 0.0 to 1.0, where higher values indicate better interface
    predictions.

    This constraint returns ``1.0 - iptm``, so lower scores indicate
    better predicted interface quality.

    **Supported tools**: AlphaFold3, Boltz2, Chai1 (NOT ESMFold)

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Examples:
        Programming a protein-protein binder with AF3:

        >>> from proto_language.language.core import Segment
        >>> target = Segment(length=200, sequence_type="protein")
        >>> binder = Segment(length=80, sequence_type="protein")
        >>> af3_iptm = Constraint(
        ...     inputs=[target, binder],
        ...     function=structure_iptm_constraint,
        ...     function_config={
        ...         "structure_tool": "alphafold3",
        ...         "alphafold3_config": {"seeds": [0, 1], "use_msa": True},
        ...     },
        ... )

        Programming a protein-DNA binder with Boltz2:

        >>> from proto_language.language.core import Segment
        >>> protein = Segment(length=100, sequence_type="protein")
        >>> aptamer = Segment(length=20, sequence_type="dna")
        >>> boltz_iptm = Constraint(
        ...     inputs=[protein, aptamer],
        ...     function=structure_iptm_constraint,
        ...     function_config={
        ...         "structure_tool": "boltz2",
        ...         "boltz2_config": {"use_msa_server": True},
        ...     },
        ... )
    """
    raw_metrics = _structure_confidence(input_sequences, config, "iptm")
    # ipTM is pretty standard, just return 1 minus the raw metric.
    return [1.0 - metric if metric is not None else 1.0 for metric in raw_metrics]


@constraint(
    key="structure-pae",
    label="Structure pAE Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate structure quality using predicted aligned error",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "alphafold3-prediction", "boltz2-prediction", "chai1-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_pae_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[float]:
    """Evaluate structure quality using predicted aligned error (pAE).

    pAE (predicted Aligned Error) measures the expected positional error
    between residue pairs. pAE values are from 0 to 31.75 Angstroms. Unlike
    most confidence metrics, lower pAE values (closer to 0) are better.
    The average pAE takes the mean of the pairwise matrix.

    This constraint transforms pAE as the normalized mean PAE, i.e., it:
        1. Computes the average of the entire pairwise pAE matrix.
        2. Normalizes by 31.75 Angstroms (the AlphaFold maximum value used
           by all major structure predictors).
        3. Returns that value without flipping the sign, as lower is better.

    **Supported tools**: ESMFold, AlphaFold3, Boltz2, Chai1

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Examples:
        Programming a protein-protein binder with AF3:

        >>> from proto_language.language.core import Segment
        >>> target = Segment(length=200, sequence_type="protein")
        >>> binder = Segment(length=80, sequence_type="protein")
        >>> af3_pae = Constraint(
        ...     inputs=[target, binder],
        ...     function=structure_pae_constraint,
        ...     function_config={
        ...         "structure_tool": "alphafold3",
        ...         "alphafold3_config": {"seeds": [0, 1], "use_msa": True},
        ...     },
        ... )
    """
    raw_metrics = _structure_confidence(input_sequences, config, "avg_pae")
    return [min(metric / PAE_MAXIMUM, 1.0) if metric is not None else 1.0 for metric in raw_metrics]


@constraint(
    key="structure-composite",
    label="Structure Composite Confidence",
    config=StructureBasedConstraintConfig,
    description="Score structure quality using a composite of plddt/iptm/ptm/pae from a single prediction call",
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "boltz2-prediction", "chai1-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_composite_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[float]:
    """Evaluate structure quality using a composite of all confidence metrics from one prediction call.

    Runs ``predict_structures`` once per batch and combines ``avg_plddt``,
    ``iptm``, ``ptm``, and ``avg_pae`` into a single scalar in ``[0, 1]`` where
    lower is better (more confident). All four raw metrics plus the resulting
    structure are also written to each proposal's ``_metadata`` so callers can
    threshold on individual metrics post-hoc (e.g., Germinal's final-filter
    gates in ``configs/filter/final/vhh.yaml``) without re-running the predictor.

    The composite is the equal-weighted mean of normalized deviations:
    ``(1 - plddt_norm + 1 - iptm + 1 - ptm + pae / PAE_MAXIMUM) / 4``.

    Versus stacking ``structure-plddt`` + ``structure-iptm`` + ``structure-ptm``
    + ``structure-pae`` as four separate constraints, this is 4x cheaper
    (one ``predict_structures`` call instead of four) and exposes all metrics
    for post-hoc threshold labeling.

    **Supported tools**: AlphaFold3, Boltz2, Chai1 (NOT ESMFold — ESMFold does
    not produce ``iptm`` and cannot handle multi-chain complexes).

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Returns:
        list[float]: Composite confidence score in ``[0, 1]`` per proposal;
            lower is better.

    Note:
        Writes the following keys onto each proposal's ``_metadata`` dict,
        **all normalized to ``[0, 1]``** so downstream threshold code is
        tool-agnostic (unlike sibling single-metric constraints, which store
        raw values and require the caller to know the tool's scale):

        - ``composite_avg_plddt``: Normalized pLDDT in ``[0, 1]`` (divided by
          100 for alphafold3).
        - ``composite_iptm``: ipTM in ``[0, 1]``.
        - ``composite_ptm``: pTM in ``[0, 1]``.
        - ``composite_avg_pae``: Normalized PAE in ``[0, 1]`` (raw Angstroms
          divided by ``PAE_MAXIMUM = 31.75``, clamped at 1).
        - ``pdb_output``: Stored PDB file handle.
        - ``structure_tool``: Tool name used for prediction.

    Examples:
        Ranking binder candidates by composite structure quality with Chai-1:

        >>> from proto_language.language.core import Segment
        >>> binder = Segment(length=80, sequence_type="protein")
        >>> target = Segment(sequence="MKTL...", sequence_type="protein")
        >>> chai1_composite = Constraint(
        ...     inputs=[binder, target],
        ...     function=structure_composite_constraint,
        ...     function_config={"structure_tool": "chai1"},
        ... )
    """
    available = TOOL_AVAILABLE_METRICS.get(config.structure_tool, set())
    if not COMPOSITE_REQUIRED_METRICS.issubset(available):
        missing = sorted(COMPOSITE_REQUIRED_METRICS - available)
        raise ValueError(
            f"structure-composite requires a tool producing all of "
            f"{sorted(COMPOSITE_REQUIRED_METRICS)}; '{config.structure_tool}' is missing {missing}."
        )

    # Build complexes from proposal tuples.
    complexes = []
    for proposal_tuple in input_sequences:
        chains = [{"sequence": seq.sequence, "entity_type": seq.sequence_type} for seq in proposal_tuple]
        complexes.append(StructurePredictionComplex(chains=chains))

    output = predict_structures(complexes, config.structure_tool, config.tool_config)

    scores: list[float] = []
    for structure, proposal_tuple in zip(output.structures, input_sequences, strict=False):
        m = structure.metrics
        plddt_raw = m.get("avg_plddt")
        if plddt_raw is None:
            plddt_raw = m.get("complex_plddt")
        iptm = m.get("iptm")
        ptm = m.get("ptm")
        pae = m.get("avg_pae")
        if pae is None:
            pae = m.get("complex_pde")

        if plddt_raw is None or iptm is None or ptm is None or pae is None:
            logger.warning(
                f"Missing composite metrics from '{config.structure_tool}': "
                f"plddt={plddt_raw}, iptm={iptm}, ptm={ptm}, pae={pae}. Returning worst score."
            )
            scores.append(1.0)
            continue

        plddt_norm = plddt_raw / 100.0 if config.structure_tool == "alphafold3" else plddt_raw
        pae_norm = min(pae / PAE_MAXIMUM, 1.0)

        score = ((1.0 - plddt_norm) + (1.0 - iptm) + (1.0 - ptm) + pae_norm) / 4.0

        if proposal_tuple:
            proposal_tuple[0].structure = structure
            proposal_tuple[0]._metadata.update(
                {
                    "composite_avg_plddt": plddt_norm,
                    "composite_iptm": iptm,
                    "composite_ptm": ptm,
                    "composite_avg_pae": pae_norm,
                    "pdb_output": store_file(structure.structure_pdb, FileType.PDB),
                    "structure_tool": config.structure_tool,
                }
            )

        scores.append(score)

    return scores
