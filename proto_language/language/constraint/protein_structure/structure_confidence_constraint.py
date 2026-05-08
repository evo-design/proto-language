"""Generic structure prediction confidence constraints.

Normalizes confidence metrics to be between 0 and 1, inclusive, where lower is
better (more confident). AlphaFold2 multimer-backed outputs are adapted to the
same predictor-style metric names as the other structure predictors, while
ColabDesign objective values are preserved separately in ``loss_*`` metadata.

Constraints:
- structure-plddt: Average predicted LDDT score
- structure-ptm: Predicted TM-score
- structure-iptm: Interface predicted TM-score (multimer)
- structure-pae: Average predicted aligned error
- structure-composite: Composite of all four above from a single prediction call.
- structure-iplddt: Interface predicted LDDT score
- structure-ipae: Interface predicted aligned error
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

from proto_tools import Structure, StructurePredictionComplex, predict_structures

from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.alphafold2_multimer import (
    af2_multimer_confidence_output_metadata,
    evaluate_af2_multimer_confidence_predictions,
)

logger = getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

TOOL_AVAILABLE_METRICS: dict[str, set[str]] = {
    "esmfold": {"avg_plddt", "ptm", "avg_pae"},
    "alphafold3": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "boltz2": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "chai1": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "protenix": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "alphafold2_multimer": {"avg_plddt", "ptm", "iptm", "avg_pae", "iplddt", "ipae"},
}
PAE_MAXIMUM: float = 31.75  # Angstroms.
COMPOSITE_REQUIRED_METRICS: frozenset[str] = frozenset({"avg_plddt", "iptm", "ptm", "avg_pae"})
COMPOSITE_SUPPORTED_TOOLS: frozenset[str] = frozenset({"alphafold3", "boltz2", "chai1", "protenix"})


@dataclass(frozen=True)
class _StructureConfidenceRecord:
    """One confidence prediction with complex-level metrics and structures.

    Most predictors return a ``Structure`` whose metrics and full-complex
    coordinates are enough. AF2 multimer needs one extra carrier because its
    adapter returns canonical metrics, the full complex for metadata/PDB
    output, and optional per-input chain structures. Keeping those fields
    together prevents each public confidence constraint from branching on AF2M.
    """

    metrics: dict[str, Any]
    complex_structure: Structure
    per_input_structures: tuple[Structure | None, ...] | None = None


# ============================================================================
# Constraints
# ============================================================================


def _predict_confidence_records(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    target_metric: str,
) -> list[_StructureConfidenceRecord]:
    """Run the configured confidence predictor once and return canonical records.

    Standard structure predictors flow through ``predict_structures`` and
    return a ``Structure`` per proposal. AF2 multimer uses the binder /
    ColabDesign API, so this adapter converts its richer output into the same
    private record shape before metric extraction and scoring happen.
    """
    if config.structure_tool == "alphafold2_multimer":
        predictions = evaluate_af2_multimer_confidence_predictions(
            proposals,
            config,
            target_metric=target_metric,
        )
        return [
            _StructureConfidenceRecord(
                metrics=prediction.metrics,
                complex_structure=prediction.structure,
                per_input_structures=prediction.structures,
            )
            for prediction in predictions
        ]

    complexes = []
    for proposal_tuple in proposals:
        chains = [{"sequence": seq.sequence, "entity_type": seq.sequence_type} for seq in proposal_tuple]
        complexes.append(StructurePredictionComplex(chains=chains))

    output = predict_structures(complexes, config.structure_tool, config.tool_config)
    return [
        _StructureConfidenceRecord(metrics=dict(structure.metrics.items()), complex_structure=structure)
        for structure in output.structures
    ]


def _structure_confidence(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    target_metric: str,
) -> list[tuple[float | None, _StructureConfidenceRecord | None]]:
    """Core helper for structure confidence constraints.

    Runs the configured structure predictor on all proposals and returns the
    requested canonical metric plus the predicted ``Structure`` per proposal,
    without mutating inputs. Callers assemble a ``ConstraintOutput`` from these
    values.

    Args:
        proposals (list[tuple[Sequence, ...]]): Per-proposal sequence tuples.
        config (StructureBasedConstraintConfig): Tool and tool-specific parameters.
        target_metric (str): Metric to extract from structure predictions.

    Returns:
        list[tuple[float | None, _StructureConfidenceRecord | None]]: ``(metric,
            record)`` per proposal. ``metric`` is ``None`` when the predictor
            omits it.

    Raises:
        ValueError: If target_metric is not available for the specified tool.
    """
    available = TOOL_AVAILABLE_METRICS.get(config.structure_tool, set())
    if target_metric not in available:
        raise ValueError(
            f"Metric '{target_metric}' is not available for tool '{config.structure_tool}'. "
            f"Available metrics: {', '.join(sorted(available))}"
        )

    records = _predict_confidence_records(proposals, config, target_metric)

    outcomes: list[tuple[float | None, _StructureConfidenceRecord | None]] = []
    for record, _proposal_tuple in zip(records, proposals, strict=True):
        metric_value = record.metrics.get(target_metric)
        if metric_value is None:
            alt = {"avg_plddt": "complex_plddt", "avg_pae": "complex_pde"}.get(target_metric)
            if alt:
                metric_value = record.metrics.get(alt)

        if metric_value is None:
            logger.warning("Metric %r not found in structure output, returning worst score.", target_metric)
            outcomes.append((None, None))
            continue

        outcomes.append((metric_value, record))

    return outcomes


def _assemble_result(
    metric: float | None,
    record: _StructureConfidenceRecord | None,
    target_metric: str,
    score: float,
    structure_tool: str,
    n_segments: int,
) -> ConstraintOutput:
    """Build a full-tuple structure result.

    Metadata describes the predicted input tuple/complex and is broadcast by
    default; the predicted Structure attaches to slot 0 as the canonical carrier.
    """
    if record is None:
        return ConstraintOutput(score=score)
    if structure_tool == "alphafold2_multimer":
        metadata = af2_multimer_confidence_output_metadata(
            record.metrics,
            output_loss=score,
            output_structure=record.complex_structure,
            target_metric=target_metric,
        )
    else:
        metadata = {
            target_metric: metric,
            "pdb_output": record.complex_structure.structure_pdb,
            "structure_tool": structure_tool,
        }
    structures = record.per_input_structures or ((record.complex_structure,) + (None,) * (n_segments - 1))
    return ConstraintOutput(score=score, metadata=metadata, structures=structures)


@constraint(
    key="structure-plddt",
    label="Structure pLDDT Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate structure quality using predicted LDDT score",
    uses_gpu=True,
    tools_called=[
        "esmfold-prediction",
        "alphafold3-prediction",
        "boltz2-prediction",
        "chai1-prediction",
        "protenix-prediction",
        "alphafold2-multimer",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_plddt_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate structure quality using predicted LDDT (pLDDT) score.

    pLDDT (predicted Local Distance Difference Test) measures per-residue
    confidence in the predicted structure. Values range from 0.0 to 100.0
    (sometimes, these are normalized from 0.0 to 1.0) where higher values
    indicate more reliable predictions.

    This constraint returns 1.0 - **normalized** pLDDT, so lower scores
    indicate better predicted structure quality.

    Note that for Boltz2, this is based on the ``"complex_plddt"`` score
    returned natively by the package.

    **Supported tools**: ESMFold, AlphaFold3, Boltz2, Chai1, Protenix, AlphaFold2 multimer

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of input sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Returns:
        list[ConstraintOutput]: Per-proposal score and ``avg_plddt`` / ``pdb_output``
            / ``structure_tool`` metadata for the predicted full input tuple;
            predicted Structure attaches to slot 0.

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
    outcomes = _structure_confidence(input_sequences, config, "avg_plddt")
    results: list[ConstraintOutput] = []
    for (metric, record), proposal_tuple in zip(outcomes, input_sequences, strict=True):
        if metric is None:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"structure_plddt_error": f"avg_plddt missing from {config.structure_tool} output"},
                )
            )
            continue
        normalized = metric / 100.0 if metric > 1.0 else metric
        results.append(
            _assemble_result(metric, record, "avg_plddt", 1.0 - normalized, config.structure_tool, len(proposal_tuple))
        )
    return results


@constraint(
    key="structure-ptm",
    label="Structure pTM Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate structure quality using predicted TM score",
    uses_gpu=True,
    tools_called=[
        "esmfold-prediction",
        "alphafold3-prediction",
        "boltz2-prediction",
        "chai1-prediction",
        "protenix-prediction",
        "alphafold2-multimer",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_ptm_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate structure quality using predicted TM-score (pTM).

    pTM (predicted Template Modeling score) measures overall structural
    accuracy of the predicted model. Values range from 0.0 to 1.0, where
    higher values indicate better global fold quality.

    This constraint returns ``1.0 - ptm``, so lower scores indicate
    better predicted structure quality.

    **Supported tools**: ESMFold, AlphaFold3, Boltz2, Chai1, Protenix, AlphaFold2 multimer

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of input sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Returns:
        list[ConstraintOutput]: Per-proposal score and ``ptm`` / ``pdb_output`` /
            ``structure_tool`` metadata; predicted Structure attaches to slot 0.

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
    outcomes = _structure_confidence(input_sequences, config, "ptm")
    results: list[ConstraintOutput] = []
    for (metric, record), proposal_tuple in zip(outcomes, input_sequences, strict=True):
        if metric is None:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"structure_ptm_error": f"ptm missing from {config.structure_tool} output"},
                )
            )
            continue
        results.append(
            _assemble_result(metric, record, "ptm", 1.0 - metric, config.structure_tool, len(proposal_tuple))
        )
    return results


@constraint(
    key="structure-iptm",
    label="Structure ipTM Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate interface quality using predicted interface TM score",
    uses_gpu=True,
    tools_called=[
        "alphafold3-prediction",
        "boltz2-prediction",
        "chai1-prediction",
        "protenix-prediction",
        "alphafold2-multimer",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_iptm_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate interface quality using predicted interface TM-score (ipTM).

    ipTM (interface predicted TM-score) specifically measures the quality
    of predicted inter-chain interfaces in multimeric complexes. Values
    range from 0.0 to 1.0, where higher values indicate better interface
    predictions.

    This constraint returns ``1.0 - iptm``, so lower scores indicate
    better predicted interface quality.

    **Supported tools**: AlphaFold3, Boltz2, Chai1, Protenix, AlphaFold2 multimer (NOT ESMFold)

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of input sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Returns:
        list[ConstraintOutput]: Per-proposal score and ``iptm`` / ``pdb_output`` /
            ``structure_tool`` metadata; predicted Structure attaches to slot 0.

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
    outcomes = _structure_confidence(input_sequences, config, "iptm")
    results: list[ConstraintOutput] = []
    for (metric, record), proposal_tuple in zip(outcomes, input_sequences, strict=True):
        if metric is None:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"structure_iptm_error": f"iptm missing from {config.structure_tool} output"},
                )
            )
            continue
        results.append(
            _assemble_result(metric, record, "iptm", 1.0 - metric, config.structure_tool, len(proposal_tuple))
        )
    return results


@constraint(
    key="structure-pae",
    label="Structure pAE Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate structure quality using predicted aligned error",
    uses_gpu=True,
    tools_called=[
        "esmfold-prediction",
        "alphafold3-prediction",
        "boltz2-prediction",
        "chai1-prediction",
        "protenix-prediction",
        "alphafold2-multimer",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_pae_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
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

    **Supported tools**: ESMFold, AlphaFold3, Boltz2, Chai1, Protenix, AlphaFold2 multimer

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of input sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Returns:
        list[ConstraintOutput]: Per-proposal score and ``avg_pae`` / ``pdb_output`` /
            ``structure_tool`` metadata; predicted Structure attaches to slot 0.

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
    outcomes = _structure_confidence(input_sequences, config, "avg_pae")
    results: list[ConstraintOutput] = []
    for (metric, record), proposal_tuple in zip(outcomes, input_sequences, strict=True):
        if metric is None:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"structure_pae_error": f"avg_pae missing from {config.structure_tool} output"},
                )
            )
            continue
        score = min(metric / PAE_MAXIMUM, 1.0)
        results.append(_assemble_result(metric, record, "avg_pae", score, config.structure_tool, len(proposal_tuple)))
    return results


@constraint(
    key="structure-iplddt",
    label="Structure Interface pLDDT",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 multimer interface pLDDT confidence.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_iplddt_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer interface pLDDT confidence.

    This reads the predictor-style ``iplddt`` metric exposed through the AF2
    multimer adapter and returns ``1.0 - iplddt`` so lower scores indicate
    better interface-local confidence. The underlying ColabDesign objective is
    still preserved separately as ``loss_iplddt`` metadata.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Per-proposal interface pLDDT confidence scores.
    """
    outcomes = _structure_confidence(input_sequences, config, "iplddt")
    results: list[ConstraintOutput] = []
    for (metric, record), proposal_tuple in zip(outcomes, input_sequences, strict=True):
        if metric is None:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"structure_iplddt_error": f"iplddt missing from {config.structure_tool} output"},
                )
            )
            continue
        normalized = metric / 100.0 if metric > 1.0 else metric
        results.append(
            _assemble_result(
                metric,
                record,
                "iplddt",
                1.0 - normalized,
                config.structure_tool,
                len(proposal_tuple),
            )
        )
    return results


@constraint(
    key="structure-ipae",
    label="Structure Interface pAE",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 multimer interface PAE confidence.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_ipae_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer interface PAE confidence.

    This reads the predictor-style ``ipae`` metric exposed through the AF2
    multimer adapter and returns normalized interface PAE, so lower scores
    indicate better interface-local confidence. The underlying ColabDesign
    objective is still preserved separately as ``loss_ipae`` metadata.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Per-proposal interface PAE confidence scores.
    """
    outcomes = _structure_confidence(input_sequences, config, "ipae")
    results: list[ConstraintOutput] = []
    for (metric, record), proposal_tuple in zip(outcomes, input_sequences, strict=True):
        if metric is None:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"structure_ipae_error": f"ipae missing from {config.structure_tool} output"},
                )
            )
            continue
        score = min(metric / PAE_MAXIMUM, 1.0)
        results.append(
            _assemble_result(
                metric,
                record,
                "ipae",
                score,
                config.structure_tool,
                len(proposal_tuple),
            )
        )
    return results


@constraint(
    key="structure-composite",
    label="Structure Composite Confidence",
    config=StructureBasedConstraintConfig,
    description="Score structure quality using a composite of plddt/iptm/ptm/pae from a single prediction call",
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "boltz2-prediction", "chai1-prediction", "protenix-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    input_labels=None,
)
def structure_composite_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate structure quality using a composite of all confidence metrics from one prediction call.

    Runs ``predict_structures`` once per batch and combines ``avg_plddt``,
    ``iptm``, ``ptm``, and ``avg_pae`` into a single scalar in ``[0, 1]`` where
    lower is better (more confident). All four raw metrics plus the resulting
    structure are also exposed via ``metadata`` / ``structures`` so callers can
    threshold on individual metrics post-hoc (e.g., Germinal's final-filter
    gates in ``configs/filter/final/vhh.yaml``) without re-running the predictor.

    The composite is the equal-weighted mean of normalized deviations:
    ``(1 - plddt_norm + 1 - iptm + 1 - ptm + pae / PAE_MAXIMUM) / 4``.

    Versus stacking ``structure-plddt`` + ``structure-iptm`` + ``structure-ptm``
    + ``structure-pae`` as four separate constraints, this is 4x cheaper
    (one ``predict_structures`` call instead of four) and exposes all metrics
    for post-hoc threshold labeling.

    **Supported tools**: AlphaFold3, Boltz2, Chai1, Protenix (NOT ESMFold — ESMFold does
    not produce ``iptm`` and cannot handle multi-chain complexes; NOT AF2
    multimer because its interface TM value is exposed as a differentiable
    objective rather than the same forward confidence metric used here).

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of input sequences.
        config (StructureBasedConstraintConfig): Constraint configuration controlling evaluation parameters.

    Returns:
        list[ConstraintOutput]: Per-proposal composite score in ``[0, 1]`` (lower
            is better). Metadata carries the four normalized components
            (``composite_avg_plddt``, ``composite_iptm``, ``composite_ptm``,
            ``composite_avg_pae``) plus ``pdb_output`` and ``structure_tool`` for
            the predicted full input tuple; the predicted Structure attaches to
            slot 0.

    Note:
        Metadata values are **all normalized to ``[0, 1]``** so downstream
        threshold code is tool-agnostic (unlike sibling single-metric
        constraints, which store raw values and require the caller to know
        the tool's scale):

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
    if config.structure_tool not in COMPOSITE_SUPPORTED_TOOLS:
        raise ValueError(
            f"structure-composite requires one of {sorted(COMPOSITE_SUPPORTED_TOOLS)}; got {config.structure_tool!r}."
        )

    records = _predict_confidence_records(input_sequences, config, "avg_plddt")

    results: list[ConstraintOutput] = []
    for record, proposal_tuple in zip(records, input_sequences, strict=True):
        m = record.metrics
        plddt_raw = m.get("avg_plddt")
        if plddt_raw is None:
            plddt_raw = m.get("complex_plddt")
        iptm = m.get("iptm")
        ptm = m.get("ptm")
        pae = m.get("avg_pae")
        if pae is None:
            pae = m.get("complex_pde")

        if plddt_raw is None or iptm is None or ptm is None or pae is None:
            err_msg = (
                f"missing composite metrics from {config.structure_tool}: "
                f"plddt={plddt_raw}, iptm={iptm}, ptm={ptm}, pae={pae}"
            )
            logger.warning("structure-composite: %s", err_msg)
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"structure_composite_error": err_msg}))
            continue

        plddt_norm = plddt_raw / 100.0 if plddt_raw > 1.0 else plddt_raw
        pae_norm = min(pae / PAE_MAXIMUM, 1.0)

        score = ((1.0 - plddt_norm) + (1.0 - iptm) + (1.0 - ptm) + pae_norm) / 4.0

        n = len(proposal_tuple)
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "composite_avg_plddt": plddt_norm,
                    "composite_iptm": iptm,
                    "composite_ptm": ptm,
                    "composite_avg_pae": pae_norm,
                    "pdb_output": record.complex_structure.structure_pdb,
                    "structure_tool": config.structure_tool,
                },
                structures=record.per_input_structures or ((record.complex_structure,) + (None,) * (n - 1)),
            )
        )

    return results
