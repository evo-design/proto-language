"""
proto_language/language/constraint/protein_structure/structure_confidence_constraint.py

Generic structure prediction confidence constraints supporting multiple tools:
ESMFold, AlphaFold3, Boltz2, and Chai1.

Normalizes confidence metrics to be between 0 and 1, inclusive, where lower is
better (more confident).

Constraints:
- structure-plddt: Average predicted LDDT score
- structure-ptm: Predicted TM-score
- structure-iptm: Interface predicted TM-score (multimer)
- structure-pae: Average predicted aligned error
"""

from __future__ import annotations

from logging import getLogger
from typing import Dict, List, Tuple

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

TOOL_AVAILABLE_METRICS: Dict[str, set] = {
    "esmfold": {"avg_plddt", "ptm", "avg_pae"},
    "alphafold3": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "boltz2": {"avg_plddt", "ptm", "iptm", "avg_pae"},
    "chai1": {"avg_plddt", "ptm", "iptm", "avg_pae"},
}
PAE_MAXIMUM: float = 31.75 # Angstroms.


# ============================================================================
# Constraints
# ============================================================================


def _structure_confidence(
    proposals: List[Tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    target_metric: str,
) -> List[float]:
    """
    Core helper for structure confidence constraints.

    Args:
        proposals (list[tuple[Sequence, ...]]): List of sequence tuples, where each tuple represents a
            complex (monomer = 1-tuple, dimer = 2-tuple, etc.).
        config (StructureBasedConstraintConfig): Configuration specifying tool and tool-specific parameters.
        target_metric (str): Metric to extract from structure predictions.

    Returns:
        list[float]: List of raw metrics requested by `target_metric`. Invalid
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
        chains = [
            {"sequence": seq.sequence, "entity_type": seq.sequence_type}
            for seq in proposal_tuple
        ]
        complexes.append(StructurePredictionComplex(chains=chains))

    # Run structure prediction.
    output = predict_structures(complexes, config.structure_tool, config.tool_config)

    # Extract and return raw requested metric.
    raw_metrics = []
    for structure, proposal_tuple in zip(output.structures, proposals):
        metric_value = structure.metrics.get(target_metric)

        if metric_value is None:
            logger.warning(
                f"Metric '{target_metric}' not found in structure output, "
                f"returning worst score."
            )
            raw_metrics.append(None)
            continue

        # Attach metadata to first sequence in tuple for visibility.
        if proposal_tuple:
            proposal_tuple[0]._metadata.update({
                target_metric: metric_value,
                "pdb_output": store_file(structure.structure_pdb, FileType.PDB),
                "structure_tool": config.structure_tool,
            })

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
    num_input_sequences_per_tuple=None,
)
def structure_plddt_constraint(
    input_sequences: List[Tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> List[float]:
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
    raw_metrics =  _structure_confidence(input_sequences, config, "avg_plddt")
    scores = []
    for metric in raw_metrics:
        if metric is None:
            scores.append(1.)
            continue
        # Each structure predictor returns differently normalized pLDDTs.
        if config.structure_tool == "alphafold3":
            metric /= 100.
        scores.append(1. - metric)
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
    num_input_sequences_per_tuple=None,
)
def structure_ptm_constraint(
    input_sequences: List[Tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> List[float]:
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
    return [ 1. - metric if metric is not None else 1. for metric in raw_metrics ]


@constraint(
    key="structure-iptm",
    label="Structure ipTM Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate interface quality using predicted interface TM score",
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "boltz2-prediction", "chai1-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    num_input_sequences_per_tuple=None,
)
def structure_iptm_constraint(
    input_sequences: List[Tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> List[float]:
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
        ...         "tool_config": {"seeds": [0, 1], "use_msa": True},
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
        ...         "tool_config": {"use_msa_server": True},
        ...     },
        ... )
    """
    raw_metrics = _structure_confidence(input_sequences, config, "iptm")
    # ipTM is pretty standard, just return 1 minus the raw metric.
    return [ 1. - metric if metric is not None else 1. for metric in raw_metrics ]


@constraint(
    key="structure-pae",
    label="Structure pAE Score",
    config=StructureBasedConstraintConfig,
    description="Evaluate structure quality using predicted aligned error",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "alphafold3-prediction", "boltz2-prediction", "chai1-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "rna", "dna", "ligand"],
    num_input_sequences_per_tuple=None,
)
def structure_pae_constraint(
    input_sequences: List[Tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> List[float]:
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
        ...         "tool_config": {"seeds": [0, 1], "use_msa": True},
        ...     },
        ... )
    """
    raw_metrics =  _structure_confidence(input_sequences, config, "avg_pae")
    scores = [
        min(metric / PAE_MAXIMUM, 1.) if metric is not None else 1.
        for metric in raw_metrics
    ]
    return scores
