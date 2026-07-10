"""Protenix adapter for grouped forward confidence scoring constraints.

Protenix returns pLDDT, pTM, ipTM, and pAE from one structure prediction. The
public API exposes those as separate constraints, so the scoring compiler groups
compatible terms into one prediction call while preserving per-constraint
metadata and weights.
"""

from __future__ import annotations

import logging
from typing import Any

from proto_tools import Complex, predict_structures
from pydantic import ValidationError

from proto_language.constraint.protein_structure.structure_confidence_constraint import (
    PAE_MAXIMUM,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
    resolve_metric,
)
from proto_language.core import Constraint
from proto_language.optimizer.constraint_compiler.base import CompiledConstraint
from proto_language.utils import MAX_ENERGY

logger = logging.getLogger(__name__)


PROTENIX_OBJECTIVE_BY_FUNCTION = {
    structure_plddt_constraint: "plddt",
    structure_ptm_constraint: "ptm",
    structure_iptm_constraint: "iptm",
    structure_pae_constraint: "pae",
}

TARGET_METRIC_BY_OBJECTIVE = {
    "plddt": "avg_plddt",
    "ptm": "ptm",
    "iptm": "iptm",
    "pae": "avg_pae",
}


def objective_key_for_constraint(constraint: Constraint) -> str | None:
    """Return the Protenix confidence objective key for ``constraint``."""
    if constraint.function is None:
        return None
    return PROTENIX_OBJECTIVE_BY_FUNCTION.get(constraint.function)


def config_for_constraint(
    constraint: Constraint,
    *,
    strict: bool = False,
) -> StructureBasedConstraintConfig | None:
    """Parse a constraint's structure config into the canonical config model."""
    config = constraint.function_config
    if isinstance(config, StructureBasedConstraintConfig):
        return config
    if isinstance(config, dict):
        try:
            return StructureBasedConstraintConfig(**config)
        except (TypeError, ValidationError):
            if strict:
                raise
            return None
    return None


def can_group_scoring_constraint(
    constraint: Constraint,
    objective_key: str | None,
    config: StructureBasedConstraintConfig | None,
) -> bool:
    """Return whether ``constraint`` can join a grouped Protenix forward call."""
    return (
        objective_key is not None
        and config is not None
        and config.structure_tool == "protenix"
        and constraint.threshold is None
    )


def scoring_group_key(constraint: Constraint, config: StructureBasedConstraintConfig) -> tuple[Any, ...]:
    """Build the identity key used to group compatible Protenix scoring constraints."""
    input_ids = tuple(id(segment) for segment in constraint.inputs)
    return (*input_ids, config.protenix_config.model_dump_json())


def evaluate_scoring_group(compiled_constraints: list[CompiledConstraint], mask: list[bool]) -> list[float]:
    """Evaluate compatible Protenix confidence constraints with one prediction batch."""
    first_constraint = compiled_constraints[0].constraint
    config = config_for_constraint(first_constraint, strict=True)
    if config is None:
        raise ValueError(f"Constraint '{first_constraint.label}' must use StructureBasedConstraintConfig.")

    inputs = first_constraint.inputs
    num_proposals = inputs[0].num_proposals
    scores = [float("nan")] * num_proposals
    proposal_indices = [idx for idx, should_eval in enumerate(mask) if should_eval]
    if not proposal_indices:
        return scores

    complexes = []
    for proposal_idx in proposal_indices:
        chains = [
            {"sequence": segment.proposal_sequences[proposal_idx].sequence, "entity_type": segment.sequence_type}
            for segment in inputs
        ]
        complexes.append(Complex(chains=chains))

    output = predict_structures(complexes, config.structure_tool, config.tool_config)
    if len(output.structures) != len(proposal_indices):
        raise ValueError(
            f"Protenix scoring returned {len(output.structures)} structures, expected {len(proposal_indices)}."
        )

    for proposal_idx, structure in zip(proposal_indices, output.structures, strict=True):
        metrics = dict(structure.metrics.items())
        term_results = [_scoring_term_score(metrics, compiled.objective_key) for compiled in compiled_constraints]
        term_scores = [score for score, _ in term_results]
        group_score = sum(
            compiled.constraint.weight * score
            for compiled, score in zip(compiled_constraints, term_scores, strict=True)
        )
        scores[proposal_idx] = group_score

        for compiled, (score, fallback_used) in zip(compiled_constraints, term_results, strict=True):
            metadata = _scoring_constraint_metadata(
                metrics,
                output_structure=structure,
                objective_key=compiled.objective_key,
                output_score=score,
                group_score=group_score,
                fallback_used=fallback_used,
            )
            compiled.constraint._write_constraint_metadata(proposal_idx, score, metadata)

        inputs[0].proposal_sequences[proposal_idx].structure = structure

    return scores


def _scoring_constraint_metadata(
    metrics: dict[str, Any],
    *,
    output_structure: Any,
    objective_key: str,
    output_score: float,
    group_score: float,
    fallback_used: bool,
) -> dict[str, Any]:
    """Build per-public-constraint metadata from a grouped Protenix scoring output."""
    target_metric = TARGET_METRIC_BY_OBJECTIVE[objective_key]
    metadata = dict(metrics)
    metric = _metric_value(metrics, target_metric)
    if metric is None:
        metadata[f"structure_{objective_key}_error"] = f"{target_metric} missing from protenix output"
    else:
        metadata[target_metric] = metric
    metadata.update(
        {
            "loss_key": objective_key,
            "output_loss": output_score,
            "group_score": group_score,
            "pdb_output": output_structure.structure_pdb,
            "structure_tool": "protenix",
            "fallback_used": fallback_used,
        }
    )
    return metadata


def _scoring_term_score(metrics: dict[str, Any], objective_key: str) -> tuple[float, bool]:
    """Compute the public forward Protenix confidence score for one objective.

    Returns:
        tuple[float, bool]: Score and whether fallback was used (MAX_ENERGY sentinel).
    """
    target_metric = TARGET_METRIC_BY_OBJECTIVE[objective_key]
    metric = _metric_value(metrics, target_metric)
    if metric is None:
        logger.warning("Metric %r not found in Protenix structure output, returning worst score.", target_metric)
        return MAX_ENERGY, True
    if objective_key == "plddt":
        normalized = metric / 100.0 if metric > 1.0 else metric
        return 1.0 - normalized, False
    if objective_key in {"ptm", "iptm"}:
        return 1.0 - metric, False
    if objective_key == "pae":
        return min(metric / PAE_MAXIMUM, 1.0), False
    raise ValueError(f"Unsupported Protenix scoring objective {objective_key!r}.")


def _metric_value(metrics: dict[str, Any], target_metric: str) -> float | None:
    """Return one numeric confidence metric, including canonical aliases."""
    value = resolve_metric(metrics, target_metric)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
