"""Malinois adapter for grouped regulatory DNA activity constraints.

Malinois predicts K562, HepG2, and SKNSH MPRA activity in one forward pass.
The public ``malinois-activity`` constraint exposes a single cell-type max/min
objective, so this provider groups compatible public constraints into one
Malinois scoring or gradient call while preserving per-constraint metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from proto_tools import (
    MalinoisGradientConfig,
    MalinoisGradientInput,
    MalinoisGradientLossTerm,
    MalinoisScoreConfig,
    MalinoisScoreInput,
    run_malinois_gradient,
    run_malinois_score,
)
from pydantic import ValidationError

from proto_language.language.constraint.sequence_annotation.malinois_activity_constraint import (
    MalinoisActivityConfig,
    malinois_activity_constraint,
    malinois_activity_score,
)
from proto_language.language.core import Constraint, Segment
from proto_language.language.core.sequence import DNA_NUCLEOTIDES
from proto_language.language.optimizer.constraint_compiler.base import (
    CompiledConstraint,
    EffectiveWeight,
    GradientProvider,
    GradientProviderOutput,
    raise_for_failed_tool_output,
)

MALINOIS_ACTIVITY_OBJECTIVE = "activity"
_OBJECTIVE_FIELDS = {"cell_type", "direction", "sigmoid_center", "sigmoid_scale"}


class MalinoisGradientProvider(GradientProvider):
    """Grouped Malinois activity provider."""

    def __init__(
        self,
        *,
        constraints: list[CompiledConstraint],
        config: MalinoisActivityConfig,
        target_segment: Segment,
    ) -> None:
        """Create a grouped Malinois gradient provider."""
        self.constraints = constraints
        self.config = config
        self.target_segment = target_segment
        self.label = _provider_label(constraints)

    def compute(
        self,
        *,
        temperature: float,
        soft: float,
        hard: float,
        step: int,
        effective_weight: EffectiveWeight,
    ) -> GradientProviderOutput:
        """Run one weighted Malinois backward pass for all proposals."""
        num_proposals = self.target_segment.num_proposals
        target_logits: list[np.ndarray] = []

        for proposal_idx in range(num_proposals):
            target_seq = self.target_segment.proposal_sequences[proposal_idx]
            if target_seq.logits is None:
                raise RuntimeError(f"{self.label} proposal {proposal_idx}: target input is missing logits.")
            target_logits.append(target_seq.logits)

        loss_terms = [
            _gradient_loss_term(compiled.constraint, effective_weight(compiled.constraint, step))
            for compiled in self.constraints
        ]
        if all(term.weight == 0.0 for term in loss_terms):
            return GradientProviderOutput(
                label=self.label,
                gradients=[np.zeros_like(logits, dtype=np.float64) for logits in target_logits],
                losses=[0.0] * num_proposals,
            )

        output = run_malinois_gradient(
            MalinoisGradientInput(logits=[logits.tolist() for logits in target_logits], temperature=temperature),
            MalinoisGradientConfig(
                loss_terms=loss_terms,
                seq_length=self.config.seq_length,
                artifact_path=self.config.artifact_path,
                artifact_url=self.config.artifact_url,
                artifact_md5=self.config.artifact_md5,
                malinois_dir=self.config.malinois_dir,
                soft=soft,
                hard=hard,
                compute_gradient=True,
                device=self.config.device,
            ),
        )
        raise_for_failed_tool_output(output, "Malinois gradient")
        if output.gradient is None:
            raise RuntimeError("Malinois compute_gradient=True must populate output.gradient.")

        gradients = [np.array(gradient, dtype=np.float64) for gradient in output.gradient]
        if len(gradients) != num_proposals:
            raise RuntimeError(f"Malinois gradient returned {len(gradients)} gradients for {num_proposals} proposals.")

        losses = output.metrics.get("losses", [])
        if len(losses) != num_proposals:
            raise RuntimeError(f"Malinois gradient returned {len(losses)} losses for {num_proposals} proposals.")

        loss_terms_by_proposal = output.metrics.get("loss_terms", [])
        if len(loss_terms_by_proposal) != num_proposals:
            raise RuntimeError(
                f"Malinois gradient returned {len(loss_terms_by_proposal)} term-metric groups "
                f"for {num_proposals} proposals."
            )

        for proposal_idx, term_metrics in enumerate(loss_terms_by_proposal):
            for compiled, term_metric in zip(self.constraints, term_metrics, strict=True):
                score = float(term_metric["score"])
                metadata = _metadata_from_term(term_metric, group_score=float(losses[proposal_idx]))
                compiled.constraint._write_constraint_metadata(proposal_idx, score, metadata)

        return GradientProviderOutput(label=self.label, gradients=gradients, losses=[float(loss) for loss in losses])


def objective_key_for_constraint(constraint: Constraint) -> str | None:
    """Return the Malinois objective key for ``constraint``."""
    if constraint.function is malinois_activity_constraint:
        return MALINOIS_ACTIVITY_OBJECTIVE
    return None


def config_for_constraint(
    constraint: Constraint,
    *,
    strict: bool = False,
) -> MalinoisActivityConfig | None:
    """Parse a constraint's Malinois config into the canonical config model."""
    config = constraint.function_config
    if isinstance(config, MalinoisActivityConfig):
        return config
    if isinstance(config, dict):
        try:
            return MalinoisActivityConfig(**config)
        except (TypeError, ValidationError):
            if strict:
                raise
            return None
    return None


def missing_config_message(constraint: Constraint) -> str:
    """Return the standard error for constraints without parseable config."""
    return f"Constraint '{constraint.label}' must use MalinoisActivityConfig."


def validate_gradient_constraint(
    constraint: Constraint,
    target_segment: Segment,
    config: MalinoisActivityConfig,
) -> None:
    """Validate that a Malinois activity constraint can be differentiated."""
    if constraint.threshold is not None:
        raise ValueError(f"Constraint '{constraint.label}' is a filter; filters are not differentiable objectives.")
    if constraint.inputs != [target_segment]:
        raise ValueError(
            f"Constraint '{constraint.label}' must take only the GradientOptimizer target_segment as input."
        )
    if target_segment.sequence_type != "dna":
        raise TypeError(
            f"Constraint '{constraint.label}' uses Malinois gradients, "
            f"but target_segment has sequence_type={target_segment.sequence_type!r}."
        )
    expected_vocab = list(DNA_NUCLEOTIDES)
    if target_segment.ordered_vocab() != expected_vocab:
        raise ValueError(
            f"Constraint '{constraint.label}' uses Malinois gradients, which require the canonical "
            f"DNA vocab {''.join(expected_vocab)} on the target segment."
        )
    if target_segment.sequence_length != config.seq_length:
        raise ValueError(
            f"Constraint '{constraint.label}' expects seq_length={config.seq_length}, "
            f"but target_segment length is {target_segment.sequence_length}."
        )


def group_key(target_segment: Segment, config: MalinoisActivityConfig) -> tuple[Any, ...]:
    """Build the identity key used to group compatible Malinois gradients."""
    config_json = config.model_dump_json(exclude=_OBJECTIVE_FIELDS)
    return (id(target_segment), config_json)


def scoring_group_key(constraint: Constraint, config: MalinoisActivityConfig) -> tuple[Any, ...]:
    """Build the identity key used to group compatible Malinois scoring calls."""
    config_json = config.model_dump_json(exclude=_OBJECTIVE_FIELDS)
    return (id(constraint.inputs[0]), config_json)


def add_gradient_constraint(provider: MalinoisGradientProvider, compiled: CompiledConstraint) -> None:
    """Attach one compiled public constraint to an existing Malinois provider."""
    provider.constraints.append(compiled)
    provider.label = _provider_label(provider.constraints)


def can_group_scoring_constraint(
    constraint: Constraint,
    objective_key: str | None,
    config: MalinoisActivityConfig | None,
) -> bool:
    """Return whether ``constraint`` can join a grouped Malinois forward call."""
    return objective_key is not None and config is not None and constraint.threshold is None


def evaluate_scoring_group(compiled_constraints: list[CompiledConstraint], mask: list[bool]) -> list[float]:
    """Evaluate compatible Malinois activity constraints with one prediction batch."""
    first_constraint = compiled_constraints[0].constraint
    config = config_for_constraint(first_constraint, strict=True)
    if config is None:
        raise ValueError(missing_config_message(first_constraint))

    segment = first_constraint.inputs[0]
    num_proposals = segment.num_proposals
    scores = [float("nan")] * num_proposals
    proposal_indices = [idx for idx, should_eval in enumerate(mask) if should_eval]
    if not proposal_indices:
        return scores

    term_configs: list[MalinoisActivityConfig] = []
    for compiled in compiled_constraints:
        term_config = config_for_constraint(compiled.constraint, strict=True)
        if term_config is None:
            raise ValueError(missing_config_message(compiled.constraint))
        term_configs.append(term_config)

    requested_cell_types: list[str] = [term_config.cell_type for term_config in term_configs]
    cell_types = _ordered_unique(requested_cell_types)
    output = run_malinois_score(
        MalinoisScoreInput(sequences=[segment.proposal_sequences[idx].sequence for idx in proposal_indices]),
        MalinoisScoreConfig(
            cell_types=cell_types,
            seq_length=config.seq_length,
            artifact_path=config.artifact_path,
            artifact_url=config.artifact_url,
            artifact_md5=config.artifact_md5,
            malinois_dir=config.malinois_dir,
            batch_size=config.batch_size,
            device=config.device,
        ),
    )
    raise_for_failed_tool_output(output, "Malinois scoring")

    for proposal_idx, result in zip(proposal_indices, output.results, strict=True):
        term_scores: list[float] = []
        term_metadata: list[dict[str, Any]] = []
        for compiled, term_config in zip(compiled_constraints, term_configs, strict=True):
            raw_score = float(result.scores[term_config.cell_type])
            score, scaled_score, sigmoid_value = malinois_activity_score(raw_score, term_config)
            term_scores.append(score)
            term_metadata.append(
                {
                    "cell_type": term_config.cell_type,
                    "direction": term_config.direction,
                    "raw_score": raw_score,
                    "scaled_score": scaled_score,
                    "sigmoid_value": sigmoid_value,
                    "score": score,
                    "weighted_score": score * compiled.constraint.weight,
                    "sigmoid_center": term_config.sigmoid_center,
                    "sigmoid_scale": term_config.sigmoid_scale,
                }
            )

        group_score = sum(
            compiled.constraint.weight * score
            for compiled, score in zip(compiled_constraints, term_scores, strict=True)
        )
        scores[proposal_idx] = group_score

        for compiled, score, metadata in zip(compiled_constraints, term_scores, term_metadata, strict=True):
            compiled.constraint._write_constraint_metadata(
                proposal_idx,
                score,
                _metadata_from_term(metadata, group_score=group_score),
            )

    return scores


def _gradient_loss_term(constraint: Constraint, weight: float) -> MalinoisGradientLossTerm:
    """Build one tool-layer loss term from a public Malinois constraint."""
    config = config_for_constraint(constraint, strict=True)
    if config is None:
        raise ValueError(missing_config_message(constraint))
    return MalinoisGradientLossTerm(
        cell_type=config.cell_type,
        direction=config.direction,
        weight=weight,
        sigmoid_center=config.sigmoid_center,
        sigmoid_scale=config.sigmoid_scale,
    )


def _metadata_from_term(term: dict[str, Any], *, group_score: float) -> dict[str, Any]:
    """Build public metadata for one Malinois term."""
    return {
        "malinois_cell_type": term["cell_type"],
        "malinois_direction": term["direction"],
        "malinois_raw_score": term["raw_score"],
        "malinois_scaled_score": term["scaled_score"],
        "malinois_sigmoid_value": term["sigmoid_value"],
        "malinois_activity_score": term["score"],
        "malinois_weighted_activity_score": term.get("weighted_score"),
        "malinois_group_score": group_score,
        "sigmoid_center": term["sigmoid_center"],
        "sigmoid_scale": term["sigmoid_scale"],
    }


def _ordered_unique(values: list[str]) -> list[str]:
    """Return unique strings in first-seen order."""
    return list(dict.fromkeys(values))


def _provider_label(constraints: list[CompiledConstraint]) -> str:
    """Return the grouped provider label shown in optimizer traces."""
    return "malinois[" + ",".join(c.constraint.label for c in constraints) + "]"
