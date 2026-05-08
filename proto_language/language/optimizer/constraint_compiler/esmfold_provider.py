"""ESMFold adapter for grouped differentiable confidence constraints.

ESMFold exposes pLDDT, pTM, and pAE from one model forward. The gradient
optimizer may receive those as separate public constraints with separate
weights and schedules, so this provider compiles compatible constraints into
one ESMFold backward call per proposal.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from proto_tools.tools.structure_prediction.esmfold import (
    ESMFoldConfig,
    ESMFoldGradientConfig,
    ESMFoldGradientInput,
    run_esmfold_gradient,
)
from pydantic import ValidationError

from proto_language.language.constraint.protein_structure.structure_confidence_constraint import (
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.language.core import Constraint, Segment
from proto_language.language.core.sequence import PROTEIN_AMINO_ACIDS
from proto_language.language.optimizer.constraint_compiler.base import (
    CompiledConstraint,
    EffectiveWeight,
    GradientProvider,
    GradientProviderOutput,
)

logger = logging.getLogger(__name__)

ESMFOLD_STRUCTURE_LOSS_BY_FUNCTION = {
    structure_plddt_constraint: "plddt",
    structure_ptm_constraint: "ptm",
    structure_pae_constraint: "pae",
}

TARGET_METRIC_BY_OBJECTIVE = {
    "plddt": "avg_plddt",
    "ptm": "ptm",
    "pae": "avg_pae",
}


class ESMFoldGradientProvider(GradientProvider):
    """Grouped ESMFold confidence provider."""

    def __init__(
        self,
        *,
        constraints: list[CompiledConstraint],
        config: ESMFoldConfig,
        inputs: list[Segment],
        target_segment: Segment,
    ):
        """Create a grouped ESMFold confidence provider."""
        self.constraints = constraints
        self.config = config
        self.inputs = inputs
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
        """Run one weighted ESMFold backward pass per proposal."""
        loss_weights = {
            compiled.objective_key: effective_weight(compiled.constraint, step) for compiled in self.constraints
        }
        target_chain_indices = [idx for idx, segment in enumerate(self.inputs) if segment is self.target_segment]
        num_proposals = self.inputs[0].num_proposals
        gradients: list[np.ndarray] = []
        losses: list[float] = []

        for proposal_idx in range(num_proposals):
            target_seq = self.target_segment.proposal_sequences[proposal_idx]
            if target_seq.logits is None:
                raise RuntimeError(f"{self.label} proposal {proposal_idx}: target input is missing logits.")
            if all(weight == 0.0 for weight in loss_weights.values()):
                gradients.append(np.zeros_like(target_seq.logits, dtype=np.float64))
                losses.append(0.0)
                continue

            chains = _proposal_chains(self.inputs, self.target_segment, proposal_idx)
            output = run_esmfold_gradient(
                ESMFoldGradientInput(
                    logits=target_seq.logits.tolist(),
                    temperature=temperature,
                    chains=chains,
                    target_chain_indices=target_chain_indices,
                ),
                ESMFoldGradientConfig(
                    **self.config.model_dump(),
                    loss_weights=loss_weights,
                    soft=soft,
                    hard=hard,
                    compute_gradient=True,
                ),
            )
            if output.gradient is None:
                raise RuntimeError("ESMFold compute_gradient=True must populate output.gradient.")

            gradients.append(np.array(output.gradient, dtype=np.float64))
            losses.append(output.loss)

            for compiled in self.constraints:
                score = _term_score(output.metrics, compiled.objective_key, output.loss)
                metadata = _constraint_metadata(
                    output.metrics,
                    output_structure=output.structure,
                    objective_key=compiled.objective_key,
                    output_loss=score,
                    group_loss=output.loss,
                )
                compiled.constraint._write_constraint_metadata(proposal_idx, score, metadata)

            self.target_segment.proposal_sequences[proposal_idx].structure = output.structure

        return GradientProviderOutput(label=self.label, gradients=gradients, losses=losses)


def objective_key_for_constraint(constraint: Constraint) -> str | None:
    """Return the ESMFold confidence objective key for ``constraint``."""
    if constraint.function is None:
        return None
    return ESMFOLD_STRUCTURE_LOSS_BY_FUNCTION.get(constraint.function)


def unsupported_gradient_reason(constraint: Constraint) -> str | None:
    """Return targeted errors for ESMFold structure constraints outside v1 support."""
    config = config_for_constraint(constraint)
    if config is not None and config.structure_tool == "esmfold":
        return (
            f"Constraint '{constraint.label}' with structure_tool='esmfold' is not differentiable in this compiler; "
            "supported ESMFold confidence gradients are structure-plddt, structure-ptm, and structure-pae."
        )
    return None


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


def missing_config_message(constraint: Constraint) -> str:
    """Return the standard error for constraints without parseable config."""
    return f"Constraint '{constraint.label}' must use StructureBasedConstraintConfig."


def validate_gradient_constraint(
    constraint: Constraint,
    target_segment: Segment,
    config: StructureBasedConstraintConfig,
) -> None:
    """Validate that an ESMFold confidence constraint can be differentiated."""
    if config.structure_tool != "esmfold":
        raise ValueError(
            f"Constraint '{constraint.label}' is not an ESMFold gradient constraint "
            f"(structure_tool={config.structure_tool!r})."
        )
    if constraint.threshold is not None:
        raise ValueError(f"Constraint '{constraint.label}' is a filter; filters are not differentiable objectives.")
    if target_segment not in constraint.inputs:
        raise ValueError(
            f"Constraint '{constraint.label}' inputs do not include the optimizer target_segment; "
            "GradientOptimizer can only differentiate constraints whose inputs contain the target."
        )
    expected_vocab = list(PROTEIN_AMINO_ACIDS)
    if target_segment.ordered_vocab() != expected_vocab:
        raise ValueError(
            f"Constraint '{constraint.label}' uses ESMFold gradients, which require the canonical "
            f"20-amino-acid vocab {''.join(expected_vocab)} on the target segment."
        )
    for idx, segment in enumerate(constraint.inputs):
        if segment.sequence_type != "protein":
            raise TypeError(
                f"Constraint '{constraint.label}' uses ESMFold gradients, "
                f"but input {idx} has sequence_type={segment.sequence_type!r}."
            )


def group_key(
    constraint: Constraint, target_segment: Segment, config: StructureBasedConstraintConfig
) -> tuple[Any, ...]:
    """Build the identity key used to group compatible ESMFold constraints."""
    input_ids = tuple(id(segment) for segment in constraint.inputs)
    config_json = config.esmfold_config.model_dump_json()
    return (*input_ids, id(target_segment), config_json)


def add_gradient_constraint(provider: ESMFoldGradientProvider, compiled: CompiledConstraint) -> None:
    """Attach one compiled public constraint to an existing ESMFold provider."""
    provider.constraints.append(compiled)
    provider.label = _provider_label(provider.constraints)


def _proposal_chains(inputs: list[Segment], target_segment: Segment, proposal_idx: int) -> list[str]:
    """Return hard chain sequences for one proposal, decoding target logits."""
    chains: list[str] = []
    for segment in inputs:
        seq = segment.proposal_sequences[proposal_idx]
        if segment is target_segment:
            if seq.logits is None:
                raise RuntimeError("ESMFold target proposal is missing logits.")
            chains.append(_decode_logits(seq.logits, segment.ordered_vocab()))
            continue
        if not seq.sequence:
            raise ValueError(
                f"ESMFold gradient fixed input segment {segment.label or '<unlabeled>'!r} "
                f"proposal {proposal_idx} has no sequence."
            )
        chains.append(seq.sequence)
    return chains


def _decode_logits(logits: np.ndarray, vocab: list[str]) -> str:
    """Decode logits to a hard sequence using the segment vocab."""
    indices = np.asarray(logits).argmax(axis=-1)
    return "".join(vocab[int(index)] for index in indices)


def _constraint_metadata(
    metrics: dict[str, Any],
    *,
    output_structure: Any,
    objective_key: str,
    output_loss: float,
    group_loss: float,
) -> dict[str, Any]:
    """Build per-public-constraint metadata from an ESMFold gradient output."""
    target_metric = TARGET_METRIC_BY_OBJECTIVE[objective_key]
    metadata = dict(metrics)
    # Keep the objective's display metric present even if the backend omitted it.
    metadata.setdefault(target_metric, None)
    metadata.update(
        {
            "loss_key": objective_key,
            "output_loss": output_loss,
            "group_loss": group_loss,
            "pdb_output": output_structure.structure_pdb,
            "structure_tool": "esmfold",
        }
    )
    return metadata


def _provider_label(constraints: list[CompiledConstraint]) -> str:
    """Return the grouped provider label shown in optimizer traces."""
    return "esmfold[" + ",".join(c.constraint.label for c in constraints) + "]"


def _term_score(metrics: dict[str, Any], objective_key: str, fallback: float) -> float:
    """Extract one unweighted ESMFold public objective score."""
    target_metric = TARGET_METRIC_BY_OBJECTIVE[objective_key]
    for key in [f"loss_{objective_key}", objective_key, target_metric]:
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    numeric_keys = sorted(key for key, value in metrics.items() if isinstance(value, int | float))
    logger.warning(
        "ESMFold metrics did not include a per-term score for objective %r. Available numeric metric keys: %s. "
        "Using grouped loss for that constraint's metadata score.",
        objective_key,
        ", ".join(numeric_keys) or "<none>",
    )
    return fallback
