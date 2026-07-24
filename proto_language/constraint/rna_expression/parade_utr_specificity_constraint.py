"""Score cell-type-specific UTR activity with the PARADE predictor.

Wraps the ``parade-activity`` tool to drive a UTR segment toward high activity in an
on-target cell line and low activity in an off-target one — the cell-type-specificity
objective from PARADE (Khoroshkin et al., 2024). The per-proposal energy averages a
maximize-on-target sigmoid term and a minimize-off-target sigmoid term, matching the
differentiable objective used during gradient optimization.

Examples:
    >>> from proto_language.core import Constraint, Segment
    >>> seg = Segment(length=50, sequence_type="dna")
    >>> c = Constraint(
    ...     inputs=[seg],
    ...     function=parade_utr_specificity_constraint,
    ...     function_config={"construct_type": "utr5", "on_target": "c2", "off_target": "c6"},
    ... )
    >>> scores = c.evaluate()  # list[float], one per proposal; lower = more specific
"""

import logging
import math
from typing import Any

import numpy as np
from proto_tools import (
    PARADE_CELL_TYPES,
    ParadeActivityConfig,
    ParadeActivityInput,
    ParadeCellType,
    ParadeConstructType,
    ParadeGradientConfig,
    ParadeGradientInput,
    ParadeGradientLossTerm,
    run_parade_activity,
    run_parade_gradient,
)
from pydantic import model_validator

from proto_language.constraint.constraint_registry import InputSlot, constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.core.constraint import GradientConstraintOutput
from proto_language.utils import MAX_ENERGY, MIN_ENERGY, sigmoid_score
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

_MIN_FLOAT32_NORMAL = 1.1754943508222875e-38


class ParadeUTRSpecificityConfig(BaseConfig):
    """Configuration for the PARADE UTR cell-type-specificity constraint.

    Attributes:
        construct_type (ParadeConstructType): Which PARADE UTR model to score with —
            ``"utr5"`` (5' UTR) or ``"utr3"`` (3' UTR). Selects the checkpoint and the
            cell-code panel.
        on_target (ParadeCellType): Cell code whose predicted activity to maximize. Must
            be in the ``construct_type`` panel.
        off_target (ParadeCellType): Cell code whose predicted activity to minimize. Must
            be in the panel and differ from ``on_target``.
        sigmoid_center (float): Raw activity mapped to the sigmoid midpoint in both the
            forward and gradient objectives (for both on- and off-target terms).
        sigmoid_scale (float): Positive width of the shared sigmoid transform in activity units.
        temperature (float): Softmax temperature for the gradient (backward) mode, which relaxes
            logits before PARADE for direct backward calls. GradientOptimizer's scheduled
            temperature takes precedence. Ignored by the forward mode.
        device (str): Device used for PARADE inference.
        batch_size (int): Number of sequences per PARADE GPU batch.
    """

    construct_type: ParadeConstructType = ConfigField(
        default="utr5",
        title="Construct Type",
        description="PARADE UTR model to score with: 'utr5' (5' UTR) or 'utr3' (3' UTR).",
    )
    on_target: ParadeCellType = ConfigField(
        default="c2",
        title="On-target Cell",
        description="PARADE cell code whose predicted activity should be maximized.",
    )
    off_target: ParadeCellType = ConfigField(
        default="c6",
        title="Off-target Cell",
        description="PARADE cell code whose predicted activity should be minimized.",
    )
    sigmoid_center: float = ConfigField(
        default=2.0,
        allow_inf_nan=False,
        title="Sigmoid Center",
        description="Raw PARADE activity at the shared forward/gradient sigmoid midpoint.",
    )
    sigmoid_scale: float = ConfigField(
        default=1.0,
        ge=_MIN_FLOAT32_NORMAL,
        allow_inf_nan=False,
        title="Sigmoid Scale",
        description="Positive float32-safe width of the shared sigmoid transform; larger is gentler.",
    )
    temperature: float = ConfigField(
        default=1.0,
        ge=_MIN_FLOAT32_NORMAL,
        allow_inf_nan=False,
        title="Gradient Temperature",
        description="Fallback softmax temperature for direct gradient calls; optimizer schedules take precedence.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device used for PARADE inference.",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences scored per PARADE GPU batch.",
    )

    @model_validator(mode="after")
    def validate_targets(self) -> "ParadeUTRSpecificityConfig":
        """Require distinct on/off cells that both belong to the construct-type panel."""
        panel = PARADE_CELL_TYPES[self.construct_type]
        offpanel = [code for code in (self.on_target, self.off_target) if code not in panel]
        if offpanel:
            raise ValueError(f"cell codes {offpanel} are not in the {self.construct_type} panel {list(panel)}")
        if self.on_target == self.off_target:
            raise ValueError("on_target and off_target must differ")
        return self


def parade_utr_specificity_gradient_backward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: ParadeUTRSpecificityConfig,
    temperature: float | None = None,
    soft: float = 1.0,
    hard: float = 0.0,
    **kwargs: Any,  # noqa: ARG001
) -> list[GradientConstraintOutput]:
    """Gradient of matched maximize-on and minimize-off PARADE terms w.r.t. relaxed UTR logits.

    The differentiable objective averages two equally weighted PARADE loss terms — maximize
    on-target activity and minimize off-target activity — so the gradient drives the on-target
    up and the off-target down simultaneously.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-segment tuple per proposal;
            each segment carries ``L x 4`` optimizer logits. For RNA, the fourth ``U``
            column is positionally equivalent to PARADE's ``T`` column.
        config (ParadeUTRSpecificityConfig): Validated constraint configuration.
        temperature (float | None): Optimizer-scheduled relaxation temperature. When
            omitted for a direct call, uses ``config.temperature``.
        soft (float): Optimizer-scheduled soft-sequence mixing coefficient.
        hard (float): Optimizer-scheduled straight-through hard-forward coefficient.
        kwargs (Any): Ignored; absorbs extra keyword arguments from the backward
            calling convention.

    Returns:
        list[GradientConstraintOutput]: One result per proposal. ``gradient`` is a one-tuple of
            the segment's ``(L, 4)`` gradient; ``loss`` is PARADE's weighted-mean objective.
    """
    if not input_sequences:
        return []

    effective_temperature = config.temperature if temperature is None else temperature
    if not math.isfinite(effective_temperature) or effective_temperature < _MIN_FLOAT32_NORMAL:
        raise ValueError(
            "PARADE gradient temperature must be finite and at least the smallest positive normal float32 value "
            f"({_MIN_FLOAT32_NORMAL}); got {effective_temperature!r}."
        )
    grouped_logits: dict[int, list[tuple[int, list[list[float]]]]] = {}
    for proposal_idx, (seq,) in enumerate(input_sequences):
        logits = seq.logits
        assert logits is not None  # noqa: S101 -- the requires_logits input slot guarantees it
        logits_array = np.asarray(logits, dtype=float)
        expected_vocab = ["A", "C", "G", "T" if seq.sequence_type == "dna" else "U"]
        actual_vocab = seq.ordered_vocab()
        if actual_vocab != expected_vocab or logits_array.ndim != 2 or logits_array.shape != (len(seq.sequence), 4):
            raise ValueError(
                "PARADE requires canonical DNA/RNA logits with shape (L, 4), "
                "where L matches the sequence length, in A,C,G,T/U order; "
                f"got shape {logits_array.shape} with vocabulary {actual_vocab}."
            )
        grouped_logits.setdefault(logits_array.shape[0], []).append((proposal_idx, logits_array.tolist()))

    results: list[GradientConstraintOutput | None] = [None] * len(input_sequences)
    for same_length_logits in grouped_logits.values():
        for start in range(0, len(same_length_logits), config.batch_size):
            batch = same_length_logits[start : start + config.batch_size]
            output = run_parade_gradient(
                ParadeGradientInput(
                    logits=[proposal_logits for _, proposal_logits in batch], temperature=effective_temperature
                ),
                ParadeGradientConfig(
                    construct_type=config.construct_type,
                    loss_terms=[
                        ParadeGradientLossTerm(
                            cell_type=config.on_target,
                            direction="max",
                            weight=0.5,
                            sigmoid_center=config.sigmoid_center,
                            sigmoid_scale=config.sigmoid_scale,
                        ),
                        ParadeGradientLossTerm(
                            cell_type=config.off_target,
                            direction="min",
                            weight=0.5,
                            sigmoid_center=config.sigmoid_center,
                            sigmoid_scale=config.sigmoid_scale,
                        ),
                    ],
                    compute_gradient=True,
                    device=config.device,
                    soft=soft,
                    hard=hard,
                ),
            )
            assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
            if len(output.gradient) != len(batch) or len(output.sample_metrics) != len(batch):
                raise ValueError(
                    "PARADE gradient output cardinality mismatch: "
                    f"expected {len(batch)}, got {len(output.gradient)} gradients and "
                    f"{len(output.sample_metrics)} sample metrics."
                )
            for (proposal_idx, proposal_logits), gradient, sample_metrics in zip(
                batch, output.gradient, output.sample_metrics, strict=True
            ):
                sample_loss = float(sample_metrics["loss"])
                gradient_array = np.asarray(gradient, dtype=np.float64)
                expected_gradient_shape = (len(proposal_logits), 4)
                if gradient_array.shape != expected_gradient_shape:
                    raise ValueError(
                        "PARADE returned a specificity gradient with shape "
                        f"{gradient_array.shape} for proposal {proposal_idx}; expected {expected_gradient_shape}."
                    )
                if not math.isfinite(sample_loss):
                    raise ValueError(
                        f"PARADE returned non-finite specificity loss {sample_loss!r} for proposal {proposal_idx}."
                    )
                if not np.isfinite(gradient_array).all():
                    raise ValueError(f"PARADE returned a non-finite specificity gradient for proposal {proposal_idx}.")
                on_activity = float(sample_metrics[config.on_target])
                off_activity = float(sample_metrics[config.off_target])
                specificity = on_activity - off_activity
                if not all(math.isfinite(value) for value in (on_activity, off_activity, specificity)):
                    raise ValueError(
                        "PARADE returned non-finite gradient specificity activities for proposal "
                        f"{proposal_idx}: {config.on_target}={on_activity!r}, "
                        f"{config.off_target}={off_activity!r}."
                    )
                sample_data = dict(sample_metrics.items())
                loss_terms = getattr(sample_metrics, "loss_terms", None)
                if loss_terms:
                    sample_data["loss_terms"] = loss_terms
                results[proposal_idx] = GradientConstraintOutput(
                    gradient=(gradient_array,),
                    loss=sample_loss,
                    metrics={
                        **sample_data,
                        "on_target": config.on_target,
                        "off_target": config.off_target,
                        "on_activity": on_activity,
                        "off_activity": off_activity,
                        "specificity": specificity,
                        "parade_loss": sample_loss,
                    },
                )

    assert all(result is not None for result in results)  # noqa: S101 -- every prepared proposal is assigned above
    return [result for result in results if result is not None]


@constraint(
    key="parade-utr-specificity",
    label="PARADE UTR cell-type specificity",
    config=ParadeUTRSpecificityConfig,
    description="Drive a UTR toward high on-target and low off-target activity using the PARADE model.",
    uses_gpu=True,
    tools_called=["parade-activity", "parade-gradient"],
    category="rna_expression",
    supported_sequence_types=["dna", "rna"],
    input_labels=[InputSlot(label="Sequence", requires_logits=True)],
    backward=parade_utr_specificity_gradient_backward,
    backward_config=ParadeUTRSpecificityConfig,
)
def parade_utr_specificity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: ParadeUTRSpecificityConfig,
) -> list[ConstraintOutput]:
    """Score UTR proposals with matched maximize-on and minimize-off activity terms.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-segment tuple per proposal.
        config (ParadeUTRSpecificityConfig): Validated constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per proposal. ``score`` is the mean of the
            maximize-on-target and minimize-off-target sigmoid energies in ``[0.0, 1.0]``;
            ``metadata`` carries both raw activities and their difference.
    """
    if not input_sequences:
        return []

    sequences = [seq.sequence for (seq,) in input_sequences]
    output = run_parade_activity(
        ParadeActivityInput(sequences=sequences),
        ParadeActivityConfig(
            construct_type=config.construct_type,
            cell_types=[config.on_target, config.off_target],
            device=config.device,
            batch_size=config.batch_size,
        ),
    )

    results: list[ConstraintOutput] = []
    for result in output.results:
        on_activity = result.scores[config.on_target]
        off_activity = result.scores[config.off_target]
        if not math.isfinite(on_activity) or not math.isfinite(off_activity):
            raise ValueError(
                "PARADE returned non-finite specificity activities: "
                f"{config.on_target}={on_activity!r}, {config.off_target}={off_activity!r}."
            )
        specificity = on_activity - off_activity
        if not math.isfinite(specificity):
            raise ValueError(
                "PARADE specificity difference is non-finite: "
                f"{config.on_target}={on_activity!r}, {config.off_target}={off_activity!r}."
            )
        on_sigmoid = sigmoid_score(on_activity, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        off_sigmoid = sigmoid_score(off_activity, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        score = 0.5 * ((1.0 - on_sigmoid) + off_sigmoid)
        results.append(
            ConstraintOutput(
                score=min(MAX_ENERGY, max(MIN_ENERGY, score)),
                metadata={
                    "on_target": config.on_target,
                    "off_target": config.off_target,
                    "on_activity": on_activity,
                    "off_activity": off_activity,
                    "specificity": specificity,
                },
            )
        )
    return results
