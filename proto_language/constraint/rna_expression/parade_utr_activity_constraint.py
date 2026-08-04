"""Score single-cell UTR activity with the PARADE predictor.

Wraps the ``parade-activity`` tool to drive a UTR segment toward high (or low) predicted
activity in one PARADE cell line. The raw activity is mapped to a ``[0, 1]`` energy through
a sigmoid centred on ``sigmoid_center``; ``direction="max"`` rewards high activity and
``direction="min"`` rewards low activity.

Examples:
    >>> from proto_language.core import Constraint, Segment
    >>> seg = Segment(length=50, sequence_type="dna")
    >>> c = Constraint(
    ...     inputs=[seg],
    ...     function=parade_utr_activity_constraint,
    ...     function_config={"construct_type": "utr5", "cell_type": "c2", "direction": "max"},
    ... )
    >>> scores = c.evaluate()  # list[float], one per proposal; lower = higher c2 activity
"""

import logging
import math
from typing import Any, Literal

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


class ParadeUTRActivityConfig(BaseConfig):
    """Configuration for the single-cell PARADE UTR activity constraint.

    Attributes:
        construct_type (ParadeConstructType): PARADE UTR model — ``"utr5"`` or ``"utr3"``.
        cell_type (ParadeCellType): Cell code whose activity to score. Must be in the
            ``construct_type`` panel.
        direction (Literal['max', 'min']): ``"max"`` rewards high activity, ``"min"``
            rewards low activity.
        sigmoid_center (float): Raw activity mapped to energy 0.5.
        sigmoid_scale (float): Positive width of the sigmoid transform in activity units.
        temperature (float): Softmax temperature for the gradient (backward) mode, which
            relaxes logits before PARADE for direct backward calls. GradientOptimizer's
            scheduled temperature takes precedence. Ignored by the forward mode.
        device (str): Device used for PARADE inference.
        batch_size (int): Number of sequences per PARADE GPU batch.
    """

    construct_type: ParadeConstructType = ConfigField(
        default="utr5",
        title="Construct Type",
        description="PARADE UTR model to score with: 'utr5' (5' UTR) or 'utr3' (3' UTR).",
    )
    cell_type: ParadeCellType = ConfigField(
        default="c2",
        title="Cell Type",
        description="PARADE cell code whose predicted activity is scored.",
    )
    direction: Literal["max", "min"] = ConfigField(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') the activity.",
    )
    sigmoid_center: float = ConfigField(
        default=2.0,
        allow_inf_nan=False,
        title="Sigmoid Center",
        description="Raw PARADE activity mapped to energy 0.5.",
    )
    sigmoid_scale: float = ConfigField(
        default=1.0,
        ge=_MIN_FLOAT32_NORMAL,
        allow_inf_nan=False,
        title="Sigmoid Scale",
        description="Positive float32-safe width of the sigmoid transform; larger is gentler.",
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
    def validate_cell_type(self) -> "ParadeUTRActivityConfig":
        """Require the cell code to belong to the construct-type panel."""
        panel = PARADE_CELL_TYPES[self.construct_type]
        if self.cell_type not in panel:
            raise ValueError(f"cell code {self.cell_type!r} is not in the {self.construct_type} panel {list(panel)}")
        return self


def parade_utr_activity_gradient_backward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: ParadeUTRActivityConfig,
    temperature: float | None = None,
    soft: float = 1.0,
    hard: float = 0.0,
    **kwargs: Any,  # noqa: ARG001
) -> list[GradientConstraintOutput]:
    """Gradient of the single-cell PARADE activity objective w.r.t. relaxed UTR logits.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-segment tuple per proposal;
            each segment carries ``L x 4`` optimizer logits. For RNA, the fourth ``U``
            column is positionally equivalent to PARADE's ``T`` column.
        config (ParadeUTRActivityConfig): Validated constraint configuration.
        temperature (float | None): Optimizer-scheduled relaxation temperature. When
            omitted for a direct call, uses ``config.temperature``.
        soft (float): Optimizer-scheduled soft-sequence mixing coefficient.
        hard (float): Optimizer-scheduled straight-through hard-forward coefficient.
        kwargs (Any): Ignored; absorbs extra keyword arguments from the backward
            calling convention.

    Returns:
        list[GradientConstraintOutput]: One result per proposal. ``gradient`` is a one-tuple
            of the segment's ``(L, 4)`` gradient; ``loss`` is PARADE's differentiable objective
            (lower is better in the requested direction).
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
                            cell_type=config.cell_type,
                            direction=config.direction,
                            sigmoid_center=config.sigmoid_center,
                            sigmoid_scale=config.sigmoid_scale,
                        )
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
                        "PARADE returned an activity gradient with shape "
                        f"{gradient_array.shape} for proposal {proposal_idx}; expected {expected_gradient_shape}."
                    )
                if not math.isfinite(sample_loss):
                    raise ValueError(
                        f"PARADE returned non-finite activity loss {sample_loss!r} for proposal {proposal_idx}."
                    )
                if not np.isfinite(gradient_array).all():
                    raise ValueError(f"PARADE returned a non-finite activity gradient for proposal {proposal_idx}.")
                activity = float(sample_metrics[config.cell_type])
                if not math.isfinite(activity):
                    raise ValueError(
                        f"PARADE returned non-finite gradient activity {activity!r} "
                        f"for cell {config.cell_type}, proposal {proposal_idx}."
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
                        "cell_type": config.cell_type,
                        "direction": config.direction,
                        "activity": activity,
                        "parade_loss": sample_loss,
                    },
                )

    assert all(result is not None for result in results)  # noqa: S101 -- every prepared proposal is assigned above
    return [result for result in results if result is not None]


@constraint(
    key="parade-utr-activity",
    label="PARADE UTR activity",
    config=ParadeUTRActivityConfig,
    description="Drive a UTR toward high or low activity in one cell line using the PARADE model.",
    uses_gpu=True,
    tools_called=["parade-activity", "parade-gradient"],
    category="rna_expression",
    supported_sequence_types=["dna", "rna"],
    input_labels=[InputSlot(label="Sequence", requires_logits=True)],
    backward=parade_utr_activity_gradient_backward,
    backward_config=ParadeUTRActivityConfig,
)
def parade_utr_activity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: ParadeUTRActivityConfig,
) -> list[ConstraintOutput]:
    """Score UTR proposals by predicted single-cell activity.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-segment tuple per proposal.
        config (ParadeUTRActivityConfig): Validated constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per proposal. ``score`` is in ``[0.0, 1.0]``
            (lower is better in the requested direction); ``metadata`` carries the raw
            ``activity``.
    """
    if not input_sequences:
        return []

    sequences = [seq.sequence for (seq,) in input_sequences]
    output = run_parade_activity(
        ParadeActivityInput(sequences=sequences),
        ParadeActivityConfig(
            construct_type=config.construct_type,
            cell_types=[config.cell_type],
            device=config.device,
            batch_size=config.batch_size,
        ),
    )

    results: list[ConstraintOutput] = []
    for result in output.results:
        activity = result.scores[config.cell_type]
        if not math.isfinite(activity):
            raise ValueError(f"PARADE returned non-finite activity {activity!r} for cell {config.cell_type}.")
        # "max": high activity -> low energy; "min": the reverse.
        sigmoid = sigmoid_score(activity, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        score = (1.0 - sigmoid) if config.direction == "max" else sigmoid
        results.append(
            ConstraintOutput(
                score=min(MAX_ENERGY, max(MIN_ENERGY, score)),
                metadata={"cell_type": config.cell_type, "direction": config.direction, "activity": activity},
            )
        )
    return results
