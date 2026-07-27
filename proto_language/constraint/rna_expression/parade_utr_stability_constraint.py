"""Score 3' UTR mRNA stability with the PARADE predictor.

Wraps the ``parade-stability`` tool to drive a 3' UTR segment toward high (or low)
predicted mRNA stability (the RNA/gDNA log-ratio). The raw log-ratio is mapped to a
``[0, 1]`` energy through a sigmoid centred on ``sigmoid_center``; ``direction="max"``
rewards high stability and ``direction="min"`` rewards low stability. PARADE trained
this model on 186-nt 3' UTRs, so score near that length.

Examples:
    >>> from proto_language.core import Constraint, Segment
    >>> seg = Segment(length=186, sequence_type="dna")
    >>> c = Constraint(
    ...     inputs=[seg],
    ...     function=parade_utr_stability_constraint,
    ...     function_config={"direction": "max"},
    ... )
    >>> scores = c.evaluate()  # list[float], one per proposal; lower = more stable
"""

import logging
import math
from typing import Literal

from proto_tools import (
    ParadeStabilityConfig,
    ParadeStabilityInput,
    run_parade_stability,
)

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY, sigmoid_score
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

_MIN_FLOAT32_NORMAL = 1.1754943508222875e-38


class ParadeUTRStabilityConfig(BaseConfig):
    """Configuration for the PARADE 3' UTR mRNA-stability constraint.

    Attributes:
        direction (Literal['max', 'min']): ``"max"`` rewards high predicted stability,
            ``"min"`` rewards low stability.
        sigmoid_center (float): Raw log-ratio mapped to energy 0.5.
        sigmoid_scale (float): Positive width of the sigmoid transform in log-ratio units.
        device (str): Device used for PARADE inference.
        batch_size (int): Number of sequences per PARADE GPU batch.
    """

    direction: Literal["max", "min"] = ConfigField(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') predicted mRNA stability.",
    )
    sigmoid_center: float = ConfigField(
        default=0.0,
        allow_inf_nan=False,
        title="Sigmoid Center",
        description="Raw PARADE log-ratio mapped to energy 0.5.",
    )
    sigmoid_scale: float = ConfigField(
        default=1.0,
        ge=_MIN_FLOAT32_NORMAL,
        allow_inf_nan=False,
        title="Sigmoid Scale",
        description="Positive float32-safe width of the sigmoid transform; larger is gentler.",
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


@constraint(
    key="parade-utr-stability",
    label="PARADE 3' UTR stability",
    config=ParadeUTRStabilityConfig,
    description="Drive a 3' UTR toward high or low predicted mRNA stability using the PARADE model.",
    uses_gpu=True,
    tools_called=["parade-stability"],
    category="rna_expression",
    supported_sequence_types=["dna", "rna"],
)
def parade_utr_stability_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: ParadeUTRStabilityConfig,
) -> list[ConstraintOutput]:
    """Score 3' UTR proposals by predicted mRNA stability.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-segment tuple per proposal.
        config (ParadeUTRStabilityConfig): Validated constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per proposal. ``score`` is in ``[0.0, 1.0]``
            (lower is better in the requested direction); ``metadata`` carries the raw
            ``log_ratio``.
    """
    if not input_sequences:
        return []

    sequences = [seq.sequence for (seq,) in input_sequences]
    output = run_parade_stability(
        ParadeStabilityInput(sequences=sequences),
        ParadeStabilityConfig(device=config.device, batch_size=config.batch_size),
    )

    results: list[ConstraintOutput] = []
    for result in output.results:
        log_ratio = result.log_ratio
        if not math.isfinite(log_ratio):
            raise ValueError(f"PARADE returned non-finite stability log-ratio {log_ratio!r}.")
        # "max": high stability -> low energy; "min": the reverse.
        sigmoid = sigmoid_score(log_ratio, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        score = (1.0 - sigmoid) if config.direction == "max" else sigmoid
        results.append(
            ConstraintOutput(
                score=min(MAX_ENERGY, max(MIN_ENERGY, score)),
                metadata={"direction": config.direction, "log_ratio": log_ratio},
            )
        )
    return results
