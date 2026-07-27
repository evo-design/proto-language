"""Score coding-sequence naturalness with the CodonFM (Encodon) codon language model.

Wraps the ``codonfm-fitness`` tool to drive a coding-sequence (CDS) segment toward high (or
low) mean codon log-likelihood under Encodon — a zero-shot proxy for a well-formed, natural
coding sequence. The raw fitness (mean per-token log-likelihood) is mapped to a ``[0, 1]``
energy through a sigmoid centred on ``sigmoid_center``; ``direction="max"`` rewards high
fitness (low energy) and ``direction="min"`` rewards low fitness.

This constraint is discrete-only (no gradient/backward mode): Encodon operates on codon-level
logits over 64 codons, which do not match the optimizer's per-nucleotide ``(L, 4)`` sequence
logits, so there is no faithful gradient bridge. Use it in discrete/semigreedy optimization.

Examples:
    >>> from proto_language.core import Constraint, Segment
    >>> seg = Segment(length=30, sequence_type="dna")  # 10 codons
    >>> c = Constraint(
    ...     inputs=[seg],
    ...     function=codonfm_fitness_constraint,
    ...     function_config={"model_checkpoint": "encodon_80m", "direction": "max"},
    ... )
    >>> scores = c.evaluate()  # list[float]; lower = more model-typical coding sequence
"""

import math
from typing import Literal

from proto_tools import (
    CodonFMCheckpoint,
    CodonFMFitnessConfig,
    CodonFMFitnessInput,
    run_codonfm_fitness,
)

from proto_language.constraint.constraint_registry import InputSlot, constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY, sigmoid_score
from proto_language.utils.base import BaseConfig, ConfigField

_MIN_FLOAT32_NORMAL = 1.1754943508222875e-38


class CodonFMFitnessConstraintConfig(BaseConfig):
    """Configuration for the CodonFM coding-sequence fitness constraint.

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Encodon checkpoint to score with.
        direction (Literal['max', 'min']): ``"max"`` rewards high fitness (natural coding
            sequence), ``"min"`` rewards low fitness.
        sigmoid_center (float): Raw mean-log-likelihood mapped to energy 0.5. Encodon's
            per-token log-likelihood is negative, so this is a negative tunable midpoint.
        sigmoid_scale (float): Positive width of the sigmoid transform in log-likelihood units.
        device (str): Device used for CodonFM inference.
        batch_size (int): Number of sequences per CodonFM GPU batch.
    """

    model_checkpoint: CodonFMCheckpoint = ConfigField(
        default="encodon_80m",
        title="Model Checkpoint",
        description="Encodon checkpoint: encodon_80m | encodon_600m | encodon_1b | encodon_1b_cdwt.",
    )
    direction: Literal["max", "min"] = ConfigField(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') the coding-sequence fitness.",
    )
    sigmoid_center: float = ConfigField(
        default=-2.0,
        allow_inf_nan=False,
        title="Sigmoid Center",
        description="Raw mean codon log-likelihood mapped to energy 0.5 (tunable; negative).",
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
        description="Device used for CodonFM inference.",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences scored per CodonFM GPU batch.",
    )


@constraint(
    key="codonfm-fitness",
    label="CodonFM Fitness",
    config=CodonFMFitnessConstraintConfig,
    description="Drive a coding sequence toward high or low naturalness using the CodonFM/Encodon model.",
    uses_gpu=True,
    tools_called=["codonfm-fitness"],
    category="sequence_scoring",
    supported_sequence_types=["dna", "rna"],
    input_labels=[InputSlot(label="Sequence")],
)
def codonfm_fitness_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: CodonFMFitnessConstraintConfig,
) -> list[ConstraintOutput]:
    """Score coding-sequence proposals by CodonFM (Encodon) mean codon log-likelihood.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-segment tuple per proposal.
        config (CodonFMFitnessConstraintConfig): Validated constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per proposal. ``score`` is in ``[0.0, 1.0]`` (lower
            is better in the requested direction); ``metadata`` carries the raw ``fitness``.
    """
    if not input_sequences:
        return []

    sequences = [seq.sequence for (seq,) in input_sequences]
    output = run_codonfm_fitness(
        CodonFMFitnessInput(sequences=sequences),
        CodonFMFitnessConfig(
            model_checkpoint=config.model_checkpoint,
            device=config.device,
            batch_size=config.batch_size,
        ),
    )

    results: list[ConstraintOutput] = []
    for result in output.results:
        fitness = result.fitness
        if not math.isfinite(fitness):
            raise ValueError(f"CodonFM returned non-finite fitness {fitness!r}.")
        # "max": high fitness -> low energy; "min": the reverse.
        sigmoid = sigmoid_score(fitness, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        score = (1.0 - sigmoid) if config.direction == "max" else sigmoid
        results.append(
            ConstraintOutput(
                score=min(MAX_ENERGY, max(MIN_ENERGY, score)),
                metadata={
                    "direction": config.direction,
                    "fitness": fitness,
                    "model_checkpoint": config.model_checkpoint,
                },
            )
        )
    return results
