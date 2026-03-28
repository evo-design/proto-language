"""
proto_language/language/constraint/sequence_composition/gc_content_constraint.py

GC content constraint for evaluating sequence GC content properties.
"""

from __future__ import annotations

from typing import List, Tuple

from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.utils import (
    MAX_ENERGY,
    MAX_GC_CONTENT,
    MIN_GC_CONTENT,
    calculate_percentage_range_deviation,
    validate_range,
)


class GCContentConfig(BaseConfig):
    """Configuration for GC content constraint.

    This class defines configuration parameters for evaluating the GC content
    in DNA or RNA sequences. This penalty scales linearly with deviation from the
    acceptable range.

    Attributes:
        min_gc (float): Minimum acceptable GC content percentage (0-100). Sequences
            with GC content below this threshold are penalized. Typical values depend
            on organism, but generally ~35% is a good lower bound.

        max_gc (float): Maximum acceptable GC content percentage (0-100). Sequences
            with GC content above this threshold are penalized. Can be used to avoid
            sequences that are GC-rich, which can form secondary structures
            or have very high melting temperatures. Higher values are more permissive.

    """
    # Required parameters
    min_gc: float = ConfigField(
        ge=0,
        le=100,
        title="Min GC",
        description="Minimum acceptable GC content percentage (0-100)",
        examples=[35]
    )
    max_gc: float = ConfigField(
        ge=0,
        le=100,
        title="Max GC",
        description="Maximum acceptable GC content percentage (0-100)",
        examples=[70]
    )

    @model_validator(mode='after')
    def validate_gc_range(self):
        """Ensure min_gc <= max_gc."""
        if self.min_gc > self.max_gc:
            raise ValueError(f"min_gc ({self.min_gc}) must be <= max_gc ({self.max_gc})")
        return self


@constraint(
    key="gc-content",
    label="GC Content",
    config=GCContentConfig,
    description="Enforce GC content within specified range",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna"],
    num_input_sequences_per_tuple=1,
)
def gc_content_constraint(input_sequences: List[Tuple[Sequence, ...]], config: GCContentConfig) -> List[float]:
    """Enforce GC content within specified range.

    This constraint function calculates the percentage of guanine (G) and cytosine
    (C) nucleotides in DNA or RNA sequences and evaluates whether it falls within
    a specified acceptable range. GC content is a fundamental sequence property
    that affects DNA stability, melting temperature, gene expression patterns,
    and technical considerations like PCR amplification efficiency.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one DNA or RNA sequence. Empty sequences receive
            maximum penalty.

        config (GCContentConfig): Configuration object containing ``min_gc``
            (minimum acceptable GC percentage) and ``max_gc`` (maximum acceptable
            GC percentage). Both values must be between 0 and 100.

    Returns:
        list[float]: Constraint scores for each sequence. A score of 0.0 indicates
            GC content is within the acceptable range [min_gc, max_gc]. Higher
            scores indicate greater deviation from the acceptable range, with
            penalties scaling linearly with the deviation distance.

    Raises:
        ValueError: If the input list is empty, if min_gc or max_gc are
            outside the range [0, 100], or if a sequence is not DNA or RNA type.

    Examples:
        Evaluating GC content constraint:

        >>> seq = Sequence("ATCGATCG", "dna")
        >>> cfg = GCContentConfig(min_gc=40.0, max_gc=60.0)
        >>> score = gc_content_constraint([(seq,)], config=cfg)
        >>> print(score[0])  # 0.0 (50% GC content is within acceptable range)
    """
    validate_range(config.min_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "min_gc")
    validate_range(config.max_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "max_gc")

    scores = []

    for (seq,) in input_sequences:
        if len(seq.sequence) == 0:
            seq._metadata["gc_content"] = 0.0
            scores.append(MAX_ENERGY)
            continue

        gc_content = (
            100.0
            * sum(nt in "GC" for nt in seq.sequence.upper())
            / max(len(seq.sequence), 1)
        )

        seq._metadata["gc_content"] = gc_content

        deviation = calculate_percentage_range_deviation(gc_content, config.min_gc, config.max_gc)
        scores.append(min(MAX_ENERGY,deviation))

    return scores
