"""
GC content constraint for evaluating sequence GC content properties.
"""

from __future__ import annotations

from pydantic import Field

from ...base import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..registry import ConstraintRegistry
from ..utils import (
    MIN_GC_CONTENT,
    MAX_GC_CONTENT,
    validate_range,
    calculate_percentage_range_deviation,
)


class GCContentConfig(BaseConfig):
    """Configuration for GC content constraint."""
    min_gc: float = Field(ge=0, le=100, description="Minimum acceptable GC content percentage (0-100)")
    max_gc: float = Field(ge=0, le=100, description="Maximum acceptable GC content percentage (0-100)")


@ConstraintRegistry.register(
    key="gc-content",
    config=GCContentConfig,
    description="Enforce GC content within specified range",
    vectorized=False,
    concatenate=True
)
def gc_content_constraint(
    input_sequence: Sequence, config: GCContentConfig
) -> float:
    """
    Evaluate whether a sequence's GC content falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration containing min_gc and max_gc parameters.

    Returns:
        Constraint score where 0.0 indicates GC content is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Raises:
        ValueError: If min_gc or max_gc are outside the range [0, 100].
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating GC content constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> cfg = GCContentConfig(min_gc=40.0, max_gc=60.0)
        >>> score = gc_content_constraint(seq, config=cfg)
        >>> print(score)  # 0.0 (50% GC content is within acceptable range)
    """
    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    validate_range(config.min_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "min_gc")
    validate_range(config.max_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "max_gc")

    gc_content = (
        100.0
        * sum(nt in "GC" for nt in input_sequence.sequence.upper())
        / max(len(input_sequence), 1)
    )

    input_sequence._metadata["gc_content"] = gc_content

    return calculate_percentage_range_deviation(gc_content, config.min_gc, config.max_gc)
