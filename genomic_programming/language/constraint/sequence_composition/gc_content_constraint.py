"""
GC content constraint for evaluating sequence GC content properties.
"""

from __future__ import annotations

from pydantic import Field
from typing import List

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import (
    MIN_GC_CONTENT,
    MAX_GC_CONTENT,
    validate_range,
    calculate_percentage_range_deviation,
    MAX_ENERGY,
    MIN_ENERGY
)


class GCContentConfig(BaseConfig):
    """Configuration for GC content constraint."""
    min_gc: float = Field(ge=0, le=100, description="Minimum acceptable GC content percentage (0-100)")
    max_gc: float = Field(ge=0, le=100, description="Maximum acceptable GC content percentage (0-100)")


@ConstraintRegistry.register(
    key="gc-content",
    label="GC Content",
    config=GCContentConfig,
    description="Enforce GC content within specified range",
    vectorized=True,
    concatenate=True
)
def gc_content_constraint(sequences: List[Sequence], config: GCContentConfig) -> List[float]:
    """
    Evaluate whether a sequence's GC content falls within a target range.

    Args:
        sequences: The sequence to evaluate.
        config: Configuration containing min_gc and max_gc parameters.

    Returns:
        Constraint scores where 0.0 indicates GC content is within acceptable range
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
    if not sequences:
        raise ValueError("Input sequence list must not be empty")
    
    validate_range(config.min_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "min_gc")
    validate_range(config.max_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "max_gc")
   
    scores = []
   
    for seq in sequences: 
        if seq.sequence_type not in {SequenceType.DNA, SequenceType.RNA}:
            raise ValueError(f"Input must be DNA or RNA sequence, found {seq.sequence_type}")
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
