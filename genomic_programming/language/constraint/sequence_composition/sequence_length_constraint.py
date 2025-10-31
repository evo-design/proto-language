"""
Sequence length constraint for evaluating sequence length properties.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import calculate_range_deviation, calculate_normalized_deviation


class SequenceLengthConfig(BaseConfig):
    """
    Configuration for sequence length constraint.
    
    Supports two modes:
    1. Range mode: Specify min_length and max_length for acceptable range
    2. Target mode: Specify target_length for exact length matching
    """
    min_length: Optional[int] = Field(
        default=None,
        gt=0,
        description="Minimum acceptable length (use with max_length for range mode)"
    )
    max_length: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum acceptable length (use with min_length for range mode)"
    )
    target_length: Optional[int] = Field(
        default=None,
        gt=0,
        description="Target length for exact matching (alternative to min/max range)"
    )
    
    @model_validator(mode='after')
    def validate_length_config(self):
        """Ensure either (min_length + max_length) OR target_length is provided."""
        has_range = self.min_length is not None and self.max_length is not None
        has_target = self.target_length is not None
        
        if not has_range and not has_target:
            raise ValueError("Must provide either (min_length + max_length) or target_length")
        
        if has_range and has_target:
            raise ValueError("Cannot provide both range (min/max) and target_length")
        
        if has_range and self.min_length > self.max_length:
            raise ValueError(f"min_length ({self.min_length}) must be <= max_length ({self.max_length})")
        
        return self


@ConstraintRegistry.register(
    key="sequence-length",
    label="Sequence Length",
    config=SequenceLengthConfig,
    description="Evaluate sequence length against target value or acceptable range",
    vectorized=True,
    concatenate=True
)
def sequence_length_constraint(sequences: List[Sequence], config: SequenceLengthConfig) -> List[float]:
    """
    Evaluate sequence length against target or range.
    
    Supports any sequence type (DNA, RNA, protein, ligand).

    Args:
        sequences: List of sequences to evaluate.
        config: Configuration with either target_length or (min_length, max_length).

    Returns:
        List of constraint scores where 0.0 indicates perfect match/within range
        and higher values indicate greater deviation.

    Examples:
        Range mode (protein):
        >>> seqs = [Sequence("MVLSP", SequenceType.PROTEIN)]
        >>> cfg = SequenceLengthConfig(min_length=4, max_length=10)
        >>> scores = sequence_length_constraint(seqs, cfg)
        
        Target mode (DNA):
        >>> seqs = [Sequence("ATCGATCG", SequenceType.DNA)]
        >>> cfg = SequenceLengthConfig(target_length=8)
        >>> scores = sequence_length_constraint(seqs, cfg)
    """
    if not sequences:
        raise ValueError("Input sequence list must not be empty")
    
    scores = []
    use_range_mode = config.min_length is not None
    
    for seq in sequences:
        actual_length = len(seq.sequence)
        seq._metadata["length"] = actual_length
        
        if use_range_mode:
            # Range mode: check if within [min, max]
            score = calculate_range_deviation(actual_length, config.min_length, config.max_length)
            seq._metadata["length_mode"] = "range"
            seq._metadata["length_min"] = config.min_length
            seq._metadata["length_max"] = config.max_length
        else:
            # Target mode: penalize deviation from exact target
            score = calculate_normalized_deviation(actual_length, config.target_length)
            seq._metadata["length_mode"] = "target"
            seq._metadata["length_target"] = config.target_length
        
        scores.append(score)
    
    return scores
