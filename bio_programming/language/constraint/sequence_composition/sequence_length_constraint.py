"""
proto_language/language/constraint/sequence_composition/sequence_length_constraint.py

Sequence length constraint for evaluating sequence length properties.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.utils import (
    calculate_normalized_deviation,
    calculate_range_deviation,
)


class SequenceLengthConfig(BaseConfig):
    """Configuration for sequence length constraint.

    This class defines configuration parameters for evaluating sequence length
    in DNA, RNA, or protein sequences. The constraint supports two modes:
    range mode (specify acceptable length range) and target mode (specify exact
    target length).

    Supports two mutually exclusive modes:
    1. **Range mode**: Specify both min_length and max_length to define an
       acceptable length range. Sequences within this range receive score 0.0,
       while those outside are penalized based on distance from the range.
    2. **Target mode**: Specify target_length for exact length matching. Sequences
       exactly matching the target receive score 0.0, while deviations are penalized
       based on proportional distance from the target.

    Attributes:
        min_length (int | None): Minimum acceptable sequence length in nucleotides
            or amino acids. Must be a positive integer. Use together with max_length
            for range mode.
            Default: None.
        max_length (int | None): Maximum acceptable sequence length in nucleotides
            or amino acids. Must be a positive integer and ≥ min_length. Use together
            with min_length for range mode. Default: None.
        target_length (int | None): Target sequence length for exact matching
            (alternative to min/max range). Must be a positive integer. Cannot be
            used together with min_length/max_length. Default: None.
    """
    # Required parameters
    min_length: Optional[int] = ConfigField(
        title="Minimum Acceptable Length",
        default=None,
        gt=0,
        description="Minimum acceptable length (use with max_length for range mode)",
    )

    # Advanced parameters
    max_length: Optional[int] = ConfigField(
        title="Maximum Acceptable Length",
        default=None,
        gt=0,
        description="Maximum acceptable length (use with min_length for range mode)",
        advanced=True,
    )
    target_length: Optional[int] = ConfigField(
        title="Target Length",
        default=None,
        gt=0,
        description="Target length for exact matching (alternative to min/max range)",
        advanced=True,
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


@constraint(
    key="sequence-length",
    label="Sequence Length",
    config=SequenceLengthConfig,
    description="Evaluate sequence length against target value or acceptable range",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna", "protein"],
    num_input_sequences_per_tuple=1,
)
def sequence_length_constraint(input_sequences: List[Tuple[Sequence, ...]], config: SequenceLengthConfig) -> List[float]:
    """Evaluate sequence length against target value or acceptable range.

    This constraint function evaluates whether sequences have appropriate lengths.
    It supports two modes: range mode (acceptable length window) and target mode
    (exact length matching).

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one sequence (DNA, RNA, or protein).

        config (SequenceLengthConfig): Configuration object containing either
            (``min_length`` AND ``max_length``) for range mode OR ``target_length``
            for target mode. Cannot specify both modes simultaneously.

    Returns:
        List[float]: Constraint scores for each sequence. A score of 0.0 indicates
            the sequence meets the length requirement (within range or at target).
            Higher scores indicate greater deviation:
            - **Range mode**: Linear penalty based on distance outside [min, max].
              Score = 0.0 if within range, else proportional to deviation distance.
            - **Target mode**: Normalized penalty as |actual - target| / target.
              For example, 10% deviation from target yields score ~0.1.

    Raises:
        ValueError: If the input list is empty, or if configuration is
            invalid (neither mode specified, both modes specified, or min > max).

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Metadata varies by mode:

        **For range mode:**
        - ``length``: Integer actual sequence length
        - ``length_mode``: String "range"
        - ``length_min``: Integer minimum acceptable length
        - ``length_max``: Integer maximum acceptable length

        **For target mode:**
        - ``length``: Integer actual sequence length
        - ``length_mode``: String "target"
        - ``length_target``: Integer target length

    Examples:
        Range mode (protein):
        >>> seqs = [(Sequence("MVLSP", "protein"),)]
        >>> cfg = SequenceLengthConfig(min_length=4, max_length=10)
        >>> scores = sequence_length_constraint(seqs, cfg)

        Target mode (DNA):
        >>> seqs = [(Sequence("ATCGATCG", "dna"),)]
        >>> cfg = SequenceLengthConfig(target_length=8)
        >>> scores = sequence_length_constraint(seqs, cfg)
    """
    scores = []
    use_range_mode = config.min_length is not None

    for (seq,) in input_sequences:
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
