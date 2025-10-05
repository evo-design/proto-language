"""
Sequence length constraint for evaluating sequence length properties.
"""

from __future__ import annotations

from pydantic import Field

from ...base import Sequence
from ...base.config import BaseConfig
from ..registry import ConstraintRegistry
from ..utils import calculate_normalized_deviation


class SequenceLengthConfig(BaseConfig):
    """Configuration for sequence length constraint."""
    target_length: int = Field(
        gt=0,
        description="Target sequence length in nucleotides or amino acids. Must be a positive integer."
    )


@ConstraintRegistry.register(
    key="sequence-length",
    config=SequenceLengthConfig,
    description="Evaluate sequence length against target value",
    vectorized=False,
    concatenate=True
)
def sequence_length_constraint(input_sequence: Sequence, config: SequenceLengthConfig) -> float:
    """
    Evaluate how well a sequence matches a target length.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration containing the target_length parameter.

    Returns:
        Constraint score where 0.0 indicates perfect length match
        and higher values indicate greater deviation from target length.

    Examples:
        Evaluating length constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> cfg = SequenceLengthConfig(target_length=8)
        >>> score = sequence_length_constraint(seq, config=cfg)
        >>> print(score)  # 0.0 (perfect match)
    """
    input_sequence._metadata["length"] = len(input_sequence)
    return calculate_normalized_deviation(len(input_sequence), config.target_length)
