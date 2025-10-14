"""
Protein length constraint function.
"""

from __future__ import annotations

from pydantic import Field

from ...core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....utils import calculate_range_deviation


class ProteinLengthConfig(BaseConfig):
    """Configuration for protein length constraint."""
    min_length: int = Field(gt=0, description="Minimum acceptable protein length")
    max_length: int = Field(gt=0, description="Maximum acceptable protein length")


@ConstraintRegistry.register(
    key="protein-length",
    config=ProteinLengthConfig,
    description="Evaluate whether protein length falls within acceptable range",
    vectorized=False,
    concatenate=True
)
def protein_length_constraint(
    input_sequence: Sequence, config: ProteinLengthConfig
) -> float:
    """
    Evaluate whether a protein sequence length falls within acceptable range.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing min_length and max_length parameters.

    Returns:
        Constraint score where 0.0 indicates length is within acceptable range
        and higher values indicate greater deviation from acceptable range.
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"

    actual_length = len(input_sequence)
    input_sequence._metadata["protein_length"] = actual_length

    return calculate_range_deviation(actual_length, config.min_length, config.max_length)
