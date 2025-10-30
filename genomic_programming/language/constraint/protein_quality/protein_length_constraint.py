"""
Protein length constraint function.
"""

from __future__ import annotations

from pydantic import Field
from typing import List
import numpy as np

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import calculate_range_deviation


class ProteinLengthConfig(BaseConfig):
    """Configuration for protein length constraint."""
    min_length: int = Field(gt=0, description="Minimum acceptable protein length")
    max_length: int = Field(gt=0, description="Maximum acceptable protein length")


@ConstraintRegistry.register(
    key="protein-length",
    label="Protein Length",
    config=ProteinLengthConfig,
    description="Evaluate whether protein length falls within acceptable range",
    vectorized=True,
    concatenate=True
)
def protein_length_constraint(
    sequences: List[Sequence], config: ProteinLengthConfig
) -> List[float]:
    """
    Evaluate whether a protein sequence length falls within acceptable range.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing min_length and max_length parameters.

    Returns:
        Constraint score where 0.0 indicates length is within acceptable range
        and higher values indicate greater deviation from acceptable range.
    """
    scores = []
    
    for seq in sequences:
        assert seq.sequence_type == SequenceType.PROTEIN, "Input must be protein"
        actual_length = len(seq)
        seq._metadata["protein_length"] = actual_length
        score = calculate_range_deviation(actual_length, config.min_length, config.max_length)
        scores.append(score)

    return scores
