"""
Protein repetitiveness constraint function.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from pydantic import Field

from ...core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....utils import MIN_ENERGY, MAX_ENERGY


def _calculate_repetitiveness_score(seq: str, min_repeat_length: int = 3) -> float:
    """
    Calculate repetitiveness score based on k-mer frequency analysis

    Args:
        seq: Protein sequence to analyze
        min_repeat_length: Minimum length of repeats to consider

    Returns:
        Maximum fraction of sequence covered by repeated k-mers (0.0 to 1.0)

    Raises:
        ValueError: If length of sequence is shorter than the minimum repeat length
    """
    if len(seq) < min_repeat_length:
        raise ValueError("Sequence must be longer that the minimum repeat length")

    seq_len = len(seq)
    seq_array = np.array(list(seq))
    max_repetitive_fraction = 0.0

    for k in range(min_repeat_length, min(min_repeat_length + 7, seq_len + 1)):
        kmers = np.lib.stride_tricks.sliding_window_view(seq_array, k)
        kmer_strings = ["".join(kmer) for kmer in kmers]
        if kmer_strings:
            max_count = max(Counter(kmer_strings).values())
            repetitive_fraction = (max_count * k) / seq_len
            max_repetitive_fraction = max(max_repetitive_fraction, repetitive_fraction)

    return max_repetitive_fraction


class ProteinRepetitivenessConfig(BaseConfig):
    """Configuration for protein repetitiveness constraint."""
    max_repetitiveness: float = Field(
        ge=0.0,
        le=1.0,
        description="Maximum acceptable repetitiveness fraction (0.0-1.0). Measures the maximum fraction of sequence covered by repeated k-mers. Typical values: 0.3-0.5."
    )
    min_repeat_length: int = Field(
        default=3,
        ge=1,
        description="Minimum k-mer length to consider as repeats. Shorter values (3-4) detect short tandem repeats, longer values (5-7) detect larger structural repeats."
    )


@ConstraintRegistry.register(
    key="protein-repetitiveness",
    config=ProteinRepetitivenessConfig,
    description="Evaluate protein sequence repetitiveness based on k-mer analysis",
    vectorized=False,
    concatenate=True
)
def protein_repetitiveness_constraint(
    input_sequence: Sequence, config: ProteinRepetitivenessConfig
) -> float:
    """
    Evaluate protein sequence repetitiveness based on k-mer analysis.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing max_repetitiveness and min_repeat_length parameters.

    Returns:
        Constraint score where 0.0 indicates acceptable repetitiveness
        and higher values indicate excessive repetitive content.
    """
    assert input_sequence.sequence_type == SequenceType.PROTEIN, "Input must be protein"

    repetitiveness_score = _calculate_repetitiveness_score(
        input_sequence.sequence, config.min_repeat_length
    )
    input_sequence._metadata["repetitiveness_score"] = repetitiveness_score
    input_sequence._metadata["max_repetitive_fraction"] = repetitiveness_score

    if repetitiveness_score <= config.max_repetitiveness:
        return MIN_ENERGY

    excess = repetitiveness_score - config.max_repetitiveness
    return min(MAX_ENERGY, excess / (1.0 - config.max_repetitiveness))
