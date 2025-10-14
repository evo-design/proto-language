"""
Balanced amino acid constraint function.
"""

from __future__ import annotations

from collections import Counter

from pydantic import Field

from ...base import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..registry import ConstraintRegistry


class BalancedAaConfig(BaseConfig):
    """Configuration for balanced amino acid constraint."""
    min_aa_frequency: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable relative frequency for any amino acid type (0.0-1.0). Amino acids below this threshold are considered underrepresented. Typical value: 0.02 (2%)."
    )
    max_underrepresented_count: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Maximum acceptable number of underrepresented amino acid types (0-20). Sequences with more underrepresented amino acids are penalized."
    )


@ConstraintRegistry.register(
    key="balanced-aa",
    config=BalancedAaConfig,
    description="Evaluate the presence of underrepresented amino acids in a protein sequence",
    vectorized=False,
    concatenate=True
)
def balanced_aa_constraint(
    input_sequence: Sequence, config: BalancedAaConfig
) -> float:
    """
    Evaluate the presence of underrepresented amino acids in a protein sequence.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing min_aa_frequency and max_underrepresented_count parameters.

    Returns:
        Constraint score from 0.0 (best, acceptable number of underrepresented amino acids) to 1.0 (worst).
        Score is scaled based on how many excess underrepresented amino acids there are and their severity.
    """
    assert input_sequence.sequence_type == SequenceType.PROTEIN, "Input must be protein"

    seq = input_sequence.sequence

    if len(seq) == 0:
        underrepresented_score = 1.0
        aa_counts = Counter()
        underrepresented_aas = []
        penalty_score = 1.0
    else:
        aa_counts = Counter(seq)
        if len(aa_counts) == 0:
            underrepresented_score = 1.0
            underrepresented_aas = []
            penalty_score = 1.0
        else:
            # Identify underrepresented amino acids (below minimum frequency threshold)
            frequency_threshold = config.min_aa_frequency * len(seq)
            underrepresented_aas = [
                aa for aa, count in aa_counts.items() if count < frequency_threshold
            ]

            # Calculate fraction of sequence that consists of underrepresented amino acids
            underrepresented_total = sum(aa_counts[aa] for aa in underrepresented_aas)
            underrepresented_score = underrepresented_total / len(seq)

            # Calculate penalty score based on count of underrepresented amino acids
            underrepresented_aa_count = len(underrepresented_aas)

            if underrepresented_aa_count <= config.max_underrepresented_count:
                penalty_score = 0.0
            else:
                # Scale penalty based on both excess count and how far amino acids are from threshold
                excess_count = underrepresented_aa_count - config.max_underrepresented_count
                max_possible_excess = (
                    20 - config.max_underrepresented_count
                )  # 20 standard amino acids

                # Calculate average "deficit" - how far underrepresented AAs are from threshold
                total_deficit = 0.0
                for aa in underrepresented_aas:
                    current_freq = aa_counts[aa] / len(seq)
                    deficit = config.min_aa_frequency - current_freq
                    total_deficit += deficit * aa_counts[aa]  # Weight by actual count

                avg_deficit = (
                    total_deficit / underrepresented_total
                    if underrepresented_total > 0
                    else 0.0
                )

                # Combine excess count with severity of underrepresentation
                count_penalty = (
                    excess_count / max_possible_excess
                    if max_possible_excess > 0
                    else 1.0
                )
                severity_penalty = (
                    avg_deficit / config.min_aa_frequency if config.min_aa_frequency > 0 else 0.0
                )
                penalty_score = min(1.0, count_penalty * (1.0 + severity_penalty))

    # Store metadata
    input_sequence._metadata["underrepresented_aa_score"] = underrepresented_score
    input_sequence._metadata["amino_acid_counts"] = dict(aa_counts)
    input_sequence._metadata["underrepresented_amino_acids"] = underrepresented_aas
    input_sequence._metadata["underrepresented_aa_count"] = len(underrepresented_aas)
    input_sequence._metadata["min_aa_frequency_threshold"] = config.min_aa_frequency

    # Return penalty score
    return penalty_score
