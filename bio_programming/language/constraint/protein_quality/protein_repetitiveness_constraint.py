"""
Protein repetitiveness constraint function.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from typing import List

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import MIN_ENERGY, MAX_ENERGY


class ProteinRepetitivenessConfig(BaseConfig):
    """Configuration for protein repetitiveness constraint.
    
    This class defines configuration parameters for evaluating repetitive content
    in protein sequences using k-mer frequency analysis. The constraint detects
    and penalizes sequences with excessive tandem repeats or repetitive motifs,
    which may indicate low-complexity regions or non-functional proteins. The
    repetitiveness score is calculated as the maximum fraction of the sequence
    covered by any repeated k-mer. For example, if "AAA" appears 10 times in a
    100-amino-acid sequence, the repetitiveness for 3-mers is (10 * 3) / 100 = 0.3
    (30% of sequence).
    
    Attributes:
        max_repetitiveness (float): Maximum acceptable repetitiveness fraction
            (0.0-1.0). Measures the maximum fraction of the sequence covered by
            repeated k-mers. For example, 0.3 means at most 30% of the sequence
            can consist of repeated motifs. Typical values range from 0.05 (strict,
            allows up to 5% repetitive content) to 0.3 (lenient, allows up to 30%).
            Lower values enforce less repetitive sequences. Default: 0.1.

        min_repeat_length (int): Minimum k-mer length to consider as repeats.
            Must be a positive integer. Smaller values detect most typical sequence
            repeats like "ATATATATA" or "MLKVMLKV", while longer values (5-7) detect larger
            structural repeats or large motif duplications. The algorithm checks k-mers
            from this length up to ``min_repeat_length + 7`` to find the most
            repetitive pattern. Default: 1.
    """
    # Required parameters
    max_repetitiveness: float = ConfigField(
        title="Max Repetitiveness Fraction",
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable repetitiveness fraction (fraction of sequence covered by repeated k-mers)",
        examples=[0.05, 0.3],
    )
    min_repeat_length: int = ConfigField(
        title="Minimum Repeat Length",
        default=1,
        ge=1,
        description="Minimum k-mer length to consider as repeats.",
        examples=[1, 3],
    )


@ConstraintRegistry.register(
    key="protein-repetitiveness",
    label="Protein Repetitiveness",
    config=ProteinRepetitivenessConfig,
    description="Evaluate protein sequence repetitiveness based on k-mer analysis",
    batched=True,
    concatenate=True,
    tools_called=[],
    category="protein quality",
    supported_sequence_types=["protein"],
)
def protein_repetitiveness_constraint(sequences: List[Sequence], config: ProteinRepetitivenessConfig) -> List[float]:
    """Evaluate protein sequence repetitiveness based on k-mer frequency analysis.
    
    This constraint function analyzes protein sequences for repetitive content by
    examining k-mer frequencies. It identifies sequences with excessive repetitive motifs,
    which may indicate low-complexity regions or non-functional proteins. The analysis
    scans multiple k-mer lengths to detect both short tandem repeats and larger sequence
    duplications. The repetitiveness score represents the maximum fraction of the sequence
    covered by any repeated k-mer. For example, if "SSS" appears 8 times in a
    60-amino-acid sequence, the repetitiveness for 3-mers is (8 * 3) / 60 = 0.4
    (40% of sequence).
    
    Args:
        sequences (List[Sequence]): List of protein sequences to evaluate. All
            sequences must have ``sequence_type == "protein"``.
            
        config (ProteinRepetitivenessConfig): Configuration object containing
            ``max_repetitiveness`` (maximum acceptable repetitiveness fraction,
            default: 0.4) and ``min_repeat_length`` (minimum k-mer length to
            consider, default: 3).

    Returns:
        List[float]: Constraint scores for each sequence, where 0.0 indicates
            acceptable repetitiveness (at or below threshold) and higher values
            indicate excessive repetitive content. Penalties scale linearly with
            excess repetitiveness: if max is 0.4 and actual is 0.6, the excess
            (0.2) is normalized by the remaining range (1.0 - 0.4 = 0.6), giving
            a score of 0.33.

    Raises:
        AssertionError: If any sequence in the input list is not a protein sequence.
        ValueError: If any sequence has length shorter than ``min_repeat_length``
            (raised by the helper function ``_calculate_repetitiveness_score``).
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:
        
        - ``repetitiveness_score``: Float repetitiveness score (0.0-1.0)
          representing the maximum fraction of sequence covered by repeated k-mers
        - ``max_repetitive_fraction``: Float identical to ``repetitiveness_score``
          (kept for backward compatibility)
    
    Examples:
        Evaluating repetitiveness with default settings:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> config = ProteinRepetitivenessConfig(max_repetitiveness=0.4, min_repeat_length=3)
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> scores = protein_repetitiveness_constraint([seq], config)
        >>> print(scores[0])  # 0.0 if repetitiveness < 40%
        >>> print(seq._metadata["repetitiveness_score"])  # e.g., 0.15
    """
    seq_strings = [seq.sequence for seq in sequences]
    repetitiveness_scores = np.array([
        _calculate_repetitiveness_score(s, config.min_repeat_length) 
        for s in seq_strings
    ])
    excess = repetitiveness_scores - config.max_repetitiveness
    scores = np.where(
        repetitiveness_scores <= config.max_repetitiveness,
        MIN_ENERGY,
        np.minimum(MAX_ENERGY, excess / (1.0 - config.max_repetitiveness))
    )
    
    for i, input_sequence in enumerate(sequences):
        input_sequence._metadata["repetitiveness_score"] = float(repetitiveness_scores[i])
        input_sequence._metadata["max_repetitive_fraction"] = float(repetitiveness_scores[i])
    
    return scores.tolist()


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
