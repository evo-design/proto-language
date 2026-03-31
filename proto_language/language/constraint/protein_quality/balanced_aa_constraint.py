"""
proto_language/language/constraint/protein_quality/balanced_aa_constraint.py

Balanced amino acid constraint function.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import PROTEIN_AMINO_ACIDS, Sequence


class BalancedAaConfig(BaseConfig):
    """Configuration for balanced amino acid constraint.

    This class defines configuration parameters for evaluating whether a protein
    sequence has balanced representation of all amino acid types. The constraint
    penalizes sequences that have too many underrepresented amino acids (those
    appearing below a minimum frequency threshold). The penalty score increases
    both with the number of underrepresented amino acids beyond the threshold and
    with the severity of under-representation (how far below min_aa_frequency each
    amino acid falls).

    Attributes:
        min_aa_frequency (float): Minimum acceptable relative frequency for any
            amino acid type in the sequence. For example, 0.02 means each amino
            acid should appear at least 2% of the time in the sequence. Valid
            range is 0.0 to 1.0. Typical values range from 0.01 (1%, lenient)
            to 0.05 (5%, strict). Default: 0.02.

        max_underrepresented_count (int): Maximum acceptable number of amino acid
            types that can be underrepresented (below ``min_aa_frequency``) before
            the sequence is penalized. Valid range is 0 to 20 (total number of
            standard amino acids). For example, if set to 3, sequences with 4 or
            more underrepresented amino acids will receive a penalty score. Lower
            values enforce stricter amino acid diversity requirements. Default: 3.

    """
    # Required parameters
    min_aa_frequency: float = ConfigField(
        title="Min Acceptable AA Frequency",
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable relative frequency for any amino acid type.",
        examples=[0.01, 0.03],
    )
    max_underrepresented_count: int = ConfigField(
        title="Max Underrepresented Count",
        default=3,
        ge=0,
        le=20,
        description="Maximum acceptable number of underrepresented amino acid types. Sequences with more are penalized.",
        examples=[2, 5],
    )


@constraint(
    key="balanced-aa",
    label="Balanced Amino Acid Representation",
    config=BalancedAaConfig,
    description="Evaluate the presence of underrepresented amino acids in a protein sequence",
    tools_called=[],
    category="protein quality",
    supported_sequence_types=["protein"],
    num_input_sequences_per_tuple=1,
)
def balanced_aa_constraint(input_sequences: List[Tuple[Sequence, ...]], config: BalancedAaConfig) -> List[float]:
    """Evaluate the presence of underrepresented amino acids in protein sequences.

    This constraint function assesses whether protein sequences have balanced
    representation of all amino acid types by identifying amino acids that appear
    below a minimum frequency threshold and penalizing sequences that have too many
    such underrepresented amino acids. The penalty is scaled based on both the
    number of excess underrepresented amino acids and the severity of their
    under-representation.

    For each input sequence, it calculates amino acid frequencies, identifies
    underrepresented amino acids, and computes a penalty score if the number
    of underrepresented amino acids exceeds the configured threshold.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one protein sequence.

        config (BalancedAaConfig): Configuration object containing ``min_aa_frequency``
            (minimum acceptable relative frequency, default: 0.02) and
            ``max_underrepresented_count`` (maximum acceptable number of
            underrepresented amino acid types, default: 3).

    Returns:
        List[float]: Constraint scores for each sequence, ranging from 0.0 (best,
            acceptable number of underrepresented amino acids) to 1.0 (worst,
            many severely underrepresented amino acids). The score is scaled based
            on how many excess underrepresented amino acids there are beyond the
            threshold and how far below the minimum frequency they fall.

    Raises:
        AssertionError: If any sequence in the input list is not a protein sequence.

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:

        - ``underrepresented_aa_score``: Float score indicating overall
          underrepresentation severity
        - ``amino_acid_counts``: Dictionary mapping amino acids to their counts
        - ``underrepresented_amino_acids``: List of amino acids that
          are underrepresented
        - ``underrepresented_aa_count``: Integer count of underrepresented amino
          acid types
        - ``min_aa_frequency_threshold``: The minimum frequency threshold used

    Examples:
        Evaluating amino acid balance in protein:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> config = BalancedAaConfig(min_aa_frequency=0.05, max_underrepresented_count=2)
        >>> seq = Sequence("AAAAAACCCCCCDDDDDD", sequence_type="protein")
        >>> scores = balanced_aa_constraint([(seq,)], config)
        >>> # This sequence has only 3 amino acid types, so 17 are underrepresented
        >>> # This exceeds max_underrepresented_count=2, resulting in a penalty
        >>> print(scores[0])  # Will be > 0.0
        >>> print(seq._metadata["underrepresented_aa_count"])  # 17
    """
    # Extract sequence strings from tuples
    seq_strings = [seq.sequence for (seq,) in input_sequences]
    seq_lengths = np.array([len(s) for s in seq_strings])
    aa_alphabet = PROTEIN_AMINO_ACIDS
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

    batch_size = len(input_sequences)
    aa_count_matrix = np.zeros((batch_size, 20), dtype=np.int32)

    for seq_idx, seq_str in enumerate(seq_strings):
        if len(seq_str) > 0:
            aa_counts = Counter(seq_str)
            for aa, count in aa_counts.items():
                if aa in aa_to_idx:
                    aa_count_matrix[seq_idx, aa_to_idx[aa]] = count

    aa_freq_matrix = aa_count_matrix / seq_lengths[:, np.newaxis].clip(min=1)
    frequency_thresholds = config.min_aa_frequency * seq_lengths
    count_thresholds = frequency_thresholds[:, np.newaxis]

    underrepresented_mask = aa_count_matrix < count_thresholds
    underrepresented_counts = underrepresented_mask.sum(axis=1)

    underrepresented_totals = (aa_count_matrix * underrepresented_mask).sum(axis=1)
    underrepresented_scores = underrepresented_totals / seq_lengths.clip(min=1)

    penalties = np.zeros(batch_size)
    excess_mask = underrepresented_counts > config.max_underrepresented_count

    if np.any(excess_mask):
        excess_counts = (underrepresented_counts - config.max_underrepresented_count).clip(min=0)
        max_possible_excess = 20 - config.max_underrepresented_count
        deficits = np.zeros(batch_size)

        for seq_idx in np.where(excess_mask)[0]:
            if underrepresented_totals[seq_idx] > 0:
                # Calculate weighted average deficit for sequences
                underrep_freqs = aa_freq_matrix[seq_idx][underrepresented_mask[seq_idx]]
                underrep_counts = aa_count_matrix[seq_idx][underrepresented_mask[seq_idx]]

                aa_deficits = config.min_aa_frequency - underrep_freqs
                weighted_deficit = (aa_deficits * underrep_counts).sum()
                deficits[seq_idx] = weighted_deficit / underrepresented_totals[seq_idx]

        # Calculate penalties for all sequences
        count_penalties = np.where(
            max_possible_excess > 0,
            excess_counts / max_possible_excess,
            1.0
        )
        severity_penalties = np.where(
            config.min_aa_frequency > 0,
            deficits / config.min_aa_frequency,
            0.0
        )

        penalties = np.where(
            excess_mask,
            np.minimum(1.0, count_penalties * (1.0 + severity_penalties)),
            0.0
        )

    for seq_idx, (input_sequence,) in enumerate(input_sequences):
        seq_str = seq_strings[seq_idx]
        aa_counts = {
            aa_alphabet[aa_idx]: int(aa_count_matrix[seq_idx, aa_idx])
            for aa_idx in range(20)
            if aa_count_matrix[seq_idx, aa_idx] > 0
        }

        # Get underrepresented AAs for sequences
        underrepresented_aas = [
            aa_alphabet[aa_idx]
            for aa_idx in range(20)
            if underrepresented_mask[seq_idx, aa_idx]
        ]

        # Store metadata
        input_sequence._metadata["underrepresented_aa_score"] = float(underrepresented_scores[seq_idx])
        input_sequence._metadata["amino_acid_counts"] = aa_counts if aa_counts else {}
        input_sequence._metadata["underrepresented_amino_acids"] = underrepresented_aas if underrepresented_aas else []
        input_sequence._metadata["underrepresented_aa_count"] = int(underrepresented_counts[seq_idx])
        input_sequence._metadata["min_aa_frequency_threshold"] = config.min_aa_frequency

    # Return penalty scores
    return penalties.tolist()
