"""
Balanced amino acid constraint function.
"""

from __future__ import annotations

from collections import Counter
from typing import List
import numpy as np
from Bio.Data import IUPACData

from pydantic import Field

from proto_language.language.core import Sequence, SequenceType,PROTEIN_AMINO_ACIDS
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry


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
    label="Balanced Amino Acid Representation",
    config=BalancedAaConfig,
    description="Evaluate the presence of underrepresented amino acids in a protein sequence",
    vectorized=True,
    concatenate=True
)
def balanced_aa_constraint(sequences: List[Sequence], config: BalancedAaConfig) -> List[float]:
    """
    Evaluate the presence of underrepresented amino acids in a protein sequence.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing min_aa_frequency and max_underrepresented_count parameters.

    Returns:
        Constraint score from 0.0 (best, acceptable number of underrepresented amino acids) to 1.0 (worst).
        Score is scaled based on how many excess underrepresented amino acids there are and their severity.
    """
    for seq in sequences:
        assert seq.sequence_type == SequenceType.PROTEIN, "Input must be protein"
    
    seq_strings = [seq.sequence for seq in sequences]
    seq_lengths = np.array([len(s) for s in seq_strings])
    aa_alphabet = PROTEIN_AMINO_ACIDS
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
    
    batch_size = len(sequences)
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

    for seq_idx, input_sequence in enumerate(sequences):
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