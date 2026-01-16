"""
Protein diversity constraint function.
"""

from __future__ import annotations

from typing import List
import numpy as np

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import MIN_ENERGY, MAX_ENERGY


class ProteinDiversityConfig(BaseConfig):
    """Configuration for protein diversity constraint.
    
    This class defines configuration parameters for evaluating amino acid diversity
    in protein sequences. The constraint measures how many different amino acid
    types are present in the sequence and penalizes sequences with insufficient
    diversity, which may indicate poor protein quality, repetitive sequences, or
    non-functional proteins.
    
    Attributes:
        min_diversity (float): Minimum acceptable amino acid diversity (0.0-1.0).
            Calculated as (unique amino acids) / 20, where 20 is the total number
            of standard amino acids. For example, 0.5 means at least 10 different
            amino acid types must be present. Typical values range from 0.6 (12
            amino acids, lenient) to 0.9 (18 amino acids, strict). Higher values
            enforce more diverse amino acid usage. Default: 0.7.
    
    Note:
        A diversity score of 1.0 means all 20 standard amino acids are present,
        while 0.0 means only one amino acid type is used (homopolymer).
    """
    min_diversity: float = ConfigField(
        title="Min Acceptable Diversity",
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable amino acid diversity. Calculated as (unique amino acids) / 20.",
        examples=[0.6, 0.9],
    )


@ConstraintRegistry.register(
    key="protein-diversity",
    label="Protein Diversity",
    config=ProteinDiversityConfig,
    description="Evaluate amino acid diversity in a protein sequence",
    batched=True,
    concatenate=True,
    tools_called=[],
    category="protein quality",
    supported_sequence_types=["protein"],
)
def protein_diversity_constraint(sequences: List[Sequence], config: ProteinDiversityConfig) -> List[float]:
    """Evaluate amino acid diversity in protein sequences.
    
    This constraint function measures the diversity of amino acid types present
    in protein sequences. It calculates diversity as the fraction of the 20
    standard amino acids that appear in the sequence, and penalizes sequences
    that fall below a minimum diversity threshold. The penalty scales linearly
    with the deficit below the minimum diversity threshold.

    Args:
        sequences (List[Sequence]): List of protein sequences to evaluate. All
            sequences must have ``sequence_type == "protein"``.
            
        config (ProteinDiversityConfig): Configuration object containing
            ``min_diversity`` (minimum acceptable amino acid diversity, default: 0.5).

    Returns:
        List[float]: Constraint scores for each sequence, where 0.0 indicates
            sufficient diversity (diversity at or above threshold) and higher
            values indicate insufficient amino acid diversity. Scores scale
            linearly with the deficit below the threshold (e.g., if min_diversity
            is 0.5 and actual diversity is 0.25, the score is 0.5), capped at 1.0.

    Raises:
        AssertionError: If any sequence in the input list is not a protein sequence.
        ValueError: If any sequence has length 0 (empty sequence).
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:
        
        - ``aa_diversity_score``: Float diversity score (0.0-1.0) calculated as
          (unique amino acids) / 20
        - ``unique_amino_acid_count``: Integer count of unique amino acid types
          present in the sequence (0-20)
        - ``unique_amino_acids``: Sorted list of amino acid characters present
          in the sequence
    
    Examples:
        Evaluating protein diversity:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> config = ProteinDiversityConfig(min_diversity=0.5)
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> scores = protein_diversity_constraint([seq], config)
        >>> print(scores[0])  # 0.0 if diversity >= 0.5
        >>> print(seq._metadata["aa_diversity_score"])  # e.g., 0.65
        >>> print(seq._metadata["unique_amino_acid_count"])  # e.g., 13
        >>> print(seq._metadata["unique_amino_acids"])  # e.g., ['A', 'D', 'E', 'F', ...]
    """
    seq_strings = [seq.sequence for seq in sequences]
    seq_lengths = np.array([len(s) for s in seq_strings])

    # Calculate amino acid diversity score
    if np.any(seq_lengths == 0):
        raise ValueError("Sequence is non-existent.")

    unique_aa_counts = np.array([len(set(s)) for s in seq_strings])
    diversity_scores = unique_aa_counts / 20.0  # 20 standard amino acids

    deficits = config.min_diversity - diversity_scores

    scores_array = np.where(
        diversity_scores >= config.min_diversity,
        MIN_ENERGY,
        np.minimum(MAX_ENERGY, deficits / config.min_diversity)
    )

    # Store metadata
    for i, input_sequence in enumerate(sequences):
        input_sequence._metadata["aa_diversity_score"] = float(diversity_scores[i])
        input_sequence._metadata["unique_amino_acid_count"] = int(unique_aa_counts[i])
        input_sequence._metadata["unique_amino_acids"] = sorted(list(set(seq_strings[i])))

    return scores_array.tolist()
