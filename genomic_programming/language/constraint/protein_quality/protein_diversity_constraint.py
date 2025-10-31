"""
Protein diversity constraint function.
"""

from __future__ import annotations

from typing import List
import numpy as np

from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import MIN_ENERGY, MAX_ENERGY


class ProteinDiversityConfig(BaseConfig):
    """Configuration for protein diversity constraint."""
    min_diversity: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum acceptable amino acid diversity (0.0-1.0). Calculated as (unique amino acids) / 20. Higher values require more diverse amino acid usage. Typical values: 0.4-0.7."
    )

@ConstraintRegistry.register(
    key="protein-diversity",
    label="Protein Diversity",
    config=ProteinDiversityConfig,
    description="Evaluate amino acid diversity in a protein sequence",
    vectorized=True,
    concatenate=True
)
def protein_diversity_constraint(sequences: List[Sequence], config: ProteinDiversityConfig) -> List[float]:
    """
    Evaluate amino acid diversity in a protein sequence.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing the min_diversity parameter.

    Returns:
        Constraint score where 0.0 indicates sufficient diversity
        and higher values indicate insufficient amino acid diversity.

    Raises:
        ValueError: If sequence has length 0
    """
    for seq in sequences:
        assert seq.sequence_type == SequenceType.PROTEIN, "Input must be protein"

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
