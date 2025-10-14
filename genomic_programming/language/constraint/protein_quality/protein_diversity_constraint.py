"""
Protein diversity constraint function.
"""

from __future__ import annotations

from pydantic import Field

from ...core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....utils import MIN_ENERGY, MAX_ENERGY


class ProteinDiversityConfig(BaseConfig):
    """Configuration for protein diversity constraint."""
    min_diversity: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum acceptable amino acid diversity (0.0-1.0). Calculated as (unique amino acids) / 20. Higher values require more diverse amino acid usage. Typical values: 0.4-0.7."
    )


@ConstraintRegistry.register(
    key="protein-diversity",
    config=ProteinDiversityConfig,
    description="Evaluate amino acid diversity in a protein sequence",
    vectorized=False,
    concatenate=True
)
def protein_diversity_constraint(
    input_sequence: Sequence, config: ProteinDiversityConfig
) -> float:
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
    assert input_sequence.sequence_type == SequenceType.PROTEIN, "Input must be protein"

    seq = input_sequence.sequence

    # Calculate amino acid diversity score
    if len(seq) == 0:
        raise ValueError("Sequence is non-existent.")

    unique_aas = len(set(seq))
    diversity_score = unique_aas / 20.0  # 20 standard amino acids

    # Store metadata
    input_sequence._metadata["aa_diversity_score"] = diversity_score
    input_sequence._metadata["unique_amino_acid_count"] = unique_aas
    input_sequence._metadata["unique_amino_acids"] = sorted(list(set(seq)))

    # Return constraint score
    if diversity_score >= config.min_diversity:
        return MIN_ENERGY

    deficit = config.min_diversity - diversity_score
    return min(MAX_ENERGY, deficit / config.min_diversity)
