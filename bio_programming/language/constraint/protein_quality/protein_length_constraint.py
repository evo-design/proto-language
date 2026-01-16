"""
Protein length constraint function.
"""

from __future__ import annotations

from typing import List

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import calculate_range_deviation


class ProteinLengthConfig(BaseConfig):
    """Configuration object for protein length constraint.
    
    This class defines configuration parameters for evaluating whether protein
    sequences fall within an acceptable length range. Length constraints are
    useful for filtering proteins that are too short or too long. The penalty scales
    linearly with the distance outside the acceptable range. For example, a protein
    10 amino acids below ``min_length`` receives a proportionally smaller penalty than
    one 50 amino acids below.
    
    Attributes:
        min_length (int): Minimum acceptable protein length in amino acids. Must
            be a positive integer. Typical values depend on the application: 80
            for general proteins, 30 for short peptides, and variable lengths 
            for function-specific requirements (e.g., 200+ for enzymes). Sequences 
            shorter than this value are penalized.

        max_length (int): Maximum acceptable protein length in amino acids. Must
            be a positive integer. Typical values: 500-800 for most proteins,
            1000+ for large proteins, or function-specific limits.
            Sequences longer than this value are penalized.
    """
    # Required parameters
    min_length: int = ConfigField(
        title="Min Acceptable Length",
        gt=0,
        description="Minimum acceptable protein length below which sequences are penalized",
    )
    max_length: int = ConfigField(
        title="Max Acceptable Length",
        gt=0,
        description="Maximum acceptable protein length above which sequences are penalized",
    )


@ConstraintRegistry.register(
    key="protein-length",
    label="Protein Length",
    config=ProteinLengthConfig,
    description="Evaluate whether protein length falls within acceptable range",
    batched=True,
    concatenate=True,
    tools_called=[],
    category="protein quality",
    supported_sequence_types=["protein"],
)
def protein_length_constraint(sequences: List[Sequence], config: ProteinLengthConfig) -> List[float]:
    """Evaluate whether protein sequence lengths fall within an acceptable range.
    
    This constraint function checks if protein sequences have lengths within a
    specified range, penalizing sequences that are too short or too long. This
    is useful for filtering out spurious ORF predictions. Penalties scale linearly
    with the distance outside the acceptable range.

    Args:
        sequences (List[Sequence]): List of protein sequences to evaluate. All
            sequences must have ``sequence_type == "protein"``.
            
        config (ProteinLengthConfig): Configuration object containing ``min_length``
            (minimum acceptable length in amino acids) and ``max_length`` (maximum
            acceptable length in amino acids).

    Returns:
        List[float]: Constraint scores for each sequence, where 0.0 indicates
            length is within the acceptable range [min_length, max_length] and
            higher values indicate greater deviation from the acceptable range.
            Penalties scale linearly: a sequence 10 amino acids outside the range
            receives half the penalty of one 20 amino acids outside.

    Raises:
        AssertionError: If any sequence in the input list is not a protein sequence.
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following key:
        
        - ``protein_length``: Integer length of the protein sequence in amino acids
    
    Examples:
        Evaluating protein length within range:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> config = ProteinLengthConfig(min_length=10, max_length=500)
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> scores = protein_length_constraint([seq], config)
        >>> print(scores[0])  # 0.0
        >>> print(seq._metadata["protein_length"])  # 37
    """
    scores = []
    
    for seq in sequences:
        actual_length = len(seq)
        seq._metadata["protein_length"] = actual_length
        score = calculate_range_deviation(actual_length, config.min_length, config.max_length)
        scores.append(score)

    return scores
