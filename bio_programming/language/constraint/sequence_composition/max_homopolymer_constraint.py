"""
Maximum homopolymer constraint for evaluating sequence homopolymer properties.
"""

from __future__ import annotations

import itertools
from typing import List

import numpy as np

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import (
    MIN_ENERGY,
    MAX_ENERGY,
    LOG_BASE,
)


class MaxHomopolymerConfig(BaseConfig):
    """Configuration for maximum homopolymer constraint.
    
    This class defines configuration parameters for limiting homopolymer length
    in DNA, RNA, or protein sequences. Homopolymers are consecutive runs of the
    same nucleotide or amino acid (e.g., "AAAAA", "GGGGGG", "SSSSSS"). This constraint
    uses logarithmic scaling for penalties to avoid extreme values while still penalizing
    very long homopolymers, providing moderate penalties for slightly exceeding the limit 
    and strong penalties for greatly exceeding the limit.

    Attributes:
        max_length (int): Maximum allowed homopolymer length in consecutive identical
            nucleotides or amino acids. Must be a positive integer. Sequences with
            homopolymers longer than this value are penalized. Typical values depend
            on application, with some examples provided below:
            - DNA synthesis: 8-10 (avoid synthesis errors)
            - PCR primers: 5-8 (prevent polymerase slippage)
            - Protein sequences: 5+ (avoid excessive amino acid repeats)
    """
    # Required parameters
    max_length: int = ConfigField(
        title="Max Homopolymer Length",
        gt=0,
        description="Max homopolymer length in consecutive identical nucleotides or amino acids (Longer penalized)",  #  Sequences with longer homopolymers are penalized.
    )


@ConstraintRegistry.register(
    key="max-homopolymer",
    label="Homopolymer Length",
    config=MaxHomopolymerConfig,
    description="Penalize sequences containing homopolymers longer than specified maximum",
    batched=True,
    concatenate=True,
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna", "protein"],
)
def max_homopolymer_constraint(sequences: List[Sequence], config: MaxHomopolymerConfig) -> List[float]:
    """Penalize sequences containing homopolymers longer than specified maximum
    
    This constraint function identifies the longest homopolymer (consecutive run
    of identical nucleotides or amino acids) in each sequence and penalizes
    sequences where this exceeds a specified maximum length.
    
    The penalty uses logarithmic scaling to provide graduated penalties: sequences
    slightly over the limit receive moderate penalties, while sequences far
    exceeding the limit receive strong penalties (capped at 1.0). This avoids
    extreme penalty values while still strongly discouraging very long homopolymers.

    Args:
        sequences (List[Sequence]): List of DNA, RNA, or protein sequences to
            evaluate. 

        config (MaxHomopolymerConfig): Configuration object containing ``max_length``
            (maximum allowed homopolymer length).

    Returns:
        List[float]: Constraint scores for each sequence. A score of 0.0 indicates
            no homopolymers exceed the maximum length (pass). Higher scores indicate
            longer homopolymers with logarithmic scaling.

    Raises:
        ValueError: If the input sequence list is empty.
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following key:
        
        - ``max_homopolymer_length``: Integer length of the longest homopolymer
          found in the sequence. For example, "ATCGAAAAAGTC" would have value 5
          (for the "AAAAA" run).
    
    Examples:
        Avoiding long A/T runs for DNA synthesis:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("ATCGATCGTAGC", "dna")
        >>> config = MaxHomopolymerConfig(max_length=4)
        >>> scores = max_homopolymer_constraint([seq], config)
        >>> print(scores[0])  # 0.0 (no runs >4)
        >>> print(seq._metadata["max_homopolymer_length"]) 
    """
    if not sequences:
        raise ValueError("Input sequence list must not be empty")
    
    scores = []
    for seq in sequences: 
        if len(seq.sequence) <= 1:
            longest_homopolymer = len(seq.sequence)
        else:
            homopolymer_lengths = [
                len(list(group)) for _, group in itertools.groupby(seq.sequence)
            ]
            longest_homopolymer = max(homopolymer_lengths)

        seq._metadata["max_homopolymer_length"] = longest_homopolymer

        if longest_homopolymer <= config.max_length:
            scores.append(MIN_ENERGY)
        else: 
            excess_length = longest_homopolymer - config.max_length
            log_ratio = np.log(1 + excess_length / config.max_length) / np.log(LOG_BASE)
            scores.append(min(MAX_ENERGY, log_ratio))
    
    return scores
