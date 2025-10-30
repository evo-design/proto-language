"""
Maximum homopolymer constraint for evaluating sequence homopolymer properties.
"""

from __future__ import annotations

import itertools
from typing import List

import numpy as np
from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import (
    MIN_ENERGY,
    MAX_ENERGY,
    LOG_BASE,
)


class MaxHomopolymerConfig(BaseConfig):
    """Configuration for maximum homopolymer constraint."""
    max_length: int = Field(
        gt=0,
        description="Maximum allowed homopolymer length in consecutive identical nucleotides or amino acids. Must be a positive integer. Sequences with longer homopolymers are penalized."
    )


@ConstraintRegistry.register(
    key="max-homopolymer",
    label="Homopolymer Length",
    config=MaxHomopolymerConfig,
    description="Penalize sequences containing homopolymers longer than specified maximum",
    vectorized=True,
    concatenate=True
)
def max_homopolymer_constraint(sequences: List[Sequence], config: MaxHomopolymerConfig) -> List[float]:
    """
    Penalize sequences containing homopolymers longer than a specified maximum.

    Args:
        sequences: Sequences to evaluate.
        config: Configuration containing the max_length parameter.

    Returns:
        Constraint score where 0.0 indicates no homopolymers exceed the maximum length
        and higher values indicate longer homopolymers with logarithmic scaling.

    Examples:
        Evaluating homopolymer constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> cfg = MaxHomopolymerConfig(max_length=3)
        >>> score = max_homopolymer_constraint(seq, config=cfg)
        >>> print(score)  # 0.0 (no long homopolymers)

    Note:
        The constraint uses logarithmic scaling to penalize excessive homopolymer lengths
        while avoiding extreme penalty values.
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
