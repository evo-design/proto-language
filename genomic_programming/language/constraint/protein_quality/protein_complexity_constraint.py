"""
Protein complexity constraint function.
"""

from __future__ import annotations

from typing import List

from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import MIN_ENERGY, MAX_ENERGY
from proto_language.tools.sequence_scoring.segmasker import (
    run_segmasker,
    SegmaskerInput,
    SegmaskerConfig,
)

class ProteinComplexityConfig(BaseConfig):
    """Configuration for protein complexity constraint."""
    max_low_complexity: float = Field(
        ge=0.0,
        le=1.0,
        description="Maximum acceptable fraction of low-complexity regions (0.0-1.0). Low-complexity regions contain repetitive or biased amino acid compositions. Typical values: 0.2-0.4."
    )
    segmasker_path: str = Field(
        default="segmasker",
        description="Path to NCBI segmasker executable for detecting low-complexity regions. Must be installed separately."
    )

@ConstraintRegistry.register(
    key="protein-complexity",
    label="Protein Complexity",
    config=ProteinComplexityConfig,
    description="Evaluate protein sequence complexity using segmasker to detect low-complexity regions",
    vectorized=True,
    concatenate=True
)
def protein_complexity_constraint(sequences: List[Sequence], config: ProteinComplexityConfig) -> List[float]:
    """
    Evaluate protein sequence complexity using segmasker to detect low-complexity regions.

    Args:
        sequences The protein sequences to evaluate.
        config: Configuration containing max_low_complexity and segmasker_path parameters.

    Returns:
        List of onstraint scores where 0.0 indicates acceptable complexity
        and higher values indicate excessive low-complexity regions.

    Raises:
        ValueError: If segmasker execution fails.
    """
    for seq in sequences:
        assert seq.sequence_type == SequenceType.PROTEIN, "Input must be protein"
    
    segmasker_inputs = SegmaskerInput(sequences=[seq.sequence for seq in sequences])
    segmasker_config = SegmaskerConfig(segmasker_path=config.segmasker_path)

    result = run_segmasker(inputs=segmasker_inputs, config=segmasker_config)

    if not result.success:
        # Tool failed - store error metadata in all sequences and raise
        scores = []
        error_msg = result.errors[0] if result.errors else "Unknown segmasker error"
        
        for seq in sequences:
            seq._metadata["low_complexity_fraction"] = 0.0
            seq._metadata["segmasker_X_count"] = 0
            seq._metadata["segmasker_error"] = True
            seq._metadata["segmasker_error_message"] = error_msg
            scores.append(MAX_ENERGY)
        
        raise ValueError(f"Segmasker analysis failed: {error_msg}")

    scores = []
    for seq, low_complexity_fraction in zip(sequences, result.low_complexity_fractions):
        seq._metadata["low_complexity_fraction"] = low_complexity_fraction
        seq._metadata["segmasker_X_count"] = int(
            low_complexity_fraction * len(seq)
        )
        seq._metadata["segmasker_error"] = False

        if low_complexity_fraction <= config.max_low_complexity:
            score = MIN_ENERGY
        else:
            excess = low_complexity_fraction - config.max_low_complexity
            score = min(MAX_ENERGY, excess / (1.0 - config.max_low_complexity))
        
        scores.append(score)

    return scores