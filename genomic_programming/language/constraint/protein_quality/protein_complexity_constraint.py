"""
Protein complexity constraint function.
"""

from __future__ import annotations

from pydantic import Field

from ...base import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..registry import ConstraintRegistry
from ....tools.utils import calculate_segmasker_score
from ..utils import MIN_ENERGY, MAX_ENERGY


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
    config=ProteinComplexityConfig,
    description="Evaluate protein sequence complexity using segmasker to detect low-complexity regions",
    vectorized=False,
    concatenate=True
)
def protein_complexity_constraint(
    input_sequence: Sequence, config: ProteinComplexityConfig
) -> float:
    """
    Evaluate protein sequence complexity using segmasker to detect low-complexity regions.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing max_low_complexity and segmasker_path parameters.

    Returns:
        Constraint score where 0.0 indicates acceptable complexity
        and higher values indicate excessive low-complexity regions.
        Returns MAX_ENERGY if segmasker fails.

    Raises:
        ValueError: If segmasker execution fails.
    """
    assert input_sequence.sequence_type == SequenceType.PROTEIN, "Input must be protein"

    try:
        low_complexity_fraction = calculate_segmasker_score(
            input_sequence.sequence, config.segmasker_path
        )

        input_sequence._metadata["low_complexity_fraction"] = low_complexity_fraction
        input_sequence._metadata["segmasker_X_count"] = int(
            low_complexity_fraction * len(input_sequence)
        )
        input_sequence._metadata["segmasker_error"] = False

        if low_complexity_fraction <= config.max_low_complexity:
            return MIN_ENERGY

        excess = low_complexity_fraction - config.max_low_complexity
        return min(MAX_ENERGY, excess / (1.0 - config.max_low_complexity))

    except ValueError as e:
        # Store error information in metadata
        input_sequence._metadata["low_complexity_fraction"] = float("nan")
        input_sequence._metadata["segmasker_X_count"] = float("nan")
        input_sequence._metadata["segmasker_error"] = True
        input_sequence._metadata["segmasker_error_message"] = str(e)

        # Re-raise the exception to propagate the error
        raise ValueError(f"Segmasker analysis failed: {str(e)}")
