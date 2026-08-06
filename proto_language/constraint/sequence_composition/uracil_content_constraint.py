"""Uracil (U) content constraint for evaluating nucleotide composition.

The metric reports the fraction of RNA uracil, or the equivalent DNA thymine fraction. Its
biological interpretation and useful range depend on the sequence context, host, formulation,
and nucleotide chemistry, so the default range is neutral and imposes no composition preference.

Examples:
    >>> config = UracilContentConfig(min_u=20.0, max_u=60.0)
    >>> result = uracil_content_constraint([(Sequence("AUGUUC", "rna"),)], config)
    >>> result[0].metadata["uracil_content"]  # 50.0
"""

from pydantic import model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, calculate_percentage_range_deviation
from proto_language.utils.base import BaseConfig, ConfigField


class UracilContentConfig(BaseConfig):
    """Configuration for the uracil (U) content constraint.

    The penalty is ``0`` inside ``[min_u, max_u]`` and scales linearly with the deviation
    outside it.

    Attributes:
        min_u (float): Minimum acceptable uracil content percentage (0-100).
        max_u (float): Maximum acceptable uracil content percentage (0-100). The default ``100``
            is neutral; choose a narrower range only when justified by the design context.
    """

    min_u: float = ConfigField(
        ge=0, le=100, default=0.0, title="Min U", description="Minimum acceptable uracil content percentage (0-100)"
    )
    max_u: float = ConfigField(
        ge=0, le=100, default=100.0, title="Max U", description="Maximum acceptable uracil content percentage (0-100)"
    )

    @model_validator(mode="after")
    def validate_u_range(self) -> "UracilContentConfig":
        """Ensure min_u <= max_u."""
        if self.min_u > self.max_u:
            raise ValueError(f"min_u ({self.min_u}) must be <= max_u ({self.max_u})")
        return self


@constraint(
    key="uracil-content",
    label="Uracil Content",
    config=UracilContentConfig,
    description="Enforce uracil/thymine content within a specified range",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna"],
)
def uracil_content_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: UracilContentConfig
) -> list[ConstraintOutput]:
    """Enforce uracil (U) content within a specified range.

    Counts uracil, treating DNA thymine (``T``) as equivalent to RNA uracil (``U``), and
    penalizes deviation from ``[min_u, max_u]``.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-sequence tuple per proposal.
        config (UracilContentConfig): Validated min/max uracil percentages.

    Returns:
        list[ConstraintOutput]: One result per sequence. ``score`` is ``0.0`` when U content is
            within range and scales linearly with deviation otherwise; ``metadata`` carries
            ``uracil_content``.
    """
    results = []
    for (seq,) in input_sequences:
        if len(seq.sequence) == 0:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"uracil_content": 0.0}))
            continue

        uracil_content = 100.0 * sum(nt in "UT" for nt in seq.sequence.upper()) / len(seq.sequence)
        deviation = calculate_percentage_range_deviation(uracil_content, config.min_u, config.max_u)
        results.append(ConstraintOutput(score=min(MAX_ENERGY, deviation), metadata={"uracil_content": uracil_content}))
    return results
