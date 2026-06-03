"""GC content constraint for evaluating sequence GC content properties."""

from pydantic import model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import (
    MAX_ENERGY,
    calculate_percentage_range_deviation,
)
from proto_language.utils.base import BaseConfig, ConfigField


class GCContentConfig(BaseConfig):
    """Configuration for GC content constraint.

    This class defines configuration parameters for evaluating the GC content
    in DNA or RNA sequences. This penalty scales linearly with deviation from the
    acceptable range.

    Attributes:
        min_gc (float): Minimum acceptable GC content percentage (0-100). Sequences
            with GC content below this threshold are penalized. Typical values depend
            on organism, but generally ~35% is a good lower bound.

        max_gc (float): Maximum acceptable GC content percentage (0-100). Sequences
            with GC content above this threshold are penalized. Can be used to avoid
            sequences that are GC-rich, which can form secondary structures
            or have very high melting temperatures. Higher values are more permissive.

    """

    # Required parameters
    min_gc: float = ConfigField(
        ge=0, le=100, title="Min GC", description="Minimum acceptable GC content percentage (0-100)", examples=[35]
    )
    max_gc: float = ConfigField(
        ge=0, le=100, title="Max GC", description="Maximum acceptable GC content percentage (0-100)", examples=[70]
    )

    @model_validator(mode="after")
    def validate_gc_range(self) -> "GCContentConfig":
        """Ensure min_gc <= max_gc."""
        if self.min_gc > self.max_gc:
            raise ValueError(f"min_gc ({self.min_gc}) must be <= max_gc ({self.max_gc})")
        return self


@constraint(
    key="gc-content",
    label="GC Content",
    config=GCContentConfig,
    description="Enforce GC content within specified range",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna"],
)
def gc_content_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: GCContentConfig
) -> list[ConstraintOutput]:
    """Enforce GC content within specified range.

    This constraint function calculates the percentage of guanine (G) and cytosine
    (C) nucleotides in DNA or RNA sequences and evaluates whether it falls within
    a specified acceptable range. GC content is a fundamental sequence property
    that affects DNA stability, melting temperature, gene expression patterns,
    and technical considerations like PCR amplification efficiency.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one DNA or RNA sequence. Empty sequences receive
            maximum penalty.

        config (GCContentConfig): Configuration object containing ``min_gc``
            (minimum acceptable GC percentage) and ``max_gc`` (maximum acceptable
            GC percentage). Both values must be between 0 and 100.

    Returns:
        list[ConstraintOutput]: One result per sequence. A score of 0.0 indicates
            GC content is within the acceptable range [min_gc, max_gc]. Higher
            scores indicate greater deviation from the acceptable range, with
            penalties scaling linearly with the deviation distance. The ``metadata``
            field carries ``gc_content``.

    Examples:
        Evaluating GC content constraint:

        >>> seq = Sequence("ATCGATCG", "dna")
        >>> cfg = GCContentConfig(min_gc=40.0, max_gc=60.0)
        >>> result = gc_content_constraint([(seq,)], config=cfg)
        >>> print(result[0].score)  # 0.0 (50% GC content is within acceptable range)
    """
    results = []

    for (seq,) in input_sequences:
        if len(seq.sequence) == 0:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"gc_content": 0.0}))
            continue

        gc_content = 100.0 * sum(nt in "GC" for nt in seq.sequence.upper()) / max(len(seq.sequence), 1)
        deviation = calculate_percentage_range_deviation(gc_content, config.min_gc, config.max_gc)
        results.append(ConstraintOutput(score=min(MAX_ENERGY, deviation), metadata={"gc_content": gc_content}))

    return results
