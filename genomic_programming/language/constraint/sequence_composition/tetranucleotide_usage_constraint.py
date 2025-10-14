"""
Tetranucleotide usage constraint for evaluating sequence tetranucleotide usage deviation.
"""

from __future__ import annotations

from pydantic import Field

from ...core import Sequence, SequenceType, DNA_NUCLEOTIDES, RNA_NUCLEOTIDES
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....utils import MIN_ENERGY, calculate_range_deviation


class TetranucleotideUsageConfig(BaseConfig):
    """Configuration for tetranucleotide usage constraint."""
    tetranucleotide: str = Field(
        min_length=4,
        max_length=4,
        description="The 4-base DNA/RNA sequence motif to analyze (e.g., 'ATCG'). Must be exactly 4 nucleotides long."
    )
    min_tud: float = Field(
        description="Minimum acceptable tetranucleotide usage deviation (TUD). TUD=1.0 means observed frequency matches expected frequency. Values <1 indicate underrepresentation, >1 indicate overrepresentation."
    )
    max_tud: float = Field(
        description="Maximum acceptable tetranucleotide usage deviation (TUD). Controls upper bound for overrepresentation of the specified tetranucleotide motif."
    )


@ConstraintRegistry.register(
    key="tetranucleotide-usage",
    config=TetranucleotideUsageConfig,
    description="Evaluate tetranucleotide usage deviation (TUD) for a specific 4-base motif",
    vectorized=False,
    concatenate=True
)
def tetranucleotide_usage_constraint(
    input_sequence: Sequence, config: TetranucleotideUsageConfig
) -> float:
    """
    Evaluate tetranucleotide usage deviation (TUD) for a specific 4-base motif.

    Args:
        input_sequence: The DNA or RNA sequence to evaluate.
        config: Configuration containing tetranucleotide, min_tud, and max_tud parameters.

    Returns:
        Constraint score where 0.0 indicates tetranucleotide usage deviation (TUD) is within acceptable range
        and higher values indicate greater deviation from the acceptable TUD range.

    Raises:
        ValueError: If tetranucleotide is not exactly 4 bases long.
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating tetranucleotide usage constraint:

        >>> seq = Sequence("ATCGATCGATCG", SequenceType.DNA)
        >>> cfg = TetranucleotideUsageConfig(tetranucleotide="ATCG", min_tud=0.5, max_tud=2.0)
        >>> score = tetranucleotide_usage_constraint(seq, config=cfg)
    """
    tetranucleotide = config.tetranucleotide.upper()

    if len(tetranucleotide) != 4:
        raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")

    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    if len(input_sequence) < 4:
        input_sequence._metadata[tetranucleotide + "_tud"] = 0.0
        return MIN_ENERGY

    valid_nucleotides = (
        DNA_NUCLEOTIDES
        if input_sequence.sequence_type == SequenceType.DNA
        else RNA_NUCLEOTIDES
    )

    # Calculate nucleotide frequencies
    seq_length = len(input_sequence)
    nucleotide_freqs = {
        nt: str(input_sequence).count(nt) / seq_length for nt in valid_nucleotides
    }

    # Count tetranucleotide occurrences
    tetra_count = sum(
        1
        for i in range(len(input_sequence) - 3)
        if str(input_sequence)[i : i + 4] == tetranucleotide
    )

    # Calculate expected frequency using zero-order Markov model
    tetra_expected_freq = 1.0
    for nt in tetranucleotide:
        if nt in nucleotide_freqs:
            tetra_expected_freq *= nucleotide_freqs[nt]
        else:
            tetra_expected_freq = 0
            break

    expected_occurrences = tetra_expected_freq * (seq_length - 3)
    tetra_tud = tetra_count / expected_occurrences if expected_occurrences > 0 else 0
    input_sequence._metadata[tetranucleotide + "_tud"] = tetra_tud

    return calculate_range_deviation(tetra_tud, config.min_tud, config.max_tud)
