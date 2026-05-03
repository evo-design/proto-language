"""Constraint for evaluating the frequency or usage deviation of a single specific k-mer."""

import math
from typing import Literal

from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, calculate_range_deviation

_FRACTIONAL_EPSILON = 1e-9


class SpecificKmerConfig(BaseConfig):
    """Configuration for evaluating a single specific k-mer.

    For evaluating all k-mers of a given length, use KmerFrequencyConstraint.

    Attributes:
        kmer (str): The specific k-mer to evaluate (e.g., 'CG', 'GATC', 'ATG').
            Must be 1-8 characters, uppercase, and valid for the sequence type.
        scoring_mode (Literal['frequency', 'usage_deviation']): Scoring metric.
            'frequency' evaluates raw k-mer frequency (count / total positions).
            'usage_deviation' evaluates observed/expected ratio using a zero-order
            Markov model. Default: 'frequency'.
        min_value (float): Minimum acceptable frequency or deviation. Must be >= 0.
        max_value (float): Maximum acceptable frequency or deviation. Must be >= 0
            and >= min_value. Capped at 1.0 for frequency mode.
    """

    kmer: str = ConfigField(
        title="K-mer",
        description="The specific k-mer to evaluate (e.g., 'CG', 'GATC', 'ATG')",
        examples=["CG", "GATC", "ATG"],
    )
    scoring_mode: Literal["frequency", "usage_deviation"] = ConfigField(
        title="Scoring Mode",
        default="frequency",
        description="Scoring mode: 'frequency' for raw counts, 'usage_deviation' for observed/expected ratio",
        examples=["frequency", "usage_deviation"],
    )
    min_value: float = ConfigField(
        title="Minimum acceptable value",
        ge=0.0,
        description="Minimum acceptable frequency/deviation based on scoring_mode",
    )
    max_value: float = ConfigField(
        title="Maximum acceptable value",
        ge=0.0,
        description="Maximum acceptable frequency/deviation based on scoring_mode",
    )

    @field_validator("kmer", mode="before")
    @classmethod
    def uppercase_kmer(cls, v: str) -> str:
        """Normalize k-mer to uppercase."""
        return v.upper()

    @model_validator(mode="after")
    def validate_config(self) -> "SpecificKmerConfig":
        """Validate configuration parameters."""
        if self.min_value > self.max_value:
            raise ValueError(f"min_value ({self.min_value}) must be <= max_value ({self.max_value})")
        if self.scoring_mode == "frequency" and self.max_value > 1.0:
            raise ValueError(f"For frequency mode, max_value must be <= 1.0, got {self.max_value}")
        if not 1 <= len(self.kmer) <= 8:
            raise ValueError(f"K-mer length must be between 1 and 8, got {len(self.kmer)}")
        return self


def _count_overlapping(seq_str: str, kmer: str) -> int:
    """Count overlapping occurrences of kmer in seq_str."""
    if not kmer:
        return 0
    count = 0
    start = 0
    while True:
        pos = seq_str.find(kmer, start)
        if pos == -1:
            return count
        count += 1
        start = pos + 1


@constraint(
    key="specific-kmer-frequency",
    label="Specific K-mer Frequency",
    config=SpecificKmerConfig,
    description="Evaluate frequency or usage deviation of a specific k-mer motif",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna", "protein"],
)
def specific_kmer_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: SpecificKmerConfig
) -> list[ConstraintOutput]:
    """Evaluate frequency or usage deviation of a specific k-mer.

    Supports two scoring modes:

    1. **Frequency mode**: Raw k-mer frequency (count / total k-mer positions).
    2. **Usage deviation mode**: Observed/expected ratio using a zero-order
       Markov model (product of individual character frequencies).

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
        config (SpecificKmerConfig): Configuration specifying the k-mer and scoring parameters.

    Returns:
        list[ConstraintOutput]: One result per input sequence. The ``metadata`` field
            carries per-mode k-mer data (see Note).

    Note:
        Metadata varies by mode:

        **Frequency mode:**
        - ``{kmer}_frequency``: Float frequency value

        **Usage deviation mode:**
        - ``{kmer}_usage_deviation``: Float observed/expected ratio
        - ``{kmer}_count``: Integer observed count
        - ``{kmer}_expected``: Float expected count
    """
    k = len(config.kmer)
    results: list[ConstraintOutput] = []

    for (seq,) in input_sequences:
        seq_str = seq.sequence.upper()
        seq_length = len(seq_str)

        if seq_length < k:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"specific_kmer_error": f"sequence length {seq_length} < kmer length {k}"},
                )
            )
            continue

        # Validate kmer characters against sequence alphabet
        valid_chars = seq.valid_chars
        if valid_chars is None or not all(c in valid_chars for c in config.kmer):
            raise ValueError(
                f"K-mer '{config.kmer}' contains characters invalid for sequence type '{seq.sequence_type}'"
            )

        kmer_count = _count_overlapping(seq_str, config.kmer)
        total_positions = seq_length - k + 1

        if config.scoring_mode == "frequency":
            frequency = kmer_count / total_positions
            score = calculate_range_deviation(frequency, config.min_value, config.max_value, _FRACTIONAL_EPSILON)
            metadata: dict[str, float | int] = {f"{config.kmer}_frequency": frequency}

        else:
            # Usage deviation: observed / expected via zero-order Markov model
            char_freqs = {c: seq_str.count(c) / seq_length for c in set(config.kmer)}
            expected_freq = math.prod(char_freqs.get(c, 0) for c in config.kmer)
            expected_occurrences = expected_freq * total_positions
            usage_deviation = kmer_count / expected_occurrences if expected_occurrences > 0 else 0

            score = calculate_range_deviation(usage_deviation, config.min_value, config.max_value, _FRACTIONAL_EPSILON)
            metadata = {
                f"{config.kmer}_usage_deviation": usage_deviation,
                f"{config.kmer}_count": kmer_count,
                f"{config.kmer}_expected": float(expected_occurrences),
            }

        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results
