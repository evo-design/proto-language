"""K-mer frequency constraint for evaluating sequence k-mer properties with arbitrary mer length."""

from typing import Literal

import numpy as np
from pydantic import model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import (
    DNA_NUCLEOTIDES,
    PROTEIN_AMINO_ACIDS,
    RNA_NUCLEOTIDES,
    ConstraintOutput,
    Sequence,
)
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField


class KmerFrequencyConfig(BaseConfig):
    """Configuration for k-mer frequency constraint.

    This class defines configuration parameters for evaluating k-mer composition
    in DNA, RNA, or protein sequences. K-mers are subsequences of length k, and
    their frequencies can indicate codon bias, tandem repeats, sequence composition
    biases, CpG islands, etc. The constraint supports two scoring modes:
    frequency-based (direct k-mer counts) and usage deviation (observed vs expected
    based on nucleotide/amino acid composition).

    Attributes:
        k (int): Length of k-mers to analyze. Must be between 1 and 8. Common values:
            - 1: Mononucleotides/amino acids (base composition)
            - 2: Dinucleotides (e.g., CpG content in DNA)
            - 3: Trinucleotides/codons (codon usage in coding sequences)
            - 4+: Longer motifs (tetranucleotide frequencies etc.)

        scoring_mode (Literal['frequency', 'usage_deviation']): Scoring metric. The
            [min_value, max_value] band applies to every k-mer that occurs in the
            sequence (a global composition band, not a single-k-mer target); for one
            specific k-mer use specific_kmer_constraint instead. Options:
            - "frequency": raw frequency (observed_count / total_kmers) of each
              observed k-mer.
            - "usage_deviation": observed/expected ratio under a zero-order Markov
              model (1.0 = expected, >1.0 over-, <1.0 underrepresented); detects
              codon bias.
            Default: "frequency".

        min_value (float): Minimum acceptable value (interpretation depends on
            scoring_mode). Must be non-negative. For frequency mode: minimum k-mer
            frequency (0.0-1.0). For usage_deviation mode: minimum acceptable
            observed/expected ratio (e.g., 0.8 = at least 80% of expected).

        max_value (float): Maximum acceptable value (interpretation depends on
            scoring_mode). Must be non-negative and ≥ min_value. For frequency
            mode: maximum k-mer frequency (0.0-1.0), capped at 1.0. For usage_deviation
            mode: maximum acceptable observed/expected ratio (e.g., 1.5 = at most
            150% of expected).

    Note:
        **Frequency mode** evaluates raw k-mer proportions (10/100 CG dinucleotides
        = 0.1). Only k-mers that occur in the sequence are scored; absent k-mers
        are not penalized.

        **Usage deviation mode** compares observed to expected frequencies under
        a zero-order Markov model. Expected frequency = product of individual
        nucleotide frequencies. For example, if a sequence is 40% G and 60% C,
        the expected CG dinucleotide frequency is 0.4 x 0.6 = 0.24. If observed
        is 0.12, usage_deviation = 0.12/0.24 = 0.5 (underrepresented).

        The penalty is the maximum deviation across observed k-mers. To evaluate a
        single specific k-mer (including penalizing its absence), use
        specific_kmer_constraint instead.
    """

    # Required parameters
    k: int = ConfigField(
        title="K-mer Length",
        ge=1,
        le=8,
        description="Length of k-mer to analyze (e.g., 2 for dinucleotide, 3 for trinucleotide).",
    )
    scoring_mode: Literal["frequency", "usage_deviation"] = ConfigField(
        title="Scoring Mode",
        default="frequency",
        description="Scoring metric: 'frequency' uses raw k-mer counts; 'usage_deviation' uses observed/expected ratios.",
        examples=["frequency", "usage_deviation"],
    )
    min_value: float = ConfigField(
        title="Min Value",
        ge=0.0,
        description="Minimum acceptable frequency/deviation based on scoring_mode",
    )
    max_value: float = ConfigField(
        title="Max Value",
        ge=0.0,
        description="Maximum acceptable frequency/deviation based on scoring_mode",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "KmerFrequencyConfig":
        """Validate configuration parameters."""
        # Validate min_value <= max_value
        if self.min_value > self.max_value:
            raise ValueError(f"min_value ({self.min_value}) must be <= max_value ({self.max_value})")

        # Validate frequency mode range
        if self.scoring_mode == "frequency" and self.max_value > 1.0:
            raise ValueError(f"For frequency mode, max_value must be <= 1.0, got {self.max_value}")

        return self


@constraint(
    key="kmer-frequency",
    label="K-mer Frequency",
    config=KmerFrequencyConfig,
    description="Evaluate k-mer frequencies or usage deviations with configurable mer length and scoring mode",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna", "protein"],
)
def kmer_frequency_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: KmerFrequencyConfig
) -> list[ConstraintOutput]:
    """Evaluate k-mer frequencies or usage deviations with configurable mer length and scoring modes.

    This constraint function analyzes k-mer (subsequences of length k) composition
    in DNA, RNA, or protein sequences using two possible scoring modes:

    1. **Frequency mode**: Evaluates raw k-mer frequencies (observed_count / total_kmers).

    2. **Usage deviation mode**: Evaluates observed/expected ratios using a zero-order
       Markov model where expected = product of individual nucleotide/amino acid
       frequencies. A ratio of 1.0 indicates observed matches expected composition,
       >1.0 indicates overrepresentation, <1.0 indicates underrepresentation.

    The penalty is the maximum deviation from the [min_value, max_value] band across
    observed k-mers; absent k-mers are not penalized. To target a single specific
    k-mer (including penalizing its absence), use specific_kmer_constraint instead.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one DNA, RNA, or protein sequence. Sequences must
            be at least k nucleotides/amino acids long. Sequences shorter than k
            receive maximum penalty.

        config (KmerFrequencyConfig): Configuration object containing ``k`` (k-mer
            length), ``scoring_mode`` (default: "frequency"), ``min_value``,
            and ``max_value``.

    Returns:
        list[ConstraintOutput]: One result per sequence. A score of 0.0 indicates
            every observed k-mer is within the acceptable range [min_value, max_value].
            Higher scores indicate the maximum deviation across observed k-mers. The
            penalty scales linearly with deviation distance from the acceptable
            range, capped at 1.0. ``metadata`` carries (over *observed* k-mers only):

            **For frequency mode:**

            - ``{k}mer_frequencies``: Dictionary mapping each observed k-mer to its
              frequency (0.0-1.0). For example, ``2mer_frequencies`` for dinucleotides.

            **For usage_deviation mode:**

            - ``{k}mer_usage_deviations``: Dictionary mapping each observed k-mer to
              its observed/expected ratio

            **For sequences too short (<k length) or with no valid k-mers:**

            - ``{k}mer_data``: Empty dictionary

    Examples:
        Analyzing codon usage (all trinucleotides):

        >>> coding_seq = Sequence("ATGAAACGTATTGCGTCG", "dna")
        >>> config = KmerFrequencyConfig(
        ...     k=3,
        ...     scoring_mode="usage_deviation",
        ...     min_value=0.5,  # Allow some underrepresentation
        ...     max_value=2.0,  # Allow some overrepresentation
        ... )
        >>> results = kmer_frequency_constraint([(coding_seq,)], config)
        >>> deviations = results[0].metadata["3mer_usage_deviations"]
        >>> for codon, ratio in sorted(deviations.items(), key=lambda x: x[1], reverse=True):
        ...     print(f"{codon}: {ratio:.2f}x expected")
    """
    results: list[ConstraintOutput] = []

    for (seq,) in input_sequences:
        # Handle sequences shorter than k
        if len(seq) < config.k:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={f"{config.k}mer_data": {}}))
            continue

        # Determine valid characters based on sequence type
        if seq.sequence_type == "dna":
            valid_bases = DNA_NUCLEOTIDES
        elif seq.sequence_type == "rna":
            valid_bases = RNA_NUCLEOTIDES
        else:  # "protein"
            valid_bases = PROTEIN_AMINO_ACIDS

        valid_base_set = set(valid_bases)

        # Extract k-mers from the (uppercased) sequence
        seq_str = seq.sequence.upper()
        seq_arr = np.frombuffer(seq_str.encode("ascii"), dtype="S1").astype(str)

        # Create sliding windows for k-mers
        if config.k == 1:
            extracted_kmers = seq_arr
        else:
            indices = np.arange(len(seq_arr) - config.k + 1)[:, None] + np.arange(config.k)
            kmer_chars = seq_arr[indices]
            extracted_kmers = np.array(["".join(kmer) for kmer in kmer_chars])

        # Filter to only valid k-mers (all characters in valid_bases)
        valid_mask = np.array([all(char in valid_base_set for char in kmer) for kmer in extracted_kmers])
        valid_kmers = extracted_kmers[valid_mask]

        if len(valid_kmers) == 0:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={f"{config.k}mer_data": {}}))
            continue

        # Score only observed k-mers, so a non-zero min_value does not saturate the
        # score to the maximum penalty via the absent k-mers.
        uniq, counts = np.unique(valid_kmers, return_counts=True)

        if config.scoring_mode == "frequency":
            # FREQUENCY MODE: Direct frequency evaluation over observed k-mers.
            total_count = counts.sum()
            values = counts / total_count
            metadata = {f"{config.k}mer_frequencies": {str(km): float(v) for km, v in zip(uniq, values, strict=True)}}
        else:
            # USAGE DEVIATION MODE: observed/expected ratio over observed k-mers.
            seq_length = len(seq_str)
            nucleotide_freqs = {nt: seq_str.count(nt) / seq_length for nt in valid_bases}
            num_windows = seq_length - config.k + 1

            values = np.zeros(len(uniq), dtype=float)
            for i, kmer in enumerate(uniq):
                # Expected occurrences under a zero-order Markov model.
                expected_freq = 1.0
                for nt in kmer:
                    expected_freq *= nucleotide_freqs.get(nt, 0.0)
                expected_occurrences = expected_freq * num_windows
                values[i] = counts[i] / expected_occurrences if expected_occurrences > 0 else 0.0

            metadata = {
                f"{config.k}mer_usage_deviations": {str(km): float(v) for km, v in zip(uniq, values, strict=True)}
            }

        # Deviation of each observed k-mer from the acceptable [min_value, max_value] band.
        below_mask = values < config.min_value
        above_mask = values > config.max_value
        deviations = np.zeros_like(values)
        deviations[below_mask] = (config.min_value - values[below_mask]) / max(config.min_value, 1e-9)
        deviations[above_mask] = (values[above_mask] - config.max_value) / max(config.max_value, 1e-9)
        deviations = np.clip(deviations, MIN_ENERGY, MAX_ENERGY)

        score = float(deviations.max()) if deviations.size > 0 else MAX_ENERGY

        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results
