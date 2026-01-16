"""
K-mer frequency constraint for evaluating sequence k-mer properties with arbitrary mer length.
"""

from __future__ import annotations

import itertools
from typing import List, Literal, Optional
import numpy as np

from pydantic import model_validator, field_validator

from proto_language.language.core import Sequence, DNA_NUCLEOTIDES, RNA_NUCLEOTIDES, PROTEIN_AMINO_ACIDS
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import MAX_ENERGY, MIN_ENERGY


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
            
        scoring_mode (Literal["frequency", "usage_deviation"]): Scoring metric to
            evaluate. Options:
            - "frequency": Evaluates if raw k-mer frequencies (observed_count / total_kmers)
              fall within a given [min_value, max_value] range. Use for direct frequency
              constraints like "AT dinucleotides should be 5-10% of all dinucleotides".
            - "usage_deviation": Evaluates observed/expected ratios where expected
              is calculated using a zero-order Markov model (independent nucleotide
              frequencies). A ratio of 1.0 means observed matches expected, >1.0
              indicates overrepresentation, <1.0 indicates underrepresentation.
              Use for detecting codon bias or sequence composition anomalies.
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

        specific_kmer (Optional[str]): If specified, only this specific k-mer is
            evaluated instead of all possible k-mers. Must be uppercase and have
            length equal to k. For example, "ATG" for start codons, or "GGGG" 
            for G-quadruplex motifs. If None, all possible k-mers are evaluated.
            Default: None.
    
    Note:
        **Frequency mode** evaluates raw k-mer proportions. For DNA dinucleotides
        with k=2, there are 16 possible k-mers (AA, AC, ..., TT). If a sequence
        has 100 dinucleotides and 10 are CG, the CG frequency is 0.1 (10%).
        
        **Usage deviation mode** compares observed to expected frequencies under
        a zero-order Markov model. Expected frequency = product of individual
        nucleotide frequencies. For example, if a sequence is 40% G and 60% C,
        the expected CG dinucleotide frequency is 0.4 × 0.6 = 0.24. If observed
        is 0.12, usage_deviation = 0.12/0.24 = 0.5 (underrepresented).
        
        The constraint returns the maximum deviation across all k-mers as the penalty
        score (unless a specific k-mer is specified).
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
        description="Scoring mode for k-mer evaluation. Specifies which metric is compared to range",
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

    # Advanced parameters
    # TODO: This should be a different constraint
    specific_kmer: Optional[str] = ConfigField(
        title="Specific K-mer",
        default=None,
        description="If specified, only this k-mer is evaluated. If None, all possible k-mers of length k are evaluated.",
        advanced=True,
    )

    @field_validator('specific_kmer', mode='before')
    @classmethod
    def uppercase_specific_kmer(cls, v):
        """Convert specific_kmer to uppercase if provided."""
        if v is not None:
            return v.upper()
        return v

    @model_validator(mode='after')
    def validate_config(self):
        """Validate configuration parameters."""
        # Validate min_value <= max_value
        if self.min_value > self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
            )

        # Validate frequency mode range
        if self.scoring_mode == "frequency":
            if self.max_value > 1.0:
                raise ValueError(
                    f"For frequency mode, max_value must be <= 1.0, got {self.max_value}"
                )

        # Validate specific_kmer length if provided
        if self.specific_kmer is not None:
            if len(self.specific_kmer) != self.k:
                raise ValueError(
                    f"specific_kmer length ({len(self.specific_kmer)}) must match k parameter ({self.k})"
                )

        return self


@ConstraintRegistry.register(
    key="kmer-frequency",
    label="K-mer Frequency",
    config=KmerFrequencyConfig,
    description="Evaluate k-mer frequencies or usage deviations with configurable mer length and scoring mode",
    batched=True,
    concatenate=True,
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna", "protein"],
)
def kmer_frequency_constraint(sequences: List[Sequence], config: KmerFrequencyConfig) -> List[float]:
    """Evaluate k-mer frequencies or usage deviations with configurable mer length and scoring modes
    
    This constraint function analyzes k-mer (subsequences of length k) composition
    in DNA, RNA, or protein sequences using two possible scoring modes:
    
    1. **Frequency mode**: Evaluates raw k-mer frequencies (observed_count / total_kmers).
    
    2. **Usage deviation mode**: Evaluates observed/expected ratios using a zero-order
       Markov model where expected = product of individual nucleotide/amino acid
       frequencies. A ratio of 1.0 indicates observed matches expected composition,
       >1.0 indicates overrepresentation, <1.0 indicates underrepresentation.
    
    The constraint can evaluate all possible k-mers of length k, or focus on a
    specific k-mer if provided. It returns the maximum deviation across all
    evaluated k-mers as the penalty score.

    Args:
        sequences (List[Sequence]): List of DNA, RNA, or protein sequences to
            evaluate. Sequences must be at least k nucleotides/amino acids long.
            Sequences shorter than k receive maximum penalty.

        config (KmerFrequencyConfig): Configuration object containing ``k`` (k-mer
            length), ``scoring_mode`` (default: "frequency"), ``min_value``,
            ``max_value``, and optional ``specific_kmer``.

    Returns:
        List[float]: Constraint scores for each sequence. A score of 0.0 indicates
            all k-mer metrics are within the acceptable range [min_value, max_value].
            Higher scores indicate the maximum deviation across all k-mers. The
            penalty scales linearly with deviation distance from the acceptable
            range, capped at 1.0.

    Raises:
        ValueError: If the input sequence list is empty, or if a sequence is not
            DNA, RNA, or PROTEIN type.
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Metadata varies by
        scoring_mode and specific_kmer:
        
        **For frequency mode (all k-mers):**
        - ``{k}mer_frequencies``: Dictionary mapping each k-mer to its frequency
          (0.0-1.0). For example, ``2mer_frequencies`` for dinucleotides.
        
        **For frequency mode (specific k-mer):**
        - ``{k}mer_frequencies``: Dictionary with only the specified k-mer
        
        **For usage_deviation mode (all k-mers):**
        - ``{k}mer_usage_deviations``: Dictionary mapping each k-mer to its
          observed/expected ratio
        
        **For usage_deviation mode (specific k-mer):**
        - ``{kmer}_usage_deviation``: Float observed/expected ratio for the k-mer
        - ``{kmer}_count``: Integer observed count
        - ``{kmer}_expected``: Float expected count
        
        **For sequences too short (<k length):**
        - ``{k}mer_data``: Empty dictionary
    
    Examples:
        Analyzing codon usage (all trinucleotides):
        
        >>> coding_seq = Sequence("ATGAAACGTATTGCGTCG", "dna")
        >>> config = KmerFrequencyConfig(
        ...     k=3,
        ...     scoring_mode="usage_deviation",
        ...     min_value=0.5,  # Allow some underrepresentation
        ...     max_value=2.0   # Allow some overrepresentation
        ... )
        >>> scores = kmer_frequency_constraint([coding_seq], config)
        >>> deviations = coding_seq._metadata["3mer_usage_deviations"]
        >>> for codon, ratio in sorted(deviations.items(), key=lambda x: x[1], reverse=True):
        ...     print(f"{codon}: {ratio:.2f}x expected")
       """ 

    if not sequences:
        raise ValueError("Input sequence list must not be empty")

    scores = []

    for seq in sequences:
        # Handle sequences shorter than k
        if len(seq) < config.k:
            seq._metadata[f"{config.k}mer_data"] = {}
            scores.append(MAX_ENERGY)
            continue

        # Determine valid characters based on sequence type
        if seq.sequence_type == "dna":
            valid_bases = DNA_NUCLEOTIDES
        elif seq.sequence_type == "rna":
            valid_bases = RNA_NUCLEOTIDES
        else:  # "protein"
            valid_bases = PROTEIN_AMINO_ACIDS

        # Generate all possible k-mers or use specific k-mer
        if config.specific_kmer is not None:
            kmers = np.array([config.specific_kmer])
        else:
            kmers = np.array(["".join(p) for p in itertools.product(valid_bases, repeat=config.k)])
        
        kmer_index = {kmer: i for i, kmer in enumerate(kmers)}

        # Extract k-mers from sequence 
        seq_arr = np.frombuffer(seq.sequence.encode("ascii"), dtype="S1").astype(str)
        
        # Create sliding windows for k-mers
        if config.k == 1:
            extracted_kmers = seq_arr
        else:
            indices = np.arange(len(seq_arr) - config.k + 1)[:, None] + np.arange(config.k)
            kmer_chars = seq_arr[indices]
            extracted_kmers = np.array([''.join(kmer) for kmer in kmer_chars])

        # Filter to only valid k-mers (all characters in valid_bases)
        valid_mask = np.array([
            all(char in valid_bases for char in kmer) 
            for kmer in extracted_kmers
        ])
        valid_kmers = extracted_kmers[valid_mask]

        if len(valid_kmers) == 0:
            seq._metadata[f"{config.k}mer_data"] = {}
            scores.append(MAX_ENERGY)
            continue

        # Count k-mer occurrences
        uniq, counts = np.unique(valid_kmers, return_counts=True)
        
        if config.scoring_mode == "frequency":
            # FREQUENCY MODE: Direct frequency evaluation
            freqs = np.zeros(len(kmers), dtype=float)
            total_count = counts.sum()
            
            for kmer, count in zip(uniq, counts):
                if kmer in kmer_index:
                    freqs[kmer_index[kmer]] = count / total_count
            
            # Calculate deviations from acceptable range
            below_mask = freqs < config.min_value
            above_mask = freqs > config.max_value
            deviations = np.zeros_like(freqs)
            
            deviations[below_mask] = (config.min_value - freqs[below_mask]) / max(config.min_value, 1e-9)
            deviations[above_mask] = (freqs[above_mask] - config.max_value) / max(config.max_value, 1e-9)
            deviations = np.clip(deviations, MIN_ENERGY, MAX_ENERGY)
            
            max_dev = deviations.max() if deviations.size > 0 else MAX_ENERGY
            score = float(max_dev)
            
            # Store frequency metadata
            seq._metadata[f"{config.k}mer_frequencies"] = {
                kmers[i]: float(freqs[i]) for i in range(len(kmers))
            }
            
        else:
            # USAGE DEVIATION MODE: usage deviation evaluation
            seq_length = len(seq)
            
            # Calculate nucleotide frequencies
            nucleotide_freqs = {
                nt: str(seq).count(nt) / seq_length for nt in valid_bases
            }
            
            if config.specific_kmer is not None:
                # Evaluate specific k-mer only
                kmer = config.specific_kmer
                
                # Count occurrences
                kmer_count = sum(1 for km in valid_kmers if km == kmer)
                
                # Calculate expected frequency
                expected_freq = 1.0
                for nt in kmer:
                    if nt in nucleotide_freqs:
                        expected_freq *= nucleotide_freqs[nt]
                    else:
                        expected_freq = 0
                        break
                
                expected_occurrences = expected_freq * (seq_length - config.k + 1)
                usage_deviation = kmer_count / expected_occurrences if expected_occurrences > 0 else 0
                
                # Calculate deviation from acceptable usage_deviation range
                if config.min_value <= usage_deviation <= config.max_value:
                    score = MIN_ENERGY
                elif usage_deviation < config.min_value:
                    score = min(MAX_ENERGY, (config.min_value - usage_deviation) / max(config.min_value, 1e-9))
                else:
                    score = min(MAX_ENERGY, (usage_deviation - config.max_value) / max(config.max_value, 1e-9))
                
                # Store usage_deviation metadata
                seq._metadata[f"{kmer}_usage_deviation"] = usage_deviation
                seq._metadata[f"{kmer}_count"] = int(kmer_count)
                seq._metadata[f"{kmer}_expected"] = float(expected_occurrences)
                
            else:
                # Evaluate all possible k-mers
                usage_deviations = np.zeros(len(kmers), dtype=float)
                
                for i, kmer in enumerate(kmers):
                    # Count occurrences
                    kmer_count = sum(1 for km in valid_kmers if km == kmer)
                    
                    # Calculate expected frequency using zero-order Markov model
                    expected_freq = 1.0
                    for nt in kmer:
                        if nt in nucleotide_freqs:
                            expected_freq *= nucleotide_freqs[nt]
                        else:
                            expected_freq = 0
                            break
                    
                    expected_occurrences = expected_freq * (seq_length - config.k + 1)
                    usage_deviations[i] = kmer_count / expected_occurrences if expected_occurrences > 0 else 0
                
                # Calculate deviations from acceptable usage_deviation range
                below_mask = usage_deviations < config.min_value
                above_mask = usage_deviations > config.max_value
                deviations = np.zeros_like(usage_deviations)
                
                deviations[below_mask] = (config.min_value - usage_deviations[below_mask]) / max(config.min_value, 1e-9)
                deviations[above_mask] = (usage_deviations[above_mask] - config.max_value) / max(config.max_value, 1e-9)
                deviations = np.clip(deviations, MIN_ENERGY, MAX_ENERGY)
                
                max_dev = deviations.max() if deviations.size > 0 else MAX_ENERGY
                score = float(max_dev)
                
                # Store usage_deviation metadata for all k-mers
                seq._metadata[f"{config.k}mer_usage_deviations"] = {
                    kmers[i]: float(usage_deviations[i]) for i in range(len(kmers))
                }
        
        scores.append(score)

    return scores
