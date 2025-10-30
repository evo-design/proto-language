"""
K-mer frequency constraint for evaluating sequence k-mer properties with arbitrary mer length.
"""

from __future__ import annotations

import itertools
from typing import List, Literal
import numpy as np

from pydantic import Field, model_validator, field_validator

from proto_language.language.core import Sequence, SequenceType, DNA_NUCLEOTIDES, RNA_NUCLEOTIDES, PROTEIN_AMINO_ACIDS
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.utils import MAX_ENERGY, MIN_ENERGY


class KmerFrequencyConfig(BaseConfig):
    """Configuration for k-mer frequency constraint."""
    
    k: int = Field(
        ge=1,
        le=8,
        description="Length of k-mer to analyze (e.g., 2 for dinucleotide, 3 for trinucleotide, 4 for tetranucleotide). Must be between 1 and 8."
    )
    
    scoring_mode: Literal["frequency", "usage_deviation"] = Field(
        default="frequency",
        description=(
            "Scoring mode for k-mer evaluation"
            "  - 'frequency': Direct frequency range constraint. "
            "Evaluates if observed k-mer frequencies fall within [min_value, max_value] range."
            "  - 'usage_deviation': Usage deviation constraint "
            "Evaluates if k-mer usage deviation (usage_deviation) falls within [min_value, max_value] range. "
            "usage_deviation = observed_freq / expected_freq, where expected_freq is calculated using a zero-order Markov model."
        )
    )
    
    min_value: float = Field(
        ge=0.0,
        description=(
            "Minimum acceptable value based on scoring_mode:"
            "  - For 'frequency' mode: Minimum acceptable frequency (0.0-1.0) for each k-mer."
            "  - For 'usage_deviation' mode: Minimum acceptable usage deviation (usage_deviation). "
            "usage_deviation=1.0 means observed frequency matches expected frequency. Values <1 indicate underrepresentation."
        )
    )
    
    max_value: float = Field(
        ge=0.0,
        description=(
            "Maximum acceptable value based on scoring_mode:\n"
            "  - For 'frequency' mode: Maximum acceptable frequency (0.0-1.0) for each k-mer. "
            "Helps prevent overly repetitive sequences.\n"
            "  - For 'usage_deviation' mode: Maximum acceptable usage deviation (usage_deviation). "
            "Values >1 indicate overrepresentation."
        )
    )
    
    specific_kmer: str | None = Field(
        default=None,
        description=(
            "Optional: Specific k-mer sequence to evaluate (e.g., 'ATCG'). "
            "If specified, only this k-mer is evaluated. If None, all possible k-mers of length k are evaluated. "
            "Length must match the 'k' parameter."
        )
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
    vectorized=True,
    concatenate=True
)
def kmer_frequency_constraint(
    sequences: List[Sequence], config: KmerFrequencyConfig
) -> List[float]:
    """
    Evaluate k-mer frequencies or usage deviations for DNA/RNA sequences.
    
    This generalized constraint supports two scoring modes:
    1. Frequency mode: Evaluates if k-mer frequencies fall within [min_value, max_value]
    2. Usage deviation mode: Evaluates if k-mer usage deviation (observed/expected) falls within range
    
    Args:
        sequences: List of DNA or RNA sequences to evaluate.
        config: Configuration containing k, scoring_mode, min_value, max_value, and optional specific_kmer.
    
    Returns:
        List of constraint scores where 0.0 indicates all k-mer metrics are within acceptable range
        and higher values indicate the maximum deviation across all k-mers.
    """

    if not sequences:
        raise ValueError("Input sequence list must not be empty")

    scores = []

    for seq in sequences:
        if seq.sequence_type not in {SequenceType.DNA, SequenceType.RNA, SequenceType.PROTEIN}:
            raise ValueError(f"Input must be a DNA, RNA, or PROTEIN sequence")

        # Handle sequences shorter than k
        if len(seq) < config.k:
            seq._metadata[f"{config.k}mer_data"] = {}
            scores.append(MAX_ENERGY)
            continue

        # Determine valid characters based on sequence type
        if seq.sequence_type == SequenceType.DNA:
            valid_bases = DNA_NUCLEOTIDES
        elif seq.sequence_type == SequenceType.RNA:
            valid_bases = RNA_NUCLEOTIDES
        else:  # SequenceType.PROTEIN
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