from abc import ABC, abstractmethod
from typing import Any, List, Dict
from language.sequence import ProgramSequence, ProgramDNASequence, ProgramRNASequence, ProgramProteinSequence

import pandas as pd
import numpy as np
import re
import itertools
import warnings


class ProgramConstraint(ABC):
    """
    Abstract base class for constraints or scoring functions applied to sequences.

    Subclasses must implement the 'evaluate' method.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the constraint, potentially with configuration parameters.

        Args:
            **kwargs (Any): Arbitrary keyword arguments for configuration.
        """
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def evaluate(self, sequences: ProgramSequence | List[ProgramSequence]) -> float:
        """
        Evaluates a list of ProgramSequence instances against the constraint.

        Args:
            sequences (List[ProgramSequence]): The sequences to evaluate.

        Returns:
            float: A score representing how well the sequences satisfy the
                   constraint. Implementations should aim for a score in the
                   interval [0.0, 1.0].
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(self, sequences: ProgramSequence | List[ProgramSequence]) -> float:
        """
        Allows calling the constraint object directly like a function.

        Args:
            sequences (List[ProgramSequence]): The sequences to evaluate.

        Returns:
            float: The score from the evaluate method.
        """
        return self.evaluate(sequences)


class ValidCharactersConstraint(ProgramConstraint):
    def __init__(self) -> None:
        super().__init__()
        self.valid_chars = {
            ProgramDNASequence: {
                'A', 'C', 'G', 'T',
            },
            ProgramRNASequence: {
                'A', 'U', 'G', 'C',
            },
            ProgramProteinSequence: {
                'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
            }
        }

    def evaluate(self, sequences: ProgramSequence | List[ProgramSequence]) -> float:
        if isinstance(sequences, ProgramSequence):
            sequences = [sequences]

        score = 1.0
        for seq in sequences:
            if isinstance(seq, ProgramDNASequence):
                valid_chars = self.valid_chars[ProgramDNASequence]
            elif isinstance(seq, ProgramRNASequence):
                valid_chars = self.valid_chars[ProgramRNASequence]
            elif isinstance(seq, ProgramProteinSequence):
                valid_chars = self.valid_chars[ProgramProteinSequence]
            else:
                raise ValueError(f"Unknown sequence type: {type(seq)}")

            has_invalid = bool(re.search(f'[^{"".join(valid_chars)}]', seq))

            # Add metadata for individual sequences
            seq._metadata['valid_characters'] = not has_invalid
            if has_invalid:
                score = 0.0
        return score


class SequenceLengthConstraint(ProgramConstraint):
    def __init__(self, target_length: int) -> None:
        super().__init__()
        self.target_length = target_length

    def evaluate(self, sequences: ProgramSequence | List[ProgramSequence]) -> float:
        if isinstance(sequences, ProgramSequence):
            sequences = [sequences]
        else:
            warnings.warn("Input is a list of sequences. Concatenating for length calculation.")
        
        for seq in sequences:
            seq._metadata['length'] = len(seq)
        
        # Calculate deviation based on total length
        full_length = len(''.join(str(seq) for seq in sequences))
        if full_length == self.target_length:
            return 0.0
        
        # Calculate normalized deviation from target length
        deviation = abs(full_length - self.target_length) / self.target_length
        return min(1.0, deviation)


class GCContentConstraint(ProgramConstraint):
    def __init__(self, target_range: tuple = (30, 60)) -> None:
        super().__init__()
        self.min_gc = min(target_range)
        self.max_gc = max(target_range)
        
        # Validate range
        if self.min_gc < 0 or self.max_gc > 100:
            raise ValueError("GC content range must be between 0 and 100 percent.")

    def evaluate(self, sequences: ProgramSequence) -> float:
        assert isinstance(sequences, (ProgramDNASequence, ProgramRNASequence)), \
               "Input must be ProgramDNASequence or ProgramRNASequence object"

        # edge case
        if len(sequences) == 0:
            sequences._metadata['gc_content'] = 0.0
            return 0.0

        # Count G and C nucleotides directly in one pass
        gc_count = 0
        for nt in str(sequences):
            if nt in 'GC':
                gc_count += 1
                    
        # Calculate GC content
        gc_content = (gc_count / len(sequences)) * 100
        sequences._metadata['gc_content'] = gc_content
            
        # return 0.0 if GC content is within the range
        if self.min_gc <= gc_content <= self.max_gc:
            return 0.0
        else:
            # return a normalized score based on distance from acceptable range
            if gc_content < self.min_gc:
                deviation = (self.min_gc - gc_content) / self.min_gc
            else:
                deviation = (gc_content - self.max_gc) / (100 - self.max_gc)
            return min(1.0, deviation)
                

class MaxHomopolymerConstraint(ProgramConstraint):
    def __init__(self, max_length: int = 10) -> None:
        super().__init__()
        self.max_length = max_length

    def evaluate(self, sequences: ProgramSequence) -> float:
        assert isinstance(sequences, ProgramSequence), \
               "Input must be ProgramSequence object"
        
        # Edge case
        if len(sequences) <= 1:
            longest_homopolymer = len(sequences)
        else:
            # Find length of each homopolymer
            homopolymer_lengths = [len(list(group)) for _, group in itertools.groupby(str(sequences))]
            longest_homopolymer = max(homopolymer_lengths)

        sequences._metadata['max_homopolymer_length'] = longest_homopolymer
        
        # Return 0.0 if longest homopolymer is within range
        if longest_homopolymer <= self.max_length:
            return 0.0
        else:
            # Use a logarithmic scale for scoring
            excess_length = longest_homopolymer - self.max_length
            log_ratio = np.log(1 + excess_length/self.max_length) / np.log(2)
            return min(1.0, log_ratio)
        

class DinucleotideFrequencyConstraint(ProgramConstraint):
    def __init__(self, freq_range: tuple = (0.03, 0.08)) -> None:
        super().__init__()
        self.min_freq = min(freq_range)
        self.max_freq = max(freq_range)
        
    def evaluate(self, sequences: ProgramSequence) -> float:
        assert isinstance(sequences, (ProgramDNASequence, ProgramRNASequence)), \
               "Input must be ProgramDNASequence or ProgramRNASequence object"
        
        # Determine valid nucleotides
        valid_nucleotides = 'ATCG' if isinstance(sequences, ProgramDNASequence) else 'AUCG'
            
        # Precompute dinucleotides
        dinucleotides = [''.join(pair) for pair in itertools.product(valid_nucleotides, repeat=2)]
        
        # Edge case
        if len(sequences) < 2:
            sequences._metadata['dinucleotide_freqs'] = {}
            return 1.0
        
        # Count dinucleotides
        dinucleotide_counts = {}
        total_count = 0
        for i in range(len(sequences) - 1):
            dinuc = str(sequences)[i:i+2]
            if all(nt in valid_nucleotides for nt in dinuc):
                dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
                total_count += 1
        
        # If no valid dinucleotides found
        if total_count == 0:
            sequences._metadata['dinucleotide_freqs'] = {}
            return 1.0
            
        # Calculate frequencies and check if they're in range
        max_deviation = 0.0
        dinucleotide_freqs = {}
        
        # Score based on deviation from target dinucleotide frequencies
        for dinuc in dinucleotides:
            freq = dinucleotide_counts.get(dinuc, 0) / total_count
            dinucleotide_freqs[dinuc] = freq
            
            # Calculate deviation if outside acceptable range
            if freq < self.min_freq:
                deviation = (self.min_freq - freq) / self.min_freq
                max_deviation = max(max_deviation, deviation)
            elif freq > self.max_freq:
                deviation = (freq - self.max_freq) / (1.0 - self.max_freq)
                max_deviation = max(max_deviation, deviation)
        
        sequences._metadata['dinucleotide_freqs'] = dinucleotide_freqs
        return min(1.0, max_deviation)
    

class TetranucleotideUsageConstraint(ProgramConstraint):
    def __init__(self, tetranucleotide: str, tud_range: tuple = (0.8, 1.2)) -> None:
        super().__init__()
        self.tetranucleotide = tetranucleotide.upper()
        self.min_tud = min(tud_range)
        self.max_tud = max(tud_range)
        
        # Validate tetranucleotide input
        if len(self.tetranucleotide) != 4:
            raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")
    
    def evaluate(self, sequences: ProgramSequence) -> float:
        assert isinstance(sequences, (ProgramDNASequence, ProgramRNASequence)), \
               "Input must be ProgramDNASequence or ProgramRNASequence object"
        
        # Set appropriate nucleotide keys based on sequence type
        nucleotide_keys = ['A', 'T', 'C', 'G'] if isinstance(sequences, ProgramDNASequence) else ['A', 'U', 'C', 'G']

        # edge case
        if len(sequences) < 4:
            sequences._metadata[self.tetranucleotide + '_tud'] = 0.0
            return 0.0
        
        # Calculate nucleotide frequencies
        nucleotide_freqs = {}
        seq_length = len(sequences)
        for nt in nucleotide_keys:
            nucleotide_freqs[nt] = str(sequences).count(nt) / seq_length

        # Count occurrences of tetranucleotide
        tetra_count = 0
        for i in range(len(sequences) - 3):
            if str(sequences)[i:i+4] == self.tetranucleotide:
                tetra_count += 1
        
        # Calculate expected frequency using zero-order Markov model
        tetra_expected_freq = 1.0
        for nt in self.tetranucleotide:
            if nt in nucleotide_freqs:
                tetra_expected_freq *= nucleotide_freqs[nt]
            else:
                # If invalid nucleotide, set to 0
                tetra_expected_freq = 0
                break
        
        # Calculate expected occurrences and TUD
        expected_occurrences = tetra_expected_freq * (seq_length - 3)
        tetra_tud = tetra_count / expected_occurrences if expected_occurrences > 0 else 0
        sequences._metadata[self.tetranucleotide + '_tud'] = tetra_tud
        
        # Score based on TUD range
        if self.min_tud <= tetra_tud <= self.max_tud:
            return 0.0
        else:
            # Calculate normalized deviation
            if tetra_tud < self.min_tud:
                deviation = (self.min_tud - tetra_tud) / self.min_tud
            else:
                deviation = (tetra_tud - self.max_tud) / self.max_tud
            
            return min(1.0, deviation)

