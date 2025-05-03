from typing import (
    Callable, List, Tuple, Dict, Any, Set, Optional,
)
import pandas as pd
import numpy as np
import re
import itertools
import warnings

from .base import ProgramConstraint, ProgramSequence
from .sequence import ProgramDNASequence, ProgramRNASequence, ProgramProteinSequence


class ValidCharactersConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence],
    ) -> None:
        
        super().__init__(inputs, 0.0) # dummy scoring function for now, ASK
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

    def evaluate(self) -> float:
        score = 1.0
        for seq in self.inputs:
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
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        target_length: int,
    ) -> None:
        
        super().__init__(inputs, 0.0) # dummy scoring function for now
        self.target_length = target_length

    def evaluate(self) -> float:
        if len(self.inputs) > 1:
            warnings.warn("Input is a list of sequences. Concatenating for length calculation.")
        
        for seq in self.inputs:
            seq._metadata['length'] = len(seq)
        
        # Calculate deviation based on total length
        full_length = len(''.join(str(seq) for seq in self.inputs))
        if full_length == self.target_length:
            return 0.0
        
        # Calculate normalized deviation from target length
        deviation = abs(full_length - self.target_length) / self.target_length
        return min(1.0, deviation)


class GCContentConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        target_range: tuple = (30.0, 60.0),
    ) -> None:
        
        super().__init__(inputs, 0.0) # dummy scoring function for now
        self.min_gc, self.max_gc = target_range
        
        # Validate range
        if self.min_gc < 0 or self.max_gc > 100:
            raise ValueError("GC content range must be between 0 and 100 percent.")

    def evaluate(self) -> float:
        assert len(self.inputs) == 1 and isinstance(self.inputs[0], (ProgramDNASequence, ProgramRNASequence)), \
               "Input must be a single ProgramDNASequence or ProgramRNASequence object"
        sequence = self.inputs[0]

        # Calculate GC content
        gc_content = 100.0 * sum(nt in "GC" for nt in str(sequence).upper()) / max(len(sequence), 1)
            
        # return 0.0 if GC content is within the range
        if self.min_gc <= gc_content <= self.max_gc:
            return 0.0
        else:
            if gc_content < self.min_gc:
                deviation = (self.min_gc - gc_content) / self.min_gc
            else:
                deviation = (gc_content - self.max_gc) / (100 - self.max_gc)
            return min(1.0, deviation)
                

class MaxHomopolymerConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        max_length: int = 10,
    ) -> None:
        
        super().__init__(inputs, 0.0) # dummy scoring function for now
        self.max_length = max_length

    def evaluate(self) -> float:
        assert len(self.inputs) == 1 and isinstance(self.inputs[0], ProgramSequence), \
               "Input must be a single ProgramSequence object"
        
        sequence = self.inputs[0]
        # Edge case
        if len(sequence) <= 1:
            longest_homopolymer = len(sequence)
        else:
            # Find length of each homopolymer
            homopolymer_lengths = [len(list(group)) for _, group in itertools.groupby(str(sequence))]
            longest_homopolymer = max(homopolymer_lengths)

        sequence._metadata['max_homopolymer_length'] = longest_homopolymer
        
        # Return 0.0 if longest homopolymer is within range
        if longest_homopolymer <= self.max_length:
            return 0.0
        else:
            # Use a logarithmic scale for scoring
            excess_length = longest_homopolymer - self.max_length
            log_ratio = np.log(1 + excess_length/self.max_length) / np.log(2)
            return min(1.0, log_ratio)
        

class DinucleotideFrequencyConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        freq_range: tuple = (0.03, 0.08),
    ) -> None:
        
        super().__init__(inputs, 0.0) # dummy scoring function for now
        self.min_freq = min(freq_range)
        self.max_freq = max(freq_range)
        
    def evaluate(self) -> float:
        assert len(self.inputs) == 1 and isinstance(self.inputs[0], (ProgramDNASequence, ProgramRNASequence)), \
               "Input must be ProgramDNASequence or ProgramRNASequence object"
        sequence = self.inputs[0]
        # Determine valid nucleotides
        valid_nucleotides = 'ATCG' if isinstance(sequence, ProgramDNASequence) else 'AUCG'
            
        # Precompute dinucleotides
        dinucleotides = [''.join(pair) for pair in itertools.product(valid_nucleotides, repeat=2)]
        
        # Edge case
        if len(sequence) < 2:
            sequence._metadata['dinucleotide_freqs'] = {}
            return 1.0
        
        # Count dinucleotides
        dinucleotide_counts = {}
        total_count = 0
        for i in range(len(sequence) - 1):
            dinuc = str(sequence)[i:i+2]
            if all(nt in valid_nucleotides for nt in dinuc):
                dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
                total_count += 1
        
        # If no valid dinucleotides found
        if total_count == 0:
            sequence._metadata['dinucleotide_freqs'] = {}
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
        
        sequence._metadata['dinucleotide_freqs'] = dinucleotide_freqs
        return min(1.0, max_deviation)
    

class TetranucleotideUsageConstraint(ProgramConstraint):
    def __init__(self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        tetranucleotide: str, 
        tud_range: tuple = (0.8, 1.2),
    ) -> None:
        
        super().__init__(inputs, 0.0) # dummy scoring function for now
        self.tetranucleotide = tetranucleotide.upper()
        self.min_tud = min(tud_range)
        self.max_tud = max(tud_range)
        
        # Validate tetranucleotide input
        if len(self.tetranucleotide) != 4:
            raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")
    
    def evaluate(self) -> float:
        assert len(self.inputs) == 1 and isinstance(self.inputs[0], (ProgramDNASequence, ProgramRNASequence)), \
               "Input must be ProgramDNASequence or ProgramRNASequence object"
        
        sequence = self.inputs[0]
        # Set appropriate nucleotide keys based on sequence type
        nucleotide_keys = ['A', 'T', 'C', 'G'] if isinstance(sequence, ProgramDNASequence) else ['A', 'U', 'C', 'G']

        # edge case
        if len(sequence) < 4:
            sequence._metadata[self.tetranucleotide + '_tud'] = 0.0
            return 0.0
        
        # Calculate nucleotide frequencies
        nucleotide_freqs = {}
        seq_length = len(sequence)
        for nt in nucleotide_keys:
            nucleotide_freqs[nt] = str(sequence).count(nt) / seq_length

        # Count occurrences of tetranucleotide
        tetra_count = 0
        for i in range(len(sequence) - 3):
            if str(sequence)[i:i+4] == self.tetranucleotide:
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
        sequence._metadata[self.tetranucleotide + '_tud'] = tetra_tud
        
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