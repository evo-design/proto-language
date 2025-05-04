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


class SequenceLengthConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        target_length: int,
    ) -> None:
        """
        Initializes a soft length constraint.

        Args:
            inputs (ProgramSequence | List[ProgramSequence]): The input variables.
            target_length (int): The targeted length.
        """
        self.target_length = target_length

        def _scoring_function(inputs: List[ProgramSequence]) -> float:
            if len(inputs) > 1:
                warnings.warn("Input is a list of sequences. Concatenating for length calculation.")

            for seq in inputs:
                seq._metadata['length'] = len(seq)

            # Calculate deviation based on total length.
            full_length = len(''.join(str(seq) for seq in inputs))
            if full_length == self.target_length:
                return 0.0

            # Calculate normalized deviation from target length.
            deviation = abs(full_length - self.target_length) / self.target_length
            return min(1.0, deviation)
        
        super().__init__(inputs, _scoring_function)


class GCContentConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        target_range: Tuple[float, float] = (30.0, 60.0),
    ) -> None:
        """
        Initializes a constraint on GC content to be within a target range.

        Args:
            inputs (ProgramSequence | List[ProgramSequence]): The input variables.
            target_range (Tuple[float, float]): The lower and upper bounds, inclusive, of the GC content target range.
        """
        self.min_gc, self.max_gc = target_range
        
        # Validate range.
        if self.min_gc < 0 or self.max_gc > 100:
            raise ValueError("GC content range must be between 0 and 100 percent.")

        def _scoring_function(inputs: List[ProgramSequence]) -> float:
            """
            If the GC content is within the range, simply returns 0.
            If not, imposes a penalty directly related to the deviation from the target GC content.
            """
            if len(inputs) > 1:
                warnings.warn("Input is a list of sequences. Concatenating for GC content calculation.")

            sequence = "".join(str(sequence) for sequence in inputs)

            # Calculate GC content.
            gc_content = 100.0 * sum(nt in "GC" for nt in sequence.upper()) / max(len(sequence), 1)

            if len(inputs) == 1:
                inputs[0]._metadata['gc_content'] = gc_content
            
            # Return 0.0 if GC content is within the range.
            if self.min_gc <= gc_content <= self.max_gc:
                return 0.0
            else:
                if gc_content < self.min_gc:
                    deviation = (self.min_gc - gc_content) / self.min_gc
                else:
                    deviation = (gc_content - self.max_gc) / (100 - self.max_gc)
                return min(1.0, deviation)

        super().__init__(inputs, _scoring_function)


class MaxHomopolymerConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        max_length: int = 10,
    ) -> None:
        """
        TODO(@dguo8412): Fill in this docstring.
        """
        self.max_length = max_length

        def _scoring_function(inputs: List[ProgramSequence]) -> float:
            if len(inputs) > 1:
                warnings.warn(
                    "Input is a list of sequences. Concatenating for homopolymer calculation."
                )

            sequence = "".join(str(sequence) for sequence in inputs)

            if len(sequence) <= 1:
                # Edge case.
                longest_homopolymer = len(sequence)
            else:
                # Find length of each homopolymer.
                homopolymer_lengths = [len(list(group)) for _, group in itertools.groupby(sequence)]
                longest_homopolymer = max(homopolymer_lengths)

            if len(inputs) == 1:
                inputs[0]._metadata['max_homopolymer_length'] = longest_homopolymer

            # Return 0.0 if longest homopolymer is within range.
            if longest_homopolymer <= self.max_length:
                return 0.0
            else:
                # Use a logarithmic scale for scoring.
                excess_length = longest_homopolymer - self.max_length
                log_ratio = np.log(1 + excess_length / self.max_length) / np.log(2)
                return min(1.0, log_ratio)

        super().__init__(inputs, _scoring_function)


class DinucleotideFrequencyConstraint(ProgramConstraint):
    def __init__(
        self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        freq_range: tuple = (0.03, 0.08),
    ) -> None:
        """
        TODO(@dguo8412): Fill in this docstring.
        """
        self.min_freq = min(freq_range)
        self.max_freq = max(freq_range)

        def _scoring_function(inputs: List[ProgramSequence]) -> float:
            assert (
                len(inputs) == 1 and
                isinstance(inputs[0], (ProgramDNASequence, ProgramRNASequence))
            ), "Input must be ProgramDNASequence or ProgramRNASequence object"

            sequence = inputs[0]

            # Edge case.
            if len(sequence) < 2:
                inputs[0]._metadata['dinucleotide_freqs'] = {}
                return 1.0

            # Determine valid nucleotides.
            valid_nucleotides = 'ATCG' if isinstance(sequence, ProgramDNASequence) else 'AUCG'

            # Precompute dinucleotides.
            dinucleotides = [''.join(pair) for pair in itertools.product(valid_nucleotides, repeat=2)]

            # Count dinucleotides.
            dinucleotide_counts = {}
            total_count = 0
            for i in range(len(sequence) - 1):
                dinuc = str(sequence)[i : i + 2]
                if all(nt in valid_nucleotides for nt in dinuc):
                    dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
                    total_count += 1

            # If no valid dinucleotides found.
            if total_count == 0:
                inputs[0]._metadata['dinucleotide_freqs'] = {}
                return 1.0

            # Calculate frequencies and check if they are in range.
            max_deviation = 0.0
            dinucleotide_freqs = {}

            # Score based on deviation from target dinucleotide frequencies.
            for dinuc in dinucleotides:
                freq = dinucleotide_counts.get(dinuc, 0) / total_count
                dinucleotide_freqs[dinuc] = freq

                # Calculate deviation if outside acceptable range.
                if freq < self.min_freq:
                    deviation = (self.min_freq - freq) / self.min_freq
                    max_deviation = max(max_deviation, deviation)
                elif freq > self.max_freq:
                    deviation = (freq - self.max_freq) / (1.0 - self.max_freq)
                    max_deviation = max(max_deviation, deviation)

            inputs[0]._metadata['dinucleotide_freqs'] = dinucleotide_freqs
            return min(1.0, max_deviation)

        super().__init__(inputs, _scoring_function)
    

class TetranucleotideUsageConstraint(ProgramConstraint):
    def __init__(self, 
        inputs: ProgramSequence | List[ProgramSequence], 
        tetranucleotide: str, 
        tud_range: tuple = (0.8, 1.2),
    ) -> None:
        """
        TODO(@dguo8412): Fill in this docstring. What is the `tetranucleotide` variable?
        """
        self.tetranucleotide = tetranucleotide.upper()
        self.min_tud = min(tud_range)
        self.max_tud = max(tud_range)

        # Validate tetranucleotide input.
        if len(self.tetranucleotide) != 4:
            raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")

        def _scoring_function(inputs: List[ProgramSequence]) -> float:
            assert (
                len(inputs) == 1 and
                isinstance(inputs[0], (ProgramDNASequence, ProgramRNASequence))
            ), "Input must be ProgramDNASequence or ProgramRNASequence object"

            sequence = inputs[0]

            # Set appropriate nucleotide keys based on sequence type.
            nucleotide_keys = (
                ['A', 'T', 'C', 'G']
                if isinstance(sequence, ProgramDNASequence) else ['A', 'U', 'C', 'G']
            )

            # Edge case.
            if len(sequence) < 4:
                inputs[0]._metadata[self.tetranucleotide + '_tud'] = 0.0
                return 0.0

            # Calculate nucleotide frequencies.
            nucleotide_freqs = {}
            seq_length = len(sequence)
            for nt in nucleotide_keys:
                nucleotide_freqs[nt] = str(sequence).count(nt) / seq_length

            # Count occurrences of tetranucleotide.
            tetra_count = 0
            for i in range(len(sequence) - 3):
                if str(sequence)[i:i+4] == self.tetranucleotide:
                    tetra_count += 1

            # Calculate expected frequency using zero-order Markov model.
            tetra_expected_freq = 1.0
            for nt in self.tetranucleotide:
                if nt in nucleotide_freqs:
                    tetra_expected_freq *= nucleotide_freqs[nt]
                else:
                    # If invalid nucleotide, set to 0
                    tetra_expected_freq = 0
                    break
        
            # Calculate expected occurrences and TUD.
            expected_occurrences = tetra_expected_freq * (seq_length - 3)
            tetra_tud = tetra_count / expected_occurrences if expected_occurrences > 0 else 0
            inputs[0]._metadata[self.tetranucleotide + '_tud'] = tetra_tud
        
            # Score based on TUD range.
            if self.min_tud <= tetra_tud <= self.max_tud:
                return 0.0
            else:
                # Calculate normalized deviation.
                if tetra_tud < self.min_tud:
                    deviation = (self.min_tud - tetra_tud) / self.min_tud
                else:
                    deviation = (tetra_tud - self.max_tud) / self.max_tud
                return min(1.0, deviation)

        super().__init__(inputs, _scoring_function)
