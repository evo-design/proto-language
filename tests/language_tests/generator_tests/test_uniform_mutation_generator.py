from __future__ import annotations
import pytest
import random
import numpy as np
import copy
from typing import Tuple

from proto_language.language.core import (
    Segment,
    SequenceType,
)
from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig


class TestUniformMutationGenerator:
    def test_initialization(self):
        """Tests the __init__ method for correct initialization."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        assert gen.num_mutations == 1

    def test_assign_and_initialization(self):
        """Tests the assign method initializes the output segment correctly."""
        seq_len = 20
        # Test assign with an empty segment (should initialize randomly)
        segment = Segment(length=seq_len, sequence_type="rna")
        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        gen.assign(segment)

        assert gen._assigned_segment is segment
        assert segment._is_assigned
        assert segment.num_selected == 1  # assign() initializes one selected sequence
        assert len(segment.selected_sequences[0].sequence) == seq_len
        assert all(c in "ACGU" for c in segment.selected_sequences[0].sequence)

        # Test assign with a pre-defined sequence
        predefined_seq = "A" * seq_len
        segment_pre = Segment(sequence=predefined_seq, sequence_type="rna")
        gen.assign(segment_pre)
        assert segment_pre.selected_sequences[0].sequence == predefined_seq


    def test_sample_mutates_sequence(self):
        """Tests the sample method introduces a single valid mutation."""
        seq_len = 25
        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        segment = Segment(sequence="A" * seq_len, sequence_type="protein")
        gen.assign(segment)

        # Create candidates before sampling (sample() mutates candidate_sequences)
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_sequence = segment.candidate_sequences[0].sequence
        gen.sample()
        mutated_sequence = segment.candidate_sequences[0].sequence

        assert len(mutated_sequence) == seq_len
        # Check that exactly one position has changed
        diff_count = sum(
            1 for a, b in zip(initial_sequence, mutated_sequence) if a != b
        )
        assert diff_count == 1
        diff_indices = [
            i
            for i, (a, b) in enumerate(zip(initial_sequence, mutated_sequence))
            if a != b
        ]
        mutated_char = mutated_sequence[diff_indices[0]]

        assert mutated_char in segment._valid_chars
        assert mutated_char != initial_sequence[diff_indices[0]]

    def test_sample_batch(self):
        """Tests that sample mutates all sequences in a batch of candidates independently."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        segment = Segment(sequence="A" * 30, sequence_type="dna")
        gen.assign(segment)

        # Create multiple candidates
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        initial_sequences = [s.sequence for s in segment.candidate_sequences]
        gen.sample()
        mutated_sequences = [s.sequence for s in segment.candidate_sequences]

        for i in range(len(initial_sequences)):
            assert initial_sequences[i] != mutated_sequences[i]
            diff_count = sum(
                1 for a, b in zip(initial_sequences[i], mutated_sequences[i]) if a != b
            )
            assert diff_count == 1
        # Check that mutations are likely different across the batch
        assert len(set(mutated_sequences)) > 1

    def test_deterministic_behavior(self):
        """Tests that with a fixed seed, the behavior is reproducible."""

        def run_with_seed(seed):
            random.seed(seed)
            config = UniformMutationGeneratorConfig(num_mutations=1)
            gen = UniformMutationGenerator(config)
            segment = Segment(length=50, sequence_type="dna")
            gen.assign(segment)
            initial_seq = segment.selected_sequences[0].sequence
            # Create one candidate and mutate it multiple times
            segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
            for _ in range(10):
                gen.sample()
                # Copy mutated candidate back to selected for next iteration
                segment.selected_sequences[0].sequence = segment.candidate_sequences[0].sequence
                segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
            final_seq = segment.candidate_sequences[0].sequence
            return initial_seq, final_seq

        init1, final1 = run_with_seed(42)
        init2, final2 = run_with_seed(42)
        init3, final3 = run_with_seed(123)

        assert init1 == init2
        assert final1 == final2
        assert init1 != init3
        assert final1 != final3

    def test_sample_len_one_sequence(self):
        """Tests that a sequence of length 1 is mutated correctly."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        segment = Segment(sequence="A", sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_char = segment.candidate_sequences[0].sequence
        gen.sample()
        mutated_char = segment.candidate_sequences[0].sequence

        assert len(mutated_char) == 1
        assert mutated_char in "CGT"
        assert mutated_char != initial_char

    def test_num_mutations_parameter(self):
        """Tests that specifying num_mutations produces exactly that many changes."""
        seq_len = 30
        num_mut = 5
        config = UniformMutationGeneratorConfig(num_mutations=num_mut)
        gen = UniformMutationGenerator(config)
        segment = Segment(sequence="A" * seq_len, sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_sequence = segment.candidate_sequences[0].sequence
        gen.sample()
        mutated_sequence = segment.candidate_sequences[0].sequence

        diff_count = sum(
            1 for a, b in zip(initial_sequence, mutated_sequence) if a != b
        )
        assert diff_count == num_mut

    def test_num_mutations_capped_by_sequence_length(self):
        """Tests that num_mutations larger than length is capped to sequence length."""
        seq_len = 3
        num_mut = 10
        config = UniformMutationGeneratorConfig(num_mutations=num_mut)
        gen = UniformMutationGenerator(config)
        segment = Segment(sequence="A" * seq_len, sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_sequence = segment.candidate_sequences[0].sequence
        gen.sample()
        mutated_sequence = segment.candidate_sequences[0].sequence

        diff_count = sum(
            1 for a, b in zip(initial_sequence, mutated_sequence) if a != b
        )
        assert diff_count == seq_len

    def test_constant_segment_rejection(self):
        """Tests that generators reject constant segments during assign()."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        
        # Create a constant segment
        constant_segment = Segment(sequence="ATCGATCGAT",
            sequence_type="dna",
            label="promoter",
            constant=True
        )
        
        # Should raise ValueError when trying to assign a constant segment
        with pytest.raises(ValueError, match="Cannot assign constant segment"):
            gen.assign(constant_segment)


class TestUniformMutationGeneratorValidation:
    """Test sequence type validation for UniformMutation generator."""

    def test_accepts_dna_segment(self):
        """UniformMutation should accept DNA segments."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(config)
        segment = Segment(length=50, sequence_type="dna")
        
        # Should not raise - universal generator supports all types
        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_accepts_rna_segment(self):
        """UniformMutation should accept RNA segments."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(config)
        segment = Segment(length=50, sequence_type="rna")
        
        # Should not raise
        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_accepts_protein_segment(self):
        """UniformMutation should accept PROTEIN segments."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(config)
        segment = Segment(length=50, sequence_type="protein")
        
        # Should not raise
        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_accepts_ligand_segment(self):
        """UniformMutation should accept LIGAND segments."""
        config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(config)
        segment = Segment(length=50, sequence_type="ligand")
        
        # Should not raise
        generator.assign(segment)
        assert generator._assigned_segment is segment
