"""Tests for the RandomNucleotideGenerator."""

import copy

import pytest
from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.language.core import Segment
from proto_language.language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)


class TestRandomNucleotideGenerator:
    def test_initialization(self):
        """Tests the __init__ method for correct initialization."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        assert gen.masking_strategy.num_mutations == 1

    def test_initialization_default(self):
        """Tests default config initialization (random 30%)."""
        config = RandomNucleotideGeneratorConfig()
        gen = RandomNucleotideGenerator(config)
        assert gen.masking_strategy.num_mutations is None
        assert gen.masking_strategy.mask_fraction is None

    def test_assign_and_initialization(self):
        """Tests the assign method initializes the output segment correctly."""
        seq_len = 20
        segment = Segment(length=seq_len, sequence_type="rna")
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        gen.assign(segment)

        assert gen._assigned_segment is segment
        assert segment.num_results == 1
        assert len(segment.proposal_sequences[0].sequence) == 0
        assert all(c in "ACGU" for c in segment.proposal_sequences[0].sequence)

        gen.sample()
        assert len(segment.proposal_sequences[0].sequence) == seq_len
        assert all(c in "ACGU" for c in segment.proposal_sequences[0].sequence)

        # Test assign with a pre-defined sequence
        predefined_seq = "A" * seq_len
        segment_pre = Segment(sequence=predefined_seq, sequence_type="rna")
        gen.assign(segment_pre)
        assert segment_pre.result_sequences[0].sequence == predefined_seq

    def test_sample_mutates_sequence(self):
        """Tests the sample method masks and fills one position."""
        seq_len = 25
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        gen._set_program_seed(42)
        segment = Segment(sequence="A" * seq_len, sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        gen.sample()
        mutated_sequence = segment.proposal_sequences[0].sequence

        assert len(mutated_sequence) == seq_len
        # All characters should be valid DNA
        assert all(c in "ACGT" for c in mutated_sequence)

    def test_sample_batch(self):
        """Tests that sample mutates all sequences in a batch of proposals independently."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=5),
        )
        gen = RandomNucleotideGenerator(config)
        segment = Segment(sequence="A" * 30, sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        gen.sample()
        mutated_sequences = [s.sequence for s in segment.proposal_sequences]

        # Each sequence should have the right length
        for seq in mutated_sequences:
            assert len(seq) == 30
        # With 5 mutations per sequence, batch should produce diverse results
        assert len(set(mutated_sequences)) > 1

    def test_sample_len_one_sequence(self):
        """Tests that a sequence of length 1 is mutated correctly."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomNucleotideGenerator(config)
        segment = Segment(sequence="A", sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        gen.sample()
        mutated_char = segment.proposal_sequences[0].sequence

        assert len(mutated_char) == 1
        assert mutated_char in "ACGT"

    def test_num_mutations_parameter(self):
        """Tests that specifying num_mutations masks exactly that many positions."""
        seq_len = 30
        num_mut = 5
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=num_mut),
        )
        gen = RandomNucleotideGenerator(config)
        segment = Segment(sequence="A" * seq_len, sequence_type="dna")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_sequence = segment.proposal_sequences[0].sequence
        gen.sample()
        mutated_sequence = segment.proposal_sequences[0].sequence

        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence, strict=False) if a != b)
        # At most num_mut positions differ (some random replacements may match original)
        assert diff_count <= num_mut
        assert len(mutated_sequence) == seq_len

    def test_substitution_scheme_parameter(self):
        """Tests that substitution_scheme parameter is passed to tool config."""
        config = RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
            substitution_scheme="R",
        )
        gen = RandomNucleotideGenerator(config)
        assert gen.substitution_scheme == "R"

    def test_seed_reproducibility(self):
        """Tests that _set_program_seed produces reproducible results with pre-masked input."""
        # Use pre-masked sequences to bypass position selection randomness
        seq = "ACG_ACG_ACG_ACGT"
        config = RandomNucleotideGeneratorConfig()

        results = []
        for _ in range(2):
            gen = RandomNucleotideGenerator(config)
            gen._set_program_seed(42)
            segment = Segment(sequence=seq, sequence_type="dna")
            gen.assign(segment)
            segment.proposal_sequences = [copy.deepcopy(segment.original_sequence)]
            gen.sample()
            results.append(segment.proposal_sequences[0].sequence)

        assert results[0] == results[1]
        # Verify mask tokens were replaced
        assert "_" not in results[0]


class TestRandomNucleotideGeneratorValidation:
    """Test sequence type validation for RandomNucleotide generator."""

    def test_accepts_dna_segment(self):
        """RandomNucleotide should accept DNA segments."""
        config = RandomNucleotideGeneratorConfig()
        generator = RandomNucleotideGenerator(config)
        segment = Segment(length=50, sequence_type="dna")

        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_accepts_rna_segment(self):
        """RandomNucleotide should accept RNA segments."""
        config = RandomNucleotideGeneratorConfig()
        generator = RandomNucleotideGenerator(config)
        segment = Segment(length=50, sequence_type="rna")

        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_rejects_protein_segment(self):
        """RandomNucleotide should reject protein segments."""
        config = RandomNucleotideGeneratorConfig()
        generator = RandomNucleotideGenerator(config)
        segment = Segment(length=50, sequence_type="protein")

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
