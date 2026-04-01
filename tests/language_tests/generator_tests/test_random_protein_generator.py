"""Tests for the RandomProteinGenerator."""

import copy

import pytest
from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.language.core import Segment
from proto_language.language.generator import (
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
)


class TestRandomProteinGenerator:
    def test_initialization(self):
        """Tests the __init__ method for correct initialization."""
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomProteinGenerator(config)
        assert gen.masking_strategy.num_mutations == 1

    def test_initialization_default(self):
        """Tests default config initialization (random 30%)."""
        config = RandomProteinGeneratorConfig()
        gen = RandomProteinGenerator(config)
        assert gen.masking_strategy.num_mutations is None
        assert gen.masking_strategy.mask_fraction is None

    def test_sample_mutates_sequence(self):
        """Tests the sample method introduces mutations."""
        seq_len = 25
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
        )
        gen = RandomProteinGenerator(config)
        segment = Segment(sequence="A" * seq_len, sequence_type="protein")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(1)]
        initial_sequence = segment.proposal_sequences[0].sequence
        gen.sample()
        mutated_sequence = segment.proposal_sequences[0].sequence

        assert len(mutated_sequence) == seq_len
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence, strict=False) if a != b)
        assert diff_count == 1

    def test_sample_batch(self):
        """Tests that sample mutates all sequences in a batch."""
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=5),
        )
        gen = RandomProteinGenerator(config)
        segment = Segment(sequence="A" * 30, sequence_type="protein")
        gen.assign(segment)

        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        gen.sample()
        mutated_sequences = [s.sequence for s in segment.proposal_sequences]

        # Each sequence should have the right length
        for seq in mutated_sequences:
            assert len(seq) == 30
        # With 5 mutations per sequence, batch should produce diverse results
        assert len(set(mutated_sequences)) > 1

    def test_codon_scheme_parameter(self):
        """Tests that codon_scheme parameter is passed to tool config."""
        config = RandomProteinGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=1),
            codon_scheme="NNK",
        )
        gen = RandomProteinGenerator(config)
        assert gen.codon_scheme == "NNK"

    def test_seed_reproducibility(self):
        """Tests that seed produces reproducible results with pre-masked input."""
        # Use pre-masked sequences to bypass position selection randomness
        seq = "MKK_LVV_GGG_AAA"
        config = RandomProteinGeneratorConfig(
            seed=42,
        )

        results = []
        for _ in range(2):
            gen = RandomProteinGenerator(config)
            segment = Segment(sequence=seq, sequence_type="protein")
            gen.assign(segment)
            segment.proposal_sequences = [copy.deepcopy(segment.original_sequence)]
            gen.sample()
            results.append(segment.proposal_sequences[0].sequence)

        assert results[0] == results[1]
        # Verify mask tokens were replaced
        assert "_" not in results[0]


class TestRandomProteinGeneratorValidation:
    """Test sequence type validation for RandomProtein generator."""

    def test_accepts_protein_segment(self):
        """RandomProtein should accept protein segments."""
        config = RandomProteinGeneratorConfig()
        generator = RandomProteinGenerator(config)
        segment = Segment(length=50, sequence_type="protein")

        generator.assign(segment)
        assert generator._assigned_segment is segment

    @pytest.mark.parametrize("seq_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, seq_type):
        """RandomProtein should reject non-protein segments."""
        config = RandomProteinGeneratorConfig()
        generator = RandomProteinGenerator(config)
        segment = Segment(length=50, sequence_type=seq_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
