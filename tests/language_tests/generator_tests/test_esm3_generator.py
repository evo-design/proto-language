"""tests/language_tests/generator_tests/test_esm3_generator.py."""

import copy

import pytest
from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.language.core import Segment
from proto_language.language.generator import ESM3Generator, ESM3GeneratorConfig


@pytest.mark.uses_gpu
class TestESM3Generator:
    def test_esm3_default_masking(self):
        """Test ESM3 generator with default masking strategy (random 30%)."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
            )
        )

        segment = Segment(length=20, sequence_type="protein")
        esm3_generator.assign(segment)

        assert esm3_generator._assigned_segment is segment

        esm3_generator.sample()

        assert segment.proposal_sequences[0].sequence is not None
        assert len(segment.proposal_sequences[0].sequence) == 20
        assert segment.proposal_sequences[0].sequence_type == "protein"

    def test_esm3_num_mutations_masking(self):
        """Test ESM3 generator with explicit num_mutations masking."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
                masking_strategy=MaskingStrategy(num_mutations=5),
            )
        )

        segment = Segment(length=20, sequence_type="protein")
        esm3_generator.assign(segment)
        esm3_generator.sample()

        assert segment.proposal_sequences[0].sequence is not None
        assert len(segment.proposal_sequences[0].sequence) == 20

    def test_esm3_batch_sampling(self):
        """Test ESM3 generator with batch processing."""
        num_proposals = 3
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
                masking_strategy=MaskingStrategy(num_mutations=5),
            )
        )

        starting_seq = "MKKLLVVGGGGAAAA"
        segment = Segment(sequence=starting_seq, sequence_type="protein")
        esm3_generator.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_proposals)]

        assert len(segment.proposal_sequences) == num_proposals

        esm3_generator.sample()

        for i in range(num_proposals):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) == 15
            assert segment.proposal_sequences[i].sequence_type == "protein"

    def test_esm3_batch_size_parameter(self):
        """Test ESM3 generator with batch_size for GPU memory management."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
                masking_strategy=MaskingStrategy(num_mutations=5),
                batch_size=2,
            )
        )

        starting_seq = "MKKLLVVGGGGAAAA"
        segment = Segment(sequence=starting_seq, sequence_type="protein")
        esm3_generator.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(3)]

        assert esm3_generator.batch_size == 2

        esm3_generator.sample()

        for i in range(3):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) == 15
            assert segment.proposal_sequences[i].sequence_type == "protein"


class TestESM3GeneratorValidation:
    """Test sequence type validation for ESM3 generator."""

    def test_valid_protein_assignment(self):
        """ESM3 should accept PROTEIN segments."""
        config = ESM3GeneratorConfig()
        generator = ESM3Generator(config)
        segment = Segment(length=50, sequence_type="protein")

        generator.assign(segment)
        assert generator._assigned_segment is segment

    @pytest.mark.parametrize("seq_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, seq_type):
        """ESM3 should reject non-protein segments."""
        config = ESM3GeneratorConfig()
        generator = ESM3Generator(config)
        segment = Segment(length=50, sequence_type=seq_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert seq_type in error_msg.lower()
