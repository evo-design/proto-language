import copy

import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig


@pytest.mark.uses_gpu
class TestESM2Generator:
    @pytest.mark.parametrize("decoding_method", ["entropy", "max_logit", "random"])
    def test_esm2_decoding_methods(self, decoding_method):
        """Test ESM2 generator with each decoding method."""
        esm2_generator = ESM2Generator(
            ESM2GeneratorConfig(
                model_checkpoint="esm2_t33_650M_UR50D",
                temperature=1.0,
                decoding_method=decoding_method,
                num_mutations=5,
            )
        )

        segment = Segment(length=20, sequence_type="protein")
        esm2_generator.assign(segment)

        assert esm2_generator._assigned_segment is segment

        esm2_generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert len(segment.candidate_sequences[0].sequence) == 20
        assert segment.candidate_sequences[0].sequence_type == "protein"

    def test_esm2_batch_sampling(self):
        """Test ESM2 generator with batch processing."""
        num_candidates = 3
        esm2_generator = ESM2Generator(
            ESM2GeneratorConfig(
                model_checkpoint="esm2_t33_650M_UR50D",
                temperature=1.0,
                decoding_method="entropy",
                num_mutations=5,
            )
        )

        starting_seq = "MKKLLVVGGGGAAAA"
        segment = Segment(sequence=starting_seq, sequence_type="protein")
        esm2_generator.assign(segment)
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_candidates)]

        assert len(segment.candidate_sequences) == num_candidates

        esm2_generator.sample()

        for i in range(num_candidates):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) == 15
            assert segment.candidate_sequences[i].sequence_type == "protein"

    def test_esm2_batch_size_parameter(self):
        """Test ESM2 generator with batch_size for GPU memory management."""
        esm2_generator = ESM2Generator(
            ESM2GeneratorConfig(
                model_checkpoint="esm2_t33_650M_UR50D",
                temperature=1.0,
                decoding_method="entropy",
                num_mutations=5,
                batch_size=2,
            )
        )

        starting_seq = "MKKLLVVGGGGAAAA"
        segment = Segment(sequence=starting_seq, sequence_type="protein")
        esm2_generator.assign(segment)
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(3)]

        assert esm2_generator.batch_size == 2

        esm2_generator.sample()

        for i in range(3):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) == 15
            assert segment.candidate_sequences[i].sequence_type == "protein"


class TestESM2GeneratorValidation:
    """Test sequence type validation for ESM2 generator."""

    def test_valid_protein_assignment(self):
        """ESM2 should accept PROTEIN segments."""
        config = ESM2GeneratorConfig()
        generator = ESM2Generator(config)
        segment = Segment(length=50, sequence_type="protein")

        generator.assign(segment)
        assert generator._assigned_segment is segment

    @pytest.mark.parametrize("seq_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, seq_type):
        """ESM2 should reject non-protein segments."""
        config = ESM2GeneratorConfig()
        generator = ESM2Generator(config)
        segment = Segment(length=50, sequence_type=seq_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert seq_type in error_msg.lower()
