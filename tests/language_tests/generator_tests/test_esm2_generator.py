import copy
import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig



@pytest.mark.uses_gpu
class TestESM2Generator:
    def test_esm2_entropy_sampling(self):
        """Test ESM2 generator with entropy-based sampling."""
        esm2_generator = ESM2Generator(
            ESM2GeneratorConfig(
                model_checkpoint="esm2_t33_650M_UR50D",
                temperature=1.0,
                decoding_method="entropy",
                num_mutations=5,
            )
        )

        # Create segment and assign to generator
        segment = Segment(length=20, sequence_type="protein")
        esm2_generator.assign(segment)

        assert esm2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()

        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == "protein"

    def test_esm2_max_logit_sampling(self):
        """Test ESM2 generator with max logit sampling."""
        esm2_generator = ESM2Generator(
            ESM2GeneratorConfig(
                model_checkpoint="esm2_t33_650M_UR50D",
                temperature=1.0,
                decoding_method="max_logit",
                num_mutations=5,
            )
        )

        # Create segment and assign to generator
        segment = Segment(length=20, sequence_type="protein")
        esm2_generator.assign(segment)

        assert esm2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()

        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == "protein"

    def test_esm2_random_sampling(self):
        """Test ESM2 generator with random sampling."""
        esm2_generator = ESM2Generator(
            ESM2GeneratorConfig(
                model_checkpoint="esm2_t33_650M_UR50D",
                temperature=1.0,
                decoding_method="random",
                num_mutations=5,
            )
        )

        # Create segment and assign to generator
        segment = Segment(length=20, sequence_type="protein")
        esm2_generator.assign(segment)

        assert esm2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()

        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == "protein"

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

        # Create segment with starting sequence for mutation-based sampling
        starting_seq = "MKKLLVVGGGGAAAA"  # 15 amino acids
        segment = Segment(sequence=starting_seq, sequence_type="protein")
        esm2_generator.assign(segment)
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_candidates)]

        assert len(segment.candidate_sequences) == num_candidates

        # Sample and check results
        esm2_generator.sample()

        for i in range(num_candidates):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) == 15
            assert segment.candidate_sequences[i].sequence_type == "protein"

    def test_constant_segment_rejection(self):
        """Tests that generators reject constant segments during assign()."""
        config = ESM2GeneratorConfig(num_mutations=1)
        gen = ESM2Generator(config)

        # Create a constant segment
        constant_segment = Segment(sequence="MMMMPPPP",
            sequence_type="protein",
            constant=True
        )

        # Should raise ValueError when trying to assign a constant segment
        with pytest.raises(ValueError, match="Cannot assign constant segment"):
            gen.assign(constant_segment)


class TestESM2GeneratorValidation:
    """Test sequence type validation for ESM2 generator."""

    def test_valid_protein_assignment(self):
        """ESM2 should accept PROTEIN segments."""
        config = ESM2GeneratorConfig()
        generator = ESM2Generator(config)
        segment = Segment(length=50, sequence_type="protein")
        
        # Should not raise
        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_rejects_dna_segment(self):
        """ESM2 should reject DNA segments."""
        config = ESM2GeneratorConfig()
        generator = ESM2Generator(config)
        segment = Segment(length=50, sequence_type="dna")
        
        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)
        
        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert "dna" in error_msg.lower()
        assert "protein" in error_msg.lower()

    def test_rejects_rna_segment(self):
        """ESM2 should reject RNA segments."""
        config = ESM2GeneratorConfig()
        generator = ESM2Generator(config)
        segment = Segment(length=50, sequence_type="rna")
        
        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)
        
        assert "does not support sequence type" in str(exc_info.value)
        assert "rna" in str(exc_info.value).lower()
