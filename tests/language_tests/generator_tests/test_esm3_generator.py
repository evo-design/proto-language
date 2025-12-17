import copy
import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import ESM3Generator, ESM3GeneratorConfig


@pytest.mark.uses_gpu
class TestESM3Generator:
    def test_esm3_entropy_sampling(self):
        """Test ESM3 generator with entropy-based sampling."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
                decoding_method="entropy",
                num_mutations=5,
            )
        )

        # Create segment and assign to generator
        segment = Segment(length=20, sequence_type="protein")
        esm3_generator.assign(segment)
        
        assert esm3_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm3_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == "protein"

    def test_esm3_max_logit_sampling(self):
        """Test ESM3 generator with max logit sampling."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
                decoding_method="max_logit",
                num_mutations=5,
            )
        )

        # Create segment and assign to generator
        segment = Segment(length=20, sequence_type="protein")
        esm3_generator.assign(segment)
        
        assert esm3_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm3_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == "protein"

    def test_esm3_random_sampling(self):
        """Test ESM3 generator with random sampling."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
                decoding_method="random",
                num_mutations=5,
            )
        )

        # Create segment and assign to generator
        segment = Segment(length=20, sequence_type="protein")
        esm3_generator.assign(segment)
        
        assert esm3_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm3_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == "protein"

    def test_esm3_batch_sampling(self):
        """Test ESM3 generator with batch processing."""
        num_candidates = 3
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                temperature=1.0,
                decoding_method="entropy",
                num_mutations=5,
            )
        )

        # Create segment with starting sequence for mutation-based sampling
        starting_seq = "MKKLLVVGGGGAAAA"  # 15 amino acids
        segment = Segment(sequence=starting_seq, sequence_type="protein")
        esm3_generator.assign(segment)
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_candidates)]

        assert len(segment.candidate_sequences) == num_candidates

        # Sample and check results
        esm3_generator.sample()
        
        for i in range(num_candidates):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) == 15
            assert segment.candidate_sequences[i].sequence_type == "protein"

    def test_constant_segment_rejection(self):
        """Tests that generators reject constant segments during assign()."""
        config = ESM3GeneratorConfig()
        gen = ESM3Generator(config)
        
        # Create a constant segment
        constant_segment = Segment(sequence="MMMMPPPP",
            sequence_type="protein",
            constant=True
        )
        
        # Should raise ValueError when trying to assign a constant segment
        with pytest.raises(ValueError, match="Cannot assign constant segment"):
            gen.assign(constant_segment)


class TestESM3GeneratorValidation:
    """Test sequence type validation for ESM3 generator."""

    def test_valid_protein_assignment(self):
        """ESM3 should accept PROTEIN segments."""
        config = ESM3GeneratorConfig()
        generator = ESM3Generator(config)
        segment = Segment(length=50, sequence_type="protein")
        
        # Should not raise
        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_rejects_dna_segment(self):
        """ESM3 should reject DNA segments."""
        config = ESM3GeneratorConfig()
        generator = ESM3Generator(config)
        segment = Segment(length=50, sequence_type="dna")
        
        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)
        
        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert "dna" in error_msg.lower()

    def test_rejects_rna_segment(self):
        """ESM3 should reject RNA segments."""
        config = ESM3GeneratorConfig()
        generator = ESM3Generator(config)
        segment = Segment(length=50, sequence_type="rna")
        
        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)
        
        assert "does not support sequence type" in str(exc_info.value)
        assert "rna" in str(exc_info.value).lower()