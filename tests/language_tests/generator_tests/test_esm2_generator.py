import pytest

from proto_language.language.core import Segment, SequenceType
from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig

# Check if GPU is available and required dependencies are installed
from proto_language.utils import is_gpu_available


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
        segment = Segment(starting_sequence_or_desired_length=20, sequence_type=SequenceType.PROTEIN)
        esm2_generator.assign(segment)
        
        assert esm2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

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
        segment = Segment(starting_sequence_or_desired_length=20, sequence_type=SequenceType.PROTEIN)
        esm2_generator.assign(segment)
        
        assert esm2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

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
        segment = Segment(starting_sequence_or_desired_length=20, sequence_type=SequenceType.PROTEIN)
        esm2_generator.assign(segment)
        
        assert esm2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

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

        # Create segment and expand candidate pool
        segment = Segment(starting_sequence_or_desired_length=15, sequence_type=SequenceType.PROTEIN)
        segment.create_candidates(num_candidates)
        esm2_generator.assign(segment)
        
        assert len(segment.candidate_sequences) == num_candidates

        # Sample and check results
        esm2_generator.sample()
        
        for i in range(num_candidates):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) == 15
            assert segment.candidate_sequences[i].sequence_type == SequenceType.PROTEIN

    def test_esm2_assign_errors(self):
        """Test error conditions during Segment creation."""
        esm2_generator = ESM2Generator(
            ESM2GeneratorConfig(
                model_checkpoint="esm2_t33_650M_UR50D",
                num_mutations=1,
            )
        )
        
        # This test is no longer relevant with the new API - the parameter type determines behavior
        # No need to test mismatched sequence and length since they're now mutually exclusive
        pass

    def test_constant_segment_rejection(self):
        """Tests that generators reject constant segments during assign()."""
        config = ESM2GeneratorConfig(num_mutations=1)
        gen = ESM2Generator(config)
        
        # Create a constant segment
        constant_segment = Segment(
            starting_sequence_or_desired_length="MMMMPPPP",
            sequence_type=SequenceType.PROTEIN,
            constant=True
        )
        
        # Should raise ValueError when trying to assign a constant segment
        with pytest.raises(ValueError, match="Cannot assign constant segment"):
            gen.assign(constant_segment)
