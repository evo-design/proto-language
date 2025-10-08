import pytest
import sys

sys.path.append(".")
from proto_language.language.base import Segment, SequenceType
from proto_language.language.generator import ESM2Generator

# Check if GPU is available and required dependencies are installed
from proto_language.utils import is_gpu_available


def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.PROTEIN
) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


@pytest.mark.uses_gpu
@pytest.mark.skipif(not is_gpu_available(), reason="GPU required for ESM2 tests")
class TestESM2Generator:
    def test_esm2_entropy_sampling(self):
        """Test ESM2 generator with entropy-based sampling."""
        esm2_generator = ESM2Generator(
            esm2_type="esm2_t33_650M_UR50D",
            sequence_length=20,
            temperature=1.0,
            decoding_method="entropy",
            top_k=5,
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm2_generator.assign(segment)
        
        assert esm2_generator._is_initialized
        assert esm2_generator._generator_output is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

    def test_esm2_max_logit_sampling(self):
        """Test ESM2 generator with max logit sampling."""
        esm2_generator = ESM2Generator(
            esm2_type="esm2_t33_650M_UR50D",
            sequence_length=20,
            temperature=1.0,
            decoding_method="max_logit",
            top_k=5,
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm2_generator.assign(segment)
        
        assert esm2_generator._is_initialized
        assert esm2_generator._generator_output is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

    def test_esm2_random_sampling(self):
        """Test ESM2 generator with random sampling."""
        esm2_generator = ESM2Generator(
            esm2_type="esm2_t33_650M_UR50D",
            sequence_length=20,
            temperature=1.0,
            decoding_method="random",
            top_k=5,
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm2_generator.assign(segment)
        
        assert esm2_generator._is_initialized
        assert esm2_generator._generator_output is segment
        assert segment._is_assigned

        # Sample and check results
        esm2_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

    def test_esm2_batch_sampling(self):
        """Test ESM2 generator with batch processing."""
        batch_size = 3
        esm2_generator = ESM2Generator(
            esm2_type="esm2_t33_650M_UR50D",
            sequence_length=15,
            temperature=1.0,
            decoding_method="entropy",
            top_k=5,
            batch_size=batch_size,
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm2_generator.assign(segment)
        
        assert segment.batch_size == batch_size
        assert esm2_generator._is_initialized

        # Sample and check results
        esm2_generator.sample()
        
        for i in range(batch_size):
            assert segment[i].sequence is not None
            assert len(segment[i].sequence) == 15
            assert segment[i].sequence_type == SequenceType.PROTEIN

    def test_esm2_assign_errors(self):
        """Test error conditions for ESM2 generator assignment."""
        esm2_generator = ESM2Generator(
            esm2_type="esm2_t33_650M_UR50D",
            sequence_length=10,
        )
        
        # Should raise error if assigned multiple segments
        with pytest.raises(ValueError):
            esm2_generator.assign([create_segment(""), create_segment("")])
