import pytest
import sys

sys.path.append(".")
from language.base import ConstructSegment, SequenceType
from language.generator import ESM2Generator

# Check if required dependencies are available
try:
    import torch
    import biotite.structure
    TORCH_AVAILABLE = True
    BIOTITE_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    BIOTITE_AVAILABLE = False


def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.PROTEIN
) -> ConstructSegment:
    """Helper to create a ConstructSegment with a single sequence."""
    return ConstructSegment(sequence=sequence, sequence_type=seq_type)


@pytest.mark.skipif(
    not TORCH_AVAILABLE or not BIOTITE_AVAILABLE, 
    reason="PyTorch and biotite required for ESM2 tests"
)
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
        
        assert len(segment) == batch_size
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
