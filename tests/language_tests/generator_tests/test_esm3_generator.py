import pytest
import sys

sys.path.append(".")
from proto_language.language.core import Segment, SequenceType
from proto_language.language.generator import ESM3Generator, ESM3GeneratorConfig

# Check if GPU is available and required dependencies are installed
from proto_language.utils import is_gpu_available


def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.PROTEIN
) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


@pytest.mark.uses_gpu
@pytest.mark.skipif(not is_gpu_available(), reason="GPU required for ESM3 tests")
class TestESM3Generator:
    def test_esm3_entropy_sampling(self):
        """Test ESM3 generator with entropy-based sampling."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                sequence_length=20,
                temperature=1.0,
                decoding_method="entropy",
                top_k=5,
            )
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm3_generator.assign(segment)
        
        assert esm3_generator._is_initialized
        assert esm3_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm3_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

    def test_esm3_max_logit_sampling(self):
        """Test ESM3 generator with max logit sampling."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                sequence_length=20,
                temperature=1.0,
                decoding_method="max_logit",
                top_k=5,
            )
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm3_generator.assign(segment)
        
        assert esm3_generator._is_initialized
        assert esm3_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm3_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

    def test_esm3_random_sampling(self):
        """Test ESM3 generator with random sampling."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                sequence_length=20,
                temperature=1.0,
                decoding_method="random",
                top_k=5,
            )
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm3_generator.assign(segment)
        
        assert esm3_generator._is_initialized
        assert esm3_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        esm3_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) == 20
        assert segment[0].sequence_type == SequenceType.PROTEIN

    def test_esm3_batch_sampling(self):
        """Test ESM3 generator with batch processing."""
        num_candidates = 3
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                sequence_length=15,
                temperature=1.0,
                decoding_method="entropy",
                top_k=5,
                num_candidates=num_candidates,
            )
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.PROTEIN)
        esm3_generator.assign(segment)
        
        assert len(segment.candidate_sequences) == num_candidates
        assert esm3_generator._is_initialized

        # Sample and check results
        esm3_generator.sample()
        
        for i in range(num_candidates):
            assert segment[i].sequence is not None
            assert len(segment[i].sequence) == 15
            assert segment[i].sequence_type == SequenceType.PROTEIN

    def test_esm3_assign_errors(self):
        """Test error conditions for ESM3 generator assignment."""
        esm3_generator = ESM3Generator(
            ESM3GeneratorConfig(
                sequence_length=10,
            )
        )
        
        # Should raise error if assigned multiple segments
        with pytest.raises(ValueError):
            esm3_generator.assign([create_segment(""), create_segment("")])
