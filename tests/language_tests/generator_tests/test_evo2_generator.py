import pytest
import sys

sys.path.append(".")
from proto_language.language.base import Segment, SequenceType
from proto_language.language.generator import Evo2Generator

# Check if GPU is available (either locally or via cloud)
from proto_language.utils import is_gpu_available


def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.DNA
) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


@pytest.mark.uses_gpu
@pytest.mark.skipif(
    not is_gpu_available(),
    reason="GPU required for Evo2 tests (local CUDA or cloud access)",
)
class TestEvo2Generator:
    def test_evo2_single_prompt_sampling(self):
        """Test Evo2 generator with a single prompt sequence."""
        prompts = ["ATCG"]
        evo2_generator = Evo2Generator(
            prompt_seqs=prompts, 
            sequence_length=100, 
            batch_size=1
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.DNA)
        evo2_generator.assign(segment)
        
        assert evo2_generator._is_initialized
        assert evo2_generator._generator_output is segment
        assert segment._is_assigned
        assert segment.batch_size == 1

        # Sample and check results
        evo2_generator.sample()
        
        assert segment[0].sequence is not None
        assert len(segment[0].sequence) > len(prompts[0])  # Should be longer than prompt
        assert segment[0].sequence_type == SequenceType.DNA

    def test_evo2_batch_sampling(self):
        """Test Evo2 generator with multiple prompt sequences."""
        prompts = ["ATCG", "AAAA"]
        batch_size = len(prompts)
        evo2_generator = Evo2Generator(
            prompt_seqs=prompts, 
            sequence_length=100, 
            batch_size=batch_size
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.DNA)
        evo2_generator.assign(segment)
        
        assert evo2_generator._is_initialized
        assert evo2_generator._generator_output is segment
        assert segment._is_assigned
        assert segment.batch_size == batch_size

        # Sample and check results
        evo2_generator.sample()
        
        # Check that each individual sequence is not None
        for i in range(batch_size):
            assert segment[i].sequence is not None
            assert len(segment[i].sequence) > len(prompts[i])  # Should be longer than prompt
            assert segment[i].sequence_type == SequenceType.DNA

    def test_evo2_assign_errors(self):
        """Test error conditions for Evo2 generator assignment."""
        prompts = ["ATCG"]
        evo2_generator = Evo2Generator(prompt_seqs=prompts, sequence_length=100)
        
        # Should raise error if assigned multiple segments
        with pytest.raises(ValueError):
            evo2_generator.assign([create_segment(""), create_segment("")])

    def test_evo2_custom_parameters(self):
        """Test Evo2 generator with custom generation parameters."""
        prompts = ["ATCGATCG"]
        evo2_generator = Evo2Generator(
            prompt_seqs=prompts,
            sequence_length=50,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            batch_size=1
        )

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.DNA)
        evo2_generator.assign(segment)
        
        assert evo2_generator.temperature == 0.8
        assert evo2_generator.top_k == 10
        assert evo2_generator.top_p == 0.9

        # Sample and check results
        evo2_generator.sample()
        
        assert segment[0].sequence is not None
        assert segment[0].sequence_type == SequenceType.DNA
