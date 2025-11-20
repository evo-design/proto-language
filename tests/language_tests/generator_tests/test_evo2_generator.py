import pytest
import time

from proto_language.language.core import Segment, SequenceType
from proto_language.language.generator import (
    Evo2Generator,
    Evo2GeneratorConfig,
)

# Check if GPU is available (either locally or via cloud)
from proto_language.utils import is_gpu_available


@pytest.mark.uses_gpu
class TestEvo2Generator:
    def test_evo2_single_prompt_sampling(self):
        """Test Evo2 generator with a single prompt sequence."""
        prompts = ["ATCG"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = Evo2GeneratorConfig(
            prompts=prompts,
        )
        evo2_generator = Evo2Generator(config)

        # Create segment and assign to generator
        segment = Segment(starting_sequence_or_desired_length=expected_length, sequence_type=SequenceType.DNA)
        evo2_generator.assign(segment)

        assert evo2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        evo2_generator.sample()

        assert segment[0].sequence is not None
        assert len(segment[0].sequence) > len(prompts[0])  # Should be longer than prompt
        assert segment[0].sequence_type == SequenceType.DNA

    def test_evo2_batch_sampling(self):
        """Test Evo2 generator with multiple prompt sequences."""
        prompts = ["ATCG", "AAAA"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = Evo2GeneratorConfig(
            prompts=prompts,
        )
        evo2_generator = Evo2Generator(config)

        # Create segment and expand candidate pool
        segment = Segment(starting_sequence_or_desired_length=expected_length, sequence_type=SequenceType.DNA)
        segment.create_candidates(len(prompts))
        evo2_generator.assign(segment)

        assert evo2_generator._assigned_segment is segment
        assert segment._is_assigned
        assert len(segment.candidate_sequences) == len(prompts)

        # Sample and check results
        evo2_generator.sample()

        # Check that each individual sequence is not None
        for i in range(len(prompts)):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) > len(prompts[i])  # Should be longer than prompt
            assert segment.candidate_sequences[i].sequence_type == SequenceType.DNA

    def test_evo2_assign_errors(self):
        """Test error conditions for Evo2 generator assignment."""
        prompts = ["ATCG"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = Evo2GeneratorConfig(prompts=prompts)
        evo2_generator = Evo2Generator(config)

        # Should raise error if number of prompts doesn't match segment candidates
        segment_two_candidates = Segment(starting_sequence_or_desired_length=expected_length, sequence_type=SequenceType.DNA)
        segment_two_candidates.create_candidates(2)
        evo2_generator.assign(segment_two_candidates)
        
        with pytest.warns(UserWarning, match="Number of prompts"):
            evo2_generator.sample()  # Will warn because 1 prompt but 2 candidates

    def test_evo2_custom_parameters(self):
        """Test Evo2 generator with custom generation parameters."""
        prompts = ["ATCGATCG"]
        num_tokens = 50
        expected_length = len(prompts[0]) + num_tokens
        config = Evo2GeneratorConfig(
            prompts=prompts,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
        )
        evo2_generator = Evo2Generator(config)

        # Create segment and assign to generator
        segment = Segment(starting_sequence_or_desired_length=expected_length, sequence_type=SequenceType.DNA)
        evo2_generator.assign(segment)

        assert evo2_generator.temperature == 0.8
        assert evo2_generator.top_k == 10
        assert evo2_generator.top_p == 0.9

        # Sample and check results
        evo2_generator.sample()

        assert segment[0].sequence is not None
        assert segment[0].sequence_type == SequenceType.DNA

    def test_constant_segment_rejection(self):
        """Tests that generators reject constant segments during assign()."""
        config = Evo2GeneratorConfig(prompts=["ATCG"])
        gen = Evo2Generator(config)
        
        # Create a constant segment
        constant_segment = Segment(
            starting_sequence_or_desired_length="ATCGATCGAT",
            sequence_type=SequenceType.DNA,
            constant=True
        )
        
        # Should raise ValueError when trying to assign a constant segment
        with pytest.raises(ValueError, match="Cannot assign constant segment"):
            gen.assign(constant_segment)