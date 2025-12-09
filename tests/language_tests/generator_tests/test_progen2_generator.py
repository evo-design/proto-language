import copy
import pytest
import time

from proto_language.language.core import Segment, SequenceType
from proto_language.language.generator import (
    ProGen2Generator,
    ProGen2GeneratorConfig,
)
from proto_language.language.generator.progen2_generator import PROGEN2_START_TOKEN

# Check if GPU is available (either locally or via cloud)
from proto_language.utils import is_gpu_available


@pytest.mark.uses_gpu
class TestProGen2Generator:
    def test_progen2_single_prompt_sampling(self):
        """Test ProGen2 generator with a single prompt sequence."""
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = ProGen2GeneratorConfig(
            prompts=prompts,
        )
        progen2_generator = ProGen2Generator(config)

        # Create segment and assign to generator
        segment = Segment(length=expected_length, sequence_type=SequenceType.PROTEIN)
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segment is segment
        assert segment._is_assigned

        # Sample and check results
        progen2_generator.sample()

        assert segment[0].sequence is not None
        assert len(segment[0].sequence) > len(prompts[0])  # Should be longer than prompt
        assert segment[0].sequence_type == SequenceType.PROTEIN

    def test_progen2_batch_sampling(self):
        """Test ProGen2 generator with multiple prompt sequences."""
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK", "1EVQLVE"]
        num_tokens = 150
        expected_length = len(prompts[0]) + num_tokens
        config = ProGen2GeneratorConfig(
            prompts=prompts,
        )
        progen2_generator = ProGen2Generator(config)

        # Create segment and expand candidate pool
        segment = Segment(length=expected_length, sequence_type=SequenceType.PROTEIN)
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segment is segment
        assert segment._is_assigned
        assert len(segment.candidate_sequences) == len(prompts)

        # Sample and check results
        progen2_generator.sample()

        # Check that each individual sequence is not None
        for i in range(len(prompts)):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) > len(prompts[i])  # Should be longer than prompt
            assert segment.candidate_sequences[i].sequence_type == SequenceType.PROTEIN
            print("Generated:", segment.candidate_sequences[i].sequence)

    def test_progen2_assign_errors(self):
        """Test error conditions for ProGen2 generator assignment."""
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = ProGen2GeneratorConfig(prompts=prompts)
        progen2_generator = ProGen2Generator(config)

        # Should raise error if number of prompts doesn't match segment candidates
        segment_two_candidates = Segment(length=expected_length, sequence_type=SequenceType.PROTEIN)
        segment_two_candidates.candidate_sequences = [copy.deepcopy(segment_two_candidates.original_sequence) for _ in range(2)]
        progen2_generator.assign(segment_two_candidates)
        
        with pytest.warns(UserWarning, match="Number of prompts"):
            progen2_generator.sample()  # Will warn because 1 prompt but 2 candidates

    def test_progen2_custom_parameters(self):
        """Test ProGen2 generator with custom generation parameters."""
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK"]
        num_tokens = 50
        expected_length = len(prompts[0]) + num_tokens
        config = ProGen2GeneratorConfig(
            prompts=prompts,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            strip_special_tokens=False,
        )
        progen2_generator = ProGen2Generator(config)

        # Create segment and assign to generator
        segment = Segment(length=expected_length, sequence_type=SequenceType.PROTEIN)
        progen2_generator.assign(segment)

        assert progen2_generator.temperature == 0.8
        assert progen2_generator.top_k == 10
        assert progen2_generator.top_p == 0.9

        # Sample and check results
        progen2_generator.sample()

        assert segment[0].sequence is not None
        assert segment[0].sequence_type == SequenceType.PROTEIN
        assert segment[0].sequence.startswith(PROGEN2_START_TOKEN)

    def test_constant_segment_rejection(self):
        """Tests that generators reject constant segments during assign()."""
        config = ProGen2GeneratorConfig(prompts=["<|pf03668|>1MEVVIVTGMSGAGK"])
        gen = ProGen2Generator(config)
        
        # Create a constant segment
        constant_segment = Segment(
            sequence="EVQLVE",
            sequence_type=SequenceType.PROTEIN,
            constant=True
        )
        
        # Should raise ValueError when trying to assign a constant segment
        with pytest.raises(ValueError, match="Cannot assign constant segment"):
            gen.assign(constant_segment)
