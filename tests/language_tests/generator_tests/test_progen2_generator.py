import copy
import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import (
    ProGen2Generator,
    ProGen2GeneratorConfig,
)
from proto_language.tools.language_models.progen2.standalone.inference import PROGEN2_START_TOKEN

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
        segment = Segment(length=expected_length, sequence_type="protein")
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segment is segment

        # Sample and check results
        progen2_generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert len(segment.candidate_sequences[0].sequence) > len(prompts[0])  # Should be longer than prompt
        assert segment.candidate_sequences[0].sequence_type == "protein"

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
        segment = Segment(length=expected_length, sequence_type="protein")
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segment is segment
        assert len(segment.candidate_sequences) == len(prompts)

        # Sample and check results
        progen2_generator.sample()

        # Check that each individual sequence is not None
        for i in range(len(prompts)):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) > len(prompts[i])  # Should be longer than prompt
            assert segment.candidate_sequences[i].sequence_type == "protein"
            print("Generated:", segment.candidate_sequences[i].sequence)

    def test_progen2_assign_errors(self):
        """Test error conditions for ProGen2 generator assignment."""
        # Multiple prompts but wrong count - should raise error
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK", "1EVQLVE", "1MKTL"]  # 3 prompts
        config = ProGen2GeneratorConfig(prompts=prompts)
        progen2_generator = ProGen2Generator(config)

        # Create segment with 2 candidates
        expected_length = 120
        segment_two_candidates = Segment(length=expected_length, sequence_type="protein")
        segment_two_candidates.candidate_sequences = [copy.deepcopy(segment_two_candidates.original_sequence) for _ in range(2)]
        progen2_generator.assign(segment_two_candidates)
        
        # 3 prompts but 2 candidates - should raise ValueError
        with pytest.raises(ValueError, match="must either be 1"):
            progen2_generator.sample()

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
        segment = Segment(length=expected_length, sequence_type="protein")
        progen2_generator.assign(segment)

        assert progen2_generator.temperature == 0.8
        assert progen2_generator.top_k == 10
        assert progen2_generator.top_p == 0.9

        # Sample and check results
        progen2_generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert segment.candidate_sequences[0].sequence_type == "protein"
        assert segment.candidate_sequences[0].sequence.startswith(PROGEN2_START_TOKEN)

class TestProGen2GeneratorValidation:
    """Test sequence type validation for ProGen2 generator."""

    def test_valid_protein_assignment(self):
        """ProGen2 should accept PROTEIN segments."""
        config = ProGen2GeneratorConfig(prompts="1MKTL")
        generator = ProGen2Generator(config)
        segment = Segment(length=100, sequence_type="protein")
        
        # Should not raise
        generator.assign(segment)
        assert generator._assigned_segment is segment

    def test_rejects_dna_segment(self):
        """ProGen2 should reject DNA segments."""
        config = ProGen2GeneratorConfig(prompts="1MKTL")
        generator = ProGen2Generator(config)
        segment = Segment(length=100, sequence_type="dna")
        
        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)
        
        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert "dna" in error_msg.lower()
        assert "protein" in error_msg.lower()

    def test_rejects_rna_segment(self):
        """ProGen2 should reject RNA segments."""
        config = ProGen2GeneratorConfig(prompts="1MKTL")
        generator = ProGen2Generator(config)
        segment = Segment(length=100, sequence_type="rna")
        
        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)
        
        assert "does not support sequence type" in str(exc_info.value)
        assert "rna" in str(exc_info.value).lower()
