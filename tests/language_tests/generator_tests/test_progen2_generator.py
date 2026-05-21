"""tests/language_tests/generator_tests/test_progen2_generator.py."""

import copy

import pytest

from proto_language.core import Segment
from proto_language.generator import ProGen2Generator, ProGen2GeneratorConfig


@pytest.mark.uses_gpu
class TestProGen2Generator:
    def test_progen2_single_prompt_sampling(self):
        """Test ProGen2 generator with a single prompt sequence."""
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK"]
        max_new_tokens = 100
        expected_length = len(prompts[0]) + max_new_tokens
        config = ProGen2GeneratorConfig(
            prompts=prompts,
        )
        progen2_generator = ProGen2Generator(config)

        segment = Segment(length=expected_length, sequence_type="protein")
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segments == (segment,)

        progen2_generator.sample()

        assert segment.proposal_sequences[0].sequence is not None
        assert len(segment.proposal_sequences[0].sequence) > len(prompts[0])
        assert segment.proposal_sequences[0].sequence_type == "protein"

    def test_progen2_batch_sampling(self):
        """Test ProGen2 generator with multiple prompt sequences."""
        prompts = ["1MEVVIVTGMSGAGK", "1EVQLVESGGGLVQP"]
        max_new_tokens = 150
        expected_length = len(prompts[0]) + max_new_tokens
        config = ProGen2GeneratorConfig(
            prompts=prompts,
        )
        progen2_generator = ProGen2Generator(config)

        segment = Segment(length=expected_length, sequence_type="protein")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segments == (segment,)
        assert len(segment.proposal_sequences) == len(prompts)

        progen2_generator.sample()

        for i in range(len(prompts)):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) > len(prompts[i])
            assert segment.proposal_sequences[i].sequence_type == "protein"

    def test_progen2_assign_errors(self):
        """Test error conditions for ProGen2 generator assignment."""
        prompts = ["1MKTL", "1EVQL", "1AAAA"]
        config = ProGen2GeneratorConfig(prompts=prompts)
        progen2_generator = ProGen2Generator(config)

        expected_length = 120
        segment_two_proposals = Segment(length=expected_length, sequence_type="protein")
        segment_two_proposals.proposal_sequences = [
            copy.deepcopy(segment_two_proposals.original_sequence) for _ in range(2)
        ]
        progen2_generator.assign(segment_two_proposals)

        with pytest.raises(ValueError, match="Expected 1 or"):
            progen2_generator.sample()

    def test_progen2_custom_parameters(self):
        """Test ProGen2 generator with custom generation parameters."""
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK"]
        max_new_tokens = 50
        expected_length = len(prompts[0]) + max_new_tokens
        config = ProGen2GeneratorConfig(
            prompts=prompts,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            strip_special_tokens=False,
        )
        progen2_generator = ProGen2Generator(config)

        segment = Segment(length=expected_length, sequence_type="protein")
        progen2_generator.assign(segment)

        assert progen2_generator.temperature == 0.8
        assert progen2_generator.top_k == 10
        assert progen2_generator.top_p == 0.9

        progen2_generator.sample()

        assert segment.proposal_sequences[0].sequence is not None
        assert segment.proposal_sequences[0].sequence_type == "protein"
        assert segment.proposal_sequences[0].sequence.startswith("1")

    def test_progen2_batch_size_parameter(self):
        """Test ProGen2 generator with batch_size for GPU memory management."""
        prompts = ["1MKTL", "1EVQL", "1AAAA", "1GGGG", "1LLLL"]
        config = ProGen2GeneratorConfig(
            prompts=prompts,
            batch_size=2,
        )
        progen2_generator = ProGen2Generator(config)

        segment = Segment(length=100, sequence_type="protein")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        progen2_generator.assign(segment)

        assert progen2_generator.batch_size == 2

        progen2_generator.sample()

        for i in range(len(prompts)):
            assert segment.proposal_sequences[i].sequence is not None
            assert segment.proposal_sequences[i].sequence_type == "protein"


class TestProGen2GeneratorValidation:
    """Test sequence type and config validation for ProGen2 generator."""

    def test_config_rejects_different_length_prompts(self):
        """Prompts with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            ProGen2GeneratorConfig(prompts=["1MKTL", "1MK"])

    def test_valid_protein_assignment(self):
        """ProGen2 should accept PROTEIN segments."""
        config = ProGen2GeneratorConfig(prompts="1MKTL")
        generator = ProGen2Generator(config)
        segment = Segment(length=100, sequence_type="protein")

        generator.assign(segment)
        assert generator._assigned_segments == (segment,)

    def test_prompt_exceeds_segment_length_with_prepend(self):
        """Prompt >= segment length with prepend_prompt=True should raise ValueError."""
        config = ProGen2GeneratorConfig(prompts="1MKTLAAAA", prepend_prompt=True)
        generator = ProGen2Generator(config)
        # Segment length shorter than prompt
        segment = Segment(length=5, sequence_type="protein")
        generator.assign(segment)

        with pytest.raises(ValueError, match=r"Prompt length.*must be less than.*segment length"):
            generator.sample()

    def test_prompt_equal_segment_length_with_prepend(self):
        """Prompt == segment length with prepend_prompt=True should raise ValueError."""
        prompt = "1MKTL"
        config = ProGen2GeneratorConfig(prompts=prompt, prepend_prompt=True)
        generator = ProGen2Generator(config)
        segment = Segment(length=len(prompt), sequence_type="protein")
        generator.assign(segment)

        with pytest.raises(ValueError, match=r"Prompt length.*must be less than.*segment length"):
            generator.sample()

    @pytest.mark.parametrize("seq_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, seq_type):
        """ProGen2 should reject non-protein segments."""
        config = ProGen2GeneratorConfig(prompts="1MKTL")
        generator = ProGen2Generator(config)
        segment = Segment(length=100, sequence_type=seq_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert seq_type in error_msg.lower()
