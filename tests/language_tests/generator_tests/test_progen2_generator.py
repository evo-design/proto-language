import copy

import pytest
from proto_tools.tools.causal_models.progen2.standalone.inference import (
    PROGEN2_START_TOKEN,
)

from proto_language.language.core import Segment
from proto_language.language.generator import ProGen2Generator, ProGen2GeneratorConfig


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

        segment = Segment(length=expected_length, sequence_type="protein")
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segment is segment

        progen2_generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert len(segment.candidate_sequences[0].sequence) > len(prompts[0])
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

        segment = Segment(length=expected_length, sequence_type="protein")
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        progen2_generator.assign(segment)

        assert progen2_generator._assigned_segment is segment
        assert len(segment.candidate_sequences) == len(prompts)

        progen2_generator.sample()

        for i in range(len(prompts)):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) > len(prompts[i])
            assert segment.candidate_sequences[i].sequence_type == "protein"

    def test_progen2_assign_errors(self):
        """Test error conditions for ProGen2 generator assignment."""
        prompts = ["<|pf03668|>1MEVVIVTGMSGAGK", "1EVQLVE", "1MKTL"]
        config = ProGen2GeneratorConfig(prompts=prompts)
        progen2_generator = ProGen2Generator(config)

        expected_length = 120
        segment_two_candidates = Segment(length=expected_length, sequence_type="protein")
        segment_two_candidates.candidate_sequences = [copy.deepcopy(segment_two_candidates.original_sequence) for _ in range(2)]
        progen2_generator.assign(segment_two_candidates)

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

        segment = Segment(length=expected_length, sequence_type="protein")
        progen2_generator.assign(segment)

        assert progen2_generator.temperature == 0.8
        assert progen2_generator.top_k == 10
        assert progen2_generator.top_p == 0.9

        progen2_generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert segment.candidate_sequences[0].sequence_type == "protein"
        assert segment.candidate_sequences[0].sequence.startswith(PROGEN2_START_TOKEN)

    def test_progen2_batch_size_parameter(self):
        """Test ProGen2 generator with batch_size for GPU memory management."""
        prompts = ["1MKTL", "1EVQL", "1AAAA", "1GGGG", "1LLLL"]
        config = ProGen2GeneratorConfig(
            prompts=prompts,
            batch_size=2,
        )
        progen2_generator = ProGen2Generator(config)

        segment = Segment(length=100, sequence_type="protein")
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        progen2_generator.assign(segment)

        assert progen2_generator.batch_size == 2

        progen2_generator.sample()

        for i in range(len(prompts)):
            assert segment.candidate_sequences[i].sequence is not None
            assert segment.candidate_sequences[i].sequence_type == "protein"


class TestProGen2GeneratorValidation:
    """Test sequence type validation for ProGen2 generator."""

    def test_valid_protein_assignment(self):
        """ProGen2 should accept PROTEIN segments."""
        config = ProGen2GeneratorConfig(prompts="1MKTL")
        generator = ProGen2Generator(config)
        segment = Segment(length=100, sequence_type="protein")

        generator.assign(segment)
        assert generator._assigned_segment is segment

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
