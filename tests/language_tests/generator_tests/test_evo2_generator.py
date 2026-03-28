"""tests/language_tests/generator_tests/test_evo2_generator.py"""
import copy
from unittest.mock import MagicMock, patch

import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig


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

        segment = Segment(length=expected_length, sequence_type="dna")
        evo2_generator.assign(segment)

        assert evo2_generator._assigned_segment is segment

        evo2_generator.sample()

        assert segment.proposal_sequences[0].sequence is not None
        assert len(segment.proposal_sequences[0].sequence) > len(prompts[0])
        assert segment.proposal_sequences[0].sequence_type == "dna"

    def test_evo2_batch_sampling(self):
        """Test Evo2 generator with multiple prompt sequences."""
        prompts = ["ATCG", "AAAA"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = Evo2GeneratorConfig(
            prompts=prompts,
        )
        evo2_generator = Evo2Generator(config)

        segment = Segment(length=expected_length, sequence_type="dna")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        evo2_generator.assign(segment)

        assert evo2_generator._assigned_segment is segment
        assert len(segment.proposal_sequences) == len(prompts)

        evo2_generator.sample()

        for i in range(len(prompts)):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) > len(prompts[i])
            assert segment.proposal_sequences[i].sequence_type == "dna"

    def test_evo2_assign_errors(self):
        """Test error conditions for Evo2 generator assignment."""
        prompts_single = ["ATCG"]
        expected_length = len(prompts_single[0]) + 100
        config_single = Evo2GeneratorConfig(prompts=prompts_single)
        evo2_generator_single = Evo2Generator(config_single)

        segment_two_proposals = Segment(length=expected_length, sequence_type="dna")
        segment_two_proposals.proposal_sequences = [copy.deepcopy(segment_two_proposals.original_sequence) for _ in range(2)]
        evo2_generator_single.assign(segment_two_proposals)
        evo2_generator_single.sample()

        prompts_multi = ["ATCG", "GGCC", "TTAA"]
        config_multi = Evo2GeneratorConfig(prompts=prompts_multi)
        evo2_generator_multi = Evo2Generator(config_multi)

        segment_two_proposals2 = Segment(length=expected_length, sequence_type="dna")
        segment_two_proposals2.proposal_sequences = [copy.deepcopy(segment_two_proposals2.original_sequence) for _ in range(2)]
        evo2_generator_multi.assign(segment_two_proposals2)

        with pytest.raises(ValueError, match="Expected 1 or"):
            evo2_generator_multi.sample()

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

        segment = Segment(length=expected_length, sequence_type="dna")
        evo2_generator.assign(segment)

        assert evo2_generator.temperature == 0.8
        assert evo2_generator.top_k == 10
        assert evo2_generator.top_p == 0.9

        evo2_generator.sample()

        assert segment.proposal_sequences[0].sequence is not None
        assert segment.proposal_sequences[0].sequence_type == "dna"

    def test_evo2_batch_size_parameter(self):
        """Test Evo2 generator with custom batch_size for GPU memory management."""
        prompts = ["ATCG"] * 5
        num_tokens = 50
        expected_length = len(prompts[0]) + num_tokens
        config = Evo2GeneratorConfig(
            prompts=prompts,
            batch_size=2,
        )
        evo2_generator = Evo2Generator(config)

        segment = Segment(length=expected_length, sequence_type="dna")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        evo2_generator.assign(segment)

        assert evo2_generator.batch_size == 2

        evo2_generator.sample()

        for i in range(len(prompts)):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) > len(prompts[i])
            assert segment.proposal_sequences[i].sequence_type == "dna"


class TestEvo2GeneratorValidation:
    """Test sequence type validation for Evo2 generator."""

    def test_valid_dna_assignment(self):
        """Evo2 should accept DNA segments."""
        config = Evo2GeneratorConfig(prompts="ATGC")
        generator = Evo2Generator(config)
        segment = Segment(length=100, sequence_type="dna")

        generator.assign(segment)
        assert generator._assigned_segment is segment

    @pytest.mark.parametrize("seq_type", ["protein", "rna"])
    def test_rejects_non_dna_segment(self, seq_type):
        """Evo2 should reject non-DNA segments."""
        config = Evo2GeneratorConfig(prompts="ATGC")
        generator = Evo2Generator(config)
        segment = Segment(length=100, sequence_type=seq_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert seq_type in error_msg.lower()

    @patch("proto_language.language.generator.evo2_generator.run_evo2_sample")
    def test_num_tokens_computed_with_prepend_override(self, mock_run):
        """num_tokens adjusts when sample() gets prepend_prompt override."""
        config = Evo2GeneratorConfig(prompts="ATCG")  # len=4
        gen = Evo2Generator(config)
        segment = Segment(length=100, sequence_type="dna")
        gen.assign(segment)

        mock_output = MagicMock()
        mock_output.sequences = ["A" * 100]
        mock_output.kv_caches = []
        mock_run.return_value = mock_output

        # prepend_prompt=True → should subtract prompt len: 100 - 4 = 96
        gen.sample(prepend_prompt=True)
        assert mock_run.call_args[1]["config"].num_tokens == 96
