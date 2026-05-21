"""tests/language_tests/generator_tests/test_evo1_generator.py."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from proto_language.core import Segment
from proto_language.generator import Evo1Generator, Evo1GeneratorConfig


@pytest.mark.uses_gpu
class TestEvo1Generator:
    def test_evo1_generation(self):
        """Test generation: custom params, batching, and prompt replication."""
        prompts = ["ATCG"]
        max_new_tokens = 50
        expected_length = len(prompts[0]) + max_new_tokens
        config = Evo1GeneratorConfig(
            prompts=prompts,
            temperature=0.8,
            top_k=10,
            batch_size=2,
        )
        gen = Evo1Generator(config)

        assert gen.temperature == 0.8
        assert gen.top_k == 10
        assert gen.batch_size == 2

        segment = Segment(length=expected_length, sequence_type="dna")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(3)]
        gen.assign(segment)
        assert gen._assigned_segments == (segment,)

        gen.sample()

        for i in range(3):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) > len(prompts[0])
            assert segment.proposal_sequences[i].sequence_type == "dna"

    def test_evo1_prompt_mismatch_raises(self):
        """3 prompts with 2 proposals should raise ValueError."""
        prompts = ["ATCG", "GGCC", "TTAA"]
        config = Evo1GeneratorConfig(prompts=prompts)
        gen = Evo1Generator(config)

        segment = Segment(length=104, sequence_type="dna")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(2)]
        gen.assign(segment)

        with pytest.raises(ValueError, match="Expected 1 or"):
            gen.sample()


class TestEvo1GeneratorValidation:
    """Test sequence type validation for Evo1 generator."""

    def test_valid_dna_assignment(self):
        """Evo1 should accept DNA segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="dna")

        gen.assign(segment)
        assert gen._assigned_segments == (segment,)

    @pytest.mark.parametrize("seq_type", ["protein", "rna"])
    def test_rejects_non_dna_segment(self, seq_type):
        """Evo1 should reject non-DNA segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type=seq_type)

        with pytest.raises(ValueError) as exc_info:
            gen.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert seq_type in error_msg.lower()

    def test_prompts_unequal_length_raises(self):
        """Prompts with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            Evo1GeneratorConfig(prompts=["ATCG", "AT"])

    @patch("proto_language.generator.evo1_generator.run_evo1_sample")
    def test_max_new_tokens_computed_with_prepend_override(self, mock_run):
        """max_new_tokens adjusts when sample() gets prepend_prompt override."""
        config = Evo1GeneratorConfig(prompts="ATCG")  # len=4
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="dna")
        gen.assign(segment)

        mock_output = MagicMock()
        mock_output.sequences = ["A" * 100]
        mock_output.scores = []
        mock_run.return_value = mock_output

        # prepend_prompt=True → should subtract prompt len: 100 - 4 = 96
        gen.sample(prepend_prompt=True)
        assert mock_run.call_args[1]["config"].max_new_tokens == 96
