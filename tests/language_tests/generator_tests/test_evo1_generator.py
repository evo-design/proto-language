import copy

import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import (
    Evo1Generator,
    Evo1GeneratorConfig,
)


@pytest.mark.uses_gpu
class TestEvo1Generator:
    def test_evo1_single_prompt_sampling(self):
        """Test Evo1 generator with a single prompt sequence."""
        prompts = ["ATCG"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = Evo1GeneratorConfig(
            prompts=prompts,
        )
        evo1_generator = Evo1Generator(config)

        # Create segment and assign to generator
        segment = Segment(length=expected_length, sequence_type="dna")
        evo1_generator.assign(segment)

        assert evo1_generator._assigned_segment is segment

        # Sample and check results
        evo1_generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert len(segment.candidate_sequences[0].sequence) > len(prompts[0])
        assert segment.candidate_sequences[0].sequence_type == "dna"

    def test_evo1_batch_sampling(self):
        """Test Evo1 generator with multiple prompt sequences."""
        prompts = ["ATCG", "AAAA"]
        num_tokens = 100
        expected_length = len(prompts[0]) + num_tokens
        config = Evo1GeneratorConfig(
            prompts=prompts,
        )
        evo1_generator = Evo1Generator(config)

        # Create segment and expand candidate pool
        segment = Segment(length=expected_length, sequence_type="dna")
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        evo1_generator.assign(segment)

        assert evo1_generator._assigned_segment is segment
        assert len(segment.candidate_sequences) == len(prompts)

        # Sample and check results
        evo1_generator.sample()

        for i in range(len(prompts)):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) > len(prompts[i])
            assert segment.candidate_sequences[i].sequence_type == "dna"

    def test_evo1_assign_errors(self):
        """Test error conditions for Evo1 generator assignment."""
        # Test 1: Single prompt with multiple candidates (should work - replicates)
        prompts_single = ["ATCG"]
        expected_length = len(prompts_single[0]) + 100
        config_single = Evo1GeneratorConfig(prompts=prompts_single)
        evo1_generator_single = Evo1Generator(config_single)

        segment_two_candidates = Segment(length=expected_length, sequence_type="dna")
        segment_two_candidates.candidate_sequences = [copy.deepcopy(segment_two_candidates.original_sequence) for _ in range(2)]
        evo1_generator_single.assign(segment_two_candidates)
        # Single prompt should be replicated for 2 candidates - no error
        evo1_generator_single.sample()

        # Test 2: Mismatched multiple prompts (should raise error)
        prompts_multi = ["ATCG", "GGCC", "TTAA"]  # 3 prompts (same length)
        config_multi = Evo1GeneratorConfig(prompts=prompts_multi)
        evo1_generator_multi = Evo1Generator(config_multi)

        segment_two_candidates2 = Segment(length=expected_length, sequence_type="dna")
        segment_two_candidates2.candidate_sequences = [copy.deepcopy(segment_two_candidates2.original_sequence) for _ in range(2)]
        evo1_generator_multi.assign(segment_two_candidates2)

        # 3 prompts but 2 candidates - should raise ValueError
        with pytest.raises(ValueError, match="must either be 1"):
            evo1_generator_multi.sample()

    def test_evo1_custom_parameters(self):
        """Test Evo1 generator with custom generation parameters."""
        prompts = ["ATCGATCG"]
        num_tokens = 50
        expected_length = len(prompts[0]) + num_tokens
        config = Evo1GeneratorConfig(
            prompts=prompts,
            temperature=0.8,
            top_k=10,
        )
        evo1_generator = Evo1Generator(config)

        # Create segment and assign to generator
        segment = Segment(length=expected_length, sequence_type="dna")
        evo1_generator.assign(segment)

        assert evo1_generator.temperature == 0.8
        assert evo1_generator.top_k == 10

        # Sample and check results
        evo1_generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert segment.candidate_sequences[0].sequence_type == "dna"

    def test_evo1_batch_size_parameter(self):
        """Test Evo1 generator with custom batch_size for GPU memory management."""
        prompts = ["ATCG"] * 5  # 5 identical prompts
        num_tokens = 50
        expected_length = len(prompts[0]) + num_tokens
        config = Evo1GeneratorConfig(
            prompts=prompts,
            batch_size=2,  # Process 2 at a time
        )
        evo1_generator = Evo1Generator(config)

        # Create segment with matching candidate pool
        segment = Segment(length=expected_length, sequence_type="dna")
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(prompts))]
        evo1_generator.assign(segment)

        assert evo1_generator.batch_size == 2

        # Sample and check all results were generated
        evo1_generator.sample()

        for i in range(len(prompts)):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) > len(prompts[i])
            assert segment.candidate_sequences[i].sequence_type == "dna"


class TestEvo1GeneratorValidation:
    """Test sequence type validation for Evo1 generator."""

    def test_valid_dna_assignment(self):
        """Evo1 should accept DNA segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="dna")

        # Should not raise
        gen.assign(segment)
        assert gen._assigned_segment is segment

    def test_rejects_protein_segment(self):
        """Evo1 should reject PROTEIN segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="protein")

        with pytest.raises(ValueError) as exc_info:
            gen.assign(segment)

        error_msg = str(exc_info.value)
        assert "does not support sequence type" in error_msg
        assert "protein" in error_msg.lower()
        assert "dna" in error_msg.lower()

    def test_rejects_rna_segment(self):
        """Evo1 should reject RNA segments."""
        config = Evo1GeneratorConfig(prompts="ATGC")
        gen = Evo1Generator(config)
        segment = Segment(length=100, sequence_type="rna")

        with pytest.raises(ValueError) as exc_info:
            gen.assign(segment)

        assert "does not support sequence type" in str(exc_info.value)
        assert "rna" in str(exc_info.value).lower()

    def test_batch_size_config(self):
        """Test that batch_size parameter is properly set in config and generator."""
        config = Evo1GeneratorConfig(prompts="ATGC", batch_size=5)
        gen = Evo1Generator(config)

        assert config.batch_size == 5
        assert gen.batch_size == 5

    def test_prompts_unequal_length_raises(self):
        """Prompts with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            Evo1GeneratorConfig(prompts=["ATCG", "AT"])
