import pytest
import sys
import time

sys.path.append(".")
from proto_language.language.core import Segment, SequenceType
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig

# Check if GPU is available (either locally or via cloud)
from proto_language.utils import is_gpu_available


def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.DNA
) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


@pytest.mark.uses_gpu
class TestEvo2Generator:
    def test_evo2_single_prompt_sampling(self):
        """Test Evo2 generator with a single prompt sequence."""
        prompts = ["ATCG"]
        config = Evo2GeneratorConfig(
            prompts=prompts, 
            num_tokens=100,
        )
        evo2_generator = Evo2Generator(config)

        # Create segment and assign to generator
        segment = create_segment("", seq_type=SequenceType.DNA)
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
        config = Evo2GeneratorConfig(
            prompts=prompts, 
            num_tokens=100,
        )
        evo2_generator = Evo2Generator(config)

        # Create segment and expand candidate pool
        segment = create_segment("", seq_type=SequenceType.DNA)
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
        config = Evo2GeneratorConfig(prompts=prompts, num_tokens=100)
        evo2_generator = Evo2Generator(config)
        
        # Should raise error if number of prompts doesn't match segment candidates
        segment_two_candidates = create_segment("", seq_type=SequenceType.DNA)
        segment_two_candidates.create_candidates(2)
        evo2_generator.assign(segment_two_candidates)
        
        with pytest.raises(ValueError, match="Number of prompts"):
            evo2_generator.sample()  # Will fail because 1 prompt but 2 candidates

    def test_evo2_custom_parameters(self):
        """Test Evo2 generator with custom generation parameters."""
        prompts = ["ATCGATCG"]
        config = Evo2GeneratorConfig(
            prompts=prompts,
            num_tokens=50,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
        )
        evo2_generator = Evo2Generator(config)

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

    def test_evo2_caching_speedup(self):
        """Test that KV caching provides speedup and verify cache is actually used."""
        from proto_language.tools.models.language_models.evo2 import (
            run_evo2_sample,
            Evo2SampleConfig,
        )

        prompt = ["ATGCGATCGATCG"]
        test_lengths = [1500, 2000]

        for continuation_length in test_lengths:
            # Test 1: Generate all at once (no cache reuse)
            config_no_cache = Evo2SampleConfig(
                prompts=prompt,
                num_tokens=continuation_length,
                cached_generation=False,
                stop_at_eos=False,
                print_generation=False,
                verbose=False,
                max_seqlen=continuation_length + 50,
                temperature=1.0,
                top_k=4,
                top_p=1.0,
                keep_on_device=True
            )
            
            start = time.time()
            result_no_cache = run_evo2_sample(config_no_cache)
            time_no_cache = time.time() - start
            
            # Test 2: Generate in two steps with cache reuse
            first_half = continuation_length // 2
            second_half = continuation_length - first_half
            
            # Generate first half and build cache
            config1 = Evo2SampleConfig(
                prompts=prompt,
                num_tokens=first_half,
                cached_generation=True,
                stop_at_eos=False,
                print_generation=False,
                verbose=False,
                max_seqlen=continuation_length + 50,
                temperature=1.0,
                top_k=4,
                top_p=1.0,
                keep_on_device=True
            )
            
            start_first_half = time.time()
            result1 = run_evo2_sample(config1)
            time_first_half = time.time() - start_first_half
            
            first_seq = result1.sequences[0]
            
            # Generate second half with cache
            config2 = Evo2SampleConfig(
                prompts=[first_seq],
                num_tokens=second_half,
                cached_generation=True,
                old_kv_cache=result1.kv_caches[0],
                force_prompt_threshold=len(first_seq),
                stop_at_eos=False,
                print_generation=False,
                verbose=False,
                max_seqlen=continuation_length + 50,
                temperature=1.0,
                top_k=4,
                top_p=1.0,
                keep_on_device=True
            )
            
            start_second_half = time.time()
            result2 = run_evo2_sample(config2)
            time_second_half = time.time() - start_second_half
            
            time_with_cache = time_first_half + time_second_half
            
            # Test 3: Generate second half without cache (control)
            config_control = Evo2SampleConfig(
                prompts=[first_seq],
                num_tokens=second_half,
                cached_generation=False,
                stop_at_eos=False,
                print_generation=False,
                verbose=False,
                max_seqlen=continuation_length + 50,
                temperature=1.0,
                top_k=4,
                top_p=1.0,
                keep_on_device=True
            )
            
            start_control = time.time()
            result_control = run_evo2_sample(config_control)
            time_control_second_half = time.time() - start_control
            
            # Calculate metrics
            overall_speedup = time_no_cache / time_with_cache
            cache_benefit = time_control_second_half / time_second_half
            
            assert overall_speedup > 1.0, (
                f"Caching should provide overall speedup for {continuation_length}bp generation. "
                f"Got {overall_speedup:.2f}x speedup (time_no_cache={time_no_cache:.3f}s, "
                f"time_with_cache={time_with_cache:.3f}s). "
                f"Cache may not be functioning correctly."
            )
            
            assert cache_benefit > 1.5, (
                f"Cached second half should be significantly faster than non-cached for {continuation_length}bp. "
                f"Got {cache_benefit:.2f}x speedup (expected >1.5x). "
                f"time_cached={time_second_half:.3f}s, time_non_cached={time_control_second_half:.3f}s. "
                f"This suggests the KV cache is not being used properly"
            )