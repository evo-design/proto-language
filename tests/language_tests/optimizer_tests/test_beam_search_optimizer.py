"""Tests for BeamSearchOptimizer - single-segment iterative beam search."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional
from unittest.mock import Mock

import pytest

from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
    GCContentConfig,
)
from proto_language.language.core import Constraint, Construct, Generator, Segment
from proto_language.language.optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
    BeamState,
)


class MockAutoregressiveGenerator(Generator):
    """Mock autoregressive generator for testing without GPU."""

    def __init__(self, use_kv_caching: bool = True):
        super().__init__()
        self.use_kv_caching = use_kv_caching
        self.kv_caches: List[Dict] = []

    def assign(self, assigned_segment: Segment) -> None:
        self._assigned_segment = assigned_segment

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prepend_prompt: Optional[bool] = None,
        num_tokens: Optional[int] = None,
        old_kv_cache: Optional[Dict] = None,
    ) -> None:
        if num_tokens is None:
            num_tokens = 100
        if prompts is None:
            prompts = [""]
        sequences = []
        for prompt in prompts:
            new_seq = "".join(random.choice("ATCG") for _ in range(num_tokens))
            sequences.append(prompt + new_seq if prepend_prompt else new_seq)
        for candidate, sequence in zip(
            self._assigned_segment.candidate_sequences, sequences, strict=True
        ):
            candidate.sequence = sequence
        if self.use_kv_caching and getattr(self, "store_kv_cache", False):
            mock_mha = Mock()
            mock_mha.key_value_memory_dict = {0: Mock(shape=(1, 2, 3))}
            mock_mha.seqlen_offset = 10
            self.kv_caches = [
                {"mha": mock_mha, "hcl": Mock()} for _ in range(len(prompts))
            ]
        else:
            self.kv_caches = []

    def replicate_cache(self, cache: Dict, n_replicates: int) -> Dict:
        return cache


class MockMutationGenerator(Generator):
    """Mock non-autoregressive generator for testing rejection."""

    def __init__(self):
        super().__init__()
        self.kv_caches: List[Dict] = []

    def assign(self, assigned_segment: Segment) -> None:
        self._assigned_segment = assigned_segment

    def sample(self, prompts=None, prepend_prompt=None, old_kv_cache=None) -> None:
        pass

    def replicate_cache(self, cache: Dict, n_replicates: int) -> Dict:
        return cache


class MockAutoregressiveGeneratorNoKVCache(Generator):
    """Mock autoregressive generator without KV caching support."""

    def __init__(self):
        super().__init__()
        # Intentionally missing kv_caches and replicate_cache

    def assign(self, assigned_segment: Segment) -> None:
        self._assigned_segment = assigned_segment

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prepend_prompt: Optional[bool] = None,
        num_tokens: Optional[int] = None,
        old_kv_cache: Optional[Dict] = None,
    ) -> None:
        if num_tokens is None:
            num_tokens = 100
        if prompts is None:
            prompts = [""]
        sequences = []
        for prompt in prompts:
            new_seq = "".join(random.choice("ATCG") for _ in range(num_tokens))
            sequences.append(prompt + new_seq if prepend_prompt else new_seq)
        for candidate, sequence in zip(
            self._assigned_segment.candidate_sequences, sequences, strict=True
        ):
            candidate.sequence = sequence


def _setup_beam_search(
    segment_length: int = 100,
    beam_length: int = 20,
    num_results: int = 3,
    candidates_per_result: int = 5,
    gc_range: tuple = (40.0, 60.0),
    use_kv_caching: bool = True,
    prompt: str = "ATCG",
    score_by: str = "mean",
    prepend_prompt: bool = True,
    mock_generator: MockAutoregressiveGenerator | None = None,
):
    """Helper to set up a BeamSearchOptimizer for testing."""
    segment = Segment(length=segment_length, sequence_type="dna")
    construct = Construct([segment])
    generator = mock_generator or MockAutoregressiveGenerator(
        use_kv_caching=use_kv_caching
    )
    generator._assigned_segment = segment
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=gc_range[0], max_gc=gc_range[1]),
    )
    config = BeamSearchOptimizerConfig(
        prompt=prompt,
        beam_length=beam_length,
        num_results=num_results,
        candidates_per_result=candidates_per_result,
        score_by=score_by,
        use_kv_caching=use_kv_caching,
        prepend_prompt=prepend_prompt,
        verbose=False,
    )
    optimizer = BeamSearchOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=[constraint],
        config=config,
        target_segment=segment,
    )
    return optimizer, generator, constraint, segment


class TestBeamSearchOptimizer:
    """Tests for BeamSearchOptimizer functionality."""

    # --- Config Validation ---
    def test_valid_config(self):
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            num_results=5,
            candidates_per_result=10,
            beam_length=2000,
            score_by="mean",
        )
        assert config.prompt == "ATCG"
        assert config.num_results == 5
        assert config.prepend_prompt is True

    def test_empty_prompt_fails(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(
                prompt="", num_results=3, candidates_per_result=5, beam_length=10
            )

    def test_invalid_num_results_fails(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(
                prompt="ATCG", num_results=0, candidates_per_result=5, beam_length=10
            )

    def test_invalid_score_by_fails(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(
                prompt="ATCG", num_results=5, candidates_per_result=10, score_by="invalid"
            )

    # --- Initialization ---
    def test_initialization(self):
        optimizer, generator, constraint, segment = _setup_beam_search()
        assert optimizer.target_segment == segment
        assert optimizer.generator == generator
        assert optimizer.num_results == 3
        assert optimizer._candidates_per_result == 5
        assert len(optimizer.beams) == optimizer.num_results
        assert all(isinstance(beam, BeamState) for beam in optimizer.beams)

    def test_multi_segment_construct_with_target_segment(self):
        """Multi-segment constructs are allowed when target_segment is specified."""
        # Target segment has no sequence (will be generated)
        target_segment = Segment(length=20, sequence_type="dna")
        # Context segments have sequences
        context_segment1 = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        context_segment2 = Segment(sequence="GCGCGCGCGCGCGCGCGCGC", sequence_type="dna")
        segments = [target_segment, context_segment1, context_segment2]
        construct = Construct(segments)
        generator = MockAutoregressiveGenerator()
        generator._assigned_segment = segments[0]
        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=10, num_results=3, candidates_per_result=5
        )
        # Should work when target_segment is specified
        optimizer = BeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=segments[0],
        )
        assert optimizer.target_segment == segments[0]

    def test_non_target_constraint_input_accepted(self):
        """BeamSearch accepts constraints referencing non-target segments."""
        target_segment = Segment(length=20, sequence_type="dna")
        context_segment = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        construct = Construct([target_segment, context_segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segment = target_segment

        non_target_constraint = Constraint(
            inputs=[context_segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=10, num_results=2, candidates_per_result=3
        )

        optimizer = BeamSearchOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[non_target_constraint],
            config=config,
        )
        assert optimizer.target_segment is target_segment

    def test_duplicate_constraint_instance_fails(self):
        """Same constraint instance cannot be passed twice."""
        segment = Segment(length=20, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=10, num_results=2, candidates_per_result=3
        )

        with pytest.raises(ValueError, match="appears multiple times"):
            BeamSearchOptimizer(
                target_segment=segment,
                constructs=[construct],
                generators=[generator],
                constraints=[constraint, constraint],
                config=config,
            )

    def test_target_segment_not_in_constructs_fails(self):
        """target_segment must belong to one of the provided constructs."""
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        other_segment = Segment(length=20, sequence_type="dna")  # Not in construct
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segment = other_segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=10, num_results=3, candidates_per_result=5
        )
        with pytest.raises(
            ValueError, match="is not in any of the provided constructs"
        ):
            BeamSearchOptimizer(
                target_segment=other_segment,
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_multi_segment_with_context_segments(self):
        """Multi-segment constructs work when non-target segments have sequences."""
        target_segment = Segment(length=20, sequence_type="dna")
        context_segment = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        construct = Construct([target_segment, context_segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segment = target_segment
        constraint = Constraint(
            inputs=[target_segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=10, num_results=3, candidates_per_result=5
        )

        # Should work - context segment has a sequence
        optimizer = BeamSearchOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
        )
        assert optimizer.target_segment == target_segment

    def test_non_autoregressive_generator_fails(self):
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockMutationGenerator()
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=10, num_results=3, candidates_per_result=5
        )
        with pytest.raises(ValueError, match="requires autoregressive generators"):
            BeamSearchOptimizer(
                target_segment=segment,
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_generator_without_kv_caching_support_fails(self):
        """Generator missing replicate_cache/kv_caches should fail when use_kv_caching=True."""
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGeneratorNoKVCache()
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=10,
            num_results=3,
            candidates_per_result=5,
            use_kv_caching=True,
        )
        with pytest.raises(ValueError, match="does not support KV caching"):
            BeamSearchOptimizer(
                target_segment=segment,
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_generator_without_kv_caching_support_works_when_disabled(self):
        """Generator missing KV caching support should work when use_kv_caching=False."""
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGeneratorNoKVCache()
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=10,
            num_results=3,
            candidates_per_result=5,
            use_kv_caching=False,
        )
        # Should not raise - KV caching is disabled
        optimizer = BeamSearchOptimizer(
            target_segment=segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
        )
        assert optimizer.use_kv_caching is False

    def test_beam_length_exceeds_segment_length_fails(self):
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=100, num_results=3, candidates_per_result=5
        )
        with pytest.raises(
            ValueError,
            match="beam_length.*cannot be greater than target_segment length",
        ):
            BeamSearchOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
                target_segment=segment,
            )

    # --- BeamState ---
    def test_beam_state_initialization(self):
        state = BeamState(running_sequence="ATCG")
        assert state.running_sequence == "ATCG"
        assert state.kv_cache is None
        assert state.beam_scores == []

    def test_optimizer_initializes_beams_with_prompt(self):
        prompt = "ATCGATCG"
        optimizer, _, _, _ = _setup_beam_search(num_results=4, prompt=prompt)
        assert len(optimizer.beams) == 4
        for beam in optimizer.beams:
            assert beam.running_sequence == prompt

    # --- Num Beams Calculation ---
    def test_num_beams_exact_division(self):
        optimizer, _, _, _ = _setup_beam_search(segment_length=100, beam_length=20)
        assert optimizer.num_beams == 5

    def test_num_beams_with_remainder(self):
        optimizer, _, _, _ = _setup_beam_search(segment_length=95, beam_length=20)
        assert optimizer.num_beams == 5

    def test_final_beam_generates_fewer_tokens(self):
        """Final beam generates remaining tokens when segment_length % beam_length != 0."""
        prompt = "ATCG"
        optimizer, _, _, segment = _setup_beam_search(
            segment_length=95, beam_length=20, prompt=prompt, num_results=2
        )
        optimizer.run()
        # 95 tokens total with prepend_prompt=True: len(prompt) + 95 = 99
        for seq in segment.selected_sequences:
            assert len(seq.sequence) == len(prompt) + 95

    # --- Score Aggregation ---
    def test_score_by_mean(self):
        optimizer, _, _, _ = _setup_beam_search(score_by="mean")
        beam = BeamState(running_sequence="ATCG", beam_scores=[0.1, 0.2, 0.3])
        assert abs(optimizer._get_aggregated_score(beam) - 0.2) < 1e-6

    def test_score_by_last(self):
        optimizer, _, _, _ = _setup_beam_search(score_by="last")
        beam = BeamState(running_sequence="ATCG", beam_scores=[0.1, 0.2, 0.3])
        assert abs(optimizer._get_aggregated_score(beam) - 0.3) < 1e-6

    def test_empty_scores_returns_inf(self):
        optimizer, _, _, _ = _setup_beam_search()
        beam = BeamState(running_sequence="ATCG", beam_scores=[])
        assert optimizer._get_aggregated_score(beam) == float('inf')

    # --- Run ---
    def test_run_generates_correct_sequence_length(self):
        prompt = "ATCG"
        segment_length = 100
        optimizer, _, _, segment = _setup_beam_search(
            segment_length=segment_length, prompt=prompt
        )
        optimizer.run()
        expected_length = len(prompt) + segment_length
        for seq in segment.selected_sequences:
            assert len(seq.sequence) == expected_length

    def test_beam_scores_accumulated(self):
        optimizer, _, _, _ = _setup_beam_search(
            segment_length=60, beam_length=20, num_results=2
        )
        optimizer.run()
        for beam in optimizer.beams:
            assert len(beam.beam_scores) == 3  # 60/20 = 3 beams

    def test_selected_sequences_populated(self):
        optimizer, _, _, segment = _setup_beam_search(num_results=4)
        optimizer.run()
        assert len(segment.selected_sequences) == optimizer.num_results

    def test_history_saved(self):
        optimizer, _, _, _ = _setup_beam_search(segment_length=60, beam_length=20)
        optimizer.run()
        assert len(optimizer.history) > 0

    # --- Prepend Prompt ---
    def test_prepend_prompt_true(self):
        prompt = "ATCGATCG"
        optimizer, _, _, segment = _setup_beam_search(
            prompt=prompt, segment_length=40, prepend_prompt=True
        )
        optimizer.run()
        for seq in segment.selected_sequences:
            assert seq.sequence.startswith(prompt)
            assert len(seq.sequence) == len(prompt) + 40

    def test_prepend_prompt_false(self):
        optimizer, _, _, segment = _setup_beam_search(
            prompt="ATCGATCG", segment_length=40, beam_length=20, prepend_prompt=False
        )
        optimizer.run()
        for seq in segment.selected_sequences:
            assert len(seq.sequence) == 40

    # --- KV Caching ---
    def test_kv_caching_disabled(self):
        optimizer, _, _, segment = _setup_beam_search(
            use_kv_caching=False, segment_length=60, beam_length=20
        )
        optimizer.run()
        assert len(segment.selected_sequences) == optimizer.num_results
        for beam in optimizer.beams:
            assert beam.kv_cache is None

    # --- Batch Size ---
    def test_batch_size_smaller_than_candidates(self):
        mock_gen = MockAutoregressiveGenerator(use_kv_caching=True)
        mock_gen.batch_size = 2
        optimizer, _, _, segment = _setup_beam_search(
            segment_length=40,
            beam_length=20,
            num_results=2,
            candidates_per_result=6,
            mock_generator=mock_gen,
        )
        assert optimizer.generator.batch_size == 2
        optimizer.run()
        assert len(segment.selected_sequences) == optimizer.num_results

    # --- Resampling ---
    def test_all_invalid_raises_error(self):
        segment = Segment(length=20, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=100.0, max_gc=100.0),
            threshold=0.0,
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=20,
            num_results=2,
            candidates_per_result=3,
            max_resample_attempts=3,
            use_kv_caching=False,
            verbose=False,
        )
        optimizer = BeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=segment,
        )
        with pytest.raises(RuntimeError, match="could not produce.*valid candidates"):
            optimizer.run()

    # --- Edge Cases ---
    def test_num_results_one(self):
        optimizer, _, _, segment = _setup_beam_search(
            num_results=1, candidates_per_result=5
        )
        optimizer.run()
        assert len(segment.selected_sequences) == 1

    def test_candidates_per_result_one(self):
        optimizer, _, _, segment = _setup_beam_search(
            num_results=3, candidates_per_result=1
        )
        optimizer.run()
        assert len(segment.selected_sequences) == 3

    # --- Verbose ---
    def test_warning_when_previous_results_discarded(self, caplog):
        """BeamSearch warns when existing sequences will be overwritten."""
        import logging

        optimizer, _, _, segment = _setup_beam_search(
            segment_length=40, beam_length=20, num_results=2, prompt="ATCG"
        )
        # Simulate previous optimizer results
        segment.selected_sequences[0].sequence = "GCTAGCTA"
        segment.selected_sequences[1].sequence = "TTTTTTTT"

        with caplog.at_level(logging.WARNING):
            optimizer.run()

        assert any("overwrites existing sequences" in msg for msg in caplog.messages)

    def test_verbose_output(self, caplog):
        import logging

        optimizer, _, _, _ = _setup_beam_search(
            segment_length=40, beam_length=20, num_results=2
        )
        optimizer.verbose = True
        with caplog.at_level(logging.DEBUG):
            optimizer.run()
        assert "Processing segment" in caplog.text


@pytest.mark.uses_gpu
@pytest.mark.slow
class TestBeamSearchOptimizerGPU:
    """GPU-dependent integration tests."""

    def test_with_evo2_generator(self):
        from proto_language.language.generator import (
            Evo2Generator,
            Evo2GeneratorConfig,
        )

        prompt = "ATCGATCGATCG"
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        gen_config = Evo2GeneratorConfig(
            prompts=[prompt], prepend_prompt=True, stop_at_eos=False
        )
        generator = Evo2Generator(config=gen_config)
        generator.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt=prompt,
            beam_length=50,
            num_results=3,
            candidates_per_result=5,
            use_kv_caching=True,
        )
        optimizer = BeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=segment,
        )
        optimizer.run()
        assert len(segment.selected_sequences) == 3
        for seq in segment.selected_sequences:
            assert len(seq.sequence) == len(prompt) + 100

    def test_kv_caching_speedup(self):
        """Benchmark KV caching speedup with real Evo2 generator."""
        import time

        from proto_language.language.generator import (
            Evo2Generator,
            Evo2GeneratorConfig,
        )

        def run_optimizer(use_kv_caching: bool) -> float:
            prompt = "ATCGATCGATCG"
            segment = Segment(length=200, sequence_type="dna")
            construct = Construct([segment])
            gen_config = Evo2GeneratorConfig(
                prompts=[prompt], prepend_prompt=True, stop_at_eos=False
            )
            generator = Evo2Generator(config=gen_config)
            generator.assign(segment)
            constraint = Constraint(
                inputs=[segment],
                function=gc_content_constraint,
                function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            )
            config = BeamSearchOptimizerConfig(
                prompt=prompt,
                beam_length=20,
                num_results=3,
                candidates_per_result=3,
                use_kv_caching=use_kv_caching,
            )
            optimizer = BeamSearchOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
                target_segment=segment,
            )
            start = time.time()
            optimizer.run()
            return time.time() - start

        time_uncached = run_optimizer(use_kv_caching=False)
        time_cached = run_optimizer(use_kv_caching=True)
        speedup = time_uncached / time_cached

        print(f"\nKV Caching Benchmark:")
        print(f"  Without caching: {time_uncached:.2f}s")
        print(f"  With caching: {time_cached:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # KV caching should be faster
        assert (
            time_cached * 1.5 < time_uncached
        ), f"KV caching should be 1.5x faster: {time_cached:.2f}s vs {time_uncached:.2f}s: {speedup:.2f}x"


class TestBeamSearchMultiStepOptimization:
    """Tests for multi-step optimization with BeamSearchOptimizer."""

    def test_multi_segment_construct_with_context_segments(self):
        """Test BeamSearchOptimizer with multi-segment construct where non-target segments provide context."""
        # Create segments - target segment and context segment (has sequence)
        target_segment = Segment(length=40, sequence_type="dna")
        context_segment = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        construct = Construct([target_segment, context_segment])

        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segment = target_segment

        # Constraint on target segment only
        constraint = Constraint(
            inputs=[target_segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=20,
            num_results=2,
            candidates_per_result=3,
            use_kv_caching=False,
        )

        optimizer = BeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=target_segment,
        )

        optimizer.run()

        # Verify target segment was optimized
        assert len(target_segment.selected_sequences) == 2
        for seq in target_segment.selected_sequences:
            assert len(seq.sequence) == 40 + 4  # prompt + segment length

        # Verify context segment was not modified
        assert context_segment.original_sequence.sequence == "ATCGATCGATCGATCGATCG"

    def test_multiple_constructs_with_target_segment(self):
        """Test BeamSearchOptimizer with multiple constructs, targeting one segment."""
        # Create two constructs
        target_segment = Segment(length=40, sequence_type="dna")
        other_segment = Segment(sequence="GCGCGCGCGC", sequence_type="dna")
        construct1 = Construct([target_segment])
        construct2 = Construct([other_segment])

        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segment = target_segment

        constraint = Constraint(
            inputs=[target_segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=20,
            num_results=2,
            candidates_per_result=3,
            use_kv_caching=False,
        )

        optimizer = BeamSearchOptimizer(
            constructs=[construct1, construct2],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=target_segment,
        )

        optimizer.run()

        # Verify target segment was optimized
        assert len(target_segment.selected_sequences) == 2

        # Verify other construct's segment was not modified
        assert other_segment.original_sequence.sequence == "GCGCGCGCGC"

    def test_target_segment_attribute(self):
        """Test that target_segment is properly stored and accessible."""
        segment = Segment(length=40, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=20,
            num_results=2,
            candidates_per_result=3,
        )

        optimizer = BeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=segment,
        )

        assert optimizer.target_segment is segment


class TestBeamSearchOptimizerRestart:
    """Tests for BeamSearchOptimizer state restart behavior."""

    def test_run_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        prompt = "ATCG"
        optimizer, generator, constraint, segment = _setup_beam_search(
            prompt=prompt,
            num_results=2,
            candidates_per_result=2,
            segment_length=40,
            beam_length=20,
        )

        # Capture original state before run
        original_selected = [copy.deepcopy(s) for s in segment.selected_sequences]

        # First run
        optimizer.run()
        assert optimizer._initial_state is not None
        first_run_beams = [b.running_sequence for b in optimizer.beams]
        # Verify beams grew beyond prompt
        assert all(len(b) > len(prompt) for b in first_run_beams)

        # Verify captured state contains original sequences (using index 0)
        assert len(optimizer._initial_state['segments']) == 1
        captured_selected = optimizer._initial_state['segments'][0]['selected']

        # Verify captured sequences match originals
        assert len(captured_selected) == len(original_selected)
        for orig, captured in zip(original_selected, captured_selected):
            assert orig.sequence == captured['sequence']

        # Manually modify sequences to invalid values to verify restore
        segment.selected_sequences[0].sequence = "G" * 44  # prompt (4) + segment_length (40)
        segment.candidate_sequences[0].sequence = "G" * 44

        # Second run should restart - beams should be reset to prompt
        optimizer.run()
        second_run_beams = [b.running_sequence for b in optimizer.beams]
        # Verify beams grew again (optimization ran)
        assert all(len(b) > len(prompt) for b in second_run_beams)

        # Verify sequences were restored (not all G's - restoration happened)
        assert any(seq.sequence != "G" * 44 for seq in segment.selected_sequences)

        # History should be fresh (cleared on restart)
        # t=0 initial snapshot + per-beam snapshots
        assert len(optimizer.history) == 3

    def test_beams_reset_on_restore(self):
        """Test that beams are reset to initial prompt on restore and sequences are restored."""
        prompt = "ATCGATCG"
        optimizer, _, _, segment = _setup_beam_search(
            prompt=prompt,
            num_results=3,
            candidates_per_result=2,
            segment_length=40,
            beam_length=20,
        )

        # Capture original sequences
        original_selected = [s.sequence for s in segment.selected_sequences]

        # First run - beams will be modified
        optimizer.run()

        # Capture modified beams
        modified_beams = [b.running_sequence for b in optimizer.beams]
        assert all(len(b) > len(prompt) for b in modified_beams)

        # Manually modify sequences to invalid values
        for seq in segment.selected_sequences:
            seq.sequence = "G" * 48  # prompt (8) + segment_length (40)

        # Trigger restore
        optimizer._restore_initial_state()

        # Beams should be reset to prompt
        assert len(optimizer.beams) == 3
        for beam in optimizer.beams:
            assert beam.running_sequence == prompt
            assert beam.kv_cache is None
            assert beam.beam_scores == []

        # Sequences should be restored to original state
        restored_sequences = [s.sequence for s in segment.selected_sequences]
        assert len(restored_sequences) == len(original_selected)
        for orig, restored in zip(original_selected, restored_sequences):
            assert orig == restored


class TestBeamSearchCandidateTracking:
    """Test candidate_results tracking in BeamSearch history."""

    def test_candidate_tracking(self):
        """History has candidate_results with 'Beam pruned' for rejected candidates."""
        optimizer, _, _, _ = _setup_beam_search(
            num_results=3, candidates_per_result=2, beam_length=10
        )
        optimizer.track_candidates = True
        optimizer.run()

        valid_rejectors = {"Beam pruned"}
        all_rejectors = set()
        for entry in optimizer.history:
            if "candidate_results" not in entry:
                continue
            for cand in entry["candidate_results"]:
                assert isinstance(cand["accepted"], bool)
                if cand["accepted"]:
                    assert cand["rejected_by"] is None
                else:
                    all_rejectors.add(cand["rejected_by"])

        assert all_rejectors.issubset(valid_rejectors)


class TestBeamSearchTrackingInterval:
    """Test tracking_interval in BeamSearchOptimizer."""

    def test_tracking_interval(self):
        """tracking_interval=2 reduces history snapshots."""
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segment = segment
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=20,
            num_results=3,
            candidates_per_result=5,
            use_kv_caching=False,
            verbose=False,
            tracking_interval=2,
        )
        optimizer = BeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=segment,
        )
        optimizer.run()

        # 100/20 = 5 beams; with interval=2: beams 2,4 saved + final (5) always saved
        saved_steps = {entry["time_step"] for entry in optimizer.history}
        # Final beam (5) is always saved
        assert optimizer.num_beams in saved_steps
        assert len(optimizer.history) < optimizer.num_beams  # Fewer than every-beam
