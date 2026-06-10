"""Tests for BeamSearchOptimizer - single-segment iterative beam search."""

import copy
import random
from collections.abc import Iterable
from unittest.mock import Mock

import pytest

from proto_language.constraint import gc_content_constraint
from proto_language.constraint.sequence_composition.gc_content_constraint import (
    GCContentConfig,
)
from proto_language.core import (
    Constraint,
    ConstraintOutput,
    Construct,
    Generator,
    GeneratorInputType,
    Segment,
)
from proto_language.optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
    BeamState,
)


class MockAutoregressiveGenerator(Generator):
    """Mock autoregressive generator for testing without GPU."""

    input_type = GeneratorInputType.PROMPT

    def __init__(self, use_kv_caching: bool = True):
        super().__init__()
        self.use_kv_caching = use_kv_caching
        self.kv_caches: list[dict] = []

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        self._assigned_segments = (segments,) if isinstance(segments, Segment) else tuple(segments)

    def _sample(
        self,
        prompts: list[str] | None = None,
        prepend_prompt: bool | None = None,
        max_new_tokens: int | None = None,
        old_kv_cache: dict | None = None,
    ) -> None:
        if max_new_tokens is None:
            max_new_tokens = 100
        if prompts is None:
            prompts = [""]
        sequences = []
        for prompt in prompts:
            new_seq = "".join(random.choice("ATCG") for _ in range(max_new_tokens))  # noqa: S311 -- non-cryptographic, test mock
            sequences.append(prompt + new_seq if prepend_prompt else new_seq)
        for proposal, sequence in zip(self.segment.proposal_sequences, sequences, strict=True):
            proposal.sequence = sequence
        if self.use_kv_caching and getattr(self, "store_kv_cache", False):
            mock_mha = Mock()
            mock_mha.key_value_memory_dict = {0: Mock(shape=(1, 2, 3))}
            mock_mha.seqlen_offset = 10
            self.kv_caches = [{"mha": mock_mha, "hcl": Mock()} for _ in range(len(prompts))]
        else:
            self.kv_caches = []

    def release_kv_cache(self, cache: dict) -> None:
        pass


class MockAutoregressiveGeneratorNoKVCacheRelease(MockAutoregressiveGenerator):
    """Mock autoregressive generator with incomplete KV caching support."""

    release_kv_cache = None


class TrackingKVCacheGenerator(Generator):
    """Deterministic autoregressive generator that tracks cache handle lifecycle."""

    input_type = GeneratorInputType.PROMPT

    batch_size = 2

    def __init__(self):
        super().__init__()
        self.kv_caches: list[str] = []
        self.sample_old_kv_caches: list[str | None] = []
        self.generated_kv_caches: list[str] = []
        self.released_kv_caches: list[str] = []
        self._sample_idx = 0

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        self._assigned_segments = (segments,) if isinstance(segments, Segment) else tuple(segments)

    def _sample(
        self,
        prompts: list[str] | None = None,
        prepend_prompt: bool | None = None,
        max_new_tokens: int | None = None,
        old_kv_cache: str | None = None,
    ) -> None:
        assert prompts is not None
        assert max_new_tokens is not None
        self.sample_old_kv_caches.append(old_kv_cache)
        self._sample_idx += 1

        suffix = "A" * max_new_tokens
        sequences = [prompt + suffix if prepend_prompt else suffix for prompt in prompts]
        for proposal, sequence in zip(self.segment.proposal_sequences, sequences, strict=True):
            proposal.sequence = sequence

        self.kv_caches = [f"cache-{self._sample_idx}-{idx}" for idx in range(len(prompts))]
        self.generated_kv_caches.extend(self.kv_caches)

    def release_kv_cache(self, cache: str) -> None:
        self.released_kv_caches.append(cache)


class MockMutationGenerator(Generator):
    """Mock non-autoregressive generator for testing rejection."""

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self):
        super().__init__()
        self.kv_caches: list[dict] = []

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        self._assigned_segments = (segments,) if isinstance(segments, Segment) else tuple(segments)

    def _sample(self) -> None:
        pass


class MockAutoregressiveGeneratorNoKVCache(Generator):
    """Mock autoregressive generator without KV caching support."""

    input_type = GeneratorInputType.PROMPT

    def __init__(self):
        super().__init__()
        # Intentionally missing kv_caches

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        self._assigned_segments = (segments,) if isinstance(segments, Segment) else tuple(segments)

    def _sample(
        self,
        prompts: list[str] | None = None,
        prepend_prompt: bool | None = None,
        max_new_tokens: int | None = None,
        old_kv_cache: dict | None = None,
    ) -> None:
        if max_new_tokens is None:
            max_new_tokens = 100
        if prompts is None:
            prompts = [""]
        sequences = []
        for prompt in prompts:
            new_seq = "".join(random.choice("ATCG") for _ in range(max_new_tokens))  # noqa: S311 -- non-cryptographic, test mock
            sequences.append(prompt + new_seq if prepend_prompt else new_seq)
        for proposal, sequence in zip(self.segment.proposal_sequences, sequences, strict=True):
            proposal.sequence = sequence


class MockAutoregressiveGeneratorMissingSampleParams(Generator):
    """Mock whose _sample() omits the required max_new_tokens / old_kv_cache parameters."""

    input_type = GeneratorInputType.PROMPT

    def __init__(self):
        super().__init__()
        self.kv_caches: list[dict] = []

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        self._assigned_segments = (segments,) if isinstance(segments, Segment) else tuple(segments)

    def _sample(self, prompts: list[str] | None = None, prepend_prompt: bool | None = None) -> None:
        pass

    def release_kv_cache(self, cache: dict) -> None:
        pass


def _setup_beam_search(
    segment_length: int = 100,
    beam_length: int = 20,
    num_results: int = 3,
    proposals_per_result: int = 5,
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
    generator = mock_generator or MockAutoregressiveGenerator(use_kv_caching=use_kv_caching)
    generator._assigned_segments = (segment,)
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=gc_range[0], max_gc=gc_range[1]),
    )
    config = BeamSearchOptimizerConfig(
        prompt=prompt,
        beam_length=beam_length,
        num_results=num_results,
        proposals_per_result=proposals_per_result,
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
    def test_empty_prompt_fails(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(prompt="", num_results=3, proposals_per_result=5, beam_length=10)

    def test_invalid_num_results_fails(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(prompt="ATCG", num_results=0, proposals_per_result=5, beam_length=10)

    def test_invalid_score_by_fails(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(prompt="ATCG", num_results=5, proposals_per_result=10, score_by="invalid")

    # --- Initialization ---
    def test_initialization(self):
        optimizer, generator, _constraint, segment = _setup_beam_search()
        assert optimizer.target_segment == segment
        assert optimizer.generator == generator
        assert optimizer.num_results == 3
        assert optimizer._proposals_per_result == 5
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
        generator._assigned_segments = (segments[0],)
        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=10, num_results=3, proposals_per_result=5)
        # Should work when target_segment is specified
        optimizer = BeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            target_segment=segments[0],
        )
        assert optimizer.target_segment == segments[0]

    def test_non_target_constraint_input_rejected(self):
        """BeamSearch rejects constraints referencing only non-target segments."""
        target_segment = Segment(length=20, sequence_type="dna")
        context_segment = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        construct = Construct([target_segment, context_segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segments = (target_segment,)

        non_target_constraint = Constraint(
            inputs=[context_segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=10, num_results=2, proposals_per_result=3)

        with pytest.raises(ValueError, match="does not include the target segment"):
            BeamSearchOptimizer(
                target_segment=target_segment,
                constructs=[construct],
                generators=[generator],
                constraints=[non_target_constraint],
                config=config,
            )

    def test_duplicate_constraint_instance_fails(self):
        """Same constraint instance cannot be passed twice."""
        segment = Segment(length=20, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=10, num_results=2, proposals_per_result=3)

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
        generator._assigned_segments = (other_segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=10, num_results=3, proposals_per_result=5)
        with pytest.raises(ValueError, match="is not in any of the provided constructs"):
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
        generator._assigned_segments = (target_segment,)
        constraint = Constraint(
            inputs=[target_segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=10, num_results=3, proposals_per_result=5)

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
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=10, num_results=3, proposals_per_result=5)
        with pytest.raises(ValueError, match="not compatible with"):
            BeamSearchOptimizer(
                target_segment=segment,
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_generator_without_kv_caching_support_fails(self):
        """Generator missing kv_caches should fail when use_kv_caching=True."""
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGeneratorNoKVCache()
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=10,
            num_results=3,
            proposals_per_result=5,
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

    def test_generator_without_kv_cache_release_fails(self):
        """Generator missing release_kv_cache should fail when use_kv_caching=True."""
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGeneratorNoKVCacheRelease()
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=10,
            num_results=3,
            proposals_per_result=5,
            use_kv_caching=True,
        )
        with pytest.raises(ValueError, match="missing release_kv_cache method"):
            BeamSearchOptimizer(
                target_segment=segment,
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_generator_with_incompatible_sample_signature_fails(self):
        """_sample() missing max_new_tokens/old_kv_cache is rejected at construct time."""
        segment = Segment(length=100, sequence_type="dna")
        generator = MockAutoregressiveGeneratorMissingSampleParams()
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG", beam_length=10, num_results=3, proposals_per_result=5, use_kv_caching=False
        )
        with pytest.raises(ValueError, match=r"_sample\(\) missing required parameter\(s\)"):
            BeamSearchOptimizer(
                target_segment=segment,
                constructs=[Construct([segment])],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_generator_without_kv_caching_support_works_when_disabled(self):
        """Generator missing KV caching support should work when use_kv_caching=False."""
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGeneratorNoKVCache()
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=10,
            num_results=3,
            proposals_per_result=5,
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
        assert generator.cached_generation is True
        assert generator.store_kv_cache is False

    def test_beam_length_exceeds_segment_length_fails(self):
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator()
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=100, num_results=3, proposals_per_result=5)
        with pytest.raises(
            ValueError,
            match=r"beam_length.*cannot be greater than target_segment length",
        ):
            BeamSearchOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
                target_segment=segment,
            )

    def test_optimizer_initializes_beams_with_prompt(self):
        prompt = "ATCGATCG"
        optimizer, _, _, _ = _setup_beam_search(num_results=4, prompt=prompt)
        assert len(optimizer.beams) == 4
        for beam in optimizer.beams:
            assert beam.running_sequence == prompt

    def test_final_beam_generates_fewer_tokens(self):
        """Final beam generates remaining tokens when segment_length % beam_length != 0."""
        prompt = "ATCG"
        optimizer, _, _, segment = _setup_beam_search(segment_length=95, beam_length=20, prompt=prompt, num_results=2)
        optimizer.run()
        # 95 tokens total with prepend_prompt=True: len(prompt) + 95 = 99
        for seq in segment.result_sequences:
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
        assert optimizer._get_aggregated_score(beam) == float("inf")

    # --- Run ---
    def test_run_generates_correct_sequence_length(self):
        prompt = "ATCG"
        segment_length = 100
        optimizer, _, _, segment = _setup_beam_search(segment_length=segment_length, prompt=prompt)
        optimizer.run()
        expected_length = len(prompt) + segment_length
        for seq in segment.result_sequences:
            assert len(seq.sequence) == expected_length

    def test_beam_scores_accumulated(self):
        optimizer, _, _, _ = _setup_beam_search(segment_length=60, beam_length=20, num_results=2)
        optimizer.run()
        for beam in optimizer.beams:
            assert len(beam.beam_scores) == 3  # 60/20 = 3 beams

    # --- Prepend Prompt ---
    def test_prepend_prompt_true(self):
        prompt = "ATCGATCG"
        optimizer, _, _, segment = _setup_beam_search(prompt=prompt, segment_length=40, prepend_prompt=True)
        optimizer.run()
        for seq in segment.result_sequences:
            assert seq.sequence.startswith(prompt)
            assert len(seq.sequence) == len(prompt) + 40

    def test_prepend_prompt_false(self):
        optimizer, _, _, segment = _setup_beam_search(
            prompt="ATCGATCG", segment_length=40, beam_length=20, prepend_prompt=False
        )
        optimizer.run()
        for seq in segment.result_sequences:
            assert len(seq.sequence) == 40

    # --- KV Caching ---
    def test_kv_caching_disabled(self):
        optimizer, _, _, segment = _setup_beam_search(use_kv_caching=False, segment_length=60, beam_length=20)
        optimizer.run()
        assert len(segment.result_sequences) == optimizer.num_results
        for beam in optimizer.beams:
            assert beam.kv_cache is None

    def test_kv_cache_handles_continue_and_release(self):
        """Beam search should pass, prune, and release opaque KV cache handles."""

        def accept_all(input_sequences, config=None):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        accept_all._constraint_supported_sequence_types = ["dna"]
        accept_all._constraint_num_input_sequences_per_tuple = 1

        segment = Segment(length=20, sequence_type="dna")
        construct = Construct([segment])
        generator = TrackingKVCacheGenerator()
        generator.assign(segment)
        constraint = Constraint(inputs=[segment], function=accept_all, function_config={})
        optimizer = BeamSearchOptimizer(
            target_segment=segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=BeamSearchOptimizerConfig(
                prompt="ATCG",
                beam_length=10,
                num_results=1,
                proposals_per_result=4,
                use_kv_caching=True,
            ),
        )

        optimizer.run()

        assert generator.sample_old_kv_caches == [None, None, "cache-1-0", "cache-1-0"]
        assert set(generator.released_kv_caches) == set(generator.generated_kv_caches)
        assert len(generator.released_kv_caches) == len(generator.generated_kv_caches)
        assert all(beam.kv_cache is None for beam in optimizer.beams)

    def test_resample_failure_releases_accepted_cache_handles(self):
        """Accepted proposal caches are released when resampling cannot fill a beam."""

        def accept_first(input_sequences, config=None):
            return [ConstraintOutput(score=0.0 if idx == 0 else float("inf")) for idx, _ in enumerate(input_sequences)]

        accept_first._constraint_supported_sequence_types = ["dna"]
        accept_first._constraint_num_input_sequences_per_tuple = 1

        segment = Segment(length=10, sequence_type="dna")
        construct = Construct([segment])
        generator = TrackingKVCacheGenerator()
        generator.assign(segment)
        constraint = Constraint(inputs=[segment], function=accept_first, function_config={}, threshold=0.0)
        optimizer = BeamSearchOptimizer(
            target_segment=segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=BeamSearchOptimizerConfig(
                prompt="ATCG",
                beam_length=10,
                num_results=1,
                proposals_per_result=4,
                max_resample_attempts=1,
                use_kv_caching=True,
            ),
        )

        with pytest.raises(RuntimeError, match="could not produce"):
            optimizer.run()

        assert set(generator.released_kv_caches) == set(generator.generated_kv_caches)

    # --- Batch Size ---
    def test_batch_size_smaller_than_proposals(self):
        mock_gen = MockAutoregressiveGenerator(use_kv_caching=True)
        mock_gen.batch_size = 2
        optimizer, _, _, segment = _setup_beam_search(
            segment_length=40,
            beam_length=20,
            num_results=2,
            proposals_per_result=6,
            mock_generator=mock_gen,
        )
        assert optimizer.generator.batch_size == 2
        optimizer.run()
        assert len(segment.result_sequences) == optimizer.num_results

    # --- Resampling ---
    def test_all_invalid_raises_error(self):
        segment = Segment(length=20, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segments = (segment,)
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
            proposals_per_result=3,
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
        with pytest.raises(RuntimeError, match=r"could not produce.*valid proposals"):
            optimizer.run()

    def test_warning_when_previous_results_discarded(self, caplog):
        """BeamSearch warns when existing sequences will be overwritten."""
        import logging

        optimizer, _, _, segment = _setup_beam_search(segment_length=40, beam_length=20, num_results=2, prompt="ATCG")
        # Simulate previous optimizer results
        segment.result_sequences[0].sequence = "GCTAGCTA"
        segment.result_sequences[1].sequence = "TTTTTTTT"

        with caplog.at_level(logging.WARNING):
            optimizer.run()

        assert any("overwrites existing sequences" in msg for msg in caplog.messages)


@pytest.mark.uses_gpu
@pytest.mark.slow
class TestBeamSearchOptimizerGPU:
    """GPU-dependent integration tests."""

    def test_with_evo2_generator(self):
        from proto_language.generator import (
            Evo2Generator,
            Evo2GeneratorConfig,
        )

        prompt = "ATCGATCGATCG"
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        gen_config = Evo2GeneratorConfig(prompts=[prompt], prepend_prompt=True, stop_at_eos=False)
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
            proposals_per_result=5,
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
        assert len(segment.result_sequences) == 3
        for seq in segment.result_sequences:
            assert len(seq.sequence) == len(prompt) + 100

    def test_with_evo1_generator(self):
        """Beam search runs end-to-end with Evo1 and use_kv_caching=False."""
        from proto_language.generator import Evo1Generator, Evo1GeneratorConfig

        prompt = "ATCGATCGATCG"
        segment = Segment(length=80, sequence_type="dna")
        gen = Evo1Generator(Evo1GeneratorConfig(prompts=[prompt], prepend_prompt=True))
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=30.0, max_gc=70.0),
        )
        optimizer = BeamSearchOptimizer(
            target_segment=segment,
            constructs=[Construct([segment])],
            generators=[gen],
            constraints=[constraint],
            config=BeamSearchOptimizerConfig(
                prompt=prompt, beam_length=40, num_results=2, proposals_per_result=3, use_kv_caching=False
            ),
        )
        optimizer.run()
        assert len(segment.result_sequences) == 2
        assert all(len(s.sequence) == len(prompt) + 80 for s in segment.result_sequences)

    def test_with_progen2_generator(self):
        """Beam search runs end-to-end with ProGen2 and use_kv_caching=False."""
        from proto_language.generator import ProGen2Generator, ProGen2GeneratorConfig

        prompt = "1MKTL"
        segment = Segment(length=40, sequence_type="protein")
        gen = ProGen2Generator(
            ProGen2GeneratorConfig(
                prompts=[prompt], prepend_prompt=True, truncate_at_stop=False, strip_special_tokens=False
            )
        )
        gen.assign(segment)

        def zero_score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        constraint = Constraint(inputs=[segment], function=zero_score, function_config=None)
        optimizer = BeamSearchOptimizer(
            target_segment=segment,
            constructs=[Construct([segment])],
            generators=[gen],
            constraints=[constraint],
            config=BeamSearchOptimizerConfig(
                prompt=prompt, beam_length=20, num_results=2, proposals_per_result=3, use_kv_caching=False
            ),
        )
        optimizer.run()
        assert len(segment.result_sequences) == 2


class TestBeamSearchOptimizerRestart:
    """Tests for BeamSearchOptimizer state restart behavior."""

    def test_run_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        prompt = "ATCG"
        optimizer, _generator, _constraint, segment = _setup_beam_search(
            prompt=prompt,
            num_results=2,
            proposals_per_result=2,
            segment_length=40,
            beam_length=20,
        )

        # Capture original state before run
        original_result = [copy.deepcopy(s) for s in segment.result_sequences]

        # First run
        optimizer.run()
        assert optimizer._initial_state is not None
        first_run_beams = [b.running_sequence for b in optimizer.beams]
        # Verify beams grew beyond prompt
        assert all(len(b) > len(prompt) for b in first_run_beams)

        # Verify captured state contains original sequences (using index 0)
        assert len(optimizer._initial_state["segments"]) == 1
        captured_result = optimizer._initial_state["segments"][0]["result"]

        # Verify captured sequences match originals
        assert len(captured_result) == len(original_result)
        for orig, captured in zip(original_result, captured_result, strict=False):
            assert orig.sequence == captured["sequence"]

        # Manually modify sequences to invalid values to verify restore
        segment.result_sequences[0].sequence = "G" * 44  # prompt (4) + segment_length (40)
        segment.proposal_sequences[0].sequence = "G" * 44

        # Second run should restart - beams should be reset to prompt
        optimizer.run()
        second_run_beams = [b.running_sequence for b in optimizer.beams]
        # Verify beams grew again (optimization ran)
        assert all(len(b) > len(prompt) for b in second_run_beams)

        # Verify sequences were restored (not all G's - restoration happened)
        assert any(seq.sequence != "G" * 44 for seq in segment.result_sequences)

        # History should be fresh (cleared on restart)
        # per-beam snapshots only (no t=0 for BeamSearch)
        assert len(optimizer.history) == 2

    def test_beams_reset_on_restore(self):
        """Test that beams are reset to initial prompt on restore and sequences are restored."""
        prompt = "ATCGATCG"
        optimizer, _, _, segment = _setup_beam_search(
            prompt=prompt,
            num_results=3,
            proposals_per_result=2,
            segment_length=40,
            beam_length=20,
        )

        # Capture original sequences
        original_result = [s.sequence for s in segment.result_sequences]

        # First run - beams will be modified
        optimizer.run()

        # Capture modified beams
        modified_beams = [b.running_sequence for b in optimizer.beams]
        assert all(len(b) > len(prompt) for b in modified_beams)

        # Manually modify sequences to invalid values
        for seq in segment.result_sequences:
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
        restored_sequences = [s.sequence for s in segment.result_sequences]
        assert len(restored_sequences) == len(original_result)
        for orig, restored in zip(original_result, restored_sequences, strict=False):
            assert orig == restored


class TestBeamSearchProposalTracking:
    """Test proposal_results tracking in BeamSearch history."""

    def test_proposal_tracking(self):
        """History has proposal_results with 'Beam pruned' for rejected proposals."""
        optimizer, _, _, _ = _setup_beam_search(num_results=3, proposals_per_result=2, beam_length=10)
        optimizer.track_proposals = True
        optimizer.run()

        valid_rejectors = {"Beam pruned"}
        all_rejectors = set()
        for entry in optimizer.history:
            if "proposal_results" not in entry:
                continue
            for cand in entry["proposal_results"]:
                assert isinstance(cand["accepted"], bool)
                if cand["accepted"]:
                    assert cand["rejected_by"] is None
                else:
                    all_rejectors.add(cand["rejected_by"])

        assert all_rejectors.issubset(valid_rejectors)

    def test_snapshots_include_beam_search_summary(self):
        optimizer, _, _, _ = _setup_beam_search(num_results=2, proposals_per_result=2, beam_length=10)
        optimizer.track_proposals = True
        optimizer.run()

        snapshot = optimizer.history[-1]
        summary = snapshot["optimizer"]

        assert summary["type"] == "beam-search"
        assert snapshot["time_step"] == optimizer.num_beams
        assert summary["beam_width"] == 2
        assert summary["proposals_per_beam"] == 2
        assert summary["proposal_count"] == len(snapshot["proposal_results"])
        assert summary["accepted_proposal_count"] == 2
        assert summary["best_energy"] == min(result["energy_score"] for result in snapshot["results"])


class TestBeamSearchNonTargetSegmentSync:
    """Tests that BeamSearch syncs non-target segment pools when resizing target.

    BeamSearch dynamically resizes target_segment.proposal_sequences during its
    run (to batch_count, N*K, etc.). _sync_proposal_pools ensures non-target
    segments stay in sync so Constraint.evaluate() sees equal pool sizes.
    """

    def test_only_target_segment_is_generated(self):
        target_segment = Segment(length=40, sequence_type="dna")
        context_segment = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        other_segment = Segment(sequence="GCGCGCGCGC", sequence_type="dna")
        construct = Construct([target_segment, context_segment])
        other_construct = Construct([other_segment])
        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segments = (target_segment,)
        constraint = Constraint(
            inputs=[target_segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        optimizer = BeamSearchOptimizer(
            target_segment=target_segment,
            constructs=[construct, other_construct],
            generators=[generator],
            constraints=[constraint],
            config=BeamSearchOptimizerConfig(
                prompt="ATCG",
                beam_length=20,
                num_results=2,
                proposals_per_result=3,
                use_kv_caching=False,
            ),
        )

        optimizer.run()

        assert len(target_segment.result_sequences) == 2
        assert context_segment.original_sequence.sequence == "ATCGATCGATCGATCGATCG"
        assert other_segment.original_sequence.sequence == "GCGCGCGCGC"

    def test_multi_segment_constraint(self):
        """Constraint reading from both target and non-target segments runs without errors."""
        target_segment = Segment(length=40, sequence_type="dna")
        context_segment = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        construct = Construct([target_segment, context_segment])

        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segments = (target_segment,)

        # Custom scoring function that reads from both segments
        def multi_seg_score(input_sequences, config=None):
            results = []
            for target_seq, context_seq in input_sequences:
                gc_target = sum(1 for c in target_seq.sequence if c in "GC") / max(len(target_seq.sequence), 1)
                gc_context = sum(1 for c in context_seq.sequence if c in "GC") / max(len(context_seq.sequence), 1)
                results.append(ConstraintOutput(score=abs(gc_target - gc_context)))
            return results

        multi_seg_score._constraint_supported_sequence_types = ["dna"]
        multi_seg_score._constraint_num_input_sequences_per_tuple = 2

        cross_segment_constraint = Constraint(
            inputs=[target_segment, context_segment],
            function=multi_seg_score,
            function_config={},
        )

        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=20,
            num_results=2,
            proposals_per_result=3,
            use_kv_caching=False,
        )

        optimizer = BeamSearchOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[cross_segment_constraint],
            config=config,
        )

        # Should run without pool-size mismatch errors
        optimizer.run()

        assert len(target_segment.result_sequences) == 2


class TestBeamSearchTrackingInterval:
    """Test tracking_interval in BeamSearchOptimizer."""

    def test_tracking_interval(self):
        """tracking_interval=2 reduces history snapshots."""
        segment = Segment(length=100, sequence_type="dna")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(use_kv_caching=False)
        generator._assigned_segments = (segment,)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        config = BeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_length=20,
            num_results=3,
            proposals_per_result=5,
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
