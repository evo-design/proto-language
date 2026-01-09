from __future__ import annotations
import pytest
import time
import random
import math
from typing import Tuple, List, Dict, Optional
from unittest.mock import Mock

from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    Generator,
    Sequence,
)
from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import SequenceLengthConfig
from proto_language.language.optimizer import (
    MultiSegmentBeamSearchOptimizer,
    MultiSegmentBeamSearchOptimizerConfig,
)


# Mock Autoregressive Generator for testing
class MockAutoregressiveGenerator(Generator):
    """
    Mock autoregressive generator for testing MultiSegmentBeamSearchOptimizer without GPU.

    Generates random DNA sequences and optionally maintains mock KV caches.
    """
    def __init__(self, num_tokens: int = 20, use_kv_caching: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.use_kv_caching = use_kv_caching
        self.kv_caches: List[Dict] = []

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a segment to this generator."""
        self._assigned_segment = assigned_segment

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prepend_prompt: Optional[bool] = None,
        old_kv_cache: Optional[Dict] = None,
    ) -> None:
        """
        Generate mock DNA sequences.

        Args:
            prompts: List of prompt strings to condition generation on
            prepend_prompt: Whether to prepend the prompt to the generated sequence
            old_kv_cache: Previous KV cache to continue from
        """
        if prompts is None:
            prompts = [""]

        num_samples = len(prompts)

        # Generate random DNA sequences
        sequences = []
        for prompt in prompts:
            # Generate random DNA sequence
            bases = "ATCG"
            new_seq = ''.join(random.choice(bases) for _ in range(self.num_tokens))

            # Prepend prompt if requested
            if prepend_prompt:
                new_seq = prompt + new_seq

            sequences.append(new_seq)

        # Set candidate sequences on assigned segment
        self._assigned_segment.candidate_sequences = [
            Sequence(sequence=seq, sequence_type="dna")
            for seq in sequences
        ]

        # Generate mock KV caches if caching is enabled
        # Check both use_kv_caching and the store_kv_cache attribute (set by MultiSegmentBeamSearchOptimizer)
        if self.use_kv_caching and getattr(self, 'store_kv_cache', False):
            self.kv_caches = [create_mock_kv_cache() for _ in range(num_samples)]
        else:
            self.kv_caches = []

    def replicate_cache(self, cache: Dict, n_replicates: int) -> Dict:
        """Replicate cache N times for beam branching."""
        return cache


# Mock Mutation Generator for testing non-autoregressive rejection
class MockMutationGenerator(Generator):
    """Mock mutation generator for testing non-autoregressive rejection."""
    def __init__(self, num_tokens: int = 20):
        super().__init__()
        self.num_tokens = num_tokens
        self.kv_caches: List[Dict] = []

    def assign(self, assigned_segment: Segment) -> None:
        self._assigned_segment = assigned_segment

    def sample(self, prompts=None, prepend_prompt=None, old_kv_cache=None) -> None:
        pass

    def replicate_cache(self, cache: Dict, n_replicates: int) -> Dict:
        return cache


def _setup_multi_segment_beam_search_components(
    num_segments: int = 3,
    seq_length: int = 20,
    beam_width: int = 3,
    candidates_per_beam: int = 5,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    use_kv_caching: bool = True,
    prompt: str = "ATCG",
):
    """Helper function to set up a basic MultiSegmentBeamSearchOptimizer for testing."""
    # 1. Create segments
    segments = [Segment(length=seq_length, sequence_type="dna") for _ in range(num_segments)]
    construct = Construct(segments)

    # 2. Create the mock generator
    generator = MockAutoregressiveGenerator(
        num_tokens=seq_length,
        use_kv_caching=use_kv_caching,
    )

    # MultiSegmentBeamSearchOptimizer requires generator to be "assigned" for validation
    # but handles assignment internally during run(), so we assign to a dummy segment
    generator._assigned_segment = segments[0]

    # 3. Create constraints (only on the first segment for simplicity)
    constraint = Constraint(
        inputs=[segments[0]],
        function=gc_content_constraint,
        function_config=GCContentConfig(
            min_gc=gc_target_range[0],
            max_gc=gc_target_range[1],
        ),
    )

    # 4. Create the MultiSegmentBeamSearchOptimizer config
    config = MultiSegmentBeamSearchOptimizerConfig(
        prompt=prompt,
        beam_width=beam_width,
        candidates_per_beam=candidates_per_beam,
        use_kv_caching=use_kv_caching,
        verbose=False,
    )

    optimizer = MultiSegmentBeamSearchOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=[constraint],
        config=config,
    )

    return optimizer, generator, constraint, segments


def create_mock_kv_cache():
    """Create a mock KV cache structure for testing without GPU dependencies."""
    # Create mock objects that simulate the cache structure without importing vortex
    # This allows CPU tests to run without GPU dependencies

    # Mock InferenceParams structure
    mock_mha = Mock()
    mock_mha.max_seqlen = 512
    mock_mha.max_batch_size = 8
    mock_mha.seqlen_offset = 10
    mock_mha.batch_size_offset = 0
    mock_mha.key_value_memory_dict = {0: Mock()}  # Mock tensor

    # Mock HyenaCascadeIIRInferenceParams structure
    mock_hcl = Mock()
    mock_hcl.fir_filter_length = 4
    mock_hcl.state_dim = 8
    mock_hcl.seqlen_offset = 10
    mock_hcl.fir_state_dict = {0: Mock()}  # Mock tensor
    mock_hcl.state_dict = {0: Mock()}  # Mock tensor

    # Mock HyenaCascadeFIRInferenceParams structure (used for both hcm and hcs)
    mock_hcm = Mock()
    mock_hcm.fir_filter_length = 4
    mock_hcm.seqlen_offset = 10
    mock_hcm.fir_inner_filter_length = 2
    mock_hcm.fir_state_dict = {0: Mock()}  # Mock tensor
    mock_hcm.fir_inner_state_dict = {0: Mock()}  # Mock tensor
    mock_hcm.state_dict = {0: Mock()}  # Mock tensor

    mock_hcs = Mock()
    mock_hcs.fir_filter_length = 4
    mock_hcs.seqlen_offset = 10
    mock_hcs.fir_inner_filter_length = 2
    mock_hcs.fir_state_dict = {0: Mock()}  # Mock tensor
    mock_hcs.fir_inner_state_dict = {0: Mock()}  # Mock tensor
    mock_hcs.state_dict = {0: Mock()}  # Mock tensor

    return {
        'mha': mock_mha,
        'hcl': mock_hcl,
        'hcm': mock_hcm,
        'hcs': mock_hcs,
    }


def create_real_kv_cache():
    """Create a real KV cache structure for GPU testing with actual vortex objects."""
    import torch
    from vortex.model.cache import InferenceParams, HyenaCascadeIIRInferenceParams, HyenaCascadeFIRInferenceParams

    # Create minimal cache data with real torch tensors
    batch_size = 1
    mock_kv = torch.randn(batch_size, 2, 4, 8, 16)  # [batch, num_heads, seq_len, head_dim, features]
    mock_fir_state = torch.randn(batch_size, 8, 16)
    mock_state = torch.randn(batch_size, 8, 16)

    return {
        'mha': InferenceParams(
            max_seqlen=512,
            max_batch_size=8,
            seqlen_offset=10,
            batch_size_offset=0,
            key_value_memory_dict={0: mock_kv.clone()},
        ),
        'hcl': HyenaCascadeIIRInferenceParams(
            fir_filter_length=4,
            state_dim=8,
            seqlen_offset=10,
            fir_state_dict={0: mock_fir_state.clone()},
            state_dict={0: mock_state.clone()},
        ),
        'hcm': HyenaCascadeFIRInferenceParams(
            fir_filter_length=4,
            seqlen_offset=10,
            fir_inner_filter_length=2,
            fir_state_dict={0: mock_fir_state.clone()},
            fir_inner_state_dict={0: mock_fir_state.clone()},
            state_dict={0: mock_state.clone()},
        ),
        'hcs': HyenaCascadeFIRInferenceParams(
            fir_filter_length=4,
            seqlen_offset=10,
            fir_inner_filter_length=2,
            fir_state_dict={0: mock_fir_state.clone()},
            fir_inner_state_dict={0: mock_fir_state.clone()},
            state_dict={0: mock_state.clone()},
        ),
    }


class TestMultiSegmentBeamSearchOptimizer:
    def test_initialization_and_validation(self):
        """Tests successful initialization and validation of MultiSegmentBeamSearchOptimizer."""
        optimizer, generator, constraint, segments = _setup_multi_segment_beam_search_components()

        assert list(optimizer.construct.segments) == segments
        assert optimizer.generator == generator
        assert optimizer.constraints == [constraint]
        assert optimizer.constraint_weights == [1.0]
        assert optimizer.beam_width == 3
        assert optimizer.candidates_per_beam == 5
        assert optimizer.use_kv_caching is True
        assert len(optimizer.running_prompts) == optimizer.beam_width
        assert len(optimizer.top_beam_kv_caches) == optimizer.beam_width
        assert all(cache is None for cache in optimizer.top_beam_kv_caches)

    def test_initialization_with_non_autoregressive_generator(self):
        """Tests that non-autoregressive generators raise an error."""
        segments = [Segment(length=20, sequence_type="dna") for _ in range(3)]
        construct = Construct(segments)

        # Create a non-autoregressive (mutation) generator 
        generator = MockMutationGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
            prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
        )

        with pytest.raises(ValueError, match="requires autoregressive generators"):
            MultiSegmentBeamSearchOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_initialization_with_multiple_constructs(self):
        """Tests that MultiSegmentBeamSearchOptimizer rejects multiple constructs."""
        segments1 = [Segment(length=20, sequence_type="dna") for _ in range(2)]
        segments2 = [Segment(length=20, sequence_type="dna") for _ in range(2)]
        construct1 = Construct(segments1)
        construct2 = Construct(segments2)

        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments1[0]

        constraint = Constraint(
            inputs=[segments1[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
        )

        with pytest.raises(ValueError, match="only supports a single construct"):
            MultiSegmentBeamSearchOptimizer(
                constructs=[construct1, construct2],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_initialization_with_multiple_generators(self):
        """Tests that MultiSegmentBeamSearchOptimizer rejects multiple generators."""
        segments = [Segment(length=20, sequence_type="dna") for _ in range(2)]
        construct = Construct(segments)

        generator1 = MockAutoregressiveGenerator(num_tokens=20)
        generator1._assigned_segment = segments[0]
        generator2 = MockAutoregressiveGenerator(num_tokens=20)
        generator2._assigned_segment = segments[0]

        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
        )

        with pytest.raises(ValueError, match="only supports a single generator"):
            MultiSegmentBeamSearchOptimizer(
                constructs=[construct],
                generators=[generator1, generator2],
                constraints=[constraint],
                config=config,
            )

    def test_initialization_with_existing_candidates_warning(self):
        """Tests that a warning is raised if segments have existing candidates."""
        segments = [Segment(sequence="ATCG", sequence_type="dna") for _ in range(2)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
        )

        with pytest.warns(UserWarning, match="will overwrite"):
            MultiSegmentBeamSearchOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

    def test_config_validation(self):
        """Tests MultiSegmentBeamSearchOptimizerConfig validation."""
        from pydantic import ValidationError

        # Valid configs
        config = MultiSegmentBeamSearchOptimizerConfig(prompt="ATCG", beam_width=5, candidates_per_beam=10)
        assert config.prompt == "ATCG"
        assert config.beam_width == 5
        assert config.candidates_per_beam == 10
        assert config.use_kv_caching is True
        assert config.verbose is False

        # Missing prompt should fail
        with pytest.raises(ValidationError):
            MultiSegmentBeamSearchOptimizerConfig(beam_width=5, candidates_per_beam=10)

        # Negative beam_width should fail
        with pytest.raises(ValidationError):
            MultiSegmentBeamSearchOptimizerConfig(prompt="ATCG", beam_width=0, candidates_per_beam=5)

        # Negative candidates_per_beam should fail
        with pytest.raises(ValidationError):
            MultiSegmentBeamSearchOptimizerConfig(prompt="ATCG", beam_width=5, candidates_per_beam=0)

    def test_constraint_weights(self):
        """Tests that constraint weights are properly handled."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components()

        # Add a second constraint
        constraint2 = Constraint(
            inputs=[segments[0]],
            function=sequence_length_constraint,
            function_config=SequenceLengthConfig(target_length=20),
            weight=2.,
        )

        # Recreate optimizer with custom weights
        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
        )

        construct = Construct(segments)
        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[optimizer.constraints[0], constraint2],
            config=config,
        )

        assert optimizer.constraint_weights == [1.0, 2.0]

    @pytest.mark.uses_gpu
    @pytest.mark.slow
    def test_replicate_cache(self):
        """Tests the replicate_cache method."""
        from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig

        segments = [Segment(length=20, sequence_type="dna") for _ in range(3)]
        construct = Construct(segments)

        gen_config = Evo2GeneratorConfig(prompts=[""], prepend_prompt=False)
        generator = Evo2Generator(config=gen_config)
        generator.assign(segments[0])

        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),

        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
            use_kv_caching=True,
            verbose=False,
        )

        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
        )

        # Create a real cache for GPU testing
        mock_cache = create_real_kv_cache()

        # Test replication
        n_replicates = 5
        replicated_cache = optimizer.generator.replicate_cache(mock_cache, n_replicates)

        # Check that all components are replicated correctly
        assert replicated_cache is not None
        assert 'mha' in replicated_cache
        assert 'hcl' in replicated_cache
        assert 'hcm' in replicated_cache
        assert 'hcs' in replicated_cache

        # Check that batch dimension is replicated
        for key, data in replicated_cache['mha'].key_value_memory_dict.items():
            assert data.shape[0] == n_replicates

        for key, data in replicated_cache['hcl'].fir_state_dict.items():
            assert data.shape[0] == n_replicates

        # Test with empty cache
        empty_replicated = optimizer.generator.replicate_cache(None, n_replicates)
        assert empty_replicated is None

        # Test with invalid n_replicates
        with pytest.raises(ValueError, match="must be at least 1"):
            optimizer.generator.replicate_cache(mock_cache, 0)

    @pytest.mark.uses_gpu
    @pytest.mark.slow
    def test_replicate_cache_validates_batch_size(self):
        """Tests that replicate_cache validates cache has batch size 1."""
        from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig

        segments = [Segment(length=20, sequence_type="dna") for _ in range(3)]
        construct = Construct(segments)

        gen_config = Evo2GeneratorConfig(prompts=[""], prepend_prompt=False)
        generator = Evo2Generator(config=gen_config)
        generator.assign(segments[0])

        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),

        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
            use_kv_caching=True,
            verbose=False,
        )

        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
        )

        # Create a cache with batch size > 1
        import torch
        from vortex.model.cache import InferenceParams

        batch_size = 3
        mock_kv = torch.randn(batch_size, 2, 4, 8, 16)

        invalid_cache = {
            'mha': InferenceParams(
                max_seqlen=512,
                max_batch_size=8,
                seqlen_offset=10,
                batch_size_offset=0,
                key_value_memory_dict={0: mock_kv},
            ),
            'hcl': Mock(),
            'hcm': Mock(),
            'hcs': Mock(),
        }

        with pytest.raises(ValueError, match="must only have one cache entry"):
            optimizer.generator.replicate_cache(invalid_cache, 5)

    def test_score_energy_active_constraints_single_segment(self):
        """Tests _score_energy_active_constraints with a single segment."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(num_segments=1)

        # Manually set candidates for the segment
        segment = segments[0]
        segment.candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna"),  # 50% GC
            Sequence(sequence="AAAAAAAAAAAAAAAAAAAA", sequence_type="dna"),  # 0% GC
            Sequence(sequence="GCGCGCGCGCGCGCGCGCGC", sequence_type="dna"),  # 100% GC
        ]

        optimizer.num_candidates = len(segment.candidate_sequences)
        optimizer._score_energy_active_constraints()

        assert len(optimizer.energy_scores) == 3
        # First sequence (50% GC) should have lowest energy (within target range)
        assert optimizer.energy_scores[0] == 0.0
        # Other sequences should have higher energy
        assert optimizer.energy_scores[1] > 0.0
        assert optimizer.energy_scores[2] > 0.0

    def test_score_energy_active_constraints_multi_segment(self):
        """Tests _score_energy_active_constraints with multi-segment constraints."""
        segments = [Segment(length=20, sequence_type="dna") for _ in range(3)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        # Create constraint that depends on segments 0 and 1
        constraint_01 = Constraint(
            inputs=[segments[0], segments[1]],
            function=sequence_length_constraint,
            function_config=SequenceLengthConfig(target_length=40),

        )

        # Create constraint that only depends on segment 0
        constraint_0 = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=2,
            candidates_per_beam=3,
        )

        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint_01, constraint_0],
            config=config,
        )

        # Initially, only segment 0 has candidates
        segments[0].candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna"),  # 50% GC
            Sequence(sequence="AAAAAAAAAAAAAAAAAAAA", sequence_type="dna"),  # 0% GC
        ]
        optimizer.num_candidates = 2

        # Only constraint_0 should be active
        optimizer._score_energy_active_constraints()

        # Should score successfully with only the active constraint
        assert len(optimizer.energy_scores) == 2

        # Now add candidates to segment 1
        segments[1].candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna"),
            Sequence(sequence="AAAAAAAAAAAAAAAAAAAA", sequence_type="dna"),
        ]

        # Now both constraints should be active
        optimizer._score_energy_active_constraints()
        assert len(optimizer.energy_scores) == 2

    def test_score_energy_active_constraints_no_active(self):
        """Tests _score_energy_active_constraints when no constraints are active."""
        segments = [Segment(length=20, sequence_type="dna") for _ in range(2)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        # Create constraint that depends on both segments
        constraint = Constraint(
            inputs=[segments[0], segments[1]],
            function=sequence_length_constraint,
            function_config=SequenceLengthConfig(target_length=40),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=2,
            candidates_per_beam=3,
        )

        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
        )

        # Only segment 0 has candidates, so constraint is not active
        segments[0].candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna"),
        ]
        # Clear segment 1 candidates to ensure constraint is not active
        segments[1].candidate_sequences = []
        optimizer.num_candidates = 1

        optimizer._score_energy_active_constraints()

        # All energy scores should be 0 since no constraints are active
        assert optimizer.energy_scores == [0.0]

    def test_select_topk(self):
        """Tests the _select_topk method."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            beam_width=3,
            candidates_per_beam=4,
            num_segments=1,
            use_kv_caching=False,  # Disable KV caching for this test
        )

        segment = segments[0]

        # Create 12 candidates (3 beams * 4 candidates per beam)
        num_candidates = optimizer.beam_width * optimizer.candidates_per_beam
        segment.candidate_sequences = [
            Sequence(sequence="A" * 20, sequence_type="dna") for _ in range(num_candidates)
        ]

        # Set up energy scores (lower is better)
        optimizer.energy_scores = [
            0.9, 0.1, 0.8, 0.2,  # Beam 0: best is idx 1
            0.7, 0.3, 0.6, 0.4,  # Beam 1: best is idx 5
            0.5, 0.05, 0.95, 0.15,  # Beam 2: best is idx 9
        ]

        # Set initial prompts (valid DNA sequences)
        optimizer.running_prompts = ["ATCG", "GCTA", "CGAT"]

        # Call _select_topk with empty KV caches (caching disabled)
        all_kv_caches = []
        top_idx = optimizer._select_topk(segment, all_kv_caches)

        # Check that we selected the top beam_width candidates
        assert len(top_idx) == optimizer.beam_width
        expected_top_idx = [9, 1, 11]  # Indices with energies [0.05, 0.1, 0.15]
        assert top_idx == expected_top_idx

        # Check that selected_sequences are set correctly
        assert len(segment.selected_sequences) == optimizer.beam_width
        # All sequences are the same in this test, so just check they exist
        assert all(len(seq.sequence) == 20 for seq in segment.selected_sequences)

        # Check that candidate_sequences are replicated
        expected_num_candidates = optimizer.beam_width * optimizer.candidates_per_beam
        assert len(segment.candidate_sequences) == expected_num_candidates

        # Check that running prompts are updated correctly
        # idx 9 came from beam 2 (9 // 4 = 2), so new prompt = "CGAT" + new_seq
        # idx 1 came from beam 0 (1 // 4 = 0), so new prompt = "ATCG" + new_seq
        # idx 11 came from beam 2 (11 // 4 = 2), so new prompt = "CGAT" + new_seq
        assert optimizer.running_prompts[0] == "CGAT" + "A" * 20
        assert optimizer.running_prompts[1] == "ATCG" + "A" * 20
        assert optimizer.running_prompts[2] == "CGAT" + "A" * 20

        # Check that energy_scores are replicated correctly
        # Should have energies [0.05, 0.1, 0.15] each replicated 4 times
        assert len(optimizer.energy_scores) == expected_num_candidates
        expected_energies = [0.05] * 4 + [0.1] * 4 + [0.15] * 4
        for expected, actual in zip(expected_energies, optimizer.energy_scores):
            assert abs(expected - actual) < 1e-6

    def test_select_topk_with_kv_caches(self):
        """Tests _select_topk with KV caches."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            beam_width=3,
            candidates_per_beam=2,
            use_kv_caching=True,
        )

        segment = segments[0]
        num_candidates = optimizer.beam_width * optimizer.candidates_per_beam
        segment.candidate_sequences = [
            Sequence(sequence="ATCGATCG", sequence_type="dna") for _ in range(num_candidates)
        ]

        optimizer.energy_scores = [0.5, 0.1, 0.3, 0.2, 0.4, 0.6]
        optimizer.running_prompts = ["ATCG", "GCTA", "CGAT"]

        # Create mock KV caches for all candidates
        all_kv_caches = [create_mock_kv_cache() for _ in range(num_candidates)]

        top_idx = optimizer._select_topk(segment, all_kv_caches)

        # Top 3: indices [1, 3, 2] with energies [0.1, 0.2, 0.3]
        assert len(top_idx) == 3
        assert top_idx == [1, 3, 2]

        # Check that KV caches are updated
        assert len(optimizer.top_beam_kv_caches) == optimizer.beam_width
        assert optimizer.top_beam_kv_caches[0] == all_kv_caches[1]
        assert optimizer.top_beam_kv_caches[1] == all_kv_caches[3]
        assert optimizer.top_beam_kv_caches[2] == all_kv_caches[2]

    def test_generate_candidates_for_single_beam(self):
        """Tests _generate_candidates_for_beam with and without KV caching."""
        # Test without KV caching
        optimizer, generator, _, segments = _setup_multi_segment_beam_search_components(
            beam_width=2,
            candidates_per_beam=3,
            num_segments=1,
            use_kv_caching=False,
        )

        segment = segments[0]

        # Mock the generator's sample method
        def mock_sample(prompts=None, prepend_prompt=None, old_kv_cache=None):
            segment.candidate_sequences = [
                Sequence(sequence="ATCGATCGATCGATCG", sequence_type="dna") for _ in range(len(prompts))
            ]
            generator.kv_caches = []

        generator.sample = mock_sample
        optimizer.running_prompts = ["ATCG", "GCTA"]

        # Generate for beam 0
        sequences, kv_caches = optimizer._generate_candidates_for_beam(segment, beam_idx=0)

        # Should have generated candidates_per_beam sequences
        assert len(sequences) == optimizer.candidates_per_beam
        # KV caches should be empty when caching is disabled
        assert len(kv_caches) == 0

        # Test with KV caching
        optimizer_cached, generator_cached, _, segments_cached = _setup_multi_segment_beam_search_components(
            beam_width=2,
            candidates_per_beam=3,
            num_segments=1,
            use_kv_caching=True,
        )

        segment_cached = segments_cached[0]
        mock_cache = create_mock_kv_cache()
        optimizer_cached.top_beam_kv_caches = [mock_cache, None]

        def mock_sample_cached(prompts=None, prepend_prompt=None, old_kv_cache=None):
            segment_cached.candidate_sequences = [
                Sequence(sequence="ATCGATCGATCG", sequence_type="dna") for _ in range(len(prompts))
            ]
            generator_cached.kv_caches = [create_mock_kv_cache() for _ in range(len(prompts))]

        generator_cached.sample = mock_sample_cached
        optimizer_cached.running_prompts = ["ATCG", "GCTA"]

        sequences_cached, kv_caches_cached = optimizer_cached._generate_candidates_for_beam(segment_cached, beam_idx=0)

        # Should have generated sequences and KV caches
        assert len(sequences_cached) == optimizer_cached.candidates_per_beam
        assert len(kv_caches_cached) == optimizer_cached.candidates_per_beam

    def test_run_single_segment(self):
        """Tests the run method with a single segment."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            num_segments=1,
            seq_length=30,
            beam_width=2,
            candidates_per_beam=3,
        )

        # Run the optimizer
        optimizer.run()

        # Check that sequences were generated
        segment = segments[0]
        assert len(segment.selected_sequences) == optimizer.beam_width
        assert all(len(seq.sequence) > 0 for seq in segment.selected_sequences)

        # Check that running prompts were updated
        assert len(optimizer.running_prompts) == optimizer.beam_width
        assert all(len(prompt) > 0 for prompt in optimizer.running_prompts)

    def test_run_multiple_segments(self):
        """Tests the run method with multiple segments."""
        initial_prompt = "ATCG"
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            num_segments=3,
            seq_length=20,
            beam_width=2,
            candidates_per_beam=3,
            prompt=initial_prompt,
        )

        optimizer.run()

        # Check that all segments have selected sequences
        for segment in segments:
            assert len(segment.selected_sequences) == optimizer.beam_width
            assert all(len(seq.sequence) > 0 for seq in segment.selected_sequences)

        # Check that running prompts accumulated all segments
        # Note: first segment includes the prompt, so total = sum of all segment sequences
        full_length = sum(len(seg.selected_sequences[0].sequence) for seg in segments)
        assert all(len(prompt) == full_length for prompt in optimizer.running_prompts)

    @pytest.mark.slow
    @pytest.mark.uses_gpu
    def test_kv_caching_speedup(self):
        """Tests that KV caching provides a speedup using real Evo2 generator."""
        import torch
        import gc
        from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig

        # Helper to create optimizer with Evo2 generator
        # Scale up to show clear caching benefits
        def setup_evo2_optimizer(use_kv_caching: bool):
            prompt = "ATCGATCGATCG"
            segments = [Segment(length=20, sequence_type="dna") for _ in range(20)]  # 20 segments
            construct = Construct(segments)

            gen_config = Evo2GeneratorConfig(
                prompts=[prompt],
                prepend_prompt=True,
                stop_at_eos=False,
            )
            generator = Evo2Generator(config=gen_config)
            generator.assign(segments[0])

            constraint = Constraint(
                inputs=[segments[0]],
                function=gc_content_constraint,
                function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            )

            config = MultiSegmentBeamSearchOptimizerConfig(
                prompt=prompt,
                beam_width=3,
                candidates_per_beam=5,
                use_kv_caching=use_kv_caching,
                verbose=False,
            )

            return MultiSegmentBeamSearchOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=[constraint],
                config=config,
            )

        # Run WITHOUT caching first (establishes baseline with clean GPU state)
        optimizer_uncached = setup_evo2_optimizer(use_kv_caching=False)

        start_uncached = time.time()
        optimizer_uncached.run()
        time_uncached = time.time() - start_uncached

        # Clean up
        del optimizer_uncached
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run WITH caching
        optimizer_cached = setup_evo2_optimizer(use_kv_caching=True)

        start_cached = time.time()
        optimizer_cached.run()
        time_cached = time.time() - start_cached

        # KV caching should provide clear speedup with scaled up parameters
        speedup_ratio = time_uncached / time_cached
        print(f"\nKV caching speedup: {speedup_ratio:.2f}x")
        print(f"Time with caching: {time_cached:.2f}s")
        print(f"Time without caching: {time_uncached:.2f}s")
        print("Parameters: beam_width=3, candidates_per_beam=5, segments=20, tokens_per_segment=100")

        # With candidates_per_beam=5, each beam's cache is reused 5 times per segment
        # Across 20 segments with growing prompts, this should show clear benefit
        # Expected: >1.1x speedup (matches beam_search_kv_caching.py 1.30x result)
        assert speedup_ratio > 1.1, (
            "Expected >1.2x speedup with cache reuse (beam_width=3, candidates_per_beam=5, 20 segments). "
            f"Got {speedup_ratio:.2f}x (time_cached={time_cached:.2f}s, time_uncached={time_uncached:.2f}s). "
            "Should match beam_search_kv_caching.py results (~1.3x speedup)."
        )

    def test_beam_search_improves_energy(self):
        """Tests that beam search finds sequences with better energy scores."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            num_segments=2,
            seq_length=40,
            beam_width=5,
            candidates_per_beam=10,
            gc_target_range=(45.0, 55.0),
        )

        optimizer.run()

        # Get the final energy scores
        segment = segments[0]
        segment.candidate_sequences = segment.selected_sequences
        optimizer.num_candidates = len(segment.selected_sequences)
        optimizer._score_energy_active_constraints()

        # At least one sequence should have low energy (close to optimal)
        best_energy = min(optimizer.energy_scores)
        assert best_energy < 0.2  # Should find reasonably good sequences

    def test_multi_segment_constraint_evaluation(self):
        """Tests that multi-segment constraints are evaluated correctly."""
        segments = [Segment(length=20, sequence_type="dna") for _ in range(3)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=30)
        generator._assigned_segment = segments[0]  # Required for validation

        # Constraint on segment 0 only
        constraint_0 = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        # Constraint on segments 0 and 1 together
        constraint_01 = Constraint(
            inputs=[segments[0], segments[1]],
            function=sequence_length_constraint,
            function_config=SequenceLengthConfig(target_length=60),
        )

        # Constraint on all three segments
        constraint_012 = Constraint(
            inputs=[segments[0], segments[1], segments[2]],
            function=sequence_length_constraint,
            function_config=SequenceLengthConfig(target_length=90),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
            verbose=False,
        )

        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint_0, constraint_01, constraint_012],
            config=config,
        )

        # Run and check that all segments are populated
        optimizer.run()

        for segment in segments:
            assert len(segment.selected_sequences) == optimizer.beam_width
            assert all(len(seq.sequence) > 0 for seq in segment.selected_sequences)

    def test_verbose_output(self, capsys):
        """Tests that verbose output is printed when enabled."""
        optimizer, _, _, _ = _setup_multi_segment_beam_search_components(
            num_segments=2,
            seq_length=20,
            beam_width=2,
            candidates_per_beam=3,
        )

        optimizer.verbose = True
        optimizer.run()

        # Check that output was printed
        captured = capsys.readouterr()
        assert "Processing" in captured.out
        assert "segments with beam search" in captured.out
        assert "Beam width:" in captured.out
        assert "KV caching:" in captured.out
        assert "Segment" in captured.out

    def test_initial_prompt_propagation(self):
        """Tests that initial prompt is correctly propagated."""
        initial_prompt = "ATCGATCG"
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            num_segments=2,
            seq_length=20,
            beam_width=3,
            candidates_per_beam=4,
            prompt=initial_prompt,
        )

        # Check initial state
        assert all(prompt == initial_prompt for prompt in optimizer.running_prompts)

        optimizer.run()

        # All running prompts should start with the initial prompt
        assert all(prompt.startswith(initial_prompt) for prompt in optimizer.running_prompts)

        # All prompts should have grown beyond the initial prompt
        assert all(len(prompt) > len(initial_prompt) for prompt in optimizer.running_prompts)

    def test_construct_joined_sequences(self):
        """Tests that construct.joined_sequences returns the final beam sequences."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            num_segments=3,
            seq_length=20,
            beam_width=4,
            candidates_per_beam=5,
        )

        optimizer.run()

        # Get the joined sequences
        joined_sequences = optimizer.construct.joined_sequences

        # Should have beam_width joined sequences
        assert len(joined_sequences) == optimizer.beam_width

        # Each joined sequence should be the concatenation of all segments
        for i, joined_seq in enumerate(joined_sequences):
            expected_sequence = "".join(
                seg.selected_sequences[i].sequence for seg in segments
            )
            assert joined_seq.sequence == expected_sequence

    def test_energy_monotonicity_within_segment(self):
        """Tests that selected sequences have better energy than rejected ones."""
        optimizer, _, _, segments = _setup_multi_segment_beam_search_components(
            num_segments=1,
            seq_length=30,
            beam_width=3,
            candidates_per_beam=10,
        )

        optimizer.run()

        # After run(), energy_scores should be replicated to match candidate_sequences
        # All replicated energies should be the same (candidates_per_beam copies of each)
        segment = segments[0]

        # Verify that energy_scores length matches candidate_sequences
        assert len(optimizer.energy_scores) == len(segment.candidate_sequences)
        assert len(optimizer.energy_scores) == optimizer.beam_width * optimizer.candidates_per_beam

        # Extract unique energies (should be beam_width unique values, each replicated candidates_per_beam times)
        unique_energies = []
        for i in range(optimizer.beam_width):
            start_idx = i * optimizer.candidates_per_beam
            end_idx = (i + 1) * optimizer.candidates_per_beam
            beam_energies = optimizer.energy_scores[start_idx:end_idx]

            # All energies in this beam should be identical
            assert all(abs(e - beam_energies[0]) < 1e-6 for e in beam_energies)
            unique_energies.append(beam_energies[0])

        # The unique energies should be sorted (best to worst)
        assert unique_energies == sorted(unique_energies)

    def test_different_beam_widths(self):
        """Tests that different beam widths produce different results."""
        # Small beam width
        optimizer_small, _, _, _ = _setup_multi_segment_beam_search_components(
            num_segments=2,
            seq_length=30,
            beam_width=2,
            candidates_per_beam=5,
        )
        optimizer_small.run()

        # Large beam width
        optimizer_large, _, _, _ = _setup_multi_segment_beam_search_components(
            num_segments=2,
            seq_length=30,
            beam_width=10,
            candidates_per_beam=5,
        )
        optimizer_large.run()

        # Larger beam width should explore more options
        assert len(optimizer_small.construct.joined_sequences) == 2
        assert len(optimizer_large.construct.joined_sequences) == 10

    @pytest.mark.uses_gpu
    @pytest.mark.slow
    def test_memory_cleanup(self):
        """Tests that KV caches are properly cleaned up (requires GPU)."""
        import gc
        import torch
        from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig

        # Set up with real Evo2 generator for memory testing
        segments = [Segment(length=30, sequence_type="dna") for _ in range(2)]
        construct = Construct(segments)

        gen_config = Evo2GeneratorConfig(prompts=[""], prepend_prompt=False)
        generator = Evo2Generator(config=gen_config)
        generator.assign(segments[0])

        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        config = MultiSegmentBeamSearchOptimizerConfig(
        prompt="ATCG",
            beam_width=3,
            candidates_per_beam=5,
            use_kv_caching=True,
            verbose=False,
        )

        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
        )

        # Track initial GPU memory if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        optimizer.run()

        # After run, only beam_width KV caches should remain
        assert len(optimizer.top_beam_kv_caches) == optimizer.beam_width

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Memory should not grow unboundedly
            # (allowing for some overhead from the sequences themselves)
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory

            # This is a rough check - memory growth should be reasonable
            print(f"\nMemory growth: {memory_growth / 1024 / 1024:.2f} MB")

    def test_infinite_energy_filtering(self):
        """Test that beam search filters out inf/NaN energies."""        
        # Use restrictive GC constraint with threshold to generate inf energies
        prompt = "ATCG"
        segment1 = Segment(length=20, sequence_type="dna")
        construct = Construct([segment1])
        
        gen = MockAutoregressiveGenerator(num_tokens=20, use_kv_caching=False)
        
        # Very restrictive: only 48-52% GC passes (threshold=0.0 converts to boolean)
        constraint = Constraint(
            inputs=[segment1],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=48.0, max_gc=52.0),
            threshold=0.0,  # Returns inf for sequences outside range
        )
        
        config = MultiSegmentBeamSearchOptimizerConfig(
            prompt=prompt,
            beam_width=3,
            candidates_per_beam=5,
            use_kv_caching=False,
            verbose=False
        )
        
        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        
        optimizer.run()
        
        # Verify all final energies are finite
        for energy in optimizer.energy_scores:
            assert not math.isinf(energy), "Found infinite energy in final results"
            assert not math.isnan(energy), "Found NaN energy in final results"
        
        # Verify we got the expected number of candidates
        assert len(segment1.selected_sequences) == config.beam_width

    def test_all_invalid_raises_error(self):
        """Test that optimizer raises error when all candidates are invalid after max attempts."""        
        prompt = "ATCG"
        segment1 = Segment(length=10, sequence_type="dna")
        construct = Construct([segment1])
        
        gen = MockAutoregressiveGenerator(num_tokens=10, use_kv_caching=False)
        
        # Impossible constraint: GC must be exactly 100% with zero tolerance
        # Random DNA will never be all GC, so all candidates will be rejected
        constraint = Constraint(
            inputs=[segment1],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=100.0, max_gc=100.0),
            threshold=0.0,  # Reject any deviation
        )
        
        config = MultiSegmentBeamSearchOptimizerConfig(
            prompt=prompt,
            beam_width=2,
            candidates_per_beam=3,
            use_kv_caching=False,
            verbose=False
        )
        
        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        
        # Should raise RuntimeError due to all invalid candidates
        with pytest.raises(RuntimeError, match="could not produce.*valid candidates"):
            optimizer.run()

    def test_partial_beam_invalidity_with_resampling(self):
        """Test that optimizer correctly handles when only some beams produce invalid candidates.
        
        Scenario: beam_width=2, candidates_per_beam=4
        - Initial generation (2 calls, one per beam): Beam 0 gets all invalid, Beam 1 gets all valid
        - Resampling (1 call): Beam 0 regenerates and gets valid candidates
        - Total: 3 calls to generator.sample()
        """        
        # Use mock generator with fixed outputs to control which beams are valid
        class ControlledMockGenerator(Generator):
            """Mock generator that produces specific sequences to control GC content."""
            def __init__(self, beam_width: int, candidates_per_beam: int):
                super().__init__()
                self.kv_caches = []
                self.call_count = 0
                self.beam_width = beam_width
                self.candidates_per_beam = candidates_per_beam
                
            def assign(self, assigned_segment: Segment) -> None:
                self._assigned_segment = assigned_segment
                
            def sample(self, prompts=None, prepend_prompt=None, old_kv_cache=None):
                """Generate sequences with varying GC content."""
                self.call_count += 1
                sequences = []
                
                # Total candidates in initial generation per beam
                beam_0_end = self.candidates_per_beam
                
                for i, prompt in enumerate(prompts):
                    # First call (initial generation): beam 0 gets invalid, beam 1 gets valid
                    # Subsequent calls (resampling): all valid
                    if self.call_count == 1 and i < beam_0_end:
                        # Beam 0 in initial generation: produce invalid low-GC sequences
                        seq = "A" * 20  # 0% GC - will be rejected by constraint
                    else:
                        # Beam 1 in initial generation or any resampling: produce valid sequences
                        seq = "ATCGATCG" * 3  # 50% GC - valid
                    
                    if prepend_prompt:
                        seq = prompt + seq
                    sequences.append(seq)
                
                self._assigned_segment.candidate_sequences = [
                    Sequence(sequence=seq, sequence_type="dna") for seq in sequences
                ]
                self.kv_caches = []
            
            def replicate_cache(self, cache, n):
                return cache
        
        prompt = "ATCG"
        beam_width = 2
        candidates_per_beam = 4
        
        segment1 = Segment(length=20, sequence_type="dna")
        construct = Construct([segment1])
        
        gen = ControlledMockGenerator(beam_width=beam_width, candidates_per_beam=candidates_per_beam)
        gen._assigned_segment = segment1
        
        # Constraint that rejects sequences with GC < 40%
        constraint = Constraint(
            inputs=[segment1],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            threshold=0.0,
        )
        
        config = MultiSegmentBeamSearchOptimizerConfig(
            prompt=prompt,
            beam_width=beam_width,
            candidates_per_beam=candidates_per_beam,
            use_kv_caching=False,
            verbose=False
        )
        
        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        
        # Run should succeed - beam 0 will need resampling, beam 1 is fine
        optimizer.run()
        
        # Verify all final energies are finite
        for energy in optimizer.energy_scores:
            assert not math.isinf(energy), "Found infinite energy after resampling"
            assert not math.isnan(energy), "Found NaN energy after resampling"
        
        # Verify resampling happened
        # Expected: beam_width calls for initial generation (2) + 1 resample for beam 0 = 3 total
        assert gen.call_count == 3, f"Expected 3 generation calls (2 for initial, 1 resample), got {gen.call_count}"
        
        # Verify we got exactly beam_width * candidates_per_beam final candidates
        expected_candidates = beam_width * candidates_per_beam
        assert len(optimizer.construct.segments[0].candidate_sequences) == expected_candidates, \
            f"Expected {expected_candidates} final candidates, got {len(optimizer.construct.segments[0].candidate_sequences)}"

    def test_resampling_with_multiple_segments(self):
        """Test that resampling works correctly across multiple segments with context accumulation."""
        class SegmentAwareMockGenerator(Generator):
            """Mock generator that produces invalid candidates for first segment, valid for subsequent.
            
            Test scenario:
            - 3 segments, beam_width=2, candidates_per_beam=3
            - Segment 0: Beam 0 produces invalid candidates (requires resampling)
            - Segments 1-2: All beams produce valid candidates
            - Verify running_prompts correctly accumulate across segments
            """
            def __init__(self, num_tokens: int):
                super().__init__()
                self.category = "autoregressive"
                self.kv_caches = []
                self.call_count_per_segment = {}
                self.current_segment_idx = 0
                self.num_tokens = num_tokens
                
            def assign(self, assigned_segment: Segment) -> None:
                self._assigned_segment = assigned_segment
                
            def sample(self, prompts=None, prepend_prompt=None, old_kv_cache=None):
                """Generate sequences with varying validity based on segment and call count."""
                # Track which segment we're on by prompt length
                if prompts and len(prompts) > 0:
                    prompt_len = len(prompts[0])
                    # Segment 0: prompt is 4 chars ("ATCG")
                    # Segment 1: prompt is 4 + num_tokens
                    # Segment 2: prompt is 4 + 2*num_tokens
                    if prompt_len == 4:
                        self.current_segment_idx = 0
                    elif prompt_len == 4 + self.num_tokens:
                        self.current_segment_idx = 1
                    elif prompt_len == 4 + 2*self.num_tokens:
                        self.current_segment_idx = 2
                
                # Track calls per segment
                seg_key = self.current_segment_idx
                self.call_count_per_segment[seg_key] = self.call_count_per_segment.get(seg_key, 0) + 1
                
                sequences = []
                for i, prompt in enumerate(prompts):
                    # For segment 0, first call, first 3 candidates (beam 0): produce invalid
                    if self.current_segment_idx == 0 and self.call_count_per_segment[seg_key] == 1 and i < 3:
                        seq = "A" * self.num_tokens  # 0% GC - invalid
                    else:
                        seq = "ATCGATCG" * (self.num_tokens // 8 + 1)  # ~50% GC - valid
                        seq = seq[:self.num_tokens]  # Trim to exact length
                    
                    if prepend_prompt:
                        seq = prompt + seq
                    sequences.append(seq)
                
                self._assigned_segment.candidate_sequences = [
                    Sequence(sequence=seq, sequence_type="dna") for seq in sequences
                ]
                self.kv_caches = []
            
            def replicate_cache(self, cache, n):
                return cache
        
        prompt = "ATCG"
        beam_width = 2
        candidates_per_beam = 3
        segment_length = 16
        num_segments = 3
        
        # Create multiple segments
        segments = [Segment(length=segment_length, sequence_type="dna") for _ in range(num_segments)]
        construct = Construct(segments)
        
        gen = SegmentAwareMockGenerator(num_tokens=segment_length)
        
        # Constraint that rejects sequences with GC < 40% on first segment only
        constraint = Constraint(
            inputs=[segments[0]],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            threshold=0.0,
        )
        
        config = MultiSegmentBeamSearchOptimizerConfig(
            prompt=prompt,
            beam_width=beam_width,
            candidates_per_beam=candidates_per_beam,
            use_kv_caching=False,
            verbose=False
        )
        
        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        
        # Run should succeed with resampling only on first segment
        optimizer.run()
        
        # Verify resampling occurred only for segment 0
        # Each segment makes beam_width calls (one per beam) for initial generation
        # Segment 0 also makes 1 additional call to resample beam 0
        assert gen.call_count_per_segment.get(0, 0) == 3, \
            f"Expected 3 calls for segment 0 (2 beams initial + 1 resample), got {gen.call_count_per_segment.get(0, 0)}"
        assert gen.call_count_per_segment.get(1, 0) == 2, \
            f"Expected 2 calls for segment 1 (2 beams, no resample), got {gen.call_count_per_segment.get(1, 0)}"
        assert gen.call_count_per_segment.get(2, 0) == 2, \
            f"Expected 2 calls for segment 2 (2 beams, no resample), got {gen.call_count_per_segment.get(2, 0)}"
        
        # Verify all segments have expected number of candidates
        expected_candidates = beam_width * candidates_per_beam
        for i, segment in enumerate(segments):
            assert len(segment.candidate_sequences) == expected_candidates, \
                f"Segment {i}: expected {expected_candidates} candidates, got {len(segment.candidate_sequences)}"
        
        # Verify running prompts accumulated correctly (length should increase with each segment)
        assert len(optimizer.running_prompts) == beam_width, \
            f"Expected {beam_width} running prompts, got {len(optimizer.running_prompts)}"
        
        expected_final_length = len(prompt) + num_segments * segment_length
        for i, running_prompt in enumerate(optimizer.running_prompts):
            assert len(running_prompt) == expected_final_length, \
                f"Running prompt {i}: expected length {expected_final_length}, got {len(running_prompt)}"
        
        # Verify all final energies are finite
        for energy in optimizer.energy_scores:
            assert not math.isinf(energy), "Found infinite energy after optimization"
            assert not math.isnan(energy), "Found NaN energy after optimization"

    def test_accumulative_resampling(self):
        """Test that resampling accumulates valid candidates across attempts and selects the best."""
        class AccumulativeTrackingGenerator(Generator):
            """Generator that tracks generation calls and produces controlled validity patterns.
            
            Test scenario:
            - beam_width=2, candidates_per_beam=5
            - Initial generation: 10 candidates (2 beams × 5 candidates)
                - Beam 0: Only 2 valid candidates (need 3 more)
                - Beam 1: All 5 valid candidates
            - Resampling generates full batch (5) but accumulates valid candidates
            - Best 5 candidates by energy are selected for beam 0
            """
            def __init__(self):
                super().__init__()
                self.category = "autoregressive"
                self.kv_caches = []
                self.call_count = 0
                self.generation_sizes = []  # Track size of each generation request
                
            def assign(self, assigned_segment: Segment) -> None:
                self._assigned_segment = assigned_segment
                
            def sample(self, prompts=None, prepend_prompt=None, old_kv_cache=None):
                """Generate sequences and track request size."""
                self.call_count += 1
                num_requested = len(prompts) if prompts else 0
                self.generation_sizes.append(num_requested)
                
                sequences = []
                for i, prompt in enumerate(prompts):
                    # First call: beam 0 (indices 0-4) gets mostly invalid, beam 1 gets all valid
                    if self.call_count == 1 and i < 5:
                        # Beam 0 in initial generation: first 2 valid, rest invalid
                        if i < 2:
                            seq = "ATCGATCG" * 3  # 50% GC - valid
                        else:
                            seq = "A" * 20  # 0% GC - invalid
                    else:
                        # Beam 1 in initial generation or any resampling: all valid
                        seq = "ATCGATCG" * 3  # 50% GC - valid
                    
                    if prepend_prompt:
                        seq = prompt + seq
                    sequences.append(seq)
                
                self._assigned_segment.candidate_sequences = [
                    Sequence(sequence=seq, sequence_type="dna") for seq in sequences
                ]
                self.kv_caches = []
            
            def replicate_cache(self, cache, n):
                return cache
        
        prompt = "ATCG"
        beam_width = 2
        candidates_per_beam = 5
        
        segment1 = Segment(length=20, sequence_type="dna")
        construct = Construct([segment1])
        
        gen = AccumulativeTrackingGenerator()
        
        # Constraint that rejects sequences with GC < 40%
        constraint = Constraint(
            inputs=[segment1],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            threshold=0.0,
        )
        
        config = MultiSegmentBeamSearchOptimizerConfig(
            prompt=prompt,
            beam_width=beam_width,
            candidates_per_beam=candidates_per_beam,
            use_kv_caching=False,
            verbose=False
        )
        
        optimizer = MultiSegmentBeamSearchOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        
        # Run optimization
        optimizer.run()
        
        # Verify generation pattern
        # Initial generation: beam_width calls of candidates_per_beam each
        # Then resampling calls as needed
        assert len(gen.generation_sizes) >= beam_width, \
            f"Expected at least {beam_width} generation calls, got {len(gen.generation_sizes)}"
        
        # First beam_width calls should be for initial generation (candidates_per_beam each)
        for i in range(beam_width):
            assert gen.generation_sizes[i] == candidates_per_beam, \
                f"Initial generation call {i} should request {candidates_per_beam} candidates, got {gen.generation_sizes[i]}"
        
        # Subsequent resampling calls should also use full batch size (candidates_per_beam)
        # This maintains efficient batching while accumulating valid candidates
        for i in range(beam_width, len(gen.generation_sizes)):
            assert gen.generation_sizes[i] == candidates_per_beam, \
                f"Resample call {i} should request full batch ({candidates_per_beam} candidates), got {gen.generation_sizes[i]}"
        
        # Verify final state: all beams have exactly candidates_per_beam valid candidates
        expected_candidates = beam_width * candidates_per_beam
        assert len(optimizer.construct.segments[0].candidate_sequences) == expected_candidates, \
            f"Expected {expected_candidates} final candidates, got {len(optimizer.construct.segments[0].candidate_sequences)}"
        
        # Verify all final energies are finite
        for energy in optimizer.energy_scores:
            assert not math.isinf(energy), "Found infinite energy after resampling"
            assert not math.isnan(energy), "Found NaN energy after resampling"
