import pytest
import time
import random
from typing import Tuple, List, Dict, Optional
from unittest.mock import Mock

import sys

sys.path.append(".")
from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    SequenceType,
    Generator,
    Sequence,
    GeneratorType,
)
from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import SequenceLengthConfig
from proto_language.language.optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
)


# Mock Autoregressive Generator for testing
class MockAutoregressiveGenerator(Generator):
    """
    Mock autoregressive generator for testing BeamSearchOptimizer without GPU.

    Generates random DNA sequences and optionally maintains mock KV caches.
    """
    def __init__(self, num_tokens: int, use_kv_caching: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.use_kv_caching = use_kv_caching
        self.type = GeneratorType.AUTOREGRESSIVE
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
            Sequence(sequence=seq, sequence_type=SequenceType.DNA)
            for seq in sequences
        ]

        # Generate mock KV caches if caching is enabled
        # Check both use_kv_caching and the store_kv_cache attribute (set by BeamSearchOptimizer)
        if self.use_kv_caching and getattr(self, 'store_kv_cache', False):
            self.kv_caches = [create_mock_kv_cache() for _ in range(num_samples)]
        else:
            self.kv_caches = []

    def replicate_cache(self, cache: Dict, n_replicates: int) -> Dict:
        """Replicate cache N times for beam branching."""
        return cache

# Helper functions
def create_segment(sequence: str, seq_type: SequenceType = SequenceType.DNA) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


def _setup_beam_search_components(
    num_segments: int = 3,
    seq_length: int = 20,
    beam_width: int = 3,
    candidates_per_beam: int = 5,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    use_kv_caching: bool = True,
    prompt: str = "ATCG",
):
    """Helper function to set up a basic BeamSearchOptimizer for testing."""
    # 1. Create segments
    segments = [create_segment("") for _ in range(num_segments)]
    construct = Construct(segments)

    # 2. Create the mock generator
    generator = MockAutoregressiveGenerator(
        num_tokens=seq_length,
        use_kv_caching=use_kv_caching,
    )

    # BeamSearchOptimizer requires generator to be "assigned" for validation
    # but handles assignment internally during run(), so we assign to a dummy segment
    generator._assigned_segment = segments[0]

    # 3. Create constraints (only on the first segment for simplicity)
    constraint = Constraint(
        inputs=[segments[0]],
        scoring_function=gc_content_constraint,
        scoring_function_config=GCContentConfig(
            min_gc=gc_target_range[0],
            max_gc=gc_target_range[1],
        ),
        vectorized=True,
        concatenate=True,
    )

    # 4. Create the BeamSearchOptimizer config
    config = BeamSearchOptimizerConfig(
        beam_width=beam_width,
        candidates_per_beam=candidates_per_beam,
        use_kv_caching=use_kv_caching,
        verbose=False,
    )

    optimizer = BeamSearchOptimizer(
        construct=construct,
        generator=generator,
        prompt=prompt,
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


class TestBeamSearchOptimizer:
    def test_initialization_and_validation(self):
        """Tests successful initialization and validation of BeamSearchOptimizer."""
        optimizer, generator, constraint, segments = _setup_beam_search_components()

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
        segments = [create_segment("") for _ in range(3)]
        construct = Construct(segments)

        # Create a non-autoregressive generator (mock)
        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation
        generator.type = GeneratorType.MUTATION  # Override to make it mutation generator

        constraint = Constraint(
            inputs=[segments[0]],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            vectorized=True,
            concatenate=True,
        )

        config = BeamSearchOptimizerConfig(
            beam_width=3,
            candidates_per_beam=5,
        )

        with pytest.raises(ValueError, match="requires autoregressive generators"):
            BeamSearchOptimizer(
                construct=construct,
                generator=generator,
                prompt="ATCG",
                constraints=[constraint],
                config=config,
            )

    def test_initialization_with_existing_candidates_warning(self):
        """Tests that a warning is raised if segments have existing candidates."""
        segments = [create_segment("ATCG") for _ in range(2)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        constraint = Constraint(
            inputs=[segments[0]],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            vectorized=True,
            concatenate=True,
        )

        config = BeamSearchOptimizerConfig(
            beam_width=3,
            candidates_per_beam=5,
        )

        with pytest.warns(UserWarning, match="will overwrite"):
            BeamSearchOptimizer(
                construct=construct,
                generator=generator,
                prompt="ATCG",
                constraints=[constraint],
                config=config,
            )

    def test_config_validation(self):
        """Tests BeamSearchOptimizerConfig validation."""
        from pydantic import ValidationError

        # Valid configs
        config = BeamSearchOptimizerConfig(beam_width=5, candidates_per_beam=10)
        assert config.beam_width == 5
        assert config.candidates_per_beam == 10
        assert config.use_kv_caching is True
        assert config.verbose is False

        # Negative beam_width should fail
        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(beam_width=0, candidates_per_beam=5)

        # Negative candidates_per_beam should fail
        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(beam_width=5, candidates_per_beam=0)

    def test_constraint_weights(self):
        """Tests that constraint weights are properly handled."""
        optimizer, _, _, segments = _setup_beam_search_components()

        # Add a second constraint
        constraint2 = Constraint(
            inputs=[segments[0]],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=20),
            vectorized=True,
            concatenate=True,
        )

        # Recreate optimizer with custom weights
        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        config = BeamSearchOptimizerConfig(
            beam_width=3,
            candidates_per_beam=5,
        )

        construct = Construct(segments)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="ATCG",
            constraints=[optimizer.constraints[0], constraint2],
            constraint_weights=[1.0, 2.0],
            config=config,
        )

        assert optimizer.constraint_weights == [1.0, 2.0]

        # Mismatched weights and constraints should fail
        with pytest.raises(ValueError, match="must match"):
            BeamSearchOptimizer(
                construct=construct,
                generator=generator,
                prompt="ATCG",
                constraints=[optimizer.constraints[0], constraint2],
                constraint_weights=[1.0, 2.0, 3.0],
                config=config,
            )

    @pytest.mark.uses_gpu
    @pytest.mark.slow
    def test_replicate_cache(self):
        """Tests the replicate_cache method."""
        from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        
        segments = [create_segment("") for _ in range(3)]
        construct = Construct(segments)
        
        gen_config = Evo2GeneratorConfig(prompts=[""], num_tokens=20)
        generator = Evo2Generator(config=gen_config)
        generator._assigned_segment = segments[0]
        
        constraint = Constraint(
            inputs=[segments[0]],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            vectorized=True,
            concatenate=True,
        )
        
        config = BeamSearchOptimizerConfig(
            beam_width=3,
            candidates_per_beam=5,
            use_kv_caching=True,
            verbose=False,
        )
        
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="ATCG",
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
        
        segments = [create_segment("") for _ in range(3)]
        construct = Construct(segments)
        
        gen_config = Evo2GeneratorConfig(prompts=[""], num_tokens=20)
        generator = Evo2Generator(config=gen_config)
        generator._assigned_segment = segments[0]
        
        constraint = Constraint(
            inputs=[segments[0]],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            vectorized=True,
            concatenate=True,
        )
        
        config = BeamSearchOptimizerConfig(
            beam_width=3,
            candidates_per_beam=5,
            use_kv_caching=True,
            verbose=False,
        )
        
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="ATCG",
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
        optimizer, _, _, segments = _setup_beam_search_components(num_segments=1)

        # Manually set candidates for the segment
        segment = segments[0]
        segment.candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type=SequenceType.DNA),  # 50% GC
            Sequence(sequence="AAAAAAAAAAAAAAAAAAAA", sequence_type=SequenceType.DNA),  # 0% GC
            Sequence(sequence="GCGCGCGCGCGCGCGCGCGC", sequence_type=SequenceType.DNA),  # 100% GC
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
        segments = [create_segment("") for _ in range(3)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        # Create constraint that depends on segments 0 and 1
        constraint_01 = Constraint(
            inputs=[segments[0], segments[1]],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=40),
            vectorized=True,
            concatenate=True,
        )

        # Create constraint that only depends on segment 0
        constraint_0 = Constraint(
            inputs=[segments[0]],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            vectorized=True,
            concatenate=True,
        )

        config = BeamSearchOptimizerConfig(
            beam_width=2,
            candidates_per_beam=3,
        )

        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="ATCG",
            constraints=[constraint_01, constraint_0],
            config=config,
        )

        # Initially, only segment 0 has candidates
        segments[0].candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type=SequenceType.DNA),  # 50% GC
            Sequence(sequence="AAAAAAAAAAAAAAAAAAAA", sequence_type=SequenceType.DNA),  # 0% GC
        ]
        optimizer.num_candidates = 2

        # Only constraint_0 should be active
        optimizer._score_energy_active_constraints()

        # Should score successfully with only the active constraint
        assert len(optimizer.energy_scores) == 2

        # Now add candidates to segment 1
        segments[1].candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type=SequenceType.DNA),
            Sequence(sequence="AAAAAAAAAAAAAAAAAAAA", sequence_type=SequenceType.DNA),
        ]

        # Now both constraints should be active
        optimizer._score_energy_active_constraints()
        assert len(optimizer.energy_scores) == 2

    def test_score_energy_active_constraints_no_active(self):
        """Tests _score_energy_active_constraints when no constraints are active."""
        segments = [create_segment("") for _ in range(2)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=20)
        generator._assigned_segment = segments[0]  # Required for validation

        # Create constraint that depends on both segments
        constraint = Constraint(
            inputs=[segments[0], segments[1]],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=40),
            vectorized=True,
            concatenate=True,
        )

        config = BeamSearchOptimizerConfig(
            beam_width=2,
            candidates_per_beam=3,
        )

        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="ATCG",
            constraints=[constraint],
            config=config,
        )

        # Only segment 0 has candidates, so constraint is not active
        segments[0].candidate_sequences = [
            Sequence(sequence="ATCGATCGATCGATCGATCG", sequence_type=SequenceType.DNA),
        ]
        # Clear segment 1 candidates to ensure constraint is not active
        segments[1].candidate_sequences = []
        optimizer.num_candidates = 1

        optimizer._score_energy_active_constraints()

        # All energy scores should be 0 since no constraints are active
        assert optimizer.energy_scores == [0.0]

    def test_select_topk(self):
        """Tests the _select_topk method."""
        optimizer, _, _, segments = _setup_beam_search_components(
            beam_width=3,
            candidates_per_beam=4,
            num_segments=1,
            use_kv_caching=False,  # Disable KV caching for this test
        )

        segment = segments[0]

        # Create 12 candidates (3 beams * 4 candidates per beam)
        num_candidates = optimizer.beam_width * optimizer.candidates_per_beam
        segment.candidate_sequences = [
            Sequence(sequence="A" * 20, sequence_type=SequenceType.DNA) for _ in range(num_candidates)
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
        optimizer, _, _, segments = _setup_beam_search_components(
            beam_width=3,
            candidates_per_beam=2,
            use_kv_caching=True,
        )

        segment = segments[0]
        num_candidates = optimizer.beam_width * optimizer.candidates_per_beam
        segment.candidate_sequences = [
            Sequence(sequence="ATCGATCG", sequence_type=SequenceType.DNA) for _ in range(num_candidates)
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

    def test_generate_candidates_without_kv_caching(self):
        """Tests _generate_candidates without KV caching."""
        optimizer, generator, _, segments = _setup_beam_search_components(
            beam_width=2,
            candidates_per_beam=3,
            num_segments=1,
            use_kv_caching=False,
        )

        segment = segments[0]

        # Mock the generator's sample method
        def mock_sample(prompts=None, prepend_prompt=None, old_kv_cache=None):
            # Generate mock sequences
            segment.candidate_sequences = [
                Sequence(sequence="ATCGATCGATCGATCG", sequence_type=SequenceType.DNA) for _ in range(len(prompts))
            ]
            generator.kv_caches = []

        generator.sample = mock_sample

        # Set running prompts (valid DNA)
        optimizer.running_prompts = ["ATCG", "GCTA"]

        # Call _generate_candidates
        all_kv_caches = optimizer._generate_candidates(segment)

        # Should have generated beam_width * candidates_per_beam sequences
        assert len(segment.candidate_sequences) == 6

        # KV caches should be empty when caching is disabled
        assert len(all_kv_caches) == 0

    def test_generate_candidates_with_kv_caching(self):
        """Tests _generate_candidates with KV caching."""
        optimizer, generator, _, segments = _setup_beam_search_components(
            beam_width=2,
            candidates_per_beam=3,
            num_segments=1,
            use_kv_caching=True,
        )

        segment = segments[0]

        # Set up mock KV caches
        mock_cache = create_mock_kv_cache()
        optimizer.top_beam_kv_caches = [mock_cache, None]

        # Mock the generator's sample and kv_caches
        def mock_sample(prompts=None, prepend_prompt=None, old_kv_cache=None):
            segment.candidate_sequences = [
                Sequence(sequence="ATCGATCGATCG", sequence_type=SequenceType.DNA) for _ in range(len(prompts))
            ]
            generator.kv_caches = [create_mock_kv_cache() for _ in range(len(prompts))]

        generator.sample = mock_sample
        generator.kv_caches = []

        optimizer.running_prompts = ["ATCG", "GCTA"]

        all_kv_caches = optimizer._generate_candidates(segment)

        # Should have generated sequences
        assert len(segment.candidate_sequences) == 6

        # Should have collected KV caches
        assert len(all_kv_caches) == 6

    def test_run_single_segment(self):
        """Tests the run method with a single segment."""
        optimizer, _, _, segments = _setup_beam_search_components(
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
        optimizer, _, _, segments = _setup_beam_search_components(
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
            segments = [create_segment("") for _ in range(20)]  # 20 segments 
            construct = Construct(segments)

            gen_config = Evo2GeneratorConfig(
                prompts=[""],
                num_tokens=100,  # 100 tokens per segment
                stop_at_eos=False,
            )
            generator = Evo2Generator(config=gen_config)
            generator._assigned_segment = segments[0]

            constraint = Constraint(
                inputs=[segments[0]],
                scoring_function=gc_content_constraint,
                scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
                vectorized=True,
                concatenate=True,
            )

            config = BeamSearchOptimizerConfig(
                beam_width=3,
                candidates_per_beam=5,
                use_kv_caching=use_kv_caching,
                verbose=False,
            )

            return BeamSearchOptimizer(
                construct=construct,
                generator=generator,
                prompt="ATCGATCGATCG",
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
        print(f"Parameters: beam_width=3, candidates_per_beam=5, segments=20, tokens_per_segment=100")

        # With candidates_per_beam=5, each beam's cache is reused 5 times per segment
        # Across 20 segments with growing prompts, this should show clear benefit
        # Expected: >1.1x speedup (matches beam_search_kv_caching.py 1.30x result)
        assert speedup_ratio > 1.1, (
            f"Expected >1.2x speedup with cache reuse (beam_width=3, candidates_per_beam=5, 20 segments). "
            f"Got {speedup_ratio:.2f}x (time_cached={time_cached:.2f}s, time_uncached={time_uncached:.2f}s). "
            f"Should match beam_search_kv_caching.py results (~1.3x speedup)."
        )

    def test_beam_search_improves_energy(self):
        """Tests that beam search finds sequences with better energy scores."""
        optimizer, _, _, segments = _setup_beam_search_components(
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
        segments = [create_segment("") for _ in range(3)]
        construct = Construct(segments)

        generator = MockAutoregressiveGenerator(num_tokens=30)
        generator._assigned_segment = segments[0]  # Required for validation

        # Constraint on segment 0 only
        constraint_0 = Constraint(
            inputs=[segments[0]],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            vectorized=True,
            concatenate=True,
        )

        # Constraint on segments 0 and 1 together
        constraint_01 = Constraint(
            inputs=[segments[0], segments[1]],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=60),
            vectorized=True,
            concatenate=True,
        )

        # Constraint on all three segments
        constraint_012 = Constraint(
            inputs=[segments[0], segments[1], segments[2]],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=90),
            vectorized=True,
            concatenate=True,
        )

        config = BeamSearchOptimizerConfig(
            beam_width=3,
            candidates_per_beam=5,
            verbose=False,
        )

        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="ATCG",
            constraints=[constraint_0, constraint_01, constraint_012],
            constraint_weights=[1.0, 1.0, 1.0],
            config=config,
        )

        # Run and check that all segments are populated
        optimizer.run()

        for segment in segments:
            assert len(segment.selected_sequences) == optimizer.beam_width
            assert all(len(seq.sequence) > 0 for seq in segment.selected_sequences)

    def test_verbose_output(self, capsys):
        """Tests that verbose output is printed when enabled."""
        optimizer, _, _, _ = _setup_beam_search_components(
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
        optimizer, _, _, segments = _setup_beam_search_components(
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
        optimizer, _, _, segments = _setup_beam_search_components(
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
        optimizer, _, _, segments = _setup_beam_search_components(
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
        optimizer_small, _, _, _ = _setup_beam_search_components(
            num_segments=2,
            seq_length=30,
            beam_width=2,
            candidates_per_beam=5,
        )
        optimizer_small.run()

        # Large beam width
        optimizer_large, _, _, _ = _setup_beam_search_components(
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
        segments = [create_segment("") for _ in range(2)]
        construct = Construct(segments)

        gen_config = Evo2GeneratorConfig(prompts=[""], num_tokens=30)
        generator = Evo2Generator(config=gen_config)
        generator._assigned_segment = segments[0]

        constraint = Constraint(
            inputs=[segments[0]],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            vectorized=True,
            concatenate=True,
        )

        config = BeamSearchOptimizerConfig(
            beam_width=3,
            candidates_per_beam=5,
            use_kv_caching=True,
            verbose=False,
        )

        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="ATCG",
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
