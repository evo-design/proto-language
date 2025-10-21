"""
Comprehensive tests for BeamSearchOptimizer.

Tests cover initialization, helper methods, edge cases, constraint filtering,
and integration scenarios.
"""

import pytest
import numpy as np
import sys

sys.path.append(".")

from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    SequenceType,
    Generator,
)
from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import SequenceLengthConfig
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
)


# Helper functions
def create_segment(sequence: str, seq_type: SequenceType = SequenceType.DNA) -> Segment:
    """Helper to create a Segment."""
    return Segment(sequence=sequence, sequence_type=seq_type)


class MockAutoregressiveGenerator(Generator):
    """Mock autoregressive generator for testing."""
    
    def __init__(self, sequence_length: int, prepend_prompt: bool = False):
        super().__init__()
        self.sequence_length = sequence_length
        self.prepend_prompt = prepend_prompt
        self.autoregressive = True
    
    def assign(self, segment: Segment):
        self._assigned_segment = segment
        self._is_initialized = True
    
    def sample(self, prompt_seqs=None):
        """Generate random DNA sequences."""
        if prompt_seqs is None:
            prompt_seqs = [""] * len(self._assigned_segment.candidate_sequences)
        
        bases = ['A', 'C', 'G', 'T']
        for i, prompt in enumerate(prompt_seqs):
            new_seq = ''.join(np.random.choice(bases) for _ in range(self.sequence_length))
            self._assigned_segment.candidate_sequences[i].sequence = new_seq


class TestBeamSearchOptimizerInitialization:
    """Test initialization and validation."""
    
    def test_basic_initialization(self):
        """Test basic initialization with valid parameters."""
        seg1 = create_segment("ATCG")
        seg2 = create_segment("GCTA")
        construct = Construct([seg1, seg2])
        
        generator = MockAutoregressiveGenerator(sequence_length=4)
        generator.assign(seg1)
        generator.assign(seg2)
        
        constraint = Constraint(
            inputs=[seg1],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0)
        )
        
        config = BeamSearchOptimizerConfig(beam_width=3, candidates_per_beam=5)
        
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        assert optimizer.beam_width == 3
        assert optimizer.candidates_per_beam == 5
        assert len(optimizer.running_prompts) == 3
        assert all(p == "" for p in optimizer.running_prompts)
        assert optimizer.num_candidates == 15  # 3 * 5
        assert optimizer.num_selected == 3
    
    def test_non_autoregressive_generator_raises(self):
        """Test that non-autoregressive generator raises ValueError."""
        segment = create_segment("ATCG")
        construct = Construct([segment])
        
        generator = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=4, num_mutations=1)
        )
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3)
        
        with pytest.raises(ValueError, match="requires autoregressive generators"):
            BeamSearchOptimizer(
                construct=construct,
                generator=generator,
                prompt="",
                constraints=[constraint],
                config=config
            )
    
    def test_initial_prompt_replication(self):
        """Test that initial prompt is replicated to beam_width."""
        segment = create_segment("ATCG")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=4)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=5, candidates_per_beam=2)
        
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="INITIAL",
            constraints=[constraint],
            config=config
        )
        
        assert len(optimizer.running_prompts) == 5
        assert all(p == "INITIAL" for p in optimizer.running_prompts)


class TestPrepareBeamPrompts:
    """Test _prepare_beam_prompts helper method."""
    
    def test_prompt_replication(self):
        """Test that prompts are replicated correctly."""
        segment = create_segment("ATCG")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(sequence_length=4)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=3, candidates_per_beam=4)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        optimizer.running_prompts = ["AAA", "BBB", "CCC"]
        prompts, beam_indices = optimizer._prepare_beam_prompts()
        
        # Should have 3 beams * 4 candidates = 12 prompts
        assert len(prompts) == 12
        assert len(beam_indices) == 12
        
        # Check prompt replication
        assert prompts[0:4] == ["AAA", "AAA", "AAA", "AAA"]
        assert prompts[4:8] == ["BBB", "BBB", "BBB", "BBB"]
        assert prompts[8:12] == ["CCC", "CCC", "CCC", "CCC"]
        
        # Check beam index tracking
        assert beam_indices[0:4] == [0, 0, 0, 0]
        assert beam_indices[4:8] == [1, 1, 1, 1]
        assert beam_indices[8:12] == [2, 2, 2, 2]
    
    def test_single_beam(self):
        """Test with beam_width=1."""
        segment = create_segment("ATCG")
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(sequence_length=4)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=1, candidates_per_beam=5)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        optimizer.running_prompts = ["SINGLE"]
        prompts, beam_indices = optimizer._prepare_beam_prompts()
        
        assert len(prompts) == 5
        assert all(p == "SINGLE" for p in prompts)
        assert all(idx == 0 for idx in beam_indices)


class TestSelectTopCandidates:
    """Test _select_topk helper method (merges selection, prompt update, and replication)."""
    
    def test_combined_operations(self):
        """Test that _select_topk performs all three operations correctly."""
        segment = create_segment("")
        
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(sequence_length=4)
        generator.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        # Create candidates and set sequences
        segment.create_candidates(6)  # 2 beams * 3 candidates
        candidate_seqs = ["AAAA", "AAAT", "AATA", "AATT", "ATAA", "ATAT"]
        for i, seq in enumerate(segment.candidate_sequences):
            seq.sequence = candidate_seqs[i]
        
        # Set initial prompts
        optimizer.running_prompts = ["GGGG", "CCCC"]
        
        # Beam indices: [0,0,0,1,1,1]
        beam_indices = [0, 0, 0, 1, 1, 1]
        
        # Mock energy scores (select candidates 1 and 4)
        optimizer.energy_scores = [5.0, 1.0, 3.0, 4.0, 2.0, 6.0]
        
        top_idx = optimizer._select_topk(segment, beam_indices)
        
        # Verify top_idx are the two lowest energy scores
        assert top_idx == [1, 4]
        
        # Verify selected sequences
        assert len(segment.selected_sequences) == 2
        assert segment.selected_sequences[0].sequence == candidate_seqs[1]
        assert segment.selected_sequences[1].sequence == candidate_seqs[4]
        
        # Verify candidate replication (2 selected * 3 candidates_per_beam)
        assert len(segment.candidate_sequences) == 6
        assert segment.candidate_sequences[0].sequence == candidate_seqs[1]
        assert segment.candidate_sequences[1].sequence == candidate_seqs[1]
        assert segment.candidate_sequences[2].sequence == candidate_seqs[1]
        assert segment.candidate_sequences[3].sequence == candidate_seqs[4]
        assert segment.candidate_sequences[4].sequence == candidate_seqs[4]
        assert segment.candidate_sequences[5].sequence == candidate_seqs[4]
        
        # Verify running prompts updated
        assert len(optimizer.running_prompts) == 2
        assert optimizer.running_prompts[0] == "GGGG" + candidate_seqs[1]
        assert optimizer.running_prompts[1] == "CCCC" + candidate_seqs[4]
    
    def test_beam_tracking_with_multiple_beams(self):
        """Test beam tracking correctness with 3 beams."""
        segment = create_segment("")
        
        construct = Construct([segment])
        generator = MockAutoregressiveGenerator(sequence_length=4)
        generator.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        config = BeamSearchOptimizerConfig(beam_width=3, candidates_per_beam=3)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        # Create candidates
        segment.create_candidates(9)  # 3 beams * 3 candidates
        candidate_seqs = ["AAAA", "AAAT", "AATA", "AATT", "ATAA", "ATAT", "ATTA", "ATTT", "TAAA"]
        for i, seq in enumerate(segment.candidate_sequences):
            seq.sequence = candidate_seqs[i]
        
        optimizer.running_prompts = ["GGGG", "CCCC", "TTTT"]
        beam_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        
        # Select one from each beam (indices 2, 4, 8)
        optimizer.energy_scores = [5.0, 4.0, 1.0, 6.0, 2.0, 5.0, 7.0, 8.0, 3.0]
        
        top_idx = optimizer._select_topk(segment, beam_indices)
        
        # Should select indices 2, 4, 8 (lowest energies)
        assert set(top_idx) == {2, 4, 8}
        
        # Verify prompts extended correctly based on beam tracking
        assert optimizer.running_prompts[0] == "GGGG" + candidate_seqs[2]
        assert optimizer.running_prompts[1] == "CCCC" + candidate_seqs[4]
        assert optimizer.running_prompts[2] == "TTTT" + candidate_seqs[8]


class TestScoreEnergyFiltered:
    """Test _score_energy_active_constraints constraint filtering logic."""
    
    def test_single_segment_constraint(self):
        """Test with constraint on only current segment."""
        segment = create_segment("ATCG")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=4)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0)
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        segment.create_candidates(6)
        for seq in segment.candidate_sequences:
            seq.sequence = "GCGCGCGC"  # 100% GC
        
        optimizer._score_energy_active_constraints()
        
        assert len(optimizer.energy_scores) == 6
        assert all(isinstance(e, float) for e in optimizer.energy_scores)
    
    def test_no_applicable_constraints(self):
        """Test when no constraints are applicable (all waiting for segments)."""
        seg1 = create_segment("A")
        seg2 = create_segment("T")
        construct = Construct([seg1, seg2])
        
        generator = MockAutoregressiveGenerator(sequence_length=1)
        generator.assign(seg1)
        generator.assign(seg2)
        
        # Constraint requires both segments
        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0)
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        # Simulate processing seg1 first: only seg1 has candidates
        seg1.create_candidates(6)
        # seg2 has no candidates yet (beam search processes segments sequentially)
        seg2.candidate_sequences = []
        
        optimizer._score_energy_active_constraints()
        
        # Should assign zero energy to all candidates since constraint requires seg2
        assert len(optimizer.energy_scores) == 6, f"Expected 6 scores, got {len(optimizer.energy_scores)}"
        assert all(e == 0.0 for e in optimizer.energy_scores), f"Expected all 0.0, got {optimizer.energy_scores}"
    
    def test_partial_constraint_applicability(self):
        """Test when some constraints are applicable and others aren't."""
        seg1 = create_segment("A")
        seg2 = create_segment("T")
        construct = Construct([seg1, seg2])
        
        generator = MockAutoregressiveGenerator(sequence_length=1)
        generator.assign(seg1)
        generator.assign(seg2)
        
        # One constraint on seg1 only, one on both
        constraint1 = Constraint(
            inputs=[seg1],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0)
        )
        constraint2 = Constraint(
            inputs=[seg1, seg2],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0)
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=2, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint1, constraint2],
            config=config,
            constraint_weights=[1.0, 2.0]
        )
        
        # Only seg1 has candidates
        seg1.create_candidates(4)
        for seq in seg1.candidate_sequences:
            seq.sequence = "GC"
        
        optimizer._score_energy_active_constraints()
        
        # Should only apply constraint1 (only on seg1)
        assert len(optimizer.energy_scores) == 4
        # Energies should be non-zero (from constraint1)
        # constraint2 is skipped because seg2 not ready


class TestBeamSearchIntegration:
    """Integration tests for full beam search workflow."""
    
    def test_single_segment_beam_search(self):
        """Test beam search with single segment."""
        segment = create_segment("AAAA")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=2)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, config: float(seq.sequence.count('A')),
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        # Set random seed for reproducibility
        np.random.seed(42)
        optimizer.run()
        
        # Should have 2 selected sequences (beam_width)
        assert len(segment.selected_sequences) == 2
        # All should have length 2 (sequence_length)
        assert all(len(seq.sequence) == 2 for seq in segment.selected_sequences)
    
    def test_multiple_segment_beam_search(self):
        """Test beam search across multiple segments."""
        seg1 = create_segment("")
        seg2 = create_segment("")
        seg3 = create_segment("")
        construct = Construct([seg1, seg2, seg3])
        
        generator = MockAutoregressiveGenerator(sequence_length=2)
        for seg in [seg1, seg2, seg3]:
            generator.assign(seg)
        
        constraint = Constraint(
            inputs=[seg1, seg2, seg3],
            scoring_function=lambda seq, config: float(len(seq.sequence)),
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=3, candidates_per_beam=4, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        # Each segment should have beam_width selected sequences
        for seg in [seg1, seg2, seg3]:
            assert len(seg.selected_sequences) == 3
            assert all(len(seq.sequence) == 2 for seq in seg.selected_sequences)
        
        # Check joined sequences
        joined = construct.joined_sequences
        assert len(joined) == 3
        # Each should be 2*3 = 6 chars (2 per segment, 3 segments)
        assert all(len(seq.sequence) == 6 for seq in joined)
    
    def test_prompt_accumulation_across_segments(self):
        """Test that prompts accumulate across segments."""
        seg1 = create_segment("")
        seg2 = create_segment("")
        construct = Construct([seg1, seg2])
        
        # Generator that returns known sequences
        class DeterministicGenerator(MockAutoregressiveGenerator):
            def sample(self, prompt_seqs=None):
                for i, seq in enumerate(self._assigned_segment.candidate_sequences):
                    seq.sequence = "A"  # Always generate "A"
        
        generator = DeterministicGenerator(sequence_length=1)
        generator.assign(seg1)
        generator.assign(seg2)
        
        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=lambda seq, config: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=2, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="GGGG",
            constraints=[constraint],
            config=config
        )
        
        optimizer.run()
        
        # After segment 1, prompts should be "GGGG" + "A"
        # After segment 2, prompts should be "GGGG" + "A" + "A"
        # Joined sequences should show accumulated prompt
        joined = construct.joined_sequences
        # But wait - the segment sequences themselves don't include the prompt
        # They're just the generated tokens
        # The running_prompts track the accumulation
        
        # Check that prompts accumulated (should be 2 prompts)
        assert len(optimizer.running_prompts) == 2
        # Each prompt should have accumulated GGGG + A (from seg1) + A (from seg2)
        assert all("GGGGA" in p for p in optimizer.running_prompts)
    
    def test_multiple_constraints_with_weights(self):
        """Test beam search with multiple weighted constraints."""
        segment = create_segment("")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=10)
        generator.assign(segment)
        
        gc_constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=45.0, max_gc=55.0)
        )
        
        len_constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=10)
        )
        
        config = BeamSearchOptimizerConfig(beam_width=3, candidates_per_beam=5, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[gc_constraint, len_constraint],
            config=config,
            constraint_weights=[1.0, 2.0]
        )
        
        np.random.seed(42)
        optimizer.run()
        
        assert len(segment.selected_sequences) == 3
        assert all(len(seq.sequence) == 10 for seq in segment.selected_sequences)
    
    def test_constraint_waiting_for_segments(self):
        """Test that constraints wait for all input segments to be ready."""
        seg1 = create_segment("")
        seg2 = create_segment("")
        seg3 = create_segment("")
        construct = Construct([seg1, seg2, seg3])
        
        generator = MockAutoregressiveGenerator(sequence_length=2)
        for seg in [seg1, seg2, seg3]:
            generator.assign(seg)
        
        # Constraint on seg1 only (always applicable)
        constraint1 = Constraint(
            inputs=[seg1],
            scoring_function=lambda seq, config: 0.1,
            scoring_function_config={}
        )
        
        # Constraint on all three (only applicable at seg3)
        constraint2 = Constraint(
            inputs=[seg1, seg2, seg3],
            scoring_function=lambda seq, config: 0.2,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3, verbose=True)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint1, constraint2],
            config=config
        )
        
        np.random.seed(42)
        # This should work without errors
        optimizer.run()
        
        # All segments should have beam_width selected sequences
        for seg in [seg1, seg2, seg3]:
            assert len(seg.selected_sequences) == 2


class TestBeamSearchEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_beam_width_one(self):
        """Test beam search with beam_width=1 (greedy search)."""
        segment = create_segment("")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=5)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, config: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=1, candidates_per_beam=10, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        assert len(segment.selected_sequences) == 1
        assert len(optimizer.running_prompts) == 1
    
    def test_empty_initial_prompt(self):
        """Test beam search starting with empty prompt."""
        segment = create_segment("")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=3)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, config: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        assert all(p == "" for p in optimizer.running_prompts)
        
        np.random.seed(42)
        optimizer.run()
        
        # Should work fine with empty initial prompt
        assert len(segment.selected_sequences) == 2
    
    def test_large_beam_width(self):
        """Test with large beam width."""
        segment = create_segment("")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=2)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, config: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=50, candidates_per_beam=2, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        assert len(segment.selected_sequences) == 50
        assert len(optimizer.running_prompts) == 50
    
    def test_identical_energies(self):
        """Test when all candidates have identical energies."""
        segment = create_segment("")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=2)
        generator.assign(segment)
        
        # Constraint that returns same score for everything
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, config: 5.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=3, candidates_per_beam=4, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        # Should still select beam_width sequences (arbitrary which ones)
        assert len(segment.selected_sequences) == 3
        # All energies should be 5.0
        optimizer.score_energy()
        assert all(abs(e - 5.0) < 0.001 for e in optimizer.energy_scores[:3])
    
    def test_top_sequences_property(self):
        """Test that construct.joined_sequences returns correct results."""
        seg1 = create_segment("")
        seg2 = create_segment("")
        construct = Construct([seg1, seg2])
        
        generator = MockAutoregressiveGenerator(sequence_length=3)
        for seg in [seg1, seg2]:
            generator.assign(seg)
        
        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=lambda seq, config: float(seq.sequence.count('G')),
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=4, candidates_per_beam=5, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        # Get top sequences
        top_seqs = construct.joined_sequences
        
        assert len(top_seqs) == 4  # beam_width
        # Each should be concatenation of two 3-char segments
        assert all(len(seq.sequence) == 6 for seq in top_seqs)


class TestBeamSearchVerboseOutput:
    """Test verbose output and logging."""
    
    def test_verbose_mode(self):
        """Test that verbose mode doesn't crash."""
        segment = create_segment("")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=2)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, config: 0.0,
            scoring_function_config={}
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3, verbose=True)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        # Should not crash with verbose=True
        optimizer.run()
        
        assert len(segment.selected_sequences) == 2
    
    def test_verbose_with_small_candidate_pool(self):
        """Test verbose output with small candidate pool (prints all candidates)."""
        segment = create_segment("")
        construct = Construct([segment])
        
        generator = MockAutoregressiveGenerator(sequence_length=2)
        generator.assign(segment)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, config: 0.0,
            scoring_function_config={}
        )
        
        # Small enough to trigger detailed printing (<= 10)
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=3, verbose=True)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        assert len(segment.selected_sequences) == 2


class TestBeamSearchConstraintInteraction:
    """Test interaction between beam search and different constraint types."""
    
    def test_per_segment_constraints(self):
        """Test with different constraints on different segments."""
        seg1 = create_segment("")
        seg2 = create_segment("")
        construct = Construct([seg1, seg2])
        
        generator = MockAutoregressiveGenerator(sequence_length=5)
        for seg in [seg1, seg2]:
            generator.assign(seg)
        
        # Constraint only on seg1
        constraint1 = Constraint(
            inputs=[seg1],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0)
        )
        
        # Constraint only on seg2
        constraint2 = Constraint(
            inputs=[seg2],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=30.0, max_gc=70.0)
        )
        
        config = BeamSearchOptimizerConfig(beam_width=3, candidates_per_beam=4, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint1, constraint2],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        assert len(seg1.selected_sequences) == 3
        assert len(seg2.selected_sequences) == 3
    
    def test_concatenated_multi_segment_constraint(self):
        """Test constraint that concatenates multiple segments."""
        seg1 = create_segment("")
        seg2 = create_segment("")
        seg3 = create_segment("")
        construct = Construct([seg1, seg2, seg3])
        
        generator = MockAutoregressiveGenerator(sequence_length=3)
        for seg in [seg1, seg2, seg3]:
            generator.assign(seg)
        
        # Constraint on concatenated sequence
        constraint = Constraint(
            inputs=[seg1, seg2, seg3],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=9),
            concatenate=True
        )
        
        config = BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=5, verbose=False)
        optimizer = BeamSearchOptimizer(
            construct=construct,
            generator=generator,
            prompt="",
            constraints=[constraint],
            config=config
        )
        
        np.random.seed(42)
        optimizer.run()
        
        # All segments should be processed
        for seg in [seg1, seg2, seg3]:
            assert len(seg.selected_sequences) == 2
        
        # Joined sequences should have correct total length
        joined = construct.joined_sequences
        assert all(len(seq.sequence) == 9 for seq in joined)


class TestBeamSearchConfigValidation:
    """Test configuration validation."""
    
    def test_invalid_beam_width(self):
        """Test that invalid beam_width raises error."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(beam_width=0, candidates_per_beam=5)
        
        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(beam_width=-1, candidates_per_beam=5)
    
    def test_invalid_candidates_per_beam(self):
        """Test that invalid candidates_per_beam raises error."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=0)
        
        with pytest.raises(ValidationError):
            BeamSearchOptimizerConfig(beam_width=2, candidates_per_beam=-5)
    
    def test_valid_config_values(self):
        """Test that valid configurations are accepted."""
        config1 = BeamSearchOptimizerConfig(beam_width=1, candidates_per_beam=1)
        assert config1.beam_width == 1
        assert config1.candidates_per_beam == 1
        
        config2 = BeamSearchOptimizerConfig(beam_width=100, candidates_per_beam=100)
        assert config2.beam_width == 100
        assert config2.candidates_per_beam == 100
        
        config3 = BeamSearchOptimizerConfig(beam_width=5, candidates_per_beam=10, verbose=False)
        assert config3.verbose is False
