"""
Tests for TopKOptimizer functionality.

Minimal tests verifying core behavior of the TopKOptimizer.
"""

import heapq
import logging
import math
import random

import pytest

from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.core import Constraint, Construct, Segment, Sequence
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig


class TestTopKOptimizerStandardMode:
    """Test TopKOptimizer in standard mode (no energy_threshold)."""

    def test_topk_optimizer_initialization(self):
        """Test basic TopKOptimizer initialization in standard mode."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = TopKOptimizerConfig(
            num_samples=10,
            k=5,
            batch_size=1,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 10
        assert optimizer.batch_size == 1
        assert optimizer.k == 5
        assert optimizer.energy_threshold is None  # Standard mode
        assert optimizer.num_selected == 5
        assert len(optimizer.constraints) == 1
        assert len(optimizer.generators) == 1

    def test_topk_returns_k_constructs(self):
        """Test that TopK optimizer returns exactly k constructs."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = TopKOptimizerConfig(
            num_samples=20,
            k=3,
            batch_size=1,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3
        assert optimizer.num_selected == 3

        # Verify energies are sorted (best first)
        for i in range(len(optimizer.energy_scores) - 1):
            assert optimizer.energy_scores[i] <= optimizer.energy_scores[i + 1]

    def test_topk_keeps_best_candidates(self):
        """Test that TopK keeps the best (lowest energy) candidates."""
        segment = Segment(sequence="AAAAAAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=2)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 80.0, "max_gc": 100.0},
        )

        config = TopKOptimizerConfig(
            num_samples=50,
            k=5,
            batch_size=1,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        best_energy = optimizer.energy_scores[0]
        worst_energy = optimizer.energy_scores[-1]
        assert best_energy <= worst_energy

    def test_topk_with_multiple_generators(self):
        """Test TopK with multiple generators applied sequentially."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen1.assign(segment)

        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen2.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = TopKOptimizerConfig(
            num_samples=10,
            k=3,
            batch_size=1,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3

    def test_topk_rounds_start_from_initial_state(self):
        """Test that each round starts from the initial state, not cumulative."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        initial_seq = segment.selected_sequences[0].sequence

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )

        config = TopKOptimizerConfig(
            num_samples=5,
            k=5,
            batch_size=1,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        for i in range(5):
            seq = segment.selected_sequences[i].sequence
            diff_count = sum(1 for a, b in zip(initial_seq, seq) if a != b)
            assert diff_count == 1, f"Expected 1 mutation, got {diff_count} differences"

    def test_run_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )

        config = TopKOptimizerConfig(
            num_samples=5,
            k=3,
            batch_size=1,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Capture original state (base class initializes selected_sequences to num_selected by cycling)
        original_seq = segment.selected_sequences[0].sequence
        assert original_seq == "ATCGATCG"
        assert len(segment.selected_sequences) == 3  # Cycled from single source

        # First run
        optimizer.run()
        assert len(segment.selected_sequences) == 3
        assert optimizer._initial_state is not None

        # Verify captured state contains cycled original sequences
        assert len(optimizer._initial_state['segments']) == 1
        captured_selected = optimizer._initial_state['segments'][0]['selected']
        assert len(captured_selected) == 3  # Cycled to num_selected
        assert all(s['sequence'] == original_seq for s in captured_selected)

        # Verify energy scores captured
        assert 'energy_scores' in optimizer._initial_state

        # Verify heap was cleared (TopK-specific state)
        assert len(optimizer._energy_heap) == 3  # Has k entries after run

        # Manually modify sequences to invalid values to verify restore
        for seq in segment.selected_sequences:
            seq.sequence = "GGGGGGGG"

        # Second run should restart - heap should be cleared and sequences restored
        optimizer.run()
        assert len(segment.selected_sequences) == 3
        assert len(optimizer._energy_heap) == 3  # Rebuilt from scratch

        # Verify sequences were restored (not all G's - restoration happened)
        assert any(seq.sequence != "GGGGGGGG" for seq in segment.selected_sequences)

    def test_topk_with_batch_size(self):
        """Test TopK with batch_size > 1 for efficient batching."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = TopKOptimizerConfig(
            num_samples=20,
            k=3,
            batch_size=5,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 20
        assert optimizer.batch_size == 5
        assert optimizer.k == 3

        optimizer.run()

        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3

        for i in range(len(optimizer.energy_scores) - 1):
            assert optimizer.energy_scores[i] <= optimizer.energy_scores[i + 1]

    def test_topk_rounds_up_num_samples(self, caplog):
        """Test TopK rounds up num_samples when not divisible by batch_size."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # 10 samples with batch_size=3 should round up to 12
        with caplog.at_level(logging.WARNING):
            config = TopKOptimizerConfig(
                num_samples=10,
                k=5,
                batch_size=3,
                verbose=False
            )
            optimizer = TopKOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint],
                config=config,
            )

        # Check that num_samples was rounded up
        assert optimizer.num_samples == 12
        assert "Rounding up to 12" in caplog.text

        optimizer.run()

        assert len(segment.selected_sequences) == 5
        assert len(optimizer.energy_scores) == 5

    def test_inf_and_nan_energy_rejection(self):
        """Test that TopK optimizer skips inf/nan energies from heap."""
        import math

        from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
            GCContentConfig,
        )

        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=3)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
            threshold=0.0,
        )

        config = TopKOptimizerConfig(
            num_samples=100,
            k=5,
            batch_size=10,
            verbose=False
        )

        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(optimizer.energy_scores) > 0
        assert len(optimizer.energy_scores) <= config.k
        for energy in optimizer.energy_scores:
            assert not math.isinf(energy)
            assert not math.isnan(energy)


class TestTopKOptimizerThresholdMode:
    """Test TopKOptimizer in threshold mode (energy_threshold set)."""

    def test_threshold_mode_initialization(self):
        """Test TopKOptimizer initialization in threshold mode."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = TopKOptimizerConfig(
            num_samples=100,
            energy_threshold=0.5,
            k=3,
            batch_size=2,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 100
        assert optimizer.energy_threshold == 0.5  # Threshold mode
        assert optimizer.k == 3

    def test_threshold_mode_stops_when_threshold_met(self):
        """Test that threshold mode stops early when threshold is met."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # High threshold that should be easily met
        config = TopKOptimizerConfig(
            num_samples=1000,
            energy_threshold=100.0,  # Very high threshold, easily met
            k=3,
            batch_size=2,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3

    def test_threshold_mode_respects_num_samples(self):
        """Test that threshold mode stops at num_samples if threshold not met."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # Very low threshold that won't be met
        config = TopKOptimizerConfig(
            num_samples=20,
            energy_threshold=0.0,  # Impossible to meet (energy would need to be negative)
            k=3,
            batch_size=5,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # Should have generated all num_samples and kept top k
        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3


class TestTopKOptimizerValidation:
    """Test TopKOptimizer config validation."""

    def test_k_cannot_exceed_num_samples(self):
        """Test that k cannot exceed num_samples."""
        with pytest.raises(ValueError, match="k \\(100\\) cannot exceed num_samples \\(10\\)"):
            _ = TopKOptimizerConfig(
                num_samples=10,
                k=100,
            )

    def test_default_is_standard_mode(self):
        """Test that default (no energy_threshold) is standard mode."""
        config = TopKOptimizerConfig(num_samples=10, k=5)
        assert config.energy_threshold is None


class TestTopKOptimizerInternals:
    """Test TopKOptimizer internal methods."""

    def test_sort_topk_by_energy(self):
        """Test _sort_topk_by_energy correctly sorts sequences by energy."""
        # Create optimizer with minimal setup
        segment1 = Segment(sequence="ATCG", sequence_type="dna")
        segment2 = Segment(sequence="GCTA", sequence_type="dna")
        construct = Construct([segment1, segment2])

        gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
        gen.assign(segment1)

        constraint = Constraint(
            inputs=[segment1],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = TopKOptimizerConfig(num_samples=5, k=3, batch_size=1)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Manually populate heap and selected_sequences with unsorted data
        # Simulate having 4 sequences with energies: [5.0, 2.0, 8.0, 1.0]
        energies = [5.0, 2.0, 8.0, 1.0]
        sequences_seg1 = [
            Sequence("ATCG", "dna"),
            Sequence("ATCC", "dna"),
            Sequence("ATCA", "dna"),
            Sequence("ATCT", "dna"),
        ]
        sequences_seg2 = [
            Sequence("GCTA", "dna"),
            Sequence("GCTC", "dna"),
            Sequence("GCTG", "dna"),
            Sequence("GCTT", "dna"),
        ]

        # Build heap with negated energies
        optimizer._energy_heap = []
        for idx, energy in enumerate(energies):
            heapq.heappush(optimizer._energy_heap, (-energy, idx))

        # Populate selected_sequences (unsorted)
        segment1.selected_sequences = sequences_seg1
        segment2.selected_sequences = sequences_seg2

        # Call _sort_topk_by_energy
        optimizer._sort_topk_by_energy()

        # Verify energy_scores are sorted (best first: lowest to highest)
        assert optimizer.energy_scores == [1.0, 2.0, 5.0, 8.0]

        # Verify selected_sequences are reordered to match sorted energies
        assert segment1.selected_sequences[0].sequence == "ATCT"  # energy 1.0
        assert segment1.selected_sequences[1].sequence == "ATCC"  # energy 2.0
        assert segment1.selected_sequences[2].sequence == "ATCG"  # energy 5.0
        assert segment1.selected_sequences[3].sequence == "ATCA"  # energy 8.0

        assert segment2.selected_sequences[0].sequence == "GCTT"  # energy 1.0
        assert segment2.selected_sequences[1].sequence == "GCTC"  # energy 2.0
        assert segment2.selected_sequences[2].sequence == "GCTA"  # energy 5.0
        assert segment2.selected_sequences[3].sequence == "GCTG"  # energy 8.0

    def test_sort_topk_by_energy_empty_heap(self):
        """Test _sort_topk_by_energy handles empty heap by padding with inf energies."""
        segment = Segment(sequence="ATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = TopKOptimizerConfig(num_samples=5, k=3, batch_size=1)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Capture initial state
        optimizer._capture_initial_state()

        # Empty heap
        optimizer._energy_heap = []
        segment.selected_sequences = []

        # Should handle gracefully by padding to k entries
        optimizer._sort_topk_by_energy()

        # Pads with inf energies and empty placeholder sequences
        assert len(optimizer.energy_scores) == 3
        assert all(math.isinf(e) for e in optimizer.energy_scores)
        assert len(segment.selected_sequences) == 3
        # All padded sequences should be empty placeholders
        assert all(seq.sequence == "" for seq in segment.selected_sequences)

    def test_all_candidates_rejected_by_filter(self):
        """Test TopK optimizer handles case where all candidates are rejected by filter.

        This is a regression test for a bug where the optimizer would crash with
        RuntimeError when all candidates had inf/nan energies.
        """
        from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
            GCContentConfig,
        )

        segment = Segment(sequence="AAAAAAAAAA", sequence_type="dna")  # 0% GC
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)  # Only 1 mutation, unlikely to reach 99% GC
        )
        gen.assign(segment)

        # Extremely strict filter - requires 99-100% GC content (effectively impossible)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=99.0, max_gc=100.0),
            threshold=0.0,  # Filter mode - rejected candidates get inf energy
        )

        config = TopKOptimizerConfig(
            num_samples=20,
            k=5,
            batch_size=5,
            verbose=False
        )

        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Should not crash - should handle gracefully by padding with inf energies
        optimizer.run()

        # Verify the optimizer completed and has k results
        assert len(optimizer.energy_scores) == 5
        assert len(segment.selected_sequences) == 5

        # All energies should be inf since no valid candidates were found
        assert all(math.isinf(e) for e in optimizer.energy_scores)

        # All sequences should be empty placeholders
        assert all(seq.sequence == "" for seq in segment.selected_sequences)

    def test_partial_candidates_rejected_by_filter(self):
        """Test TopK optimizer handles case where some but not all candidates pass filter."""
        from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
            GCContentConfig,
        )

        segment = Segment(sequence="ATCGATCGAT", sequence_type="dna")  # 40% GC
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=2)
        )
        gen.assign(segment)

        # Moderate filter - requires 30-70% GC (some will pass, some won't)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=30.0, max_gc=70.0),
            threshold=0.0,  # Filter mode
        )

        config = TopKOptimizerConfig(
            num_samples=50,
            k=10,
            batch_size=5,
            verbose=False
        )

        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # Should have k results
        assert len(optimizer.energy_scores) == 10
        assert len(segment.selected_sequences) == 10


class TestTopKOptimizerTrajectoryPreservation:
    """Test that TopK preserves trajectory diversity from handoff."""

    def test_topk_preserves_input_diversity(self):
        """Test that TopK uses each candidate's own initial sequence, not just the first.

        This verifies the fix for the single-seed bug where TopK was discarding
        diversity by always using candidates[0] as the mutation seed.
        """
        # Create segment with 3 distinct initial sequences (simulating handoff from previous optimizer)
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])

        # Pre-populate selected_sequences with diverse seeds (simulating previous optimizer output)
        segment.selected_sequences = [
            Sequence("AAAA", "dna"),
            Sequence("CCCC", "dna"),
            Sequence("GGGG", "dna"),
        ]

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=0)  # No mutations - keeps original
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        # k=6 with 3 source sequences → cycling produces [A, C, G, A, C, G]
        # num_samples must be >= k, and batch_size is the pool size
        config = TopKOptimizerConfig(
            num_samples=6,  # Generate 6 samples total
            k=6,            # Keep top 6
            batch_size=6,   # All at once
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Initialize pools (this cycles through the 3 seeds to fill 6 slots)
        optimizer._initialize_sequence_pools()
        optimizer._capture_initial_state()

        # Run one sampling round (with 0 mutations, sequences should stay as their seeds)
        optimizer._run_sampling_round(0)

        # Verify that candidates come from different seeds (cycled pattern)
        # With the fix: candidates should be [AAAA, CCCC, GGGG, AAAA, CCCC, GGGG]
        # With the bug: candidates would all be [AAAA, AAAA, AAAA, AAAA, AAAA, AAAA]
        candidates = [seq.sequence for seq in segment.candidate_sequences]

        # At least 2 unique sequences should be present (proving diversity is preserved)
        unique_candidates = set(candidates)
        assert len(unique_candidates) >= 2, (
            f"TopK should preserve input diversity but found only {unique_candidates}. "
            f"This suggests all candidates are seeded from the first sequence."
        )

        # Verify the expected cycling pattern
        assert candidates == ["AAAA", "CCCC", "GGGG", "AAAA", "CCCC", "GGGG"], (
            f"Expected cycled pattern but got {candidates}"
        )

    def test_topk_batch_coherence_across_segments(self):
        """Test that batch coherence is maintained across multiple segments.

        Each batch index should use the same source index across all segments,
        preserving the semantic pairing from the previous optimizer.
        """
        # Create two segments with matching diverse seeds
        segment1 = Segment(sequence="AAAA", sequence_type="dna", label="seg1")
        segment2 = Segment(sequence="TTTT", sequence_type="dna", label="seg2")
        construct = Construct([segment1, segment2])

        # Pre-populate with paired sequences (index 0 pairs: AAAA-TTTT, index 1 pairs: CCCC-GGGG)
        segment1.selected_sequences = [
            Sequence("AAAA", "dna"),
            Sequence("CCCC", "dna"),
        ]
        segment2.selected_sequences = [
            Sequence("TTTT", "dna"),
            Sequence("GGGG", "dna"),
        ]

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=0)
        )
        gen1.assign(segment1)

        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=0)
        )
        gen2.assign(segment2)

        # Use separate constraints for each segment
        constraint1 = Constraint(
            inputs=[segment1],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )
        constraint2 = Constraint(
            inputs=[segment2],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = TopKOptimizerConfig(
            num_samples=4,
            k=4,
            batch_size=4,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[constraint1, constraint2],
            config=config,
        )

        optimizer._initialize_sequence_pools()
        optimizer._capture_initial_state()
        optimizer._run_sampling_round(0)

        # Verify batch coherence: index i in segment1 should pair with index i in segment2
        candidates1 = [seq.sequence for seq in segment1.candidate_sequences]
        candidates2 = [seq.sequence for seq in segment2.candidate_sequences]

        # Expected: [AAAA, CCCC, AAAA, CCCC] and [TTTT, GGGG, TTTT, GGGG]
        assert candidates1 == ["AAAA", "CCCC", "AAAA", "CCCC"]
        assert candidates2 == ["TTTT", "GGGG", "TTTT", "GGGG"]

        # Verify pairing is preserved (same index = same source trajectory)
        for i in range(4):
            # Index 0,2 should both be from source 0 (AAAA-TTTT pair)
            # Index 1,3 should both be from source 1 (CCCC-GGGG pair)
            expected_source = i % 2
            if expected_source == 0:
                assert candidates1[i] == "AAAA" and candidates2[i] == "TTTT"
            else:
                assert candidates1[i] == "CCCC" and candidates2[i] == "GGGG"


class TestTopKCustomLogging:
    """Regression: custom_logging must not corrupt heap indices (Bug 1).

    Previously, ``_log_round_progress`` called ``_sort_topk_by_energy()`` when
    ``custom_logging`` was set, reordering ``selected_sequences`` in-place while
    the heap still held the old indices.
    """

    def test_custom_logging_does_not_corrupt_results(self):
        """Results with custom_logging must match results without it (same seed)."""
        seed = 42

        def run_topk(custom_logging_fn=None):
            random.seed(seed)
            segment = Segment(sequence="ATCGATCG", sequence_type="dna")
            construct = Construct([segment])
            gen = UniformMutationGenerator(
                UniformMutationGeneratorConfig(num_mutations=2)
            )
            gen.assign(segment)
            constraint = Constraint(
                inputs=[segment],
                function=gc_content_constraint,
                function_config={"min_gc": 40.0, "max_gc": 60.0},
            )
            config = TopKOptimizerConfig(
                num_samples=30, k=5, batch_size=1, verbose=False,
            )
            optimizer = TopKOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint],
                config=config,
                custom_logging=custom_logging_fn,
            )
            optimizer.run()
            return (
                [s.sequence for s in segment.selected_sequences],
                optimizer.energy_scores[:],
            )

        seqs_no_log, energies_no_log = run_topk(custom_logging_fn=None)

        log_calls = []
        seqs_with_log, energies_with_log = run_topk(
            custom_logging_fn=lambda r, s: log_calls.append(r)
        )

        assert sorted(seqs_no_log) == sorted(seqs_with_log)
        assert sorted(energies_no_log) == sorted(energies_with_log)
        assert len(log_calls) > 0

    def test_custom_logging_callback_receives_segments(self):
        """Verify the custom_logging callback receives the correct arguments."""
        received = []

        def logger_fn(round_idx, segments):
            received.append((round_idx, len(segments)))

        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 8},
        )
        config = TopKOptimizerConfig(
            num_samples=5, k=3, batch_size=1, verbose=False,
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
            custom_logging=logger_fn,
        )
        optimizer.run()

        assert len(received) == 5
        for round_idx, num_segments in received:
            assert isinstance(round_idx, int)
            assert num_segments == 1


class TestTopKLabelDeduplication:
    """Regression: optimizer must deduplicate constraint labels (Bug 3)."""

    def test_duplicate_constraint_labels_are_deduplicated(self):
        """Two constraints with the same label should be auto-renamed."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 20.0, "max_gc": 80.0},
        )
        assert constraint1.label == constraint2.label

        config = TopKOptimizerConfig(
            num_samples=5, k=3, batch_size=1, verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint1, constraint2],
            config=config,
        )
        optimizer.run()
        assert constraint1.label != constraint2.label

    def test_deduplication_is_idempotent(self):
        """Calling _deduplicate_constraint_labels twice must not accumulate suffixes."""
        segment = Segment(sequence="AAAA", sequence_type="dna")
        construct = Construct([segment])
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 20.0, "max_gc": 80.0},
        )

        config = TopKOptimizerConfig(
            num_samples=5, k=3, batch_size=1, verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint1, constraint2],
            config=config,
        )

        optimizer._deduplicate_constraint_labels()
        label_after_first = constraint2.label
        optimizer._deduplicate_constraint_labels()
        label_after_second = constraint2.label

        assert label_after_first == label_after_second
        assert label_after_first.count("_1") == 1


class TestTopKCandidateTracking:
    """Test candidate_results tracking in TopK history."""

    def test_candidate_tracking(self):
        """History has candidate_results with 'Not in top-k' for rejected candidates."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen.assign(segment)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=TopKOptimizerConfig(
                num_samples=20, k=3, batch_size=5, verbose=False
            ),
        )
        optimizer.run()

        valid_rejectors = {"Not in top-k"}
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
