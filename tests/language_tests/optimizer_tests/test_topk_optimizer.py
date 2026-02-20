"""
Tests for TopKOptimizer functionality.

Minimal tests verifying core behavior of the TopKOptimizer.
"""

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
            num_results=5,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 10
        assert optimizer.samples_per_round == 1  # Default
        assert optimizer.energy_threshold is None  # Standard mode
        assert optimizer.num_results == 5
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
            num_results=3,
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
        assert optimizer.num_results == 3

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
            num_results=5,
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
            num_results=3,
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
            num_results=5,
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
            num_results=3,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Capture original state (base class initializes selected_sequences to num_results by cycling)
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
        assert len(captured_selected) == 3  # Cycled to num_results
        assert all(s['sequence'] == original_seq for s in captured_selected)

        # Verify energy scores captured
        assert 'energy_scores' in optimizer._initial_state

        # Verify sorted list was populated (TopK-specific state)
        assert len(optimizer._selected_energies) == 3  # Has k entries after run

        # Manually modify sequences to invalid values to verify restore
        for seq in segment.selected_sequences:
            seq.sequence = "GGGGGGGG"

        # Second run should restart - sorted list should be cleared and sequences restored
        optimizer.run()
        assert len(segment.selected_sequences) == 3
        assert len(optimizer._selected_energies) == 3  # Rebuilt from scratch

        # Verify sequences were restored (not all G's - restoration happened)
        assert any(seq.sequence != "GGGGGGGG" for seq in segment.selected_sequences)

    def test_topk_with_candidates_per_round(self):
        """Test TopK with samples_per_round > 1 for efficient batching."""
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
            num_results=3,
            samples_per_round=5,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.num_samples == 20
        assert optimizer.samples_per_round == 5
        assert optimizer.num_results == 3

        optimizer.run()

        assert len(segment.selected_sequences) == 3
        assert len(optimizer.energy_scores) == 3

        for i in range(len(optimizer.energy_scores) - 1):
            assert optimizer.energy_scores[i] <= optimizer.energy_scores[i + 1]

    def test_topk_rounds_up_num_samples(self):
        """Test TopK rounds num_samples up to nearest samples_per_round multiple."""
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

        # 10 samples with samples_per_round=3 → rounded up to 12 (4 rounds)
        config = TopKOptimizerConfig(
            num_samples=10,
            num_results=5,
            samples_per_round=3,
            verbose=False
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # num_samples rounded up to nearest multiple of samples_per_round
        assert optimizer.num_samples == 12
        assert optimizer.num_samples // optimizer.samples_per_round == 4

        optimizer.run()

        assert len(segment.selected_sequences) == 5
        assert len(optimizer.energy_scores) == 5

    def test_inf_and_nan_energy_rejection(self):
        """Test that TopK optimizer skips inf/nan energies."""
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
            num_results=5,
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
        assert len(optimizer.energy_scores) <= config.num_results
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
            num_results=3,
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
        assert optimizer.num_results == 3

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
            num_results=3,
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
            num_results=3,
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

    def test_num_results_cannot_exceed_num_samples(self):
        """Test that num_results cannot exceed num_samples."""
        with pytest.raises(ValueError, match="num_results \\(100\\) cannot exceed num_samples \\(10\\)"):
            _ = TopKOptimizerConfig(
                num_samples=10,
                num_results=100,
            )

    def test_default_is_standard_mode(self):
        """Test that default (no energy_threshold) is standard mode."""
        config = TopKOptimizerConfig(num_samples=10, num_results=5)
        assert config.energy_threshold is None


class TestTopKOptimizerInternals:
    """Test TopKOptimizer internal methods."""

    def test_selected_sequences_always_sorted(self):
        """Test that selected_sequences are always sorted by energy (ascending)."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=2))
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        config = TopKOptimizerConfig(num_samples=30, num_results=5)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # energy_scores should be sorted ascending
        assert optimizer.energy_scores == sorted(optimizer.energy_scores)
        # _selected_energies should match energy_scores
        assert optimizer._selected_energies == optimizer.energy_scores

    def test_empty_selected_when_all_rejected(self):
        """Test that TopK returns empty lists when all candidates are rejected."""
        segment = Segment(sequence="ATCG", sequence_type="dna")
        construct = Construct([segment])

        gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config={"target_length": 4},
        )

        config = TopKOptimizerConfig(num_samples=5, num_results=3)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Capture initial state and clear selected
        optimizer._capture_initial_state()

        # Verify empty state
        assert optimizer._selected_energies == []
        assert optimizer.energy_scores == []
        assert segment.selected_sequences == []

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
            num_results=5,
            verbose=False
        )

        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        # Should not crash - returns empty results when no valid candidates found
        optimizer.run()

        # No valid candidates found — empty results (no padding)
        assert len(optimizer.energy_scores) == 0
        assert len(segment.selected_sequences) == 0

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
            num_results=10,
            verbose=False
        )

        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # Should have results (up to k, may be fewer if some were rejected)
        assert len(optimizer.energy_scores) <= 10
        assert len(segment.selected_sequences) == len(optimizer.energy_scores)


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

        # num_results=6 with 3 source sequences → cycling produces [A, C, G, A, C, G]
        # samples_per_round=6 means 6 candidates per round
        config = TopKOptimizerConfig(
            num_samples=6,              # Generate 6 samples total
            num_results=6,              # Keep top 6
            samples_per_round=6,
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
            num_results=4,
            samples_per_round=4,
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
    """Regression: custom_logging must not corrupt results (Bug 1).

    Previously, logging could corrupt the sorted top-k list by reordering
    ``selected_sequences`` while indices were still in use.
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
                num_samples=30, num_results=5, verbose=False,
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

        def logger_fn(step, segments):
            received.append((step, len(segments)))

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
            num_samples=5, num_results=3, verbose=False,
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
        for step, num_segments in received:
            assert isinstance(step, int)
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
            num_samples=5, num_results=3, verbose=False
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
            num_samples=5, num_results=3, verbose=False
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
                num_samples=20,
                num_results=3,
                verbose=False,
                track_candidates=True,
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


class TestTopKTrackingInterval:
    """Test tracking_interval in TopK optimizer."""

    def test_tracking_interval(self):
        """tracking_interval=2 reduces history snapshots."""
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
            num_samples=10,
            num_results=3,
            verbose=False,
            tracking_interval=2,
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        optimizer.run()

        # 10 rounds with interval=2: rounds 2,4,6,8,10 + step 0
        saved_steps = {entry["time_step"] for entry in optimizer.history}
        assert saved_steps == {0, 2, 4, 6, 8, 10}

    def test_tracking_interval_with_threshold_early_exit(self):
        """Threshold early-exit forces a final snapshot even on non-interval rounds."""
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
            num_samples=1000,
            num_results=3,
            verbose=False,
            tracking_interval=5,
            energy_threshold=100.0,  # Very high — easily met early
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )
        optimizer.run()

        # Threshold should be met well before round 1000
        num_sampling_rounds = config.num_samples // optimizer.samples_per_round
        assert len(optimizer.history) < num_sampling_rounds

        # The last snapshot should reflect the round where threshold was met
        saved_steps = sorted(entry["time_step"] for entry in optimizer.history)
        last_saved = saved_steps[-1]
        # Final snapshot must exist and be > 0 (not just the initial snapshot)
        assert last_saved > 0
        # If threshold was met on a non-interval round, we still get a snapshot
        # (the fix ensures this)


class TestTopKMetadata:
    """Test metadata preservation through TopK optimization."""

    def test_topk_preserves_initial_metadata_on_selected(self):
        """Initial user metadata should survive through TopK rounds to selected_sequences."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna", metadata={"user_key": "user_value"})
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
            num_results=3,
            verbose=False,
        )
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        for seq in segment.selected_sequences:
            assert seq._metadata.get("user_key") == "user_value"
