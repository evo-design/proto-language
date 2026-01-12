"""
Tests for TopKOptimizer functionality.

Minimal tests verifying core behavior of the TopKOptimizer.
"""

import pytest
import logging
from proto_language.language.core import (
    Construct, Segment, Constraint)
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig
)
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig
from proto_language.language.constraint import gc_content_constraint, sequence_length_constraint


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

    def test_infinite_energy_rejection(self):
        """Test that TopK optimizer skips inf/nan energies from heap."""
        import math
        from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig

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
