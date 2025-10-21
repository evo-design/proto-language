"""
Tests for TopKOptimizer functionality.

Minimal tests verifying core behavior of the TopKOptimizer.
"""

import pytest
from proto_language.language.core import (
    Construct, Segment, Constraint, SequenceType)
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig
)
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig
from proto_language.language.constraint import gc_content_constraint, sequence_length_constraint


class TestTopKOptimizer:
    """Test core TopKOptimizer functionality."""

    def test_topk_optimizer_initialization(self):
        """Test basic TopKOptimizer initialization."""
        segment = Segment(sequence="AAAA", sequence_type=SequenceType.DNA)
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=4, num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": 4}
        )

        config = TopKOptimizerConfig(rounds=10, k=5, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        assert optimizer.rounds == 10
        assert optimizer.k == 5
        assert optimizer.num_selected == 5
        assert len(optimizer.constraints) == 1
        assert len(optimizer.generators) == 1

    def test_topk_returns_k_constructs(self):
        """Test that TopK optimizer returns exactly k constructs."""
        segment = Segment(sequence="ATCGATCG", sequence_type=SequenceType.DNA)
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=8,
                num_mutations=1
            )
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0}
        )

        config = TopKOptimizerConfig(rounds=20, k=3, verbose=False)
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
        segment = Segment(sequence="AAAAAAAA", sequence_type=SequenceType.DNA)
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=8,
                num_mutations=2
            )
        )
        gen.assign(segment)

        # Constraint that prefers higher GC content (80-100%)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 80.0, "max_gc": 100.0}
        )

        config = TopKOptimizerConfig(rounds=50, k=5, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # Best candidate should have lower energy than worst
        best_energy = optimizer.energy_scores[0]
        worst_energy = optimizer.energy_scores[-1]

        assert best_energy <= worst_energy

    def test_topk_k_capped_at_rounds(self):
        """Test that k cannot exceed number of rounds."""
        segment = Segment(sequence="ATCG", sequence_type=SequenceType.DNA)
        construct = Construct([segment])

        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=4, num_mutations=1)
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": 4}
        )

        config = TopKOptimizerConfig(rounds=10, k=100, verbose=False)

        with pytest.raises(ValueError, match="k \\(100\\) cannot be greater than rounds \\(10\\)"):
            optimizer = TopKOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint],
                config=config,
            )

    def test_topk_with_multiple_generators(self):
        """Test TopK with multiple generators applied sequentially."""
        segment = Segment(sequence="AAAA", sequence_type=SequenceType.DNA)
        construct = Construct([segment])

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=4,
                num_mutations=1
            )
        )
        gen1.assign(segment)

        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=4,
                num_mutations=1
            )
        )
        gen2.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0}
        )

        config = TopKOptimizerConfig(rounds=10, k=3, verbose=False)
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
        segment = Segment(sequence="ATCGATCG", sequence_type=SequenceType.DNA)
        construct = Construct([segment])

        initial_seq = segment.selected_sequences[0].sequence

        # Generator that mutates 1 position
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=8,
                num_mutations=1
            )
        )
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": 8}
        )

        config = TopKOptimizerConfig(rounds=5, k=5, verbose=False)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=config,
        )

        optimizer.run()

        # All top-k sequences should differ from initial by only 1 mutation
        # (since each round starts fresh and applies 1 mutation)
        for i in range(5):
            seq = segment.selected_sequences[i].sequence
            diff_count = sum(1 for a, b in zip(initial_seq, seq) if a != b)
            assert diff_count == 1, f"Expected 1 mutation, got {diff_count} differences"
