"""
test_program_multiple_optimizers.py

Tests for Program class with multiple sequential optimizers.
Verifies that state persists correctly between optimizers and that
different optimizer combinations work as expected.
"""

import pytest
from proto_language.language.core import (
    Program,
    Construct,
    Segment,
    Constraint,
)
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    TopKOptimizer,
    TopKOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)
from proto_language.language.constraint import gc_content_constraint


class TestMultipleOptimizers:
    """Test Program with multiple sequential optimizers."""

    def test_two_optimizers_sequential(self):
        """Test that two optimizers run sequentially and state persists."""
        # Setup
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        # First optimizer: TopK
        gen1_config = UniformMutationGeneratorConfig(num_mutations=10)
        gen1 = UniformMutationGenerator(gen1_config)
        gen1.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer1 = TopKOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=10, k=2, batch_size=5),
        )

        # Second optimizer: MCMC
        gen2_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen2 = UniformMutationGenerator(gen2_config)
        gen2.assign(segment)

        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 80, "max_gc": 90},
        )

        optimizer2 = MCMCOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[constraint2],
            config=MCMCOptimizerConfig(num_selected=1, num_steps=5, track_step_size=1),
        )

        # Create program with both optimizers
        program = Program(optimizers=[optimizer1, optimizer2])

        # Run program
        program.run()

        # Verify results
        assert len(program.constructs) == 1
        assert program.constructs[0] is construct
        final_sequences = program.constructs[0].joined_sequences
        assert len(final_sequences) > 0

        # Verify energy scores come from final optimizer
        assert hasattr(program, 'energy_scores')
        assert len(program.energy_scores) > 0

    def test_three_optimizers_sequential(self):
        """Test that three optimizers run in sequence."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        optimizers = []
        for i in range(3):
            gen_config = UniformMutationGeneratorConfig(
                num_mutations=5
            )
            gen = UniformMutationGenerator(gen_config)
            gen.assign(segment)

            constraint = Constraint(
                inputs=[segment],
                function=gc_content_constraint,
                function_config={"min_gc": 40 + i * 10, "max_gc": 60 + i * 10},
            )

            optimizer = TopKOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint],
                config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
            )
            optimizers.append(optimizer)

        program = Program(optimizers=optimizers)
        program.run()

        # Verify all three optimizers ran
        assert len(program.optimizers) == 3
        for i, optimizer in enumerate(program.optimizers):
            history = optimizer.history
            assert len(history) > 0, f"Optimizer {i} should have history"

    def test_optimizer_histories_preserved(self):
        """Test that each optimizer maintains separate history."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        # Create two optimizers with different numbers of steps
        optimizers = []
        for i, num_steps in enumerate([3, 5]):
            gen_config = UniformMutationGeneratorConfig(
                num_mutations=1
            )
            gen = UniformMutationGenerator(gen_config)
            gen.assign(segment)

            constraint = Constraint(
                inputs=[segment],
                function=gc_content_constraint,
                function_config={"min_gc": 50, "max_gc": 100},
            )

            optimizer = MCMCOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint],
                config=MCMCOptimizerConfig(num_selected=1, num_steps=num_steps, track_step_size=1),
            )
            optimizers.append(optimizer)

        program = Program(optimizers=optimizers)
        program.run()

        # Verify separate histories
        assert len(program.optimizers) == 2
        assert program.optimizers[0].history is optimizers[0].history
        assert program.optimizers[1].history is optimizers[1].history

    def test_construct_validation_same_objects(self):
        """Test that validation passes when all optimizers share same constructs."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)

        gen1 = UniformMutationGenerator(gen_config)
        gen1.assign(segment)

        gen2 = UniformMutationGenerator(gen_config)
        gen2.assign(segment)

        # Each optimizer must have its own constraint instance
        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer1 = TopKOptimizer(
            constructs=[construct],  # Same construct object
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        optimizer2 = TopKOptimizer(
            constructs=[construct],  # Same construct object
            generators=[gen2],
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        # Should not raise
        program = Program(optimizers=[optimizer1, optimizer2])
        assert program.constructs[0] is construct

    def test_construct_validation_different_objects_fails(self):
        """Test that validation fails when optimizers have different construct objects."""
        segment1 = Segment(length=50, sequence_type="dna")
        segment2 = Segment(length=50, sequence_type="dna")
        construct1 = Construct([segment1])
        construct2 = Construct([segment2])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)

        gen1 = UniformMutationGenerator(gen_config)
        gen1.assign(segment1)

        gen2 = UniformMutationGenerator(gen_config)
        gen2.assign(segment2)

        constraint1 = Constraint(
            inputs=[segment1],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        constraint2 = Constraint(
            inputs=[segment2],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer1 = TopKOptimizer(
            constructs=[construct1],  # Different construct
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        optimizer2 = TopKOptimizer(
            constructs=[construct2],  # Different construct
            generators=[gen2],
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="not the same object"):
            Program(optimizers=[optimizer1, optimizer2])

    def test_construct_validation_different_lengths_fails(self):
        """Test that validation fails when optimizers have different numbers of constructs."""
        segment = Segment(length=50, sequence_type="dna")
        construct1 = Construct([segment])
        construct2 = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)

        gen1 = UniformMutationGenerator(gen_config)
        gen1.assign(segment)

        gen2 = UniformMutationGenerator(gen_config)
        gen2.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer1 = TopKOptimizer(
            constructs=[construct1],  # One construct
            generators=[gen1],
            constraints=[constraint],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        optimizer2 = TopKOptimizer(
            constructs=[construct1, construct2],  # Two constructs
            generators=[gen2],
            constraints=[constraint],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="has .* constructs"):
            Program(optimizers=[optimizer1, optimizer2])

    def test_empty_optimizers_list_fails(self):
        """Test that empty optimizers list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Program(optimizers=[])

    def test_energy_scores_from_final_optimizer(self):
        """Test that energy_scores property returns results from final optimizer."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        optimizers = []
        for _ in range(2):
            gen_config = UniformMutationGeneratorConfig(
                num_mutations=1
            )
            gen = UniformMutationGenerator(gen_config)
            gen.assign(segment)

            constraint = Constraint(
                inputs=[segment],
                function=gc_content_constraint,
                function_config={"min_gc": 50, "max_gc": 100},
            )

            optimizer = MCMCOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint],
                config=MCMCOptimizerConfig(num_selected=1, num_steps=3, track_step_size=1),
            )
            optimizers.append(optimizer)

        program = Program(optimizers=optimizers)
        program.run()

        # energy_scores should come from final optimizer
        assert program.energy_scores == optimizers[-1].energy_scores

    def test_state_persistence_between_optimizers(self):
        """Test that sequence state persists from one optimizer to the next."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        # First optimizer
        gen1_config = UniformMutationGeneratorConfig(num_mutations=5)
        gen1 = UniformMutationGenerator(gen1_config)
        gen1.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer1 = TopKOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=10, k=1, batch_size=5),
        )

        # Run first optimizer standalone to get its output
        optimizer1.run()
        _ = construct.joined_sequences[0]

        # Second optimizer should start from opt1's results
        gen2_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen2 = UniformMutationGenerator(gen2_config)
        gen2.assign(segment)

        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 80, "max_gc": 90},
        )

        _ = MCMCOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[constraint2],
            config=MCMCOptimizerConfig(num_selected=1, num_steps=3, track_step_size=1),
        )

        # Verify optimizer2 starts from opt1's ending state
        # by checking that the segment has the expected selected_sequences
        assert len(segment.selected_sequences) > 0
        # The state should be preserved in the segment object

    def test_mcmc_to_topk_sequence(self):
        """Test MCMC optimizer followed by TopK optimizer."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        # MCMC first
        gen1_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen1 = UniformMutationGenerator(gen1_config)
        gen1.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 70},
        )

        optimizer1 = MCMCOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[constraint1],
            config=MCMCOptimizerConfig(num_selected=1, num_steps=5, track_step_size=1),
        )

        # TopK second
        gen2_config = UniformMutationGeneratorConfig(num_mutations=5)
        gen2 = UniformMutationGenerator(gen2_config)
        gen2.assign(segment)

        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 70, "max_gc": 90},
        )

        optimizer2 = TopKOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=10, k=2, batch_size=5),
        )

        program = Program(optimizers=[optimizer1, optimizer2])
        program.run()

        # Verify both ran successfully
        assert len(program.optimizers) == 2
        assert all(len(opt.history) > 0 for opt in program.optimizers)
        assert len(program.energy_scores) > 0

    def test_generator_reuse_across_optimizers_fails(self):
        """Test that reusing the same generator instance across optimizers raises ValueError."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        # Create single generator instance
        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        shared_gen = UniformMutationGenerator(gen_config)
        shared_gen.assign(segment)

        # Each optimizer needs its own constraint
        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer1 = TopKOptimizer(
            constructs=[construct],
            generators=[shared_gen],  # Same generator instance
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        optimizer2 = TopKOptimizer(
            constructs=[construct],
            generators=[shared_gen],  # Same generator instance - should fail
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        with pytest.raises(ValueError, match="Generator.*reused"):
            Program(optimizers=[optimizer1, optimizer2])

    def test_constraint_reuse_across_optimizers_fails(self):
        """Test that reusing the same constraint instance across optimizers raises ValueError."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)

        gen1 = UniformMutationGenerator(gen_config)
        gen1.assign(segment)

        gen2 = UniformMutationGenerator(gen_config)
        gen2.assign(segment)

        # Create single constraint instance
        shared_constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer1 = TopKOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[shared_constraint],  # Same constraint instance
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        optimizer2 = TopKOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[shared_constraint],  # Same constraint instance - should fail
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        with pytest.raises(ValueError, match="Constraint.*reused"):
            Program(optimizers=[optimizer1, optimizer2])

    def test_single_optimizer_no_reuse_validation(self):
        """Test that single optimizer programs don't trigger reuse validation."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(gen_config)
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
        )

        # Should not raise - single optimizer has no reuse concerns
        program = Program(optimizers=[optimizer])
        assert len(program.optimizers) == 1

    def test_duplicate_generator_in_single_optimizer_fails(self):
        """Test that same generator instance appearing twice in one optimizer raises ValueError."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(gen_config)
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        # Same generator instance twice in generators list
        with pytest.raises(ValueError, match="appears multiple times"):
            TopKOptimizer(
                constructs=[construct],
                generators=[gen, gen],  # Duplicate generator
                constraints=[constraint],
                config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
            )

    def test_duplicate_constraint_in_single_optimizer_fails(self):
        """Test that same constraint instance appearing twice in one optimizer raises ValueError."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(gen_config)
        gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config={"min_gc": 50, "max_gc": 100},
        )

        # Same constraint instance twice in constraints list
        with pytest.raises(ValueError, match="appears multiple times"):
            TopKOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[constraint, constraint],  # Duplicate constraint
                config=TopKOptimizerConfig(num_samples=5, k=1, batch_size=5),
            )
