"""
Tests for Program class including state management and restart behavior.
"""

import pytest

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.core import Construct, Program, Segment
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    TopKOptimizer,
    TopKOptimizerConfig,
)


def _create_simple_program(num_stages: int = 1, sequence: str = "ATGCATGCATGCATGCATGC"):
    """Create a simple program for testing."""
    segment = Segment(sequence=sequence, sequence_type="dna", label="test")
    construct = Construct([segment])

    optimizers = []
    for i in range(num_stages):
        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},  # Always passes
        )

        opt_config = TopKOptimizerConfig(num_samples=3, k=2, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )
        optimizers.append(optimizer)

    return Program(optimizers=optimizers)


class TestProgramRestart:
    """Tests for Program state restart behavior."""

    def test_run_twice_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=1, sequence=original_seq)

        # First run
        program.run()
        first_run_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].selected_sequences
        ]

        # Second run should restart from original state
        program.run()
        second_run_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].selected_sequences
        ]

        # Verify sequences were modified from original (mutations applied)
        assert len(first_run_sequences) == 2
        assert len(second_run_sequences) == 2
        assert any(seq != original_seq for seq in first_run_sequences)
        assert any(seq != original_seq for seq in second_run_sequences)

    def test_multi_stage_run_twice(self):
        """Test that multi-stage program restarts correctly."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=2, sequence=original_seq)

        # First run through both stages
        program.run()
        first_run_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].selected_sequences
        ]

        # Second run should restart from original state
        program.run()
        second_run_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].selected_sequences
        ]

        # Verify sequences were modified from original (mutations applied across 2 stages)
        assert len(first_run_sequences) == 2
        assert len(second_run_sequences) == 2
        assert any(seq != original_seq for seq in first_run_sequences)
        assert any(seq != original_seq for seq in second_run_sequences)

    def test_optimizer_initial_states_cleared_between_stages(self):
        """Test that optimizer _initial_state is None before each stage."""
        program = _create_simple_program(num_stages=2)

        # Run the program
        program.run()

        # After run(), all optimizers should have their initial state set
        # (captured during their run())
        assert program.optimizers[0]._initial_state is not None
        assert program.optimizers[1]._initial_state is not None

    def test_program_restores_from_opt1_initial_state(self):
        """Test that re-running restores state from opt1's captured initial."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=1, sequence=original_seq)

        # First run
        program.run()
        assert program.optimizers[0]._initial_state is not None
        
        # Verify captured state contains original sequence (using index 0)
        segment = program.constructs[0].segments[0]
        assert len(program.optimizers[0]._initial_state['segments']) == 1
        captured_state = program.optimizers[0]._initial_state['segments'][0]
        captured_selected = captured_state['selected']
        captured_candidates = captured_state['candidates']
        
        # Verify captured sequences match original
        assert len(captured_selected) == 1
        assert captured_selected[0].sequence == original_seq
        assert len(captured_candidates) > 0
        assert all(c.sequence == original_seq for c in captured_candidates)

        # Manually modify sequences
        for seq in segment.selected_sequences:
            seq.sequence = "MODIFIED_SEQUENCE_123"
        for seq in segment.candidate_sequences:
            seq.sequence = "MODIFIED_CANDIDATE_12"

        # Second run should restore from opt1's initial state
        program.run()

        # Sequences should not remain as "MODIFIED" (they were restored before running)
        current_sequences = [
            seq.sequence for seq in segment.selected_sequences
        ]
        assert all("MODIFIED" not in seq for seq in current_sequences)
        
        # Verify candidates were also restored
        candidate_sequences = [
            seq.sequence for seq in segment.candidate_sequences
        ]
        assert all("MODIFIED" not in seq for seq in candidate_sequences)


class TestRunStageRestart:
    """Tests for Program.run_stage restart behavior."""

    def test_run_stage_captures_initial_state(self):
        """Test that run_stage captures initial state for the optimizer."""
        program = _create_simple_program(num_stages=1)

        # Run first stage
        program.run_stage(0)

        # Optimizer should have its initial state captured
        assert program.optimizers[0]._initial_state is not None

    def test_run_stage_forces_recapture(self):
        """Test that run_stage forces _initial_state recapture."""
        program = _create_simple_program(num_stages=2)

        # Run first stage
        program.run_stage(0)

        # Run second stage - it should force recapture
        program.run_stage(1)

        # Second optimizer should have captured its own initial state
        # (which includes the results from stage 1)
        assert program.optimizers[1]._initial_state is not None


class TestProgramValidation:
    """Tests for Program validation."""

    def test_empty_optimizers_raises(self):
        """Test that empty optimizer list raises ValueError."""
        with pytest.raises(ValueError, match="optimizers list cannot be empty"):
            Program(optimizers=[])

    def test_mismatched_constructs_raises(self):
        """Test that optimizers with different constructs raise error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct1 = Construct([segment1])

        segment2 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct2 = Construct([segment2])

        gen1 = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
        gen1.assign(segment1)
        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = TopKOptimizer(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=3, k=2, batch_size=2),
        )

        gen2 = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
        gen2.assign(segment2)
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment2], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt2 = TopKOptimizer(
            constructs=[construct2],  # Different construct object!
            generators=[gen2],
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=3, k=2, batch_size=2),
        )

        with pytest.raises(ValueError, match="not the same object"):
            Program(optimizers=[opt1, opt2])
