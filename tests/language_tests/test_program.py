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

    def test_run_stage_rerun_previous_stage(self):
        """Test that re-running a previous stage resets the pipeline."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=2, sequence=original_seq)

        # Run both stages
        program.run_stage(0)
        program.run_stage(1)
        assert program.current_stage == 2
        assert len(program._stage_results) == 2

        # Re-run stage 0 - should reset pipeline
        program.run_stage(0)
        assert program.current_stage == 1  # Back to after stage 0
        assert len(program._stage_results) == 1  # Stage 1 results wiped
        # Stage 1's initial state should be wiped
        assert program.optimizers[1]._initial_state is None

        # Can now run stage 1 again
        program.run_stage(1)
        assert program.current_stage == 2
        assert len(program._stage_results) == 2

    def test_run_stage_cannot_skip_forward(self):
        """Test that skipping stages forward raises an error."""
        program = _create_simple_program(num_stages=2)

        # Can't skip to stage 1 without running stage 0
        with pytest.raises(RuntimeError, match="Cannot skip"):
            program.run_stage(1)

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

    def test_run_stage_returns_none(self):
        """Test that run_stage returns None (results accessed via stage_results)."""
        program = _create_simple_program(num_stages=1)

        result = program.run_stage(0)

        assert result is None

    def test_run_stage_stores_results_in_stage_results(self):
        """Test that run_stage stores results accessible via get_stage_results."""
        program = _create_simple_program(num_stages=2)

        program.run_stage(0)

        # Results accessible via getter
        result = program.get_stage_results(0)

        # Validate result structure
        assert "batch_results" in result
        assert "best_batch_idx" in result
        assert isinstance(result["batch_results"], list)
        assert isinstance(result["best_batch_idx"], int)

        # Validate batch_results structure
        batch = result["batch_results"][0]
        assert "batch_idx" in batch
        assert "constructs" in batch
        assert "energy_score" in batch
        assert "metadata" in batch

    def test_get_stage_results_raises_for_unrun_stage(self):
        """Test that get_stage_results raises IndexError for unrun stages."""
        program = _create_simple_program(num_stages=2)

        with pytest.raises(IndexError, match="Stage 0 not available"):
            program.get_stage_results(0)

        program.run_stage(0)

        # Stage 0 now available, stage 1 not yet
        program.get_stage_results(0)  # Should not raise
        with pytest.raises(IndexError, match="Stage 1 not available"):
            program.get_stage_results(1)

    def test_run_returns_none(self):
        """Test that run() returns None."""
        program = _create_simple_program(num_stages=1)

        result = program.run()

        assert result is None

    def test_run_populates_stage_results(self):
        """Test that run() populates stage_results for all stages."""
        program = _create_simple_program(num_stages=2)

        program.run()

        # Both stages accessible via getter
        for i in range(2):
            result = program.get_stage_results(i)
            assert "batch_results" in result
            assert "best_batch_idx" in result


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


class TestSerializeRestoreState:
    """Tests for Program.serialize_state and restore_state (cross-task persistence)."""

    def test_serialize_state_structure(self):
        """Test that serialize_state returns correct structure."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)

        state = program.serialize_state()

        assert "current_stage" in state
        assert "segments" in state
        assert "energy_scores" in state
        assert state["current_stage"] == 1
        assert len(state["segments"]) == 1  # One segment

    def test_serialize_state_captures_sequences(self):
        """Test that serialize_state captures selected_sequences correctly."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=1, sequence=original_seq)
        program.run_stage(0)

        state = program.serialize_state()

        # Check segment state structure
        seg_state = state["segments"][0]
        assert "selected_sequences" in seg_state
        assert len(seg_state["selected_sequences"]) == 2  # k=2 from config

        # Check sequence data structure
        for seq_data in seg_state["selected_sequences"]:
            assert "sequence" in seq_data
            assert "metadata" in seq_data
            assert isinstance(seq_data["sequence"], str)
            assert isinstance(seq_data["metadata"], dict)

    def test_serialize_state_captures_energy_scores(self):
        """Test that serialize_state captures energy_scores from last optimizer."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)

        state = program.serialize_state()

        assert len(state["energy_scores"]) == 2  # k=2
        assert all(isinstance(s, float) for s in state["energy_scores"])

    def test_restore_state_restores_sequences(self):
        """Test that restore_state correctly restores selected_sequences."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=2, sequence=original_seq)

        # Run stage 0 and serialize
        program.run_stage(0)
        state = program.serialize_state()
        stage0_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].selected_sequences
        ]

        # Create fresh program and restore
        fresh_program = _create_simple_program(num_stages=2, sequence=original_seq)
        fresh_program.restore_state(state)

        # Verify sequences were restored
        restored_sequences = [
            seq.sequence for seq in fresh_program.constructs[0].segments[0].selected_sequences
        ]
        assert restored_sequences == stage0_sequences

    def test_restore_state_restores_current_stage(self):
        """Test that restore_state correctly restores current_stage."""
        program = _create_simple_program(num_stages=2)
        program.run_stage(0)
        state = program.serialize_state()

        fresh_program = _create_simple_program(num_stages=2)
        assert fresh_program.current_stage == 0

        fresh_program.restore_state(state)
        assert fresh_program.current_stage == 1

    def test_restore_state_restores_metadata(self):
        """Test that restore_state correctly restores sequence metadata."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)

        # Add custom metadata
        segment = program.constructs[0].segments[0]
        segment.selected_sequences[0]._metadata["custom_key"] = "custom_value"

        state = program.serialize_state()

        # Restore to fresh program
        fresh_program = _create_simple_program(num_stages=1)
        fresh_program.restore_state(state)

        # Verify metadata was restored
        restored_segment = fresh_program.constructs[0].segments[0]
        assert restored_segment.selected_sequences[0]._metadata.get("custom_key") == "custom_value"

    def test_restore_state_validates_segment_count(self):
        """Test that restore_state raises error on segment count mismatch."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)
        state = program.serialize_state()

        # Add extra segment to state
        state["segments"].append(state["segments"][0].copy())

        fresh_program = _create_simple_program(num_stages=1)
        with pytest.raises(ValueError, match="State mismatch"):
            fresh_program.restore_state(state)

    def test_serialize_restore_roundtrip(self):
        """Test full serialize -> restore -> continue optimization flow."""
        original_seq = "ATGCATGCATGCATGCATGC"

        # Run stage 0
        program1 = _create_simple_program(num_stages=2, sequence=original_seq)
        program1.run_stage(0)
        state = program1.serialize_state()
        stage0_sequences = [
            seq.sequence for seq in program1.constructs[0].segments[0].selected_sequences
        ]

        # Restore and run stage 1
        program2 = _create_simple_program(num_stages=2, sequence=original_seq)
        program2.restore_state(state)

        # Verify state was restored before running stage 1
        pre_stage1_sequences = [
            seq.sequence for seq in program2.constructs[0].segments[0].selected_sequences
        ]
        assert pre_stage1_sequences == stage0_sequences

        # Run stage 1
        program2.run_stage(1)

        # Verify stage 1 completed
        assert program2.current_stage == 2

    def test_serialize_state_is_json_compatible(self):
        """Test that serialize_state output can be JSON serialized."""
        import json

        program = _create_simple_program(num_stages=1)
        program.run_stage(0)
        state = program.serialize_state()

        # Should not raise
        json_str = json.dumps(state)
        assert len(json_str) > 0

        # Should round-trip
        restored = json.loads(json_str)
        assert restored["current_stage"] == state["current_stage"]
        assert len(restored["segments"]) == len(state["segments"])
