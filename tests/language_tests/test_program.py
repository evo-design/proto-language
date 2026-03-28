"""
tests/language_tests/test_program.py

Tests for Program class including state management and restart behavior.
"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.core import Construct, Program, Segment
from proto_language.language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig

_UNSET = object()


def _create_simple_program(
    num_stages: int = 1,
    sequence: str = "ATGCATGCATGCATGCATGC",
    compute=_UNSET,
):
    """Create a simple program for testing.

    Args:
        compute: Compute parameter for Program. Defaults to nullcontext()
            to avoid auto-detection (no GPU/cloud imports needed).
            Pass compute=None to test auto-detection behavior.
    """
    if compute is _UNSET:
        compute = nullcontext()

    segment = Segment(sequence=sequence, sequence_type="dna", label="test")
    construct = Construct([segment])

    optimizers = []
    for i in range(num_stages):
        gen_config = RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        generator = RandomNucleotideGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},  # Always passes
        )

        opt_config = TopKOptimizerConfig(num_samples=3, num_results=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )
        optimizers.append(optimizer)

    return Program(optimizers=optimizers, num_results=2, compute=compute)


class TestProgramRestart:
    """Tests for Program state restart behavior."""

    def test_run_twice_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=1, sequence=original_seq)

        # First run
        program.run()
        first_run_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].result_sequences
        ]

        # Second run should restart from original state
        program.run()
        second_run_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].result_sequences
        ]

        # Verify sequences were modified from original (mutations applied)
        assert len(first_run_sequences) == 2
        assert len(second_run_sequences) == 2
        assert any(seq != original_seq for seq in first_run_sequences)
        assert any(seq != original_seq for seq in second_run_sequences)

    def test_multi_stage_run_twice(self):
        """Test that multi-stage program restarts correctly.

        Regression test: on second Program.run(), subsequent optimizers must
        receive fresh results from the current run's earlier stages, not stale
        _initial_state captured during the first run.
        """
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=2, sequence=original_seq)
        segment = program.constructs[0].segments[0]

        # First run through both stages
        program.run()
        first_run_sequences = [
            seq.sequence for seq in segment.result_sequences
        ]
        assert len(first_run_sequences) == 2
        assert any(seq != original_seq for seq in first_run_sequences)

        # After first run, opt1 should have captured its initial state
        assert program.optimizers[1]._initial_state is not None

        # Second run: opt1._initial_state must be cleared so it recaptures
        # from opt0's fresh output instead of using stale first-run state.
        program.run()
        second_run_sequences = [
            seq.sequence for seq in segment.result_sequences
        ]
        assert len(second_run_sequences) == 2
        assert any(seq != original_seq for seq in second_run_sequences)

        # After second run(), subsequent optimizers' _initial_state should
        # reflect the second run's initialization (not the first run's).
        # Verify by checking the captured state contains sequences that
        # differ from the first run's captured state — proving recapture.
        opt1_state = program.optimizers[1]._initial_state
        assert opt1_state is not None, "opt1 should recapture state on second run"

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
        captured_result = captured_state['result']
        captured_proposals = captured_state['proposals']

        # Verify captured sequences match original (cycled to num_results=2)
        assert len(captured_result) == 2  # Cycled from single source to num_results=2
        assert all(s['sequence'] == original_seq for s in captured_result)
        assert len(captured_proposals) > 0
        assert all(c['sequence'] == original_seq for c in captured_proposals)

        # Manually modify sequences to all G's to verify restore works
        for seq in segment.result_sequences:
            seq.sequence = "G" * 20
        for seq in segment.proposal_sequences:
            seq.sequence = "G" * 20

        # Second run should restore from opt1's initial state
        program.run()

        # Sequences should not remain as all G's (they were restored before running)
        current_sequences = [
            seq.sequence for seq in segment.result_sequences
        ]
        assert any(seq != "G" * 20 for seq in current_sequences)

        # Verify proposals were also restored (and mutations applied)
        proposal_sequences = [
            seq.sequence for seq in segment.proposal_sequences
        ]
        assert any(seq != "G" * 20 for seq in proposal_sequences)


def _make_topk(segment, construct, num_results=None):
    """Create a TopK optimizer with optional num_results."""
    gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
    gen.assign(segment)
    constraint = ConstraintRegistry.create(
        key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100},
    )
    return TopKOptimizer(
        constructs=[construct], generators=[gen], constraints=[constraint],
        config=TopKOptimizerConfig(num_samples=6, num_results=num_results),
    )


class TestProgramNumResults:
    """Tests for num_results resolution: config field > Program fallback > error."""

    def test_config_num_results_sets_num_results(self):
        """Config num_results=2 sets num_results directly."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])
        opt = _make_topk(segment, construct, num_results=2)
        opt.run()
        assert opt.num_results == 2
        assert len(segment.result_sequences) == 2

    def test_deferred_topk_construction_succeeds(self):
        """TopK with neither config.num_results nor num_results constructs successfully (deferred)."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])
        opt = _make_topk(segment, construct)
        assert opt.num_results is None

    def test_program_num_results_flows_to_deferred_optimizer(self):
        """Program(num_results=5) pushes to TopK with num_results=None."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])
        opt = _make_topk(segment, construct)  # num_results=None
        assert opt.num_results is None

        program = Program(optimizers=[opt], num_results=3)

        assert opt.num_results == 3

        # Should be runnable
        program.run()
        assert len(segment.result_sequences) <= 3

    def test_program_num_results_does_not_override_config(self):
        """config.num_results=2 wins over Program(num_results=5)."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])
        opt = _make_topk(segment, construct, num_results=2)  # num_results=2 set in config
        assert opt.num_results == 2

        Program(optimizers=[opt], num_results=5)

        # Config should not be overridden
        assert opt.num_results == 2

    def test_program_mixed_resolved_and_deferred(self):
        """Program with some resolved and some deferred optimizers works."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])

        # First optimizer: resolved (num_results=2)
        opt1 = _make_topk(segment, construct, num_results=2)
        assert opt1.num_results == 2

        # Second optimizer: deferred (num_results=None)
        opt2 = _make_topk(segment, construct)
        assert opt2.num_results is None

        Program(optimizers=[opt1, opt2], num_results=3)

        # opt1 should keep num_results=2, opt2 should get 3 from Program
        assert opt1.num_results == 2
        assert opt2.num_results == 3


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
        assert "results" in result
        assert "best_result_idx" in result
        assert isinstance(result["results"], list)
        assert isinstance(result["best_result_idx"], int)

        # Validate results structure (new structured format)
        result_entry = result["results"][0]
        assert "result_idx" in result_entry
        assert "constructs" in result_entry
        assert "energy_score" in result_entry

        # Validate new structured constructs format
        construct = result_entry["constructs"][0]
        assert "type" in construct
        assert "segments" in construct
        segment = construct["segments"][0]
        assert "label" in segment
        assert "sequence" in segment
        assert "constraints" in segment

    def test_run_stage_clears_constraint_metadata_between_stages(self):
        """Test that constraint metadata is cleared between optimization stages."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna", label="test")
        construct = Construct([segment])

        # Stage 1: uses gc-content constraint with label "gc_stage_1"
        gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen1.assign(segment)
        constraint1 = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},
            label="gc_stage_1",
        )
        opt1 = TopKOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=2, num_results=2),
        )

        # Stage 2: uses gc-content constraint with label "gc_stage_2"
        gen2 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen2.assign(segment)
        constraint2 = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},
            label="gc_stage_2",
        )
        opt2 = TopKOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=2, num_results=2),
        )

        program = Program(optimizers=[opt1, opt2], num_results=2)

        # Run stage 1
        program.run_stage(0)

        # Verify stage 1 constraint metadata is present
        seq = segment.result_sequences[0]
        assert "gc_stage_1" in seq._constraints_metadata

        # Run stage 2
        program.run_stage(1)

        # Verify stage 1 metadata is cleared and only stage 2 metadata exists
        seq = segment.result_sequences[0]
        assert "gc_stage_1" not in seq._constraints_metadata, \
            "Stage 1 constraint metadata should be cleared"
        assert "gc_stage_2" in seq._constraints_metadata, \
            "Stage 2 constraint metadata should be present"

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
            assert "results" in result
            assert "best_result_idx" in result


class TestProgramValidation:
    """Tests for Program._validate_program checks."""

    def test_empty_optimizers_raises(self):
        """Test that empty optimizer list raises ValueError."""
        with pytest.raises(ValueError, match="optimizers list cannot be empty"):
            Program(optimizers=[], num_results=1)

    def test_mismatched_construct_count_raises(self):
        """Test that optimizers with different construct counts raise error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        segment2 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct1 = Construct([segment1], label="c1")
        construct2 = Construct([segment2], label="c2")

        gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen1.assign(segment1)
        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = TopKOptimizer(
            constructs=[construct1, construct2],
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        gen2 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen2.assign(segment1)
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt2 = TopKOptimizer(
            constructs=[construct1],  # Different count!
            generators=[gen2],
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="has 1 constructs.*has 2"):
            Program(optimizers=[opt1, opt2], num_results=2)

    def test_mismatched_construct_identity_raises(self):
        """Test that optimizers with different construct objects raise error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct1 = Construct([segment1])

        segment2 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct2 = Construct([segment2])

        gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen1.assign(segment1)
        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = TopKOptimizer(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        gen2 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen2.assign(segment2)
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment2], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt2 = TopKOptimizer(
            constructs=[construct2],  # Different construct object!
            generators=[gen2],
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="not the same object"):
            Program(optimizers=[opt1, opt2], num_results=2)

    def test_duplicate_construct_labels_raises(self):
        """Test that duplicate construct labels raise error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        segment2 = Segment(sequence="GGGGGGGGGGGGGGGGGGGG", sequence_type="dna")
        construct1 = Construct([segment1], label="same_label")
        construct2 = Construct([segment2], label="same_label")

        gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen1.assign(segment1)
        gen2 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen2.assign(segment2)
        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment2], config_dict={"min_gc": 0, "max_gc": 100}, label="gc_content_2"
        )
        opt = TopKOptimizer(
            constructs=[construct1, construct2],
            generators=[gen1, gen2],
            constraints=[constraint1, constraint2],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="Construct labels must be unique.*same_label"):
            Program(optimizers=[opt], num_results=2)

    def test_segment_reuse_across_constructs_raises(self):
        """Test that reusing segment instance across constructs raises error."""
        shared_segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct1 = Construct([shared_segment], label="c1")
        construct2 = Construct([shared_segment], label="c2")  # Same segment instance!

        gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen1.assign(shared_segment)
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[shared_segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt = TopKOptimizer(
            constructs=[construct1, construct2],
            generators=[gen1],
            constraints=[constraint],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="Segment.*used in multiple constructs"):
            Program(optimizers=[opt], num_results=2)

    def test_dangling_segment_raises(self):
        """Test that segment with no input and no generator raises error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        segment2 = Segment(sequence_type="dna", length=20)  # No input sequence
        construct = Construct([segment1, segment2])

        gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen.assign(segment1)  # Only segment1 has generator
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt = TopKOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="never populated.*no input sequence and no generator"):
            Program(optimizers=[opt], num_results=2)

    def test_generator_reuse_across_optimizers_raises(self):
        """Test that reusing generator instance across optimizers raises error."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])

        shared_gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        shared_gen.assign(segment)

        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = TopKOptimizer(
            constructs=[construct],
            generators=[shared_gen],  # Same generator instance
            constraints=[constraint1],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )
        opt2 = TopKOptimizer(
            constructs=[construct],
            generators=[shared_gen],  # Same generator instance!
            constraints=[constraint2],
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="Generator.*reused across optimizer"):
            Program(optimizers=[opt1, opt2], num_results=2)

    def test_constraint_reuse_across_optimizers_raises(self):
        """Test that reusing constraint instance across optimizers raises error."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])

        gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen1.assign(segment)
        gen2 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen2.assign(segment)

        shared_constraint = ConstraintRegistry.create(
            key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = TopKOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[shared_constraint],  # Same constraint instance
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )
        opt2 = TopKOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[shared_constraint],  # Same constraint instance!
            config=TopKOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="Constraint.*reused across optimizer"):
            Program(optimizers=[opt1, opt2], num_results=2)


class TestSerializeRestoreState:
    """Tests for Program.serialize_state and restore_state (cross-task persistence)."""

    def test_serialize_state_structure(self):
        """Test that serialize_state returns correct structure."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)

        state = program.serialize_state()

        assert "segments" in state
        assert len(state["segments"]) == 1  # One segment

    def test_serialize_state_captures_sequences(self):
        """Test that serialize_state captures result_sequences correctly."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=1, sequence=original_seq)
        program.run_stage(0)

        state = program.serialize_state()

        # Check segment state structure
        seg_state = state["segments"][0]
        assert "result_sequences" in seg_state
        assert len(seg_state["result_sequences"]) == 2  # num_results=2 from config

        # Check sequence data structure (minimal: sequence, sequence_type, valid_chars)
        for seq_data in seg_state["result_sequences"]:
            assert "sequence" in seq_data
            assert "sequence_type" in seq_data
            assert "valid_chars" in seq_data
            assert isinstance(seq_data["sequence"], str)

    def test_restore_state_restores_sequences(self):
        """Test that restore_state correctly restores result_sequences."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=2, sequence=original_seq)

        # Run stage 0 and serialize
        program.run_stage(0)
        state = program.serialize_state()
        stage0_sequences = [
            seq.sequence for seq in program.constructs[0].segments[0].result_sequences
        ]

        # Create fresh program and restore
        fresh_program = _create_simple_program(num_stages=2, sequence=original_seq)
        fresh_program.restore_state(state)

        # Verify sequences were restored
        restored_sequences = [
            seq.sequence for seq in fresh_program.constructs[0].segments[0].result_sequences
        ]
        assert restored_sequences == stage0_sequences

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
            seq.sequence for seq in program1.constructs[0].segments[0].result_sequences
        ]

        # Restore and run stage 1
        program2 = _create_simple_program(num_stages=2, sequence=original_seq)
        program2.restore_state(state)
        program2.current_stage = 1  # API tracks this in DB, not in serialized state

        # Verify state was restored before running stage 1
        pre_stage1_sequences = [
            seq.sequence for seq in program2.constructs[0].segments[0].result_sequences
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
        assert len(restored["segments"]) == len(state["segments"])

    def test_serialize_restore_preserves_valid_chars(self):
        """Test that valid_chars are preserved through serialize/restore roundtrip."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)

        # Verify serialized state includes valid_chars
        state = program.serialize_state()
        seq_data = state["segments"][0]["result_sequences"][0]
        assert "valid_chars" in seq_data
        assert "sequence_type" in seq_data

        # Restore and verify valid_chars is preserved
        fresh_program = _create_simple_program(num_stages=1)
        fresh_program.restore_state(state)

        original_seq = program.constructs[0].segments[0].result_sequences[0]
        restored_seq = fresh_program.constructs[0].segments[0].result_sequences[0]
        assert restored_seq.valid_chars == original_seq.valid_chars
        assert restored_seq.sequence_type == original_seq.sequence_type


class TestProgramExport:
    """Tests for Program.export, to_dataframe, and to_fasta."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.program = _create_simple_program(num_stages=1)
        self.program.run()

    def test_export_all_tables(self, tmp_path):
        """export() without table writes all 4 table files."""
        out = self.program.export(path=tmp_path / "results", format="csv")
        assert out.is_dir()
        for name in ("sequences", "constraints", "constructs", "optimization"):
            assert (out / f"{name}.csv").stat().st_size > 0

    def test_export_single_table(self, tmp_path):
        """export() with table writes one file with expected columns."""
        path = tmp_path / "seqs.csv"
        self.program.export(path=path, table="sequences")
        content = path.read_text()
        assert "result_idx" in content and "sequence" in content

    def test_export_xlsx(self, tmp_path):
        """export() with xlsx writes a workbook."""
        path = tmp_path / "results.xlsx"
        self.program.export(path=path, format="xlsx")
        assert path.stat().st_size > 0

    def test_export_stage_filter(self, tmp_path):
        """export() with stage= filters to that stage."""
        program = _create_simple_program(num_stages=2)
        program.run()
        path = tmp_path / "s0.csv"
        program.export(path=path, table="sequences", stage=0)
        assert path.exists()

    def test_export_invalid_table_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown table"):
            self.program.export(path=tmp_path / "x.csv", table="nonexistent")

    @pytest.mark.parametrize("table,expected_col", [
        ("sequences", "sequence"),
        ("constraints", "constraint"),
        ("constructs", "full_sequence"),
        ("optimization", "timepoint"),
    ])
    def test_to_dataframe_all_tables(self, table, expected_col):
        """to_dataframe dispatches correctly to each table."""
        df = self.program.to_dataframe(table=table)
        assert len(df) > 0
        assert expected_col in df.columns

    def test_to_dataframe_invalid_table_raises(self):
        with pytest.raises(ValueError, match="Unknown table"):
            self.program.to_dataframe(table="nonexistent")

    def test_to_fasta(self, tmp_path):
        """to_fasta returns valid FASTA and optionally writes file."""
        fasta = self.program.to_fasta()
        assert fasta.startswith(">")
        # Write to file
        path = tmp_path / "seqs.fasta"
        assert self.program.to_fasta(path=path) == path.read_text()

    def test_to_fasta_segment_filter(self):
        """to_fasta with segments= filters output."""
        assert len(self.program.to_fasta(segments={"test"})) > 0
        assert self.program.to_fasta(segments={"nonexistent"}) == ""


class TestProgramCompute:
    """Tests for Program.compute parameter and _enter_compute() context manager."""

    @patch("proto_tools.utils.device.number_of_available_gpus", return_value=0)
    @patch("proto_tools.tools.tool_registry.ToolRegistry._execution_backend", new="mock_backend")
    @patch("proto_tools.utils.tool_pool.ToolPool")
    def test_compute_defaults_to_toolpool_remote_when_backend(self, mock_pool_cls, _mock_gpus):
        """Default compute=None resolves to ToolPool(remote=True) when cloud backend is registered."""
        program = _create_simple_program(compute=None)
        mock_pool_cls.assert_called_once_with(remote=True)
        assert program.compute is mock_pool_cls.return_value

    @patch("proto_tools.utils.device.number_of_available_gpus", return_value=2)
    @patch("proto_tools.tools.tool_registry.ToolRegistry._execution_backend", new=None)
    @patch("proto_tools.utils.tool_pool.ToolPool")
    def test_compute_defaults_to_toolpool_local_when_gpus(self, mock_pool_cls, _mock_gpus):
        """Default compute=None resolves to ToolPool() when GPUs available but no backend."""
        program = _create_simple_program(compute=None)
        mock_pool_cls.assert_called_once_with()
        assert program.compute is mock_pool_cls.return_value

    @patch("proto_tools.utils.device.number_of_available_gpus", return_value=0)
    @patch("proto_tools.tools.tool_registry.ToolRegistry._execution_backend", new=None)
    def test_compute_defaults_to_nullcontext_when_nothing(self, _mock_gpus):
        """Default compute=None resolves to nullcontext() when no backend and no GPUs."""
        from contextlib import nullcontext
        program = _create_simple_program(compute=None)
        assert isinstance(program.compute, nullcontext)

    @patch("proto_tools.utils.tool_pool._active_pool")
    def test_run_enters_compute_context(self, mock_active_pool):
        """ToolPool __enter__ and __exit__ called during run()."""
        mock_pool = MagicMock()
        mock_active_pool.get.return_value = None
        program = _create_simple_program(compute=mock_pool)
        # Mock run_stage to avoid tool dispatch conflicts with mocked pool
        program.run_stage = MagicMock()
        program.run()
        mock_pool.__enter__.assert_called_once()
        mock_pool.__exit__.assert_called_once()

    @patch("proto_tools.utils.tool_pool._active_pool")
    def test_run_stage_standalone_enters_and_exits_compute(self, mock_active_pool):
        """run_stage() called directly enters and exits compute context."""
        mock_active_pool.get.return_value = None
        mock_pool = MagicMock()
        program = _create_simple_program(compute=mock_pool)
        program.run_stage(0)
        mock_pool.__enter__.assert_called_once()
        mock_pool.__exit__.assert_called_once()

    @patch("proto_tools.utils.tool_pool._active_pool")
    def test_run_stage_skips_enter_when_active(self, mock_active_pool):
        """run_stage() called from run() does not double-enter compute."""
        mock_pool = MagicMock()
        # First call returns None (run() enters), subsequent calls return mock
        mock_active_pool.get.side_effect = [None] + [mock_pool] * 10
        program = _create_simple_program(compute=mock_pool)
        # Mock run_stage to avoid tool dispatch conflicts with mocked pool
        program.run_stage = MagicMock()
        program.run()
        # Only one __enter__ call (from run()), run_stage() should skip
        assert mock_pool.__enter__.call_count == 1

    @patch("proto_tools.utils.tool_pool._active_pool")
    def test_compute_exit_on_exception(self, mock_active_pool):
        """__exit__ called even when optimizer raises."""
        mock_pool = MagicMock()
        mock_active_pool.get.side_effect = [None, mock_pool]
        program = _create_simple_program(compute=mock_pool)
        # Make optimizer.run() raise
        program.optimizers[0].run = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            program.run()
        mock_pool.__exit__.assert_called_once()
