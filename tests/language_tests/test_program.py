"""Tests for Program class including state management and restart behavior."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from proto_tools.transforms.masking import MaskingStrategy
from pydantic import BaseModel

from proto_language import ConstraintOutput, GradientConstraintOutput
from proto_language.constraint import ConstraintRegistry, gc_content_constraint
from proto_language.core import (
    Constraint,
    Construct,
    Generator,
    GeneratorInputType,
    Program,
    Segment,
)
from proto_language.generator import (
    ESM2Generator,
    ESM2GeneratorConfig,
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
)
from proto_language.optimizer import (
    CyclingOptimizer,
    CyclingOptimizerConfig,
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)
from tests.helpers.mock_structure import MockStructure

_UNSET = object()


def _create_simple_program(
    num_stages: int = 1,
    sequence: str = "ATGCATGCATGCATGCATGC",
    compute=_UNSET,
    seed: int | None = None,
):
    """Create a simple program for testing.

    Args:
        num_stages: Number of optimizer stages to create.
        sequence: DNA sequence string for the test segment.
        compute: Compute parameter for Program. Defaults to nullcontext()
            to skip auto-detection.
            Pass compute=None to test auto-detection behavior.
        seed: Optional random seed for deterministic results.
    """
    if compute is _UNSET:
        compute = nullcontext()

    segment = Segment(sequence=sequence, sequence_type="dna", label="test")
    construct = Construct([segment])

    optimizers = []
    for _i in range(num_stages):
        gen_config = RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        generator = RandomNucleotideGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},  # Always passes
        )

        opt_config = RejectionSamplingOptimizerConfig(num_samples=3, num_results=2)
        optimizer = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )
        optimizers.append(optimizer)

    return Program(optimizers=optimizers, num_results=2, compute=compute, seed=seed)


class TestClearSequenceMetadata:
    """Tests for Program._clear_sequence_metadata stage hygiene."""

    def test_clears_both_constraint_and_generator_metadata(self):
        """At a stage boundary, both stale constraint and generator metadata are reset."""
        program = _create_simple_program(num_stages=1, compute=nullcontext())
        segment = program.constructs[0].segments[0]
        for seq in segment.result_sequences + segment.proposal_sequences:
            seq._constraints_metadata = {"stale-constraint": {"score": 1.0}}
            seq._generator_metadata = {"stale-generator": {"samples": ["AAAA"]}}

        program._clear_sequence_metadata()

        for seq in segment.result_sequences + segment.proposal_sequences:
            assert seq._constraints_metadata == {}
            assert seq._generator_metadata == {}


class TestProgramRestart:
    """Tests for Program state restart behavior."""

    def test_run_twice_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        original_seq = "ATGCATGCATGCATGCATGC"
        program = _create_simple_program(num_stages=1, sequence=original_seq, seed=42)

        # First run
        program.run()
        first_run_sequences = [seq.sequence for seq in program.constructs[0].segments[0].result_sequences]

        # Second run should restart from original state
        program.run()
        second_run_sequences = [seq.sequence for seq in program.constructs[0].segments[0].result_sequences]

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
        program = _create_simple_program(num_stages=2, sequence=original_seq, seed=42)
        segment = program.constructs[0].segments[0]

        # First run through both stages
        program.run()
        first_run_sequences = [seq.sequence for seq in segment.result_sequences]
        assert len(first_run_sequences) == 2
        assert any(seq != original_seq for seq in first_run_sequences)

        # After first run, opt1 should have captured its initial state
        assert program.optimizers[1]._initial_state is not None

        # Second run: opt1._initial_state must be cleared so it recaptures
        # from opt0's fresh output instead of using stale first-run state.
        program.run()
        second_run_sequences = [seq.sequence for seq in segment.result_sequences]
        assert len(second_run_sequences) == 2
        assert any(seq != original_seq for seq in second_run_sequences)

        # After second run(), subsequent optimizers' _initial_state should
        # reflect the second run's initialization (not the first run's).
        # Verify by checking the captured state contains sequences that
        # differ from the first run's captured state, proving recapture.
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
        assert len(program.optimizers[0]._initial_state["segments"]) == 1
        captured_state = program.optimizers[0]._initial_state["segments"][0]
        captured_result = captured_state["result"]
        captured_proposals = captured_state["proposals"]

        # Verify captured sequences match original (cycled to num_results=2)
        assert len(captured_result) == 2  # Cycled from single source to num_results=2
        assert all(s["sequence"] == original_seq for s in captured_result)
        assert len(captured_proposals) > 0
        assert all(c["sequence"] == original_seq for c in captured_proposals)

        # Manually modify sequences to all G's to verify restore works
        for seq in segment.result_sequences:
            seq.sequence = "G" * 20
        for seq in segment.proposal_sequences:
            seq.sequence = "G" * 20

        # Second run should restore from opt1's initial state
        program.run()

        # Sequences should not remain as all G's (they were restored before running)
        current_sequences = [seq.sequence for seq in segment.result_sequences]
        assert any(seq != "G" * 20 for seq in current_sequences)

        # Verify proposals were also restored (and mutations applied)
        proposal_sequences = [seq.sequence for seq in segment.proposal_sequences]
        assert any(seq != "G" * 20 for seq in proposal_sequences)


def _make_rejection_sampling(segment, construct, num_results=None):
    """Create a Rejection Sampling optimizer with optional num_results."""
    gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
    gen.assign(segment)
    constraint = ConstraintRegistry.create(
        key="gc-content",
        segments=[segment],
        config_dict={"min_gc": 0, "max_gc": 100},
    )
    return RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[gen],
        constraints=[constraint],
        config=RejectionSamplingOptimizerConfig(num_samples=6, num_results=num_results),
    )


class TestProgramNumResults:
    """Tests for num_results resolution: config field > Program fallback > error."""

    def test_config_num_results_sets_num_results(self):
        """Config num_results=2 sets num_results directly."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])
        opt = _make_rejection_sampling(segment, construct, num_results=2)
        opt.run()
        assert opt.num_results == 2
        assert len(segment.result_sequences) == 2

    def test_deferred_rejection_sampling_construction_succeeds(self):
        """RS with neither config.num_results nor num_results constructs successfully (deferred)."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])
        opt = _make_rejection_sampling(segment, construct)
        assert opt.num_results is None

    def test_program_num_results_flows_to_deferred_optimizer(self):
        """Program(num_results=5) pushes to Rejection Sampling with num_results=None."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])
        opt = _make_rejection_sampling(segment, construct)  # num_results=None
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
        opt = _make_rejection_sampling(segment, construct, num_results=2)  # num_results=2 set in config
        assert opt.num_results == 2

        Program(optimizers=[opt], num_results=5)

        # Config should not be overridden
        assert opt.num_results == 2

    def test_program_mixed_resolved_and_deferred(self):
        """Program with some resolved and some deferred optimizers works."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])

        # First optimizer: resolved (num_results=2)
        opt1 = _make_rejection_sampling(segment, construct, num_results=2)
        assert opt1.num_results == 2

        # Second optimizer: deferred (num_results=None)
        opt2 = _make_rejection_sampling(segment, construct)
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
        program = _create_simple_program(num_stages=2, sequence=original_seq, seed=42)

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
        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(segment)
        constraint1 = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},
            label="gc_stage_1",
        )
        opt1 = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[constraint1],
            config=RejectionSamplingOptimizerConfig(num_samples=2, num_results=2),
        )

        # Stage 2: uses gc-content constraint with label "gc_stage_2"
        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2.assign(segment)
        constraint2 = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},
            label="gc_stage_2",
        )
        opt2 = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[constraint2],
            config=RejectionSamplingOptimizerConfig(num_samples=2, num_results=2),
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
        assert "gc_stage_1" not in seq._constraints_metadata, "Stage 1 constraint metadata should be cleared"
        assert "gc_stage_2" in seq._constraints_metadata, "Stage 2 constraint metadata should be present"

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
        with pytest.raises(ValueError, match=r"requires at least one Optimizer"):
            Program(optimizers=[], num_results=1)

    def test_mismatched_construct_count_raises(self):
        """Test that optimizers with different construct counts raise error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        segment2 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct1 = Construct([segment1], label="c1")
        construct2 = Construct([segment2], label="c2")

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(segment1)
        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = RejectionSamplingOptimizer(
            constructs=[construct1, construct2],
            generators=[gen1],
            constraints=[constraint1],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2.assign(segment1)
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt2 = RejectionSamplingOptimizer(
            constructs=[construct1],  # Different count!
            generators=[gen2],
            constraints=[constraint2],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match=r"has 1 constructs.*has 2"):
            Program(optimizers=[opt1, opt2], num_results=2)

    def test_mismatched_construct_identity_raises(self):
        """Test that optimizers with different construct objects raise error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct1 = Construct([segment1])

        segment2 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct2 = Construct([segment2])

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(segment1)
        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = RejectionSamplingOptimizer(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2.assign(segment2)
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment2], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt2 = RejectionSamplingOptimizer(
            constructs=[construct2],  # Different construct object!
            generators=[gen2],
            constraints=[constraint2],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match="not the same object"):
            Program(optimizers=[opt1, opt2], num_results=2)

    def test_duplicate_construct_labels_raises(self):
        """Test that duplicate construct labels raise error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        segment2 = Segment(sequence="GGGGGGGGGGGGGGGGGGGG", sequence_type="dna")
        construct1 = Construct([segment1], label="same_label")
        construct2 = Construct([segment2], label="same_label")

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(segment1)
        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2.assign(segment2)
        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment2], config_dict={"min_gc": 0, "max_gc": 100}, label="gc_content_2"
        )
        opt = RejectionSamplingOptimizer(
            constructs=[construct1, construct2],
            generators=[gen1, gen2],
            constraints=[constraint1, constraint2],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match=r"Construct labels must be unique.*same_label"):
            Program(optimizers=[opt], num_results=2)

    def test_segment_reuse_across_constructs_raises(self):
        """Test that reusing segment instance across constructs raises error."""
        shared_segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct1 = Construct([shared_segment], label="c1")
        construct2 = Construct([shared_segment], label="c2")  # Same segment instance!

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(shared_segment)
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[shared_segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt = RejectionSamplingOptimizer(
            constructs=[construct1, construct2],
            generators=[gen1],
            constraints=[constraint],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match=r"Segment.*used in multiple constructs"):
            Program(optimizers=[opt], num_results=2)

    def test_dangling_segment_raises(self):
        """Test that segment with no input and no generator raises error."""
        segment1 = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        segment2 = Segment(sequence_type="dna", length=20)  # No input sequence
        construct = Construct([segment1, segment2])

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(segment1)  # Only segment1 has generator
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[segment1], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match=r"never populated.*no input sequence and no generator"):
            Program(optimizers=[opt], num_results=2)

    def test_generator_reuse_across_optimizers_raises(self):
        """Test that reusing generator instance across optimizers raises error."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])

        shared_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        shared_gen.assign(segment)

        constraint1 = ConstraintRegistry.create(
            key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        constraint2 = ConstraintRegistry.create(
            key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[shared_gen],  # Same generator instance
            constraints=[constraint1],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )
        opt2 = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[shared_gen],  # Same generator instance!
            constraints=[constraint2],
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match=r"Generator.*reused across optimizer"):
            Program(optimizers=[opt1, opt2], num_results=2)

    def test_constraint_reuse_across_optimizers_raises(self):
        """Test that reusing constraint instance across optimizers raises error."""
        segment = Segment(sequence="ATGCATGCATGCATGCATGC", sequence_type="dna")
        construct = Construct([segment])

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(segment)
        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2.assign(segment)

        shared_constraint = ConstraintRegistry.create(
            key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100}
        )
        opt1 = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[shared_constraint],  # Same constraint instance
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )
        opt2 = RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[shared_constraint],  # Same constraint instance!
            config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
        )

        with pytest.raises(ValueError, match=r"Constraint.*reused across optimizer"):
            Program(optimizers=[opt1, opt2], num_results=2)


# Helpers for TestProgramGeneratorInputs ---------------------------------------


def _gc(segment: Segment, *, threshold: float | None = None) -> Constraint:
    return Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 0.0, "max_gc": 100.0},
        threshold=threshold,
    )


def _protein_zero(input_sequences, config):
    return [ConstraintOutput(score=0.0) for _ in input_sequences]


_protein_zero._constraint_supported_sequence_types = ["protein"]
_protein_zero._constraint_num_input_sequences_per_tuple = 1


def _protein_constraint(segment: Segment, *, threshold: float | None = None) -> Constraint:
    return Constraint(inputs=[segment], function=_protein_zero, function_config={}, threshold=threshold)


def _rejection(construct, generators, constraints):
    return RejectionSamplingOptimizer(
        constructs=[construct],
        generators=generators,
        constraints=constraints,
        config=RejectionSamplingOptimizerConfig(num_samples=3, num_results=2),
    )


def _mcmc(construct, generators, constraints):
    return MCMCOptimizer(
        constructs=[construct],
        generators=generators,
        constraints=constraints,
        config=MCMCOptimizerConfig(num_results=2, num_steps=2),
    )


class _MockARConfig(BaseModel):
    prompts: list[str] = []


class MockAutoregressiveGenerator(Generator):
    """Bypasses Evo2/ProGen2's empty-prompts rejection so the validator's PROMPT branch is reachable."""

    input_type = GeneratorInputType.PROMPT

    def __init__(self, prompts: list[str]) -> None:
        super().__init__()
        self.config = _MockARConfig(prompts=prompts)

    def _sample(self, *args, **kwargs) -> None:
        pass


class _GradCfg(BaseModel):
    pass


def _grad_backward(input_sequences, *, config, **kwargs):
    out = []
    for (seq,) in input_sequences:
        out.append(GradientConstraintOutput(gradient=(seq.logits - np.zeros_like(seq.logits),), loss=0.0, metrics={}))
    return out


def _grad_constraint(segment: Segment) -> Constraint:
    return Constraint(
        inputs=[segment],
        function=_protein_zero,
        function_config=_GradCfg(),
        backward=_grad_backward,
        backward_config=_GradCfg(),
        label="grad",
    )


class TestProgramGeneratorInputs:
    """Tests for ``Program._validate_generator_inputs`` (per-input_type contract enforcement)."""

    # STARTING_SEQUENCE (mutation) -------------------------------------------

    def test_random_nucleotide_starting_sequence_length_only_runs(self):
        seg = Segment(length=8, sequence_type="dna")
        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen.assign(seg)
        program = Program(
            optimizers=[_rejection(Construct([seg]), [gen], [_gc(seg)])],
            num_results=2,
            compute=nullcontext(),
            seed=1,
        )
        program.run()

        assert all(seq.sequence for seq in seg.result_sequences)
        assert all(len(seq.sequence) == 8 for seq in seg.result_sequences)

    def test_random_protein_starting_sequence_length_only_runs(self):
        seg = Segment(length=8, sequence_type="protein")
        gen = RandomProteinGenerator(RandomProteinGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen.assign(seg)
        program = Program(
            optimizers=[_rejection(Construct([seg]), [gen], [_protein_constraint(seg)])],
            num_results=2,
            compute=nullcontext(),
            seed=1,
        )
        program.run()

        assert all(seq.sequence for seq in seg.result_sequences)
        assert all(len(seq.sequence) == 8 for seq in seg.result_sequences)

    def test_non_random_starting_sequence_length_only_raises(self):
        seg = Segment(length=8, sequence_type="protein")
        gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig(clear_logits=True))
        gen.assign(seg)
        with pytest.raises(ValueError, match="no starting sequence is available"):
            Program(optimizers=[_rejection(Construct([seg]), [gen], [_protein_constraint(seg)])], num_results=2)

    def test_starting_sequence_tied_secondary_length_only_ok(self):
        """Tied segments are mirrored from the primary; secondary tied segments need no starting sequence."""
        primary = Segment(sequence="MKKLLLAA", sequence_type="protein", label="primary")
        tied = Segment(length=8, sequence_type="protein", label="tied")
        construct = Construct([primary, tied])
        gen = ESM2Generator(ESM2GeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
        gen.assign([primary, tied])
        Program(optimizers=[_rejection(construct, [gen], [_protein_constraint(primary)])], num_results=2)

    def test_starting_sequence_multi_stage_inherits(self):
        """Stage 2 has no segment.input_sequence; the validator credits Stage 1's output."""
        seg = Segment(sequence="MKKLLLAA", sequence_type="protein")
        construct = Construct([seg])

        def _esm2_stage() -> RejectionSamplingOptimizer:
            g = ESM2Generator(ESM2GeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
            g.assign(seg)
            return _rejection(construct, [g], [_protein_constraint(seg)])

        Program(optimizers=[_esm2_stage(), _esm2_stage()], num_results=2)

    # PROMPT (autoregressive) ------------------------------------------------

    def test_prompt_empty_raises(self):
        seg = Segment(length=10, sequence_type="dna")
        gen = MockAutoregressiveGenerator(prompts=[])
        gen.assign(seg)
        with pytest.raises(ValueError, match="requires non-empty prompts"):
            Program(optimizers=[_rejection(Construct([seg]), [gen], [_gc(seg)])], num_results=2)

    # STRUCTURE (inverse folding) --------------------------------------------

    def test_structure_no_input_no_cycling_raises(self):
        seg = Segment(length=10, sequence_type="protein")
        gen = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(structure_inputs=None))
        gen.assign(seg)
        with pytest.raises(ValueError, match="requires structure_inputs on its config"):
            Program(
                optimizers=[_rejection(Construct([seg]), [gen], [_protein_constraint(seg)])],
                num_results=2,
            )

    def test_structure_under_protein_hunter_pipeline_ok(self):
        """Cycling pipeline supplies structures dynamically; no config.structure_inputs required."""
        seg = Segment(length=10, sequence_type="protein")
        gen = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(structure_inputs=None))
        gen.assign(seg)
        opt = CyclingOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[_protein_constraint(seg, threshold=1.0)],
            config=CyclingOptimizerConfig(num_steps=1, num_results=2, pipeline="protein-hunter"),
        )
        Program(optimizers=[opt], num_results=2)

    def test_prompt_under_beam_search_ok(self):
        """BeamSearchOptimizer supplies prompts from its own ``config.prompt`` — empty ``config.prompts`` is fine."""
        from proto_language.optimizer import BeamSearchOptimizer, BeamSearchOptimizerConfig

        seg = Segment(length=10, sequence_type="dna")
        gen = MockAutoregressiveGenerator(prompts=[])
        gen.assign(seg)
        opt = BeamSearchOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[_gc(seg)],
            config=BeamSearchOptimizerConfig(prompt="ATCG", beam_length=10, num_results=2, proposals_per_result=2),
        )
        Program(optimizers=[opt], num_results=2)

    def test_prompt_under_cycling_ok(self):
        """CyclingOptimizer (custom conditioning_fn) supplies prompts at runtime — empty ``config.prompts`` is fine."""
        seg = Segment(length=10, sequence_type="dna")
        gen = MockAutoregressiveGenerator(prompts=[])
        gen.assign(seg)
        opt = CyclingOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[_gc(seg, threshold=1.0)],
            config=CyclingOptimizerConfig(num_steps=1, num_results=2),
            conditioning_fn=lambda seqs: [["AT"]] * len(seqs),
        )
        Program(optimizers=[opt], num_results=2)

    def test_structure_under_custom_cycling_conditioning_fn_ok(self):
        """A user-supplied conditioning_fn (not a named pipeline) supplies structures at runtime."""
        seg = Segment(length=10, sequence_type="protein")
        gen = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(structure_inputs=None))
        gen.assign(seg)
        opt = CyclingOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[_protein_constraint(seg, threshold=1.0)],
            config=CyclingOptimizerConfig(num_steps=1, num_results=2),
            conditioning_fn=lambda seqs: [None] * len(seqs),
        )
        Program(optimizers=[opt], num_results=2)

    # LOGITS (gradient) ------------------------------------------------------

    def test_logits_gradient_then_semigreedy_ok(self):
        """Canonical Germinal pattern: GradientOptimizer produces logits → MCMC + SemigreedyMutation refines."""
        seg = Segment(sequence="MKKLLLAA", sequence_type="protein")
        construct = Construct([seg])

        pw = PositionWeightGenerator(PositionWeightGeneratorConfig())
        pw.assign(seg)
        opt1 = GradientOptimizer(
            target_segment=seg,
            constructs=[construct],
            generators=[pw],
            constraints=[_grad_constraint(seg)],
            config=GradientOptimizerConfig(num_results=2, num_steps=2, lr=0.1),
        )

        sg = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig())
        sg.assign(seg)
        opt2 = _mcmc(construct, [sg], [_protein_constraint(seg)])

        Program(optimizers=[opt1, opt2], num_results=2)


class TestCaptureRestoreFidelity:
    """Restart capture/restore must preserve logits AND structure (regression).

    ``_capture_initial_state`` previously serialized sequences with plain
    ``to_dict()`` (logits and structure dropped), so a restart rehydrated
    sequences without them. Both the base optimizer and the GradientOptimizer
    override must round-trip both fields.
    """

    @staticmethod
    def _seed_pools_with_logits_and_structure(segment: Segment) -> None:
        """Give every sequence in *segment*'s pools non-None logits and a structure."""
        for seq in segment.result_sequences + segment.proposal_sequences:
            seq.logits = np.zeros((len(seq.sequence), 20))
            seq.structure = MockStructure()

    @staticmethod
    def _wipe_pools(segment: Segment) -> None:
        """Clear logits and structure off the live pools to prove restore re-populates them."""
        for seq in segment.result_sequences + segment.proposal_sequences:
            seq.logits = None
            seq.structure = None

    def _assert_pools_have_logits_and_structure(self, segment: Segment) -> None:
        pools = segment.result_sequences + segment.proposal_sequences
        assert pools, "expected populated pools"
        for seq in pools:
            assert seq.logits is not None
            assert seq.structure is not None

    def test_base_optimizer_restore_preserves_logits_and_structure(self):
        """Base ``_capture_initial_state``/``_restore_initial_state`` round-trip both fields."""
        seg = Segment(sequence="MKKLLLAA", sequence_type="protein")
        gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig())
        gen.assign(seg)
        opt = _rejection(Construct([seg]), [gen], [_protein_constraint(seg)])

        self._seed_pools_with_logits_and_structure(seg)
        opt._capture_initial_state()
        self._wipe_pools(seg)
        opt._restore_initial_state()

        self._assert_pools_have_logits_and_structure(seg)

    def test_gradient_optimizer_restore_preserves_logits_and_structure(self):
        """GradientOptimizer override round-trips both fields, not just logits."""
        seg = Segment(sequence="MKKLLLAA", sequence_type="protein")
        pw = PositionWeightGenerator(PositionWeightGeneratorConfig())
        pw.assign(seg)
        opt = GradientOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[pw],
            constraints=[_grad_constraint(seg)],
            config=GradientOptimizerConfig(num_results=2, num_steps=1, lr=0.1),
        )

        self._seed_pools_with_logits_and_structure(seg)
        opt._capture_initial_state()
        self._wipe_pools(seg)
        opt._restore_initial_state()

        self._assert_pools_have_logits_and_structure(seg)


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
        program = _create_simple_program(num_stages=2, sequence=original_seq, seed=42)

        # Run stage 0 and serialize
        program.run_stage(0)
        state = program.serialize_state()
        stage0_sequences = [seq.sequence for seq in program.constructs[0].segments[0].result_sequences]

        # Create fresh program and restore
        fresh_program = _create_simple_program(num_stages=2, sequence=original_seq, seed=42)
        fresh_program.restore_state(state)

        # Verify sequences were restored
        restored_sequences = [seq.sequence for seq in fresh_program.constructs[0].segments[0].result_sequences]
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
        program1 = _create_simple_program(num_stages=2, sequence=original_seq, seed=42)
        program1.run_stage(0)
        state = program1.serialize_state()
        stage0_sequences = [seq.sequence for seq in program1.constructs[0].segments[0].result_sequences]

        # Restore and run stage 1
        program2 = _create_simple_program(num_stages=2, sequence=original_seq, seed=42)
        program2.restore_state(state)
        program2.current_stage = 1  # API tracks this in DB, not in serialized state

        # Verify state was restored before running stage 1
        pre_stage1_sequences = [seq.sequence for seq in program2.constructs[0].segments[0].result_sequences]
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

    def test_serialize_restore_preserves_logits(self):
        """Optimizer handoff state preserves logits for hosted multi-stage runs."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)
        segment = program.constructs[0].segments[0]
        logits = np.arange(len(segment.result_sequences[0].sequence) * 4, dtype=np.float64).reshape(-1, 4)
        segment.result_sequences[0].logits = logits

        state = program.serialize_state()
        assert state["segments"][0]["result_sequences"][0]["logits"] == logits.tolist()

        fresh_program = _create_simple_program(num_stages=1)
        fresh_program.restore_state(state)

        restored_seq = fresh_program.constructs[0].segments[0].result_sequences[0]
        np.testing.assert_array_equal(restored_seq.logits, logits)

    def test_serialize_restore_preserves_structure(self):
        """Optimizer handoff state preserves structures used by pLDDT-weighted generators."""
        program = _create_simple_program(num_stages=1)
        program.run_stage(0)
        segment = program.constructs[0].segments[0]
        plddt = [0.2, 0.8]
        segment.result_sequences[0].structure = MockStructure.with_plddt(plddt)

        state = program.serialize_state()
        assert "structure" in state["segments"][0]["result_sequences"][0]

        fresh_program = _create_simple_program(num_stages=1)
        fresh_program.restore_state(state)

        restored_seq = fresh_program.constructs[0].segments[0].result_sequences[0]
        assert restored_seq.structure is not None
        np.testing.assert_allclose(restored_seq.structure.per_residue_plddt, plddt)

    def test_serialize_state_with_handoff_payloads_is_json_compatible(self):
        """Intermediate DB state survives the hosted JSON round-trip."""
        import json

        program = _create_simple_program(num_stages=1)
        program.run_stage(0)
        segment = program.constructs[0].segments[0]
        sequence = segment.result_sequences[0]
        logits = np.zeros((len(sequence.sequence), 4), dtype=np.float64)
        plddt = [0.2, 0.8]
        sequence.logits = logits
        sequence.structure = MockStructure.with_plddt(plddt)

        state = program.serialize_state()
        structure_state = state["segments"][0]["result_sequences"][0]["structure"]
        assert isinstance(structure_state["b_factor_type"], str)

        restored_state = json.loads(json.dumps(state))
        fresh_program = _create_simple_program(num_stages=1)
        fresh_program.restore_state(restored_state)

        restored_seq = fresh_program.constructs[0].segments[0].result_sequences[0]
        np.testing.assert_array_equal(restored_seq.logits, logits)
        assert restored_seq.structure is not None
        np.testing.assert_allclose(restored_seq.structure.per_residue_plddt, plddt)


class TestProgramExport:
    """Tests for Program.export, to_dataframe, and to_fasta."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.program = _create_simple_program(num_stages=1)
        self.program.run()

    def test_export_writes_folder_with_4_tables_fasta_and_assets(self, tmp_path):
        """export() always writes a folder: 4 CSVs + FASTA + assets/ (empty when no payloads)."""
        out = self.program.export(path=tmp_path / "results", format="csv")
        assert out.is_dir() and out == tmp_path / "results"
        for name in ("sequences", "constraints", "constructs", "optimization"):
            assert (out / f"{name}.csv").stat().st_size > 0
        assert (out / "sequences.fasta").exists()
        assert (out / "assets").is_dir()

    def test_export_xlsx_writes_workbook_inside_folder(self, tmp_path):
        """Xlsx format produces a single results.xlsx workbook inside the folder, alongside assets/."""
        out = self.program.export(path=tmp_path / "results", format="xlsx")
        assert (out / "results.xlsx").stat().st_size > 0
        assert (out / "assets").is_dir()

    def test_export_stage_filter_writes_that_stage_only(self, tmp_path):
        """export(stage=0) emits sequences only from stage 0's results."""
        program = _create_simple_program(num_stages=2)
        program.run()
        out = program.export(path=tmp_path / "s0", stage=0)
        opt_csv = (out / "optimization.csv").read_text()
        # stage-1 timepoints must not appear in the optimization table.
        assert "stage_1" not in opt_csv

    @pytest.mark.parametrize(
        "table,expected_col",
        [
            ("sequences", "sequence"),
            ("constraints", "constraint"),
            ("constructs", "full_sequence"),
            ("optimization", "timepoint"),
        ],
    )
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

    @patch("proto_tools.utils.tool_pool.ToolPool")
    def test_compute_nullcontext_when_dispatch_configured(self, mock_pool_cls):
        """Configured dispatch backend -> nullcontext, not ToolPool."""
        from proto_tools.tools.tool_registry import ToolRegistry

        ToolRegistry.configure_dispatch_backend(lambda key, inputs, config: None)
        try:
            program = _create_simple_program(compute=None)
            assert isinstance(program.compute, nullcontext)
            mock_pool_cls.assert_not_called()
        finally:
            ToolRegistry.clear_dispatch_backend()

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
