from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from typing import List

from proto_language.language.core import (
    Optimizer,
    Construct,
    Segment,
    Constraint,
    Generator,
    Sequence,
)


# Concrete implementation for testing the abstract base class
class ConcreteOptimizer(Optimizer):
    """Concrete implementation of Optimizer for testing purposes."""
    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        num_candidates: int,
        num_selected: int,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_candidates=num_candidates,
            num_selected=num_selected,
            clear_tool_cache=clear_tool_cache,
            verbose=verbose
        )

    def run(self) -> None:
        """Dummy run implementation."""
        pass


class MockGenerator(Generator):
    """Minimal mock generator for testing."""
    def __init__(self):
        super().__init__()
        self._assigned_segment = None

    def assign(self, segment: Segment) -> None:
        self._assigned_segment = segment

    def sample(self, *args, **kwargs) -> None:
        pass


def _setup_optimizer_components(num_candidates: int = 4):
    """Helper function to set up components for testing Optimizer."""
    segment = Segment(sequence="ATCG", sequence_type="dna")
    construct = Construct([segment])

    generator = MockGenerator()
    generator.assign(segment)

    constraint = MagicMock(spec=Constraint)
    constraint.inputs = [segment]
    constraint.label = "MockConstraint"
    constraint.threshold = None
    constraint.weight = 1.0
    constraint.evaluate.return_value = [1.0] * num_candidates

    return construct, generator, constraint, segment


class TestOptimizerInitialization:
    """Tests for optimizer initialization and validation."""

    def test_valid_initialization(self):
        """Tests that optimizer initializes correctly with valid inputs."""
        construct, generator, constraint, _ = _setup_optimizer_components()

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=4,
            num_selected=2,
        )
        assert optimizer.num_candidates == 4
        assert optimizer.num_selected == 2

    def test_empty_lists_raise_errors(self):
        """Tests that empty constructs/generators/constraints raise ValueError."""
        construct, generator, constraint, _ = _setup_optimizer_components()

        with pytest.raises(ValueError, match="Constructs list cannot be empty"):
            ConcreteOptimizer([], [generator], [constraint], 4, 2)

        with pytest.raises(ValueError, match="Generators list cannot be empty"):
            ConcreteOptimizer([construct], [], [constraint], 4, 2)

        with pytest.raises(ValueError, match="Constraints list cannot be empty"):
            ConcreteOptimizer([construct], [generator], [], 4, 2)

    def test_unassigned_generator_raises_error(self):
        """Tests that unassigned generator raises RuntimeError."""
        construct, _, constraint, _ = _setup_optimizer_components()
        unassigned_gen = MockGenerator()

        with pytest.raises(RuntimeError, match="has no segment assigned"):
            ConcreteOptimizer([construct], [unassigned_gen], [constraint], 4, 2)

    def test_constraint_with_no_inputs_raises_error(self):
        """Tests that constraint with no inputs raises RuntimeError."""
        construct, generator, _, _ = _setup_optimizer_components()
        empty_constraint = MagicMock(spec=Constraint)
        empty_constraint.inputs = []

        with pytest.raises(RuntimeError, match="has no input segment"):
            ConcreteOptimizer([construct], [generator], [empty_constraint], 4, 2)


class TestSequencePoolInitialization:
    """Tests for sequence pool initialization behavior."""

    def test_fresh_initialization(self):
        """Tests initialization when no previous candidates exist."""
        original_seq = "ATCG"
        segment = Segment(sequence=original_seq, sequence_type="dna")
        segment.candidate_sequences = []
        segment.selected_sequences = []
        construct = Construct([segment])
        generator = MockGenerator()
        generator.assign(segment)
        constraint = MagicMock(spec=Constraint)
        constraint.inputs = [segment]
        constraint.label = "MockConstraint"
        constraint.threshold = None
        constraint.weight = 1.0
        constraint.evaluate.return_value = [1.0] * 5

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=5,
            num_selected=2,
        )
        assert optimizer.constructs == [construct]

        # Selected sequences initialized from original
        assert len(segment.selected_sequences) == 2
        assert all(s.sequence == original_seq for s in segment.selected_sequences)

        # Candidate sequences initialized from original
        assert len(segment.candidate_sequences) == 5
        assert all(s.sequence == original_seq for s in segment.candidate_sequences)

        # Verify independence (deep copy)
        segment.selected_sequences[0].sequence = "GGGG"
        assert segment.candidate_sequences[0].sequence == original_seq

    def test_chained_initialization(self):
        """Tests initialization when inheriting from previous optimizer."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=5)

        # Simulate previous optimizer results
        prev_best = Sequence(sequence="AAAA", sequence_type="dna")
        prev_second = Sequence(sequence="TTTT", sequence_type="dna")
        segment.selected_sequences = [prev_best, prev_second]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=5,
            num_selected=3,  # Requesting 3, but only 2 available
        )
        assert optimizer.constructs == [construct]

        # Uses existing selected sequences
        assert segment.selected_sequences[0].sequence == "AAAA"
        assert segment.selected_sequences[1].sequence == "TTTT"

        # Pads with copies of best sequence
        assert len(segment.selected_sequences) == 3
        assert segment.selected_sequences[2].sequence == "AAAA"

        # Candidates initialized from best sequence
        assert len(segment.candidate_sequences) == 5
        assert all(s.sequence == "AAAA" for s in segment.candidate_sequences)


class TestScoreEnergy:
    """Tests for energy scoring functionality."""

    def test_add_operation(self):
        """Tests energy scoring with 'add' operation."""
        construct, generator, constraint1, segment = _setup_optimizer_components(num_candidates=2)

        constraint1.weight = 1.0
        constraint1.threshold = None
        constraint1.evaluate.return_value = [10.0, 20.0]

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.weight = 1.0
        constraint2.threshold = None
        constraint2.evaluate.return_value = [4.0, 8.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint1, constraint2],
            num_candidates=2,
            num_selected=2,
        )
        segment.candidate_sequences = [Sequence("A"), Sequence("C")]
        optimizer.score_energy(operation="add")

        # Expected: [10+4, 20+8] = [14, 28]
        assert optimizer.energy_scores == [14.0, 28.0]

    def test_multiply_operation(self):
        """Tests energy scoring with 'multiply' operation."""
        construct, generator, constraint1, segment = _setup_optimizer_components(num_candidates=2)

        constraint1.weight = 1.0
        constraint1.evaluate.return_value = [2.0, 3.0]

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.weight = 1.0
        constraint2.threshold = None
        constraint2.evaluate.return_value = [4.0, 5.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint1, constraint2],
            num_candidates=2,
            num_selected=2,
        )
        segment.candidate_sequences = [Sequence("A"), Sequence("C")]
        optimizer.score_energy(operation="multiply")

        # Expected: [2*4, 3*5] = [8, 15]
        assert optimizer.energy_scores == [8.0, 15.0]

    def test_invalid_operation_raises_error(self):
        """Tests that invalid operation raises ValueError."""
        construct, generator, constraint, segment = _setup_optimizer_components()
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)
        segment.candidate_sequences = [Sequence("A"), Sequence("C")]

        with pytest.raises(ValueError, match="Operation must be 'add' or 'multiply'"):
            optimizer.score_energy(operation="invalid")


class TestFilterConstraints:
    """Tests for filter constraint behavior in scoring."""

    def test_filter_rejection_applies_penalty(self):
        """Tests that filter constraints reject candidates and apply penalty."""
        construct, generator, filter_constraint, segment = _setup_optimizer_components(num_candidates=3)
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # Filter rejects candidate 1
        filter_constraint.threshold = 0.5
        filter_constraint.evaluate.return_value = [True, False, True]

        # Scoring constraint (returns NaN for skipped candidate 1)
        scoring_constraint = MagicMock(spec=Constraint)
        scoring_constraint.inputs = [segment]
        scoring_constraint.threshold = None
        scoring_constraint.weight = 1.0
        scoring_constraint.evaluate.return_value = [10.0, float('nan'), 20.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[filter_constraint, scoring_constraint],
            num_candidates=3,
            num_selected=2,
        )

        optimizer.score_energy(operation="add", filter_penalty=999.0)

        # Candidate 0: passed → 10.0
        # Candidate 1: rejected → 999.0 (penalty)
        # Candidate 2: passed → 20.0
        assert optimizer.energy_scores == [10.0, 999.0, 20.0]

    def test_filter_skips_subsequent_evaluation(self):
        """Tests that rejected candidates skip subsequent constraint evaluations."""
        construct, generator, filter_constraint, segment = _setup_optimizer_components(num_candidates=3)
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # Filter rejects candidate 1
        filter_constraint.threshold = 0.0
        filter_constraint.evaluate.return_value = [True, False, True]

        # Scoring constraint
        scoring_constraint = MagicMock(spec=Constraint)
        scoring_constraint.inputs = [segment]
        scoring_constraint.threshold = None
        scoring_constraint.evaluate.return_value = [1.0, float('nan'), 1.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[filter_constraint, scoring_constraint],
            num_candidates=3,
            num_selected=2,
        )

        optimizer.score_energy()

        # Verify scoring constraint received mask reflecting filter rejection
        _, kwargs = scoring_constraint.evaluate.call_args
        assert kwargs['mask'] == [True, False, True]

    def test_inconsistent_state_raises_error(self):
        """Tests that passed candidate with NaN score raises RuntimeError."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=3)
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # No filter - all candidates should pass
        constraint.threshold = None
        constraint.weight = 1.0
        # Bug: returns NaN for candidate 1 even though it wasn't filtered
        constraint.evaluate.return_value = [1.0, float('nan'), 1.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_candidates=3,
            num_selected=2,
        )

        with pytest.raises(RuntimeError, match="Inconsistent state: candidate 1 passed all filters but has NaN score"):
            optimizer.score_energy(operation="add")


class TestToolCacheClearing:
    """Tests for tool cache clearing functionality."""

    def test_cache_clearing_modes(self):
        """Tests different cache clearing configurations."""
        construct, generator, constraint, _ = _setup_optimizer_components()

        with patch('proto_language.language.core.optimizer.ToolCache') as MockCache:
            mock_cache = MockCache.return_value
            mock_cache.current_size = 200

            # Case 1: int threshold (prune if exceeded)
            optimizer = ConcreteOptimizer([construct], [generator], [constraint], 4, 2, clear_tool_cache=100)
            optimizer._clear_tool_cache()
            mock_cache.prune.assert_called_with(100)

            # Case 2: bool True (clear all)
            mock_cache.prune.reset_mock()
            optimizer.clear_tool_cache = True
            optimizer._clear_tool_cache()
            mock_cache.clear.assert_called_once()

            # Case 3: List[str] (clear specific tools)
            mock_cache.clear.reset_mock()
            optimizer.clear_tool_cache = ["tool_A"]
            optimizer._clear_tool_cache()
            mock_cache.clear.assert_called_with("tool_A")


class TestProgressSnapshot:
    """Tests for history/progress tracking functionality."""

    def test_snapshot_captures_state(self):
        """Tests that history snapshots capture correct state."""
        construct, generator, constraint, _ = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        # energy_scores must have length num_selected before snapshot
        optimizer.energy_scores = [0.1]
        optimizer._save_progress_snapshot(time_step=5)

        assert len(optimizer.history) == 1
        snapshot = optimizer.history[0]

        assert snapshot["time_step"] == 5
        assert snapshot["energy_scores"] == [0.1]
        assert "constructs" in snapshot
        # Should be deep copy
        assert snapshot["constructs"] is not optimizer.constructs

    def test_snapshot_validates_energy_scores_length(self):
        """Tests that snapshot raises error if energy_scores length != num_selected."""
        construct, generator, constraint, _ = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        # energy_scores has wrong length (num_candidates instead of num_selected)
        optimizer.energy_scores = [0.1, 0.9]

        with pytest.raises(RuntimeError, match="energy_scores has length 2, expected 1"):
            optimizer._save_progress_snapshot(time_step=5)


class TestStateRestartBehavior:
    """Tests for optimizer state capture and restore on re-run."""

    def test_prepare_run_captures_state_on_first_call(self):
        """Tests that _prepare_run captures state on first call."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        assert optimizer._initial_state is None

        optimizer._prepare_run()

        assert optimizer._initial_state is not None
        assert 'segments' in optimizer._initial_state
        assert 'energy_scores' in optimizer._initial_state

    def test_prepare_run_restores_state_on_subsequent_calls(self):
        """Tests that _prepare_run restores state on subsequent calls."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        # Capture initial state
        original_seq = segment.candidate_sequences[0].sequence
        optimizer._prepare_run()

        # Modify state
        segment.candidate_sequences[0].sequence = "GGGG"
        segment.selected_sequences[0].sequence = "CCCC"
        optimizer.energy_scores = [999.0, 999.0]
        optimizer.history = [{"test": "data"}]

        # Restore
        optimizer._prepare_run()

        # Verify state restored
        assert segment.candidate_sequences[0].sequence == original_seq
        assert segment.selected_sequences[0].sequence == original_seq
        assert optimizer.energy_scores == [float("inf"), float("inf")]
        assert optimizer.history == []

    def test_restore_initial_state_clears_history(self):
        """Tests that _restore_initial_state clears history."""
        construct, generator, constraint, _ = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        optimizer._capture_initial_state()
        optimizer.history = [{"step": 1}, {"step": 2}]

        optimizer._restore_initial_state()

        assert optimizer.history == []

    def test_state_independence_via_deepcopy(self):
        """Tests that captured state is independent of current state."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        optimizer._capture_initial_state()

        # Modify current state
        segment.candidate_sequences[0].sequence = "XXXX"

        # Captured state should be unchanged (using index 0)
        captured_seq = optimizer._initial_state['segments'][0]['candidates'][0].sequence
        assert captured_seq == "ATCG"
