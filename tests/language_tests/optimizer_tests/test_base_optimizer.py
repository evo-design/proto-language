from __future__ import annotations

import logging
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
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


class TestOptimizerValidation:
    """Tests for Optimizer._validate_optimizer checks."""

    # 1. Non-empty lists
    def test_empty_constructs_raises(self):
        """Tests that empty constructs list raises ValueError."""
        _, generator, constraint, _ = _setup_optimizer_components()
        with pytest.raises(ValueError, match="Constructs list cannot be empty"):
            ConcreteOptimizer([], [generator], [constraint], 4, 2)

    def test_empty_generators_raises(self):
        """Tests that empty generators list raises ValueError."""
        construct, _, constraint, _ = _setup_optimizer_components()
        with pytest.raises(ValueError, match="Generators list cannot be empty"):
            ConcreteOptimizer([construct], [], [constraint], 4, 2)

    def test_empty_constraints_raises(self):
        """Tests that empty constraints list raises ValueError."""
        construct, generator, _, _ = _setup_optimizer_components()
        with pytest.raises(ValueError, match="Constraints list cannot be empty"):
            ConcreteOptimizer([construct], [generator], [], 4, 2)

    # 2. Type validation
    def test_invalid_generator_type_raises(self):
        """Tests that non-Generator type raises TypeError."""
        construct, _, constraint, _ = _setup_optimizer_components()
        with pytest.raises(TypeError, match="expected Generator"):
            ConcreteOptimizer([construct], ["not_a_generator"], [constraint], 4, 2)

    def test_invalid_constraint_type_raises(self):
        """Tests that non-Constraint type raises TypeError."""
        construct, generator, _, _ = _setup_optimizer_components()
        with pytest.raises(TypeError, match="expected Constraint"):
            ConcreteOptimizer([construct], [generator], ["not_a_constraint"], 4, 2)

    # 3. Structure validation
    def test_construct_with_no_segments_raises(self):
        """Tests that construct with no segments raises ValueError."""
        _, generator, constraint, _ = _setup_optimizer_components()
        empty_construct = MagicMock(spec=Construct)
        empty_construct.segments = []
        with pytest.raises(ValueError, match="has no segments"):
            ConcreteOptimizer([empty_construct], [generator], [constraint], 4, 2)

    def test_unassigned_generator_raises(self):
        """Tests that unassigned generator raises RuntimeError."""
        construct, _, constraint, _ = _setup_optimizer_components()
        unassigned_gen = MockGenerator()
        with pytest.raises(RuntimeError, match="has no segment assigned"):
            ConcreteOptimizer([construct], [unassigned_gen], [constraint], 4, 2)

    def test_constraint_with_no_inputs_raises(self):
        """Tests that constraint with no inputs raises RuntimeError."""
        construct, generator, _, _ = _setup_optimizer_components()
        empty_constraint = MagicMock(spec=Constraint)
        empty_constraint.inputs = []
        with pytest.raises(RuntimeError, match="has no input segment"):
            ConcreteOptimizer([construct], [generator], [empty_constraint], 4, 2)

    # 4. No duplicate instances
    def test_duplicate_generator_instance_raises(self):
        """Tests that same generator instance appearing twice raises ValueError."""
        construct, generator, constraint, segment = _setup_optimizer_components()
        # Use same generator instance twice
        with pytest.raises(ValueError, match="appears multiple times.*can only be used once"):
            ConcreteOptimizer([construct], [generator, generator], [constraint], 4, 2)

    def test_duplicate_constraint_instance_raises(self):
        """Tests that same constraint instance appearing twice raises ValueError."""
        construct, generator, constraint, _ = _setup_optimizer_components()
        # Use same constraint instance twice
        with pytest.raises(ValueError, match="appears multiple times.*can only be used once"):
            ConcreteOptimizer([construct], [generator], [constraint, constraint], 4, 2)

    # 5. Unique constraint labels per segment (auto-renamed on collision)
    def test_duplicate_constraint_labels_same_segment_auto_renames(self):
        """Tests that duplicate constraint labels on same segment are auto-renamed."""
        construct, generator, _, segment = _setup_optimizer_components()

        constraint1 = MagicMock(spec=Constraint)
        constraint1.inputs = [segment]
        constraint1.label = "same_label"
        constraint1.threshold = None
        constraint1.weight = 1.0

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.label = "same_label"  # Duplicate label on same segment - will be auto-renamed
        constraint2.threshold = None
        constraint2.weight = 1.0

        ConcreteOptimizer([construct], [generator], [constraint1, constraint2], 4, 2)

        # First keeps original, second gets renamed
        assert constraint1.label == "same_label"
        assert constraint2.label == "same_label_1"

    def test_multiple_duplicate_constraint_labels_auto_renames(self):
        """Tests that multiple duplicate labels get incrementing suffixes."""
        construct, generator, _, segment = _setup_optimizer_components()

        constraint1 = MagicMock(spec=Constraint)
        constraint1.inputs = [segment]
        constraint1.label = "gc_content"
        constraint1.threshold = None
        constraint1.weight = 1.0

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.label = "gc_content"
        constraint2.threshold = None
        constraint2.weight = 1.0

        constraint3 = MagicMock(spec=Constraint)
        constraint3.inputs = [segment]
        constraint3.label = "gc_content"
        constraint3.threshold = None
        constraint3.weight = 1.0

        ConcreteOptimizer([construct], [generator], [constraint1, constraint2, constraint3], 4, 2)

        # First keeps original, subsequent get incrementing suffixes
        assert constraint1.label == "gc_content"
        assert constraint2.label == "gc_content_1"
        assert constraint3.label == "gc_content_2"

    def test_duplicate_constraint_labels_different_segments_allowed(self):
        """Tests that same constraint labels on different segments are allowed."""
        segment1 = Segment(sequence="ATCG", sequence_type="dna")
        segment2 = Segment(sequence="GCTA", sequence_type="dna")
        construct = Construct([segment1, segment2])

        generator1 = MockGenerator()
        generator1.assign(segment1)
        generator2 = MockGenerator()
        generator2.assign(segment2)

        constraint1 = MagicMock(spec=Constraint)
        constraint1.inputs = [segment1]
        constraint1.label = "same_label"
        constraint1.threshold = None
        constraint1.weight = 1.0

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment2]
        constraint2.label = "same_label"  # Same label but different segment - OK!
        constraint2.threshold = None
        constraint2.weight = 1.0

        # Should not raise
        ConcreteOptimizer([construct], [generator1, generator2], [constraint1, constraint2], 4, 2)

    # 6. Valid constraint inputs
    def test_constraint_references_unpopulated_segment_raises(self):
        """Tests that constraint referencing unpopulated segment without generator raises."""
        segment_with_gen = Segment(sequence="ATCG", sequence_type="dna")
        segment_no_gen = Segment(sequence_type="dna", length=4)  # No input sequence
        construct = Construct([segment_with_gen, segment_no_gen])

        generator = MockGenerator()
        generator.assign(segment_with_gen)  # Only first segment has generator

        constraint = MagicMock(spec=Constraint)
        constraint.inputs = [segment_no_gen]  # References unpopulated segment!
        constraint.label = "bad_constraint"
        constraint.threshold = None
        constraint.weight = 1.0

        with pytest.raises(RuntimeError, match="no populated sequence and no generator assigned"):
            ConcreteOptimizer([construct], [generator], [constraint], 4, 2)

    # 7. Homo-oligomer constraints (same segment repeated in inputs)
    def test_homo_oligomer_constraint_same_segment_multiple_times(self):
        """Tests that constraints with same segment repeated don't accumulate _1 suffixes.

        This is a regression test for a bug where homo-oligomer constraints (e.g., trimers)
        that have the same segment multiple times in inputs would have their labels
        incorrectly renamed on each iteration through the inputs list.
        """
        construct, generator, _, segment = _setup_optimizer_components()

        # Homo-trimer constraint: same segment appears 3 times (common for structure prediction)
        trimer_constraint = MagicMock(spec=Constraint)
        trimer_constraint.inputs = [segment, segment, segment]  # Same segment 3x
        trimer_constraint.label = "structure_plddt_constraint"
        trimer_constraint.threshold = None
        trimer_constraint.weight = 1.0

        ConcreteOptimizer([construct], [generator], [trimer_constraint], 4, 2)

        # Label should NOT have _1 suffix - it's the same segment, not a collision
        assert trimer_constraint.label == "structure_plddt_constraint"

    def test_label_deduplication_is_idempotent(self):
        """Tests that calling _validate_optimizer multiple times doesn't accumulate suffixes.

        This is a regression test for a bug where score_energy() called _validate_optimizer(),
        causing constraint labels to accumulate _1_1_1... suffixes on each iteration.
        """
        construct, generator, _, segment = _setup_optimizer_components()

        # Two constraints with same label on same segment (collision case)
        constraint1 = MagicMock(spec=Constraint)
        constraint1.inputs = [segment]
        constraint1.label = "my_constraint"
        constraint1.threshold = None
        constraint1.weight = 1.0
        constraint1.evaluate.return_value = [1.0, 1.0, 1.0, 1.0]

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.label = "my_constraint"  # Same label - will be renamed to _1
        constraint2.threshold = None
        constraint2.weight = 1.0
        constraint2.evaluate.return_value = [1.0, 1.0, 1.0, 1.0]

        optimizer = ConcreteOptimizer([construct], [generator], [constraint1, constraint2], 4, 2)

        # After __init__, labels should be deduplicated
        assert constraint1.label == "my_constraint"
        assert constraint2.label == "my_constraint_1"

        # Call score_energy multiple times (simulating optimization iterations)
        for _ in range(5):
            optimizer.score_energy()

        # Labels should NOT accumulate more suffixes
        assert constraint1.label == "my_constraint"
        assert constraint2.label == "my_constraint_1"  # Still _1, not _1_1_1_1_1

    def test_homo_oligomer_with_multiple_constraints_same_label(self):
        """Tests combination of homo-oligomer and duplicate label handling.

        When multiple constraints with the same label each have the same segment
        repeated, the deduplication should only apply between different constraints,
        not within a single constraint's inputs.
        """
        construct, generator, _, segment = _setup_optimizer_components()

        # Two homo-trimer constraints with same base label
        constraint1 = MagicMock(spec=Constraint)
        constraint1.inputs = [segment, segment, segment]  # Trimer
        constraint1.label = "structure_constraint"
        constraint1.threshold = None
        constraint1.weight = 1.0
        constraint1.evaluate.return_value = [1.0, 1.0, 1.0, 1.0]

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment, segment, segment]  # Also trimer, same label
        constraint2.label = "structure_constraint"
        constraint2.threshold = None
        constraint2.weight = 1.0
        constraint2.evaluate.return_value = [1.0, 1.0, 1.0, 1.0]

        optimizer = ConcreteOptimizer([construct], [generator], [constraint1, constraint2], 4, 2)

        # First constraint keeps original label
        assert constraint1.label == "structure_constraint"
        # Second constraint gets _1 suffix (collision between constraints)
        assert constraint2.label == "structure_constraint_1"

        # Multiple score_energy calls shouldn't change anything
        for _ in range(3):
            optimizer.score_energy()

        assert constraint1.label == "structure_constraint"
        assert constraint2.label == "structure_constraint_1"

    def test_four_constraints_same_label_sequential_suffixes(self):
        """Tests that 4+ constraints with same label on same segment get sequential suffixes.

        Regression test for a bug where label mutation mid-iteration caused the
        deduplication key to use the already-renamed label instead of the original,
        leading to missed collisions.
        """
        construct, generator, _, segment = _setup_optimizer_components()

        constraints = []
        for _ in range(4):
            c = MagicMock(spec=Constraint)
            c.inputs = [segment]
            c.label = "energy"
            c.threshold = None
            c.weight = 1.0
            c.evaluate.return_value = [1.0, 1.0, 1.0, 1.0]
            constraints.append(c)

        ConcreteOptimizer([construct], [generator], constraints, 4, 2)

        assert constraints[0].label == "energy"
        assert constraints[1].label == "energy_1"
        assert constraints[2].label == "energy_2"
        assert constraints[3].label == "energy_3"

    def test_multi_segment_constraint_label_dedup_uses_base_label(self):
        """Tests that a multi-segment constraint's label dedup uses the original base label.

        Regression test: if constraint A has inputs [seg1, seg2] and constraint B
        also has label "X" on seg2, after A processes seg1 and gets renamed to "X_1",
        it should still collide with B on seg2 using base label "X" (not "X_1").
        """
        segment1 = Segment(sequence="ATCG", sequence_type="dna")
        segment2 = Segment(sequence="GCTA", sequence_type="dna")
        construct = Construct([segment1, segment2])

        gen1 = MockGenerator()
        gen1.assign(segment1)
        gen2 = MockGenerator()
        gen2.assign(segment2)

        # Constraint A: references both segments, label "score"
        constraint_a = MagicMock(spec=Constraint)
        constraint_a.inputs = [segment1, segment2]
        constraint_a.label = "score"
        constraint_a.threshold = None
        constraint_a.weight = 1.0
        constraint_a.evaluate.return_value = [1.0, 1.0, 1.0, 1.0]

        # Constraint B: references seg1 only, same label "score"
        constraint_b = MagicMock(spec=Constraint)
        constraint_b.inputs = [segment1]
        constraint_b.label = "score"
        constraint_b.threshold = None
        constraint_b.weight = 1.0
        constraint_b.evaluate.return_value = [1.0, 1.0, 1.0, 1.0]

        ConcreteOptimizer(
            [construct], [gen1, gen2], [constraint_a, constraint_b], 4, 2
        )

        # constraint_a collides with constraint_b on seg1 → constraint_b gets renamed
        assert constraint_a.label == "score"
        assert constraint_b.label == "score_1"


class TestOptimizerInitialization:
    """Tests for optimizer initialization behavior."""

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

        # Selected sequences padded by cycling: [AAAA, TTTT, AAAA]
        assert len(segment.selected_sequences) == 3
        assert segment.selected_sequences[0].sequence == "AAAA"
        assert segment.selected_sequences[1].sequence == "TTTT"
        assert segment.selected_sequences[2].sequence == "AAAA"  # Cycles back

        # Candidates also initialized by cycling: [AAAA, TTTT, AAAA, TTTT, AAAA]
        assert len(segment.candidate_sequences) == 5
        expected = ["AAAA", "TTTT", "AAAA", "TTTT", "AAAA"]
        for i, seq in enumerate(segment.candidate_sequences):
            assert seq.sequence == expected[i], f"candidate[{i}] expected {expected[i]}, got {seq.sequence}"


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
        constraint2.label = "MockConstraint2"
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
        constraint2.label = "MockConstraint2"
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
        scoring_constraint.label = "ScoringConstraint"
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
        scoring_constraint.label = "ScoringConstraint"
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

    def test_filter_only_constraints_warns(self, caplog):
        """Tests that filter-only constraints (no scoring constraints) logs a warning."""
        construct, generator, filter_constraint, segment = _setup_optimizer_components(num_candidates=2)
        segment.candidate_sequences = [Sequence("A"), Sequence("C")]

        # Only a filter constraint, no scoring constraints
        filter_constraint.threshold = 0.5
        filter_constraint.evaluate.return_value = [True, True]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[filter_constraint],
            num_candidates=2,
            num_selected=2,
        )

        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.optimizer"):
            optimizer.score_energy()

        assert any("All constraints are filters" in msg for msg in caplog.messages)
        # Passing candidates get energy 0.0 (sum of empty list)
        assert optimizer.energy_scores == [0.0, 0.0]

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

    def test_verbose_logs_rejection_reasons(self, caplog):
        """Tests that verbose logging accurately attributes rejections to the filter that evaluated them.

        Filter 1 (pLDDT): passes candidate 0, rejects candidate 1
        Filter 2 (Length): rejects candidate 2 (candidate 1 was already masked, so Length never evaluated it)

        Expected: candidate 1 → "pLDDT Filter" only, candidate 2 → "Length Filter" only.
        """
        construct, generator, _, segment = _setup_optimizer_components(num_candidates=3)
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        filter1 = MagicMock(spec=Constraint)
        filter1.inputs = [segment]
        filter1.label = "pLDDT Filter"
        filter1.threshold = 0.5
        filter1.weight = 1.0
        filter1.evaluate.return_value = [True, False, True]

        filter2 = MagicMock(spec=Constraint)
        filter2.inputs = [segment]
        filter2.label = "Length Filter"
        filter2.threshold = 0.5
        filter2.weight = 1.0
        filter2.evaluate.return_value = [True, True, False]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[filter1, filter2],
            num_candidates=3,
            num_selected=2,
            verbose=True,
        )

        with caplog.at_level(logging.INFO, logger="proto_language.language.core.optimizer"):
            optimizer.score_energy()

        # Candidate 0 passed both filters
        assert any("Candidate 0" in m and "ACCEPTED" in m for m in caplog.messages)
        # Candidate 1 rejected by pLDDT Filter only (Length Filter never evaluated it)
        assert any(
            "Candidate 1" in m and "REJECTED by pLDDT Filter" in m and "Length Filter" not in m
            for m in caplog.messages
        )
        # Candidate 2 rejected by Length Filter only (pLDDT Filter passed it)
        assert any(
            "Candidate 2" in m and "REJECTED by Length Filter" in m and "pLDDT" not in m
            for m in caplog.messages
        )


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
        assert "batch_results" in snapshot
        assert len(snapshot["batch_results"]) == 1
        assert snapshot["batch_results"][0]["energy_score"] == 0.1
        assert isinstance(snapshot["batch_results"][0]["constructs"], list)

    def test_snapshot_validates_energy_scores_matches_selected(self):
        """Tests that snapshot raises error if energy_scores length != selected_sequences length."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        # energy_scores doesn't match selected_sequences length (1)
        optimizer.energy_scores = [0.1, 0.9]

        with pytest.raises(RuntimeError, match="energy_scores has length 2, expected 1"):
            optimizer._save_progress_snapshot(time_step=5)

    def test_snapshot_allows_partial_selected(self):
        """Tests that snapshot allows partial selected_sequences (e.g. TopK mid-run)."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=4)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 4, 3)

        # Simulate partial state: only 2 of 3 selected
        segment.selected_sequences = segment.selected_sequences[:2]
        optimizer.energy_scores = [0.1, 0.5]

        # Should NOT raise — relaxed validation allows partial
        optimizer._save_progress_snapshot(time_step=1)
        assert len(optimizer.history) == 1
        assert len(optimizer.history[0]["batch_results"]) == 2


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

    def test_state_independence_via_serialization(self):
        """Tests that captured state is independent of current state."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        optimizer._capture_initial_state()

        # Modify current state
        segment.candidate_sequences[0].sequence = "XXXX"

        # Captured state should be unchanged (now stored as serialized dicts)
        captured_seq = optimizer._initial_state['segments'][0]['candidates'][0]['sequence']
        assert captured_seq == "ATCG"


class TestCandidateTracking:
    """Tests for _candidate_outcomes and candidate_results in snapshots."""

    def test_filter_outcomes_with_and_logic(self):
        """score_energy() sets outcomes to 'accepted' or the first failing filter's label."""
        construct, generator, _, segment = _setup_optimizer_components(num_candidates=3)
        segment.candidate_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        filter1 = MagicMock(spec=Constraint)
        filter1.inputs = [segment]
        filter1.label = "Filter1"
        filter1.threshold = 0.5
        filter1.weight = 1.0
        filter1.evaluate.return_value = [True, True, False]  # Candidate 2 fails

        filter2 = MagicMock(spec=Constraint)
        filter2.inputs = [segment]
        filter2.label = "Filter2"
        filter2.threshold = 0.5
        filter2.weight = 1.0
        filter2.evaluate.return_value = [True, False, True]  # Candidate 1 fails

        optimizer = ConcreteOptimizer(
            [construct], [generator], [filter1, filter2], 3, 2
        )
        optimizer.score_energy()

        assert optimizer._candidate_outcomes == ["accepted", "Filter2", "Filter1"]

    def test_snapshot_includes_candidate_results(self):
        """_save_progress_snapshot includes candidate_results derived from _candidate_outcomes."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        optimizer.energy_scores = [0.5]
        optimizer._candidate_outcomes = ["accepted", "GC Filter"]

        optimizer._save_progress_snapshot(time_step=1)

        snapshot = optimizer.history[0]
        assert "candidate_results" in snapshot
        assert len(snapshot["candidate_results"]) == 2
        assert snapshot["candidate_results"][0]["accepted"] is True
        assert snapshot["candidate_results"][0]["rejected_by"] is None
        assert snapshot["candidate_results"][1]["accepted"] is False
        assert snapshot["candidate_results"][1]["rejected_by"] == "GC Filter"

    def test_snapshot_omits_candidate_results_when_outcomes_empty(self):
        """_save_progress_snapshot omits candidate_results before any scoring."""
        construct, generator, constraint, _ = _setup_optimizer_components(num_candidates=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        optimizer.energy_scores = [0.5]
        optimizer._save_progress_snapshot(time_step=0)

        assert "candidate_results" not in optimizer.history[0]
