"""tests/language_tests/optimizer_tests/test_base_optimizer.py."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.language.core.constraint import ConstraintOutput, GradientConstraintOutput
from proto_language.language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
)
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.language.generator.proteinmpnn_generator import ProteinMPNNGenerator, ProteinMPNNGeneratorConfig
from proto_language.language.optimizer import (
    GradientOptimizer,
    GradientOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry


# Concrete implementation for testing the abstract base class
class ConcreteOptimizer(Optimizer):
    """Concrete implementation of Optimizer for testing purposes."""

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        num_proposals: int | None,
        num_results: int | None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
        verbose: bool = False,
        tracking_interval: int = 1,
        track_proposals: bool = False,
    ) -> None:
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_proposals=num_proposals,
            num_results=num_results,
            clear_tool_cache=clear_tool_cache,
            verbose=verbose,
            tracking_interval=tracking_interval,
            track_proposals=track_proposals,
        )

    def run(self) -> None:
        """Dummy run implementation."""
        self._prepare_run()


class MockGenerator(Generator):
    """Minimal mock generator for testing."""

    def __init__(self):
        super().__init__()
        self._assigned_segment = None

    def assign(self, segment: Segment) -> None:
        self._assigned_segment = segment

    def sample(self, *args, **kwargs) -> None:
        pass


def _setup_optimizer_components(num_proposals: int = 4):
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
    constraint.evaluate.return_value = [1.0] * num_proposals

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

    # num_results / num_proposals validation
    @pytest.mark.parametrize("num_results", [0, -1, -100])
    def test_invalid_num_results_raises(self, num_results):
        """Tests that num_results < 1 raises ValueError."""
        construct, generator, constraint, _ = _setup_optimizer_components()
        with pytest.raises(ValueError, match="num_results must be >= 1"):
            ConcreteOptimizer([construct], [generator], [constraint], 4, num_results)

    @pytest.mark.parametrize("num_proposals", [0, -1, -100])
    def test_invalid_num_proposals_raises(self, num_proposals):
        """Tests that num_proposals < 1 raises ValueError."""
        construct, generator, constraint, _ = _setup_optimizer_components()
        with pytest.raises(ValueError, match="num_proposals must be >= 1"):
            ConcreteOptimizer([construct], [generator], [constraint], num_proposals, 2)

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
        construct, generator, constraint, _segment = _setup_optimizer_components()
        # Use same generator instance twice
        with pytest.raises(ValueError, match=r"appears multiple times.*can only be used once"):
            ConcreteOptimizer([construct], [generator, generator], [constraint], 4, 2)

    def test_duplicate_constraint_instance_raises(self):
        """Tests that same constraint instance appearing twice raises ValueError."""
        construct, generator, constraint, _ = _setup_optimizer_components()
        # Use same constraint instance twice
        with pytest.raises(ValueError, match=r"appears multiple times.*can only be used once"):
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

        ConcreteOptimizer([construct], [generator], [constraint1, constraint2], 4, 2)

        # First constraint keeps original label
        assert constraint1.label == "structure_constraint"
        # Second constraint gets _1 suffix (collision between constraints)
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

        ConcreteOptimizer([construct], [gen1, gen2], [constraint_a, constraint_b], 4, 2)

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
            num_proposals=4,
            num_results=2,
        )
        assert optimizer.num_proposals == 4
        assert optimizer.num_results == 2


class TestSequencePoolInitialization:
    """Tests for sequence pool initialization behavior."""

    def test_fresh_initialization(self):
        """Tests initialization when no previous proposals exist."""
        original_seq = "ATCG"
        segment = Segment(sequence=original_seq, sequence_type="dna")
        segment.proposal_sequences = []
        segment.result_sequences = []
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
            num_proposals=5,
            num_results=2,
        )
        assert optimizer.constructs == [construct]

        # Result sequences initialized from original
        assert len(segment.result_sequences) == 2
        assert all(s.sequence == original_seq for s in segment.result_sequences)

        # Proposal sequences initialized from original
        assert len(segment.proposal_sequences) == 5
        assert all(s.sequence == original_seq for s in segment.proposal_sequences)

        # Verify independence (deep copy)
        segment.result_sequences[0].sequence = "GGGG"
        assert segment.proposal_sequences[0].sequence == original_seq

    def test_chained_initialization(self):
        """Tests initialization when inheriting from previous optimizer."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_proposals=5)

        # Simulate previous optimizer results
        prev_best = Sequence(sequence="AAAA", sequence_type="dna")
        prev_second = Sequence(sequence="TTTT", sequence_type="dna")
        segment.result_sequences = [prev_best, prev_second]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_proposals=5,
            num_results=3,  # Requesting 3, but only 2 available
        )
        assert optimizer.constructs == [construct]

        # Result sequences padded by cycling: [AAAA, TTTT, AAAA]
        assert len(segment.result_sequences) == 3
        assert segment.result_sequences[0].sequence == "AAAA"
        assert segment.result_sequences[1].sequence == "TTTT"
        assert segment.result_sequences[2].sequence == "AAAA"  # Cycles back

        # Proposals also initialized by cycling: [AAAA, TTTT, AAAA, TTTT, AAAA]
        assert len(segment.proposal_sequences) == 5
        expected = ["AAAA", "TTTT", "AAAA", "TTTT", "AAAA"]
        for i, seq in enumerate(segment.proposal_sequences):
            assert seq.sequence == expected[i], f"proposal[{i}] expected {expected[i]}, got {seq.sequence}"


class TestScoreEnergy:
    """Tests for energy scoring functionality."""

    def test_add_operation(self):
        """Tests energy scoring with 'add' operation."""
        construct, generator, constraint1, segment = _setup_optimizer_components(num_proposals=2)

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
            num_proposals=2,
            num_results=2,
        )
        segment.proposal_sequences = [Sequence("A"), Sequence("C")]
        optimizer.score_energy(operation="add")

        # Expected: [10+4, 20+8] = [14, 28]
        assert optimizer.energy_scores == [14.0, 28.0]

    def test_multiply_operation(self):
        """Tests energy scoring with 'multiply' operation."""
        construct, generator, constraint1, segment = _setup_optimizer_components(num_proposals=2)

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
            num_proposals=2,
            num_results=2,
        )
        segment.proposal_sequences = [Sequence("A"), Sequence("C")]
        optimizer.score_energy(operation="multiply")

        # Expected: [2*4, 3*5] = [8, 15]
        assert optimizer.energy_scores == [8.0, 15.0]

    def test_invalid_operation_raises_error(self):
        """Tests that invalid operation raises ValueError."""
        construct, generator, constraint, segment = _setup_optimizer_components()
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)
        segment.proposal_sequences = [Sequence("A"), Sequence("C")]

        with pytest.raises(ValueError, match="Operation must be 'add' or 'multiply'"):
            optimizer.score_energy(operation="invalid")


class TestFilterConstraints:
    """Tests for filter constraint behavior in scoring."""

    def test_filter_rejection_applies_penalty(self):
        """Tests that filter constraints reject proposals and apply penalty."""
        construct, generator, filter_constraint, segment = _setup_optimizer_components(num_proposals=3)
        segment.proposal_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # Filter rejects proposal 1
        filter_constraint.threshold = 0.5
        filter_constraint.evaluate.return_value = [True, False, True]

        # Scoring constraint (returns NaN for skipped proposal 1)
        scoring_constraint = MagicMock(spec=Constraint)
        scoring_constraint.inputs = [segment]
        scoring_constraint.label = "ScoringConstraint"
        scoring_constraint.threshold = None
        scoring_constraint.weight = 1.0
        scoring_constraint.evaluate.return_value = [10.0, float("nan"), 20.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[filter_constraint, scoring_constraint],
            num_proposals=3,
            num_results=2,
        )

        optimizer.score_energy(operation="add", filter_penalty=999.0)

        # Proposal 0: passed → 10.0
        # Proposal 1: rejected → 999.0 (penalty)
        # Proposal 2: passed → 20.0
        assert optimizer.energy_scores == [10.0, 999.0, 20.0]

    def test_filter_skips_subsequent_evaluation(self):
        """Tests that rejected proposals skip subsequent constraint evaluations."""
        construct, generator, filter_constraint, segment = _setup_optimizer_components(num_proposals=3)
        segment.proposal_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # Filter rejects proposal 1
        filter_constraint.threshold = 0.0
        filter_constraint.evaluate.return_value = [True, False, True]

        # Scoring constraint
        scoring_constraint = MagicMock(spec=Constraint)
        scoring_constraint.inputs = [segment]
        scoring_constraint.label = "ScoringConstraint"
        scoring_constraint.threshold = None
        scoring_constraint.evaluate.return_value = [1.0, float("nan"), 1.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[filter_constraint, scoring_constraint],
            num_proposals=3,
            num_results=2,
        )

        optimizer.score_energy()

        # Verify scoring constraint received mask reflecting filter rejection
        _, kwargs = scoring_constraint.evaluate.call_args
        assert kwargs["mask"] == [True, False, True]

    def test_filter_only_constraints_warns(self, caplog):
        """Tests that filter-only constraints (no scoring constraints) logs a warning."""
        construct, generator, filter_constraint, segment = _setup_optimizer_components(num_proposals=2)
        segment.proposal_sequences = [Sequence("A"), Sequence("C")]

        # Only a filter constraint, no scoring constraints
        filter_constraint.threshold = 0.5
        filter_constraint.evaluate.return_value = [True, True]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[filter_constraint],
            num_proposals=2,
            num_results=2,
        )

        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.optimizer"):
            optimizer.score_energy()

        assert any("All constraints are filters" in msg for msg in caplog.messages)
        # Passing proposals get energy 0.0 (sum of empty list)
        assert optimizer.energy_scores == [0.0, 0.0]

    def test_inconsistent_state_raises_error(self):
        """Tests that passed proposal with NaN score raises RuntimeError."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_proposals=3)
        segment.proposal_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        # No filter - all proposals should pass
        constraint.threshold = None
        constraint.weight = 1.0
        # Bug: returns NaN for proposal 1 even though it wasn't filtered
        constraint.evaluate.return_value = [1.0, float("nan"), 1.0]

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_proposals=3,
            num_results=2,
        )

        with pytest.raises(RuntimeError, match="Inconsistent state: proposal 1 passed all filters but has NaN score"):
            optimizer.score_energy(operation="add")

    def test_verbose_logs_rejection_reasons(self, caplog):
        """Tests that verbose logging accurately attributes rejections to the filter that evaluated them.

        Filter 1 (pLDDT): passes proposal 0, rejects proposal 1
        Filter 2 (Length): rejects proposal 2 (proposal 1 was already masked, so Length never evaluated it)

        Expected: proposal 1 → "pLDDT Filter" only, proposal 2 → "Length Filter" only.
        """
        construct, generator, _, segment = _setup_optimizer_components(num_proposals=3)
        segment.proposal_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

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
            num_proposals=3,
            num_results=2,
            verbose=True,
        )

        with caplog.at_level(logging.INFO, logger="proto_language.language.core.optimizer"):
            optimizer.score_energy()

        # Proposal 0 passed both filters
        assert any("Proposal 0" in m and "ACCEPTED" in m for m in caplog.messages)
        # Proposal 1 rejected by pLDDT Filter only (Length Filter never evaluated it)
        assert any(
            "Proposal 1" in m and "REJECTED by pLDDT Filter" in m and "Length Filter" not in m for m in caplog.messages
        )
        # Proposal 2 rejected by Length Filter only (pLDDT Filter passed it)
        assert any("Proposal 2" in m and "REJECTED by Length Filter" in m and "pLDDT" not in m for m in caplog.messages)


class TestToolCacheClearing:
    """Tests for tool cache clearing functionality."""

    def test_cache_clearing_modes(self):
        """Tests different cache clearing configurations."""
        construct, generator, constraint, _ = _setup_optimizer_components()

        with patch("proto_language.language.core.optimizer.ToolCache") as MockCache:
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
        construct, generator, constraint, _ = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        # energy_scores must have length num_results before snapshot
        optimizer.energy_scores = [0.1]
        optimizer._save_progress_snapshot(time_step=5, optimizer_metadata={"type": "test"})

        assert len(optimizer.history) == 1
        snapshot = optimizer.history[0]

        assert snapshot["time_step"] == 5
        assert snapshot["optimizer"] == {"type": "test"}
        assert "results" in snapshot
        assert len(snapshot["results"]) == 1
        assert snapshot["results"][0]["energy_score"] == 0.1
        assert isinstance(snapshot["results"][0]["constructs"], list)

    def test_snapshot_validates_energy_scores_matches_result(self):
        """Tests that snapshot raises error if energy_scores length != result_sequences length."""
        construct, generator, constraint, _segment = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        # energy_scores doesn't match result_sequences length (1)
        optimizer.energy_scores = [0.1, 0.9]

        with pytest.raises(RuntimeError, match="energy_scores has length 2, expected 1"):
            optimizer._save_progress_snapshot(time_step=5, optimizer_metadata={"type": "test"})

    def test_snapshot_allows_partial_result(self):
        """Tests that snapshot allows partial result_sequences (e.g. Rejection Sampling mid-run)."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_proposals=4)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 4, 3)

        # Simulate partial state: only 2 of 3 result
        segment.result_sequences = segment.result_sequences[:2]
        optimizer.energy_scores = [0.1, 0.5]

        # Should NOT raise; relaxed validation allows partial
        optimizer._save_progress_snapshot(time_step=1, optimizer_metadata={"type": "test"})
        assert len(optimizer.history) == 1
        assert len(optimizer.history[0]["results"]) == 2


class TestStateRestartBehavior:
    """Tests for optimizer state capture and restore on re-run."""

    def test_prepare_run_captures_state_on_first_call(self):
        """Tests that _prepare_run captures state on first call."""
        construct, generator, constraint, _segment = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        assert optimizer._initial_state is None

        optimizer._prepare_run()

        assert optimizer._initial_state is not None
        assert "segments" in optimizer._initial_state
        assert "energy_scores" in optimizer._initial_state

    def test_prepare_run_restores_state_on_subsequent_calls(self):
        """Tests that _prepare_run restores state on subsequent calls."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        # Capture initial state
        original_seq = segment.proposal_sequences[0].sequence
        optimizer._prepare_run()

        # Modify state
        segment.proposal_sequences[0].sequence = "GGGG"
        segment.result_sequences[0].sequence = "CCCC"
        optimizer.energy_scores = [999.0, 999.0]
        optimizer.history = [{"test": "data"}]
        optimizer._proposal_outcomes = ["accepted", "GC Filter"]
        optimizer._proposal_energy_scores = [0.5, float("inf")]

        # Restore
        optimizer._prepare_run()

        # Verify state restored
        assert segment.proposal_sequences[0].sequence == original_seq
        assert segment.result_sequences[0].sequence == original_seq
        assert optimizer.energy_scores == [float("inf"), float("inf")]
        assert optimizer.history == []
        assert optimizer._proposal_outcomes == []
        assert optimizer._proposal_energy_scores == []

    def test_state_independence_via_serialization(self):
        """Tests that captured state is independent of current state."""
        construct, generator, constraint, segment = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 2)

        optimizer._capture_initial_state()

        # Modify current state
        segment.proposal_sequences[0].sequence = "XXXX"

        # Captured state should be unchanged (now stored as serialized dicts)
        captured_seq = optimizer._initial_state["segments"][0]["proposals"][0]["sequence"]
        assert captured_seq == "ATCG"

    def test_labels_deduplicated_resets_on_restore(self):
        """Tests that _labels_deduplicated flag resets on _restore_initial_state."""
        construct, generator, _, segment = _setup_optimizer_components(num_proposals=2)

        constraint1 = MagicMock(spec=Constraint)
        constraint1.inputs = [segment]
        constraint1.label = "my_label"
        constraint1.threshold = None
        constraint1.weight = 1.0
        constraint1.evaluate.return_value = [1.0, 1.0]

        constraint2 = MagicMock(spec=Constraint)
        constraint2.inputs = [segment]
        constraint2.label = "my_label"
        constraint2.threshold = None
        constraint2.weight = 1.0
        constraint2.evaluate.return_value = [1.0, 1.0]

        optimizer = ConcreteOptimizer([construct], [generator], [constraint1, constraint2], 2, 2)

        # After init, labels deduplicated and flag is set
        assert optimizer._labels_deduplicated is True
        assert constraint1.label == "my_label"
        assert constraint2.label == "my_label_1"

        # Capture state, then restore
        optimizer._capture_initial_state()
        optimizer._restore_initial_state()

        # Flag should be reset
        assert optimizer._labels_deduplicated is False


class TestProposalTracking:
    """Tests for _proposal_outcomes and proposal_results in snapshots."""

    def test_filter_outcomes_with_and_logic(self):
        """score_energy() sets outcomes to 'accepted' or the first failing filter's label."""
        construct, generator, _, segment = _setup_optimizer_components(num_proposals=3)
        segment.proposal_sequences = [Sequence("A"), Sequence("G"), Sequence("C")]

        filter1 = MagicMock(spec=Constraint)
        filter1.inputs = [segment]
        filter1.label = "Filter1"
        filter1.threshold = 0.5
        filter1.weight = 1.0
        filter1.evaluate.return_value = [True, True, False]  # Proposal 2 fails

        filter2 = MagicMock(spec=Constraint)
        filter2.inputs = [segment]
        filter2.label = "Filter2"
        filter2.threshold = 0.5
        filter2.weight = 1.0
        filter2.evaluate.return_value = [True, False, True]  # Proposal 1 fails

        optimizer = ConcreteOptimizer([construct], [generator], [filter1, filter2], 3, 2)
        optimizer.score_energy()

        assert optimizer._proposal_outcomes == ["accepted", "Filter2", "Filter1"]

    def test_snapshot_includes_proposal_results(self):
        """_save_progress_snapshot includes proposal_results when track_proposals=True."""
        construct, generator, constraint, _segment = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)
        optimizer.track_proposals = True

        optimizer.energy_scores = [0.5]
        optimizer._proposal_outcomes = ["accepted", "GC Filter"]
        optimizer._proposal_energy_scores = [0.5, float("inf")]

        optimizer._save_progress_snapshot(time_step=1, optimizer_metadata={"type": "test"})

        snapshot = optimizer.history[0]
        assert "proposal_results" in snapshot
        assert len(snapshot["proposal_results"]) == 2
        assert snapshot["proposal_results"][0]["accepted"] is True
        assert snapshot["proposal_results"][0]["rejected_by"] is None
        assert snapshot["proposal_results"][0]["energy_score"] == 0.5
        assert snapshot["proposal_results"][1]["accepted"] is False
        assert snapshot["proposal_results"][1]["rejected_by"] == "GC Filter"
        assert snapshot["proposal_results"][1]["energy_score"] is None  # inf → None

    def test_snapshot_omits_proposal_results_by_default(self):
        """_save_progress_snapshot omits proposal_results when track_proposals=False (default)."""
        construct, generator, constraint, _ = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)

        optimizer.energy_scores = [0.5]
        optimizer._proposal_outcomes = ["accepted", "GC Filter"]
        optimizer._proposal_energy_scores = [0.5, float("inf")]
        optimizer._save_progress_snapshot(time_step=1, optimizer_metadata={"type": "test"})

        assert "proposal_results" not in optimizer.history[0]

    def test_snapshot_omits_proposal_results_when_outcomes_empty(self):
        """_save_progress_snapshot omits proposal_results before any scoring."""
        construct, generator, constraint, _ = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)
        optimizer.track_proposals = True

        optimizer.energy_scores = [0.5]
        optimizer._save_progress_snapshot(time_step=0, optimizer_metadata={"type": "test"})

        assert "proposal_results" not in optimizer.history[0]

    def test_tracking_interval_gates_snapshots(self):
        """tracking_interval>1 reduces the number of history snapshots."""
        construct, generator, constraint, _ = _setup_optimizer_components(num_proposals=2)
        optimizer = ConcreteOptimizer([construct], [generator], [constraint], 2, 1)
        optimizer.tracking_interval = 3

        # Simulate saving snapshots for steps 0..10
        for step in range(11):
            optimizer.energy_scores = [0.5]
            if step % optimizer.tracking_interval == 0 or step == 10:
                optimizer._save_progress_snapshot(time_step=step, optimizer_metadata={"type": "test"})

        # Steps saved: 0, 3, 6, 9, 10 = 5 snapshots
        assert len(optimizer.history) == 5
        saved_steps = {entry["time_step"] for entry in optimizer.history}
        assert saved_steps == {0, 3, 6, 9, 10}


class TestDeferredNumResults:
    """Tests for deferred num_results resolution."""

    def test_deferred_num_results_skips_pool_init(self):
        """Optimizer with num_results=None constructs without error."""
        construct, generator, constraint, segment = _setup_optimizer_components()

        # Segment starts with 1 sequence in each pool from its constructor
        pre_result_len = len(segment.result_sequences)
        pre_proposal_len = len(segment.proposal_sequences)

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_proposals=None,
            num_results=None,
        )

        assert optimizer.num_results is None
        assert optimizer.num_proposals is None
        assert optimizer.energy_scores == []
        # Pools should NOT have been resized by _initialize_sequence_pools
        assert len(segment.result_sequences) == pre_result_len
        assert len(segment.proposal_sequences) == pre_proposal_len

    def test_deferred_run_raises_runtime_error(self):
        """Calling run() on deferred optimizer raises RuntimeError."""
        construct, generator, constraint, _ = _setup_optimizer_components()

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_proposals=None,
            num_results=None,
        )

        with pytest.raises(RuntimeError, match="num_results must be set"):
            optimizer.run()

    def test_resolve_num_results_initializes_pools(self):
        """_resolve_num_results sets num_results and initializes pools."""
        construct, generator, constraint, segment = _setup_optimizer_components()

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_proposals=None,
            num_results=None,
        )

        # Manually set num_proposals before resolving (subclasses do this in override)
        optimizer.num_proposals = 4
        optimizer._resolve_num_results(2)

        assert optimizer.num_results == 2
        assert optimizer.num_proposals == 4
        assert len(optimizer.energy_scores) == 4
        assert len(segment.result_sequences) == 2
        assert len(segment.proposal_sequences) == 4

    def test_resolve_num_results_validates_value(self):
        """_resolve_num_results raises ValueError for invalid values."""
        construct, generator, constraint, _ = _setup_optimizer_components()

        optimizer = ConcreteOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            num_proposals=None,
            num_results=None,
        )
        optimizer.num_proposals = 4

        with pytest.raises(ValueError, match="num_results must be >= 1"):
            optimizer._resolve_num_results(0)


class TestOptimizerExport:
    """Tests for Optimizer.export, to_dataframe, and to_fasta."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from proto_language.language.constraint import ConstraintRegistry

        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        construct = Construct([segment])
        generator = MockGenerator()
        generator.assign(segment)
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 0, "max_gc": 100},
        )
        self.optimizer = ConcreteOptimizer(
            [construct],
            [generator],
            [constraint],
            num_proposals=2,
            num_results=2,
        )
        segment.proposal_sequences = [Sequence("ATCGATCG"), Sequence("GCTAGCTA")]
        self.optimizer.score_energy()
        segment.result_sequences = list(segment.proposal_sequences)
        self.optimizer._save_progress_snapshot(time_step=0, optimizer_metadata={"type": "test"})

    def test_export_all_tables(self, tmp_path):
        """export() without table creates directory with all table files."""
        out = self.optimizer.export(path=tmp_path / "results", format="csv")
        assert out.is_dir()
        for name in ("sequences", "constraints", "constructs", "optimization"):
            assert (out / f"{name}.csv").stat().st_size > 0

    def test_export_single_table(self, tmp_path):
        """export() with table writes one file."""
        path = tmp_path / "seqs.csv"
        self.optimizer.export(path=path, table="sequences")
        content = path.read_text()
        assert "result_idx" in content and "sequence" in content

    @pytest.mark.parametrize(
        "table,expected_col",
        [
            ("sequences", "sequence"),
            ("constraints", "constraint"),
            ("constructs", "full_sequence"),
            ("optimization", "timepoint"),
        ],
    )
    def test_to_dataframe(self, table, expected_col):
        """to_dataframe dispatches correctly to each table."""
        df = self.optimizer.to_dataframe(table=table)
        assert len(df) > 0
        assert expected_col in df.columns

    def test_to_fasta(self):
        """to_fasta returns valid FASTA string."""
        fasta = self.optimizer.to_fasta()
        assert fasta.startswith(">")
        assert "\n" in fasta


class TestOptimizerRegistry:
    """Tests for OptimizerRegistry metadata fields."""

    def test_beam_search_compatible_generators_match_autoregressive(self):
        spec = OptimizerRegistry.get("beam-search")
        autoregressive_keys = sorted(s.key for s in GeneratorRegistry.list_all() if s.category == "autoregressive")
        assert sorted(spec.compatible_generators) == autoregressive_keys

    def test_general_optimizers_accept_all_generators(self):
        for key in ("mcmc", "rejection-sampling", "cycling"):
            assert OptimizerRegistry.get(key).compatible_generators is None

    def test_gradient_required_constraint_mode(self):
        assert OptimizerRegistry.get("gradient").required_constraint_mode == "gradient"

    def test_default_required_constraint_mode_is_none(self):
        assert OptimizerRegistry.get("mcmc").required_constraint_mode is None

    def test_requires_generators_on_mpnn_perplexity(self):
        assert ConstraintRegistry.get("mpnn-perplexity").requires_generators == ["proteinmpnn"]

    def test_requires_generators_default_none(self):
        assert ConstraintRegistry.get("gc-content").requires_generators is None


def _compat_scorer(input_sequences: list[tuple[Sequence, ...]], config) -> list[ConstraintOutput]:
    return [ConstraintOutput(score=0.5) for _ in input_sequences]


def _compat_backward(inputs: tuple[Sequence, ...], *, config, **kwargs) -> GradientConstraintOutput:
    import numpy as np

    return GradientConstraintOutput(gradient=(np.zeros((6, 20)),), loss=0.0)


class TestComponentCompatibility:
    """Tests for centralized component dependency validation."""

    def test_missing_required_generator_raises(self):
        seg = Segment(sequence="ACDEFG", sequence_type="protein")
        gen = RandomProteinGenerator(RandomProteinGeneratorConfig())
        gen.assign(seg)
        con = ConstraintRegistry.create(
            key="mpnn-perplexity",
            segments=[seg],
            config_dict={"top_k": 5},
            label="mpnn_prescreen",
            threshold=0.0,
        )
        with pytest.raises(ValueError, match="proteinmpnn"):
            RejectionSamplingOptimizer(
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=RejectionSamplingOptimizerConfig(num_results=1, num_samples=5),
            )

    def test_present_required_generator_passes(self):
        seg = Segment(sequence="ACDEFG", sequence_type="protein")
        gen = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig())
        gen.assign(seg)
        con = ConstraintRegistry.create(
            key="mpnn-perplexity",
            segments=[seg],
            config_dict={"top_k": 5},
            label="mpnn_prescreen",
            threshold=0.0,
        )
        RejectionSamplingOptimizer(
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[con],
            config=RejectionSamplingOptimizerConfig(num_results=1, num_samples=5),
        )

    def test_incompatible_generator_rejected(self):
        seg = Segment(sequence="ACDEFG", sequence_type="protein")
        gen = RandomProteinGenerator(RandomProteinGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], backward=_compat_backward, backward_config={}, label="g")
        with pytest.raises(ValueError, match="not compatible with"):
            GradientOptimizer(
                target_segment=seg,
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )

    def test_discrete_constraint_rejected_by_gradient_optimizer(self):
        seg = Segment(sequence="ACDEFG", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], function=_compat_scorer, function_config={}, label="d")
        with pytest.raises(ValueError, match="gradient"):
            GradientOptimizer(
                target_segment=seg,
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )
