"""Tests for Constraint evaluation, validation, and gradient support."""

import copy
import math
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from constraint_tests.utils import (
    mock_dna_only_scoring_function,
    mock_multi_input_scoring_function,
    mock_multi_input_scoring_function_disjoint,
    mock_single_input_scoring_function,
)
from proto_language.language.core import Constraint, Segment, Sequence
from proto_language.language.core.constraint import GradientResult
from tests.helpers.mock_structure import MockStructure


class MockConfig(BaseModel):
    """Empty config for mock constraint functions."""


class MockBackwardConfig(BaseModel):
    """Separate config for mock backward function."""

    loss_type: str = "plddt"


def _make_segment_with_proposals(sequences: list[str], seq_type: str = "dna") -> Segment:
    """Helper to create a segment with multiple proposal sequences for testing."""
    segment = Segment(sequence=sequences[0], sequence_type=seq_type)
    segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(sequences))]
    for i, seq_str in enumerate(sequences):
        segment.proposal_sequences[i].sequence = seq_str
    return segment


def _mock_backward(inputs: tuple, *, config: BaseModel, temperature: float = 1.0, **kwargs: Any) -> GradientResult:
    """Mock backward that reads logits from the first input Sequence."""
    logits = inputs[0].logits
    return GradientResult(
        gradient=(-logits * temperature,), loss=float(np.mean(logits**2)), metrics={"temperature": temperature}
    )


def _make_gradient_constraint(segment: Segment | None = None, **kwargs: object) -> Constraint:
    if segment is None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
    defaults: dict[str, object] = {
        "inputs": [segment],
        "function": mock_single_input_scoring_function,
        "function_config": MockConfig(),
        "backward": _mock_backward,
        "backward_config": MockConfig(),
    }
    defaults.update(kwargs)
    return Constraint(**defaults)


# =============================================================================
# TESTS FOR CONSTRAINT EVALUATION MODES
# =============================================================================


class TestConstraintEvaluation:
    """Tests for constraint evaluation with different input configurations."""

    @pytest.mark.parametrize(
        "sequences,expected_scores",
        [
            (["ACTGACTG"], [0.25]),  # Single sequence: 2 T's out of 8
            (["ACTGACTG", "TCTGTCTG", "TTTGTTTG", "TTTTTTTT"], [0.25, 0.5, 0.75, 1.0]),  # Batch
        ],
    )
    def test_constraint_evaluation_contiguous(self, sequences, expected_scores):
        """Single and batched evaluation return per-proposal scores in order."""
        segment = _make_segment_with_proposals(sequences, "dna")
        constraint = Constraint(
            inputs=[segment], function=mock_single_input_scoring_function, function_config=MockConfig()
        )
        assert constraint.evaluate() == expected_scores

    def test_constraint_metadata_propagation(self):
        """Tests that metadata is correctly propagated back to sequences."""
        sequences = ["ACTGACTG", "TTTTTTTT"]
        segment = _make_segment_with_proposals(sequences, "dna")
        config = MockConfig()

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=config,
        )
        constraint.evaluate()

        # Check metadata was propagated in nested format
        for _i, seq in enumerate(segment.proposal_sequences):
            constraints = seq._constraints_metadata
            assert "mock_multi_input_scoring_function" in constraints
            assert "t_count" in constraints["mock_multi_input_scoring_function"]["data"]
            assert "total_length" in constraints["mock_multi_input_scoring_function"]["data"]
            assert "t_fraction" in constraints["mock_multi_input_scoring_function"]["data"]

    def test_disjoint_mode(self):
        """Tests constraint evaluation in disjoint mode (separate sequences)."""
        sequences_a = ["AAAA", "AAAT", "AATT", "ATTT", "TTTT"]
        sequences_b = ["AAAA", "AAAC", "AACC", "ACCC", "CCCC"]

        seg_a = _make_segment_with_proposals(sequences_a, "dna")
        seg_b = _make_segment_with_proposals(sequences_b, "dna")
        config = MockConfig()

        constraint = Constraint(
            inputs=[seg_a, seg_b],
            function=mock_multi_input_scoring_function_disjoint,
            function_config=config,
        )
        scores = constraint.evaluate()

        # Score: (T% in first + C% in second) / 2
        expected_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert scores == expected_scores

        # Each segment should have its own metadata in nested format
        for i in range(len(sequences_a)):
            constraints_a = seg_a.proposal_sequences[i]._constraints_metadata
            constraints_b = seg_b.proposal_sequences[i]._constraints_metadata
            assert "mock_multi_input_scoring_function_disjoint" in constraints_a
            assert "mock_multi_input_scoring_function_disjoint" in constraints_b
            assert "t_percent" in constraints_a["mock_multi_input_scoring_function_disjoint"]["data"]
            assert "c_percent" in constraints_b["mock_multi_input_scoring_function_disjoint"]["data"]

    def test_structure_and_logits_propagate_to_original(self):
        """Regression for #1180: fields set on the scored dummy must reach the original proposal."""
        attached_structure = MockStructure.with_plddt([0.1, 0.95, 0.95])
        attached_logits = np.ones((3, 20))

        def attaches(input_sequences: list[tuple[Sequence, ...]], config: BaseModel) -> list[float]:
            for (seq,) in input_sequences:
                seq.structure = attached_structure
                seq.logits = attached_logits
            return [0.0] * len(input_sequences)

        attaches._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]

        segment = _make_segment_with_proposals(["ACD"], seq_type="protein")
        Constraint(inputs=[segment], function=attaches, function_config=MockConfig()).evaluate()

        assert segment.proposal_sequences[0].structure is attached_structure
        assert np.array_equal(segment.proposal_sequences[0].logits, attached_logits)

    def test_structure_and_logits_pass_through_when_constraint_leaves_them_alone(self):
        """Constraint both sees the existing structure/logits (read path) and leaves the original references intact (no silent rebind)."""
        existing_structure = MockStructure.with_plddt([0.5, 0.5, 0.5])
        existing_logits = np.ones((3, 20))
        seen: dict[str, object] = {}

        def inspects(input_sequences: list[tuple[Sequence, ...]], config: BaseModel) -> list[float]:
            for (seq,) in input_sequences:
                seen["structure"] = seq.structure
                seen["logits"] = seq.logits
            return [0.0] * len(input_sequences)

        inspects._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]

        segment = _make_segment_with_proposals(["ACD"], seq_type="protein")
        segment.proposal_sequences[0].structure = existing_structure
        segment.proposal_sequences[0].logits = existing_logits

        Constraint(inputs=[segment], function=inspects, function_config=MockConfig()).evaluate()

        # Read path: constraint saw the exact objects on its dummy (not defensive copies).
        assert seen["structure"] is existing_structure
        assert seen["logits"] is existing_logits
        # Write path: no silent rebind when the constraint doesn't touch the fields.
        assert segment.proposal_sequences[0].structure is existing_structure
        assert segment.proposal_sequences[0].logits is existing_logits


# =============================================================================
# TESTS FOR INPUT VALIDATION
# =============================================================================


class TestConstraintValidation:
    """Tests for constraint input validation."""

    def test_empty_inputs_raises_error(self):
        """Test that empty inputs list raises ValueError."""
        with pytest.raises(ValueError, match="At least one segment must be provided"):
            Constraint(
                inputs=[],
                function=mock_single_input_scoring_function,
                function_config=MockConfig(),
            )

    def test_mixed_batch_sizes_raises_error(self):
        """Test that inconsistent proposal pool sizes raise ValueError."""
        seg1 = _make_segment_with_proposals(["ATCG", "GGGG"])  # 2 proposals
        seg2 = _make_segment_with_proposals(["TTTT"])  # 1 proposal

        # Use multi-input constraint to test batch size validation
        with pytest.raises(ValueError, match="All segments must have the same number of proposal sequences"):
            Constraint(
                inputs=[seg1, seg2],
                function=mock_multi_input_scoring_function_disjoint,
                function_config=MockConfig(),
            )

    def test_multi_input_constraint_allows_mixed_sequence_types(self):
        """Test that multi-input constraints can have different sequence types (e.g., protein + ligand)."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna")
        seg2 = Segment(sequence="MVLS", sequence_type="protein")

        # Multi-input constraints can legitimately have different sequence types
        # (e.g., protein-ligand binding constraints)
        constraint = Constraint(
            inputs=[seg1, seg2],
            function=mock_multi_input_scoring_function_disjoint,
            function_config=MockConfig(),
        )
        assert len(constraint.inputs) == 2

    def test_unsupported_sequence_type_raises_error(self):
        """Validation rejects a segment whose type isn't in the constraint's supported types."""
        protein_seg = Segment(sequence="MVLSPADKTN", sequence_type="protein")

        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[protein_seg],
                function=mock_dna_only_scoring_function,
                function_config=MockConfig(),
            )


# =============================================================================
# TESTS FOR CUSTOM LABEL HANDLING
# =============================================================================


class TestConstraintLabel:
    """Tests for custom label functionality."""

    def test_custom_label_in_metadata(self):
        """Test that custom label overrides function name in metadata."""
        segment = Segment(sequence="ATCGACTG", sequence_type="dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_single_input_scoring_function,
            function_config=MockConfig(),
            label="my_custom_label",
        )
        constraint.evaluate()

        # Metadata should use custom label in nested format
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "my_custom_label" in constraints
        assert "t_count" in constraints["my_custom_label"]["data"]
        # Should NOT use function name
        assert "mock_single_input_scoring_function" not in constraints


# =============================================================================
# TESTS FOR MASK-BASED EVALUATION
# =============================================================================


class TestConstraintMask:
    """Tests for mask-based selective evaluation."""

    def test_mask_skips_unevaluated_proposals(self):
        """Test that mask correctly skips evaluation of masked proposals."""
        sequences = ["ATTTTTTT", "AAAAAAAA", "TTTTTTTT", "AAAATTTT", "ATATATAT"]
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConfig(),
        )

        # Only evaluate proposals 0, 2, 4
        mask = [True, False, True, False, True]
        scores = constraint.evaluate(mask=mask)

        assert len(scores) == 5
        assert scores[0] == pytest.approx(0.875)  # 7/8
        assert math.isnan(scores[1])  # Skipped
        assert scores[2] == pytest.approx(1.0)  # 8/8
        assert math.isnan(scores[3])  # Skipped
        assert scores[4] == pytest.approx(0.5)  # 4/8

        # Verify metadata only propagated to evaluated proposals (nested under "constraints")
        constraint_label = "mock_multi_input_scoring_function"
        assert constraint_label in segment.proposal_sequences[0]._constraints_metadata
        # Skipped proposal should have no constraint metadata (constraints dict always exists but is empty)
        assert constraint_label not in segment.proposal_sequences[1]._constraints_metadata

    def test_mask_all_false_returns_nan(self):
        """Test that all-false mask returns NaN for all proposals."""
        sequences = ["ATCG", "GGGG", "TTTT"]
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConfig(),
        )

        scores = constraint.evaluate(mask=[False, False, False])
        assert len(scores) == 3
        assert all(math.isnan(s) for s in scores)

    def test_mask_invalid_length_raises_error(self):
        """Test that mask with incorrect length raises ValueError (both shorter and longer)."""
        sequences = ["ATCG", "GGGG", "TTTT"]
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConfig(),
        )

        # Longer than pool
        with pytest.raises(ValueError, match=r"Mask length .* does not match"):
            constraint.evaluate(mask=[True, False, True, True])

        # Shorter than pool
        with pytest.raises(ValueError, match=r"Mask length .* does not match"):
            constraint.evaluate(mask=[True, True])


# =============================================================================
# TESTS FOR THRESHOLD-BASED FILTERING
# =============================================================================


class TestConstraintThreshold:
    """Tests for threshold-based filtering functionality."""

    def test_threshold_converts_scores_to_boolean(self):
        """Test that threshold converts float scores to boolean filters."""

        def mock_scoring(input_sequences, config=None):
            return [len(seq_tuple[0].sequence) / 10.0 for seq_tuple in input_sequences]

        mock_scoring._constraint_config_class = MockConfig
        mock_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["ATCG", "ATCGATCG", "AT"]  # lengths 4, 8, 2 → scores 0.4, 0.8, 0.2
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_scoring,
            function_config=MockConfig(),
            threshold=0.5,
        )
        results = constraint.evaluate()

        # Scores <= threshold pass: 0.4 <= 0.5 (True), 0.8 <= 0.5 (False), 0.2 <= 0.5 (True)
        assert results == [True, False, True]
        assert all(isinstance(r, bool) for r in results)

    def test_no_threshold_returns_float_scores(self):
        """Test that constraints without threshold return float scores."""

        def mock_scoring(input_sequences, config=None):
            return [0.4, 0.8]

        mock_scoring._constraint_config_class = MockConfig
        mock_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["ATCG", "GGGG"]
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_scoring,
            function_config=MockConfig(),
        )
        results = constraint.evaluate()

        assert results == [0.4, 0.8]
        assert all(isinstance(r, float) for r in results)


# =============================================================================
# TESTS FOR WEIGHT PARAMETER
# =============================================================================


class TestConstraintWeight:
    """Tests for weight parameter functionality."""

    def test_weight_defaults_to_one(self):
        """Test that weight defaults to 1.0."""
        segment = Segment(sequence="ATCG", sequence_type="dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_single_input_scoring_function,
            function_config=MockConfig(),
        )
        assert constraint.weight == 1.0

    def test_weight_multiplies_scores(self):
        """Test that weight correctly multiplies raw scores."""

        def mock_scoring(input_sequences, config=None):
            return [0.2, 0.5]

        mock_scoring._constraint_config_class = MockConfig
        mock_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["AT", "GC"]
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_scoring,
            function_config=MockConfig(),
            weight=2.0,
        )
        results = constraint.evaluate()

        assert results == pytest.approx([0.4, 1.0])

    def test_weight_and_threshold_mutually_exclusive(self):
        """Test that setting both weight and threshold raises ValueError."""
        segment = Segment(sequence="ATCG", sequence_type="dna")

        with pytest.raises(ValueError, match=r"Both threshold .* and weight .* are set"):
            Constraint(
                inputs=[segment],
                function=mock_single_input_scoring_function,
                function_config=MockConfig(),
                threshold=0.5,
                weight=2.0,
            )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestConstraintEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_reserved_key_collision_raises_error(self):
        """Test that writing a reserved key to seq._metadata raises ValueError."""

        def collision_scoring_function(input_sequences, config):
            scores = []
            for (seq,) in input_sequences:
                # Write a reserved key (this should be caught)
                seq._metadata["score"] = 0.5
                scores.append(0.1)
            return scores

        collision_scoring_function._constraint_supported_sequence_types = {"dna"}

        segment = _make_segment_with_proposals(["ATCG"], "dna")
        constraint = Constraint(
            inputs=[segment],
            function=collision_scoring_function,
            function_config=MockConfig(),
        )

        with pytest.raises(ValueError, match="reserved key"):
            constraint.evaluate()

    def test_out_of_range_scores_warn(self, caplog):
        """Test that constraint scores outside [0, 1] log a warning."""

        def negative_scoring(input_sequences, config=None):
            return [-0.5, 1.5, 0.5]

        negative_scoring._constraint_config_class = MockConfig
        negative_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["ATCG", "GGGG", "TTTT"]
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=negative_scoring,
            function_config=MockConfig(),
        )

        import logging

        with caplog.at_level(logging.WARNING):
            scores = constraint.evaluate()

        # Out-of-range scores pass through (not clamped), warnings logged
        assert scores == [-0.5, 1.5, 0.5]
        warning_msgs = [
            r.message for r in caplog.records if r.levelno == logging.WARNING and "out-of-range" in r.message
        ]
        assert len(warning_msgs) == 2
        assert "out-of-range score -0.5" in warning_msgs[0]
        assert "out-of-range score 1.5" in warning_msgs[1]

    def test_non_reserved_key_allowed(self):
        """Test that writing non-reserved keys to seq._metadata works fine."""

        def safe_scoring_function(input_sequences, config):
            scores = []
            for (seq,) in input_sequences:
                seq._metadata["gc_content"] = 50.0
                seq._metadata["my_custom_metric"] = 42
                scores.append(0.1)
            return scores

        safe_scoring_function._constraint_supported_sequence_types = {"dna"}

        segment = _make_segment_with_proposals(["ATCG"], "dna")
        constraint = Constraint(
            inputs=[segment],
            function=safe_scoring_function,
            function_config=MockConfig(),
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        # Custom data should be in constraints_metadata under "data"
        cdata = segment.proposal_sequences[0]._constraints_metadata["safe_scoring_function"]
        assert cdata["data"]["gc_content"] == 50.0
        assert cdata["data"]["my_custom_metric"] == 42


# =============================================================================
# TESTS FOR GRADIENT RESULT
# =============================================================================


class TestGradientResult:
    @pytest.mark.parametrize(
        "shapes",
        [
            pytest.param([(2, 2)], id="single-segment"),
            pytest.param([(4, 20), (3, 20)], id="multi-segment"),
        ],
    )
    def test_construction_preserves_gradient_shapes_and_default_metrics(self, shapes: list[tuple[int, int]]) -> None:
        result = GradientResult(gradient=tuple(np.zeros(s) for s in shapes), loss=0.5)
        assert result.loss == 0.5
        assert result.metrics == {}
        assert [g.shape for g in result.gradient] == shapes

    def test_custom_metrics_stored_and_repr_shows_shape_not_array(self) -> None:
        result = GradientResult(gradient=(np.zeros((5, 20)),), loss=0.5, metrics={"plddt": 0.85})
        assert result.metrics["plddt"] == pytest.approx(0.85)
        # repr must elide the array (huge) and surface the shape + loss for debugging.
        r = repr(result)
        assert "(5, 20)" in r
        assert "loss=0.5" in r

    def test_frozen(self) -> None:
        result = GradientResult(gradient=(np.zeros((5, 20)),), loss=1.0)
        with pytest.raises(AttributeError):
            result.loss = 2.0  # type: ignore[misc]


# =============================================================================
# TESTS FOR GRADIENT SUPPORT
# =============================================================================


class TestConstraintGradientSupport:
    def test_backward_config(self) -> None:
        bwd_config = MockBackwardConfig(loss_type="ptm")
        c = _make_gradient_constraint(backward_config=bwd_config)
        assert c.backward_config is bwd_config
        assert c.backward_config.loss_type == "ptm"  # type: ignore[union-attr]

    def test_backward_property(self) -> None:
        c = _make_gradient_constraint()
        assert c.backward is _mock_backward

    def test_weight_not_applied_to_gradient(self) -> None:
        """Weight is NOT applied in compute_gradient — optimizer handles weighting during merging."""
        segment = _make_segment_with_proposals(["ACTGACTG"])
        logits = np.ones((8, 4))
        segment.proposal_sequences[0].logits = logits
        c = _make_gradient_constraint(segment=segment, weight=2.0)
        results = c.compute_gradient(temperature=1.0)
        raw = _mock_backward((segment.proposal_sequences[0],), config=MockConfig(), temperature=1.0)
        np.testing.assert_array_almost_equal(results[0].gradient, raw.gradient)
        assert results[0].loss == pytest.approx(raw.loss)

    def test_gradient_support_discovery(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        regular = Constraint(
            inputs=[segment], function=mock_single_input_scoring_function, function_config=MockConfig()
        )
        grad = _make_gradient_constraint(segment=segment)

        assert grad.supports_gradient
        assert grad.supports_discrete
        assert not regular.supports_gradient
        assert regular.supports_discrete

    def test_compute_gradient(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        logits = np.random.randn(8, 4)
        segment.proposal_sequences[0].logits = logits
        c = _make_gradient_constraint(segment=segment)
        results = c.compute_gradient(temperature=1.0)
        assert len(results) == 1
        assert isinstance(results[0], GradientResult)
        assert results[0].gradient[0].shape == (8, 4)
        np.testing.assert_array_almost_equal(results[0].gradient[0], -logits)

    def test_compute_gradient_propagates_metadata(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        segment.proposal_sequences[0].logits = np.ones((8, 4))
        c = _make_gradient_constraint(segment=segment, label="test_grad")
        c.compute_gradient(temperature=1.0)

        metadata = segment.proposal_sequences[0]._constraints_metadata["test_grad"]
        assert metadata["score"] == pytest.approx(1.0)  # mean(ones**2) = 1.0
        assert metadata["weight"] == 1.0
        assert metadata["data"]["temperature"] == 1.0

    def test_temperature_affects_gradient(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        logits = np.ones((8, 4))
        segment.proposal_sequences[0].logits = logits
        c = _make_gradient_constraint(segment=segment)
        hot = c.compute_gradient(temperature=2.0)
        cold = c.compute_gradient(temperature=0.5)
        assert np.abs(hot[0].gradient).mean() > np.abs(cold[0].gradient).mean()

    def test_backward_config_forwarded(self) -> None:
        received: list[BaseModel] = []

        def capturing_backward(inputs: tuple, *, config: BaseModel, **kwargs: Any) -> GradientResult:
            received.append(config)
            return GradientResult(gradient=(np.zeros_like(inputs[0].logits),), loss=0.0)

        segment = _make_segment_with_proposals(["ACTGACTG"])
        segment.proposal_sequences[0].logits = np.zeros((8, 4))
        bwd_config = MockBackwardConfig(loss_type="ptm")
        c = _make_gradient_constraint(segment=segment, backward=capturing_backward, backward_config=bwd_config)
        c.compute_gradient(temperature=1.0)
        assert received == [bwd_config]

    def test_compute_gradient_raises_without_logits(self) -> None:
        """compute_gradient raises when no segment has logits set."""
        segment = _make_segment_with_proposals(["ACTGACTG"])
        c = _make_gradient_constraint(segment=segment)
        with pytest.raises(RuntimeError, match="no input segment has logits"):
            c.compute_gradient(temperature=1.0)

    @pytest.mark.parametrize(
        ("backward", "expected_exception", "match"),
        [
            (
                lambda inputs, *, config, **kwargs: {"gradient": np.zeros_like(inputs[0].logits), "loss": 0.0},
                TypeError,
                r"must return GradientResult",
            ),
            (
                lambda inputs, *, config, **kwargs: GradientResult(gradient=(np.zeros((4, 8)),), loss=0.0),
                ValueError,
                r"gradient 0 shape",
            ),
            (
                lambda inputs, *, config, **kwargs: GradientResult(
                    gradient=(np.zeros((8, 4)), np.zeros((8, 4))), loss=0.0
                ),
                ValueError,
                r"got 2 gradient\(s\), expected 1",
            ),
        ],
    )
    def test_compute_gradient_validates_backward_result(
        self,
        backward: object,
        expected_exception: type[Exception],
        match: str,
    ) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        segment.proposal_sequences[0].logits = np.zeros((8, 4))
        c = _make_gradient_constraint(segment=segment, backward=backward)
        with pytest.raises(expected_exception, match=match):
            c.compute_gradient(temperature=1.0)


# =============================================================================
# TESTS FOR BACKWARD-ONLY AND CALLABLE REQUIREMENTS
# =============================================================================


class TestConstraintCallableRequirements:
    def test_backward_only(self) -> None:
        """Backward-only constraint: can compute gradients, evaluate raises."""
        segment = _make_segment_with_proposals(["ACTGACTG"])
        segment.proposal_sequences[0].logits = np.ones((8, 4))
        c = Constraint(inputs=[segment], backward=_mock_backward, backward_config=MockConfig())
        assert c.supports_gradient
        assert not c.supports_discrete
        assert c.label == "_mock_backward"

        results = c.compute_gradient(temperature=1.0)
        assert len(results) == 1
        assert isinstance(results[0], GradientResult)
        assert results[0].gradient[0].shape == (8, 4)

        with pytest.raises(RuntimeError, match="does not support discrete evaluation"):
            c.evaluate()

    def test_function_only_gradient_raises(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        segment.proposal_sequences[0].logits = np.zeros((8, 4))
        c = Constraint(inputs=[segment], function=mock_single_input_scoring_function, function_config=MockConfig())
        with pytest.raises(RuntimeError, match="does not support gradient computation"):
            c.compute_gradient(temperature=1.0)

    def test_neither_callable_raises(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        with pytest.raises(ValueError, match="At least one of"):
            Constraint(inputs=[segment])
