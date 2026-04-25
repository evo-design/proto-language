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
from proto_language.language.constraint.constraint_registry import InputSlot
from proto_language.language.core import Constraint, ConstraintOutput, Segment, Sequence
from proto_language.language.core.constraint import GradientConstraintOutput
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


def _mock_backward(
    input_sequences: list[tuple[Sequence, ...]], *, config: BaseModel, temperature: float = 1.0, **kwargs: Any
) -> list[GradientConstraintOutput]:
    """Mock backward that reads logits from each input Sequence."""
    results: list[GradientConstraintOutput] = []
    for (seq,) in input_sequences:
        logits = seq.logits
        results.append(
            GradientConstraintOutput(
                gradient=(-logits * temperature,), loss=float(np.mean(logits**2)), metrics={"temperature": temperature}
            )
        )
    return results


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

        # By default, flat metadata is written to each input segment.
        for i in range(len(sequences_a)):
            constraints_a = seg_a.proposal_sequences[i]._constraints_metadata
            constraints_b = seg_b.proposal_sequences[i]._constraints_metadata
            assert "mock_multi_input_scoring_function_disjoint" in constraints_a
            assert "mock_multi_input_scoring_function_disjoint" in constraints_b
            assert "t_percent" in constraints_a["mock_multi_input_scoring_function_disjoint"]["data"]
            assert "c_percent" in constraints_b["mock_multi_input_scoring_function_disjoint"]["data"]
            assert "c_percent" in constraints_a["mock_multi_input_scoring_function_disjoint"]["data"]
            assert "t_percent" in constraints_b["mock_multi_input_scoring_function_disjoint"]["data"]

    @pytest.mark.parametrize("use_input_slots", [True, False], ids=["input-slots", "segment-labels"])
    def test_metadata_recipient_label_limits_metadata_to_one_segment(self, use_input_slots: bool):
        """metadata_recipient keeps user metadata on the selected labeled input only."""
        seg_a = _make_segment_with_proposals(["AAAA"], "dna")
        seg_b = _make_segment_with_proposals(["CCCC"], "dna")
        input_slots = None
        if use_input_slots:
            input_slots = [InputSlot(label="Query"), InputSlot(label="Reference")]
        else:
            seg_a.label = "Query"
            seg_b.label = "Reference"

        def metadata_to_query(input_sequences, config):
            return [ConstraintOutput(score=0.0, metadata={"query_metric": 1.0}, metadata_recipient="Query")]

        metadata_to_query._constraint_supported_sequence_types = ["dna"]

        Constraint(
            inputs=[seg_a, seg_b],
            function=metadata_to_query,
            function_config=MockConfig(),
            input_slots=input_slots,
        ).evaluate()

        data_a = seg_a.proposal_sequences[0]._constraints_metadata["metadata_to_query"]["data"]
        data_b = seg_b.proposal_sequences[0]._constraints_metadata["metadata_to_query"]["data"]
        assert data_a == {"query_metric": 1.0}
        assert data_b == {}

    def test_structure_and_logits_propagate_to_original(self):
        """Regression for #1180: structures/logits on the returned result reach the original proposal."""
        attached_structure = MockStructure.with_plddt([0.1, 0.95, 0.95])
        attached_logits = np.ones((3, 20))

        def attaches(input_sequences: list[tuple[Sequence, ...]], config: BaseModel) -> list[ConstraintOutput]:
            return [
                ConstraintOutput(score=0.0, structures=(attached_structure,), logits=(attached_logits,))
                for _ in input_sequences
            ]

        attaches._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]

        segment = _make_segment_with_proposals(["ACD"], seq_type="protein")
        Constraint(inputs=[segment], function=attaches, function_config=MockConfig()).evaluate()

        assert segment.proposal_sequences[0].structure is attached_structure
        assert np.array_equal(segment.proposal_sequences[0].logits, attached_logits)

    def test_constraint_receives_original_proposal_by_identity(self):
        """Forward constraints read the real proposal sequences (no defensive dummy copy)."""
        existing_structure = MockStructure.with_plddt([0.5, 0.5, 0.5])
        existing_logits = np.ones((3, 20))
        seen: dict[str, object] = {}

        def inspects(input_sequences: list[tuple[Sequence, ...]], config: BaseModel) -> list[ConstraintOutput]:
            for (seq,) in input_sequences:
                seen["sequence_obj"] = seq
                seen["structure"] = seq.structure
                seen["logits"] = seq.logits
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        inspects._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]

        segment = _make_segment_with_proposals(["ACD"], seq_type="protein")
        segment.proposal_sequences[0].structure = existing_structure
        segment.proposal_sequences[0].logits = existing_logits

        Constraint(inputs=[segment], function=inspects, function_config=MockConfig()).evaluate()

        assert seen["sequence_obj"] is segment.proposal_sequences[0]
        assert seen["structure"] is existing_structure
        assert seen["logits"] is existing_logits
        # Empty structures=()/logits=() on the returned result is a no-op.
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
        with pytest.raises(ValueError, match=r"Mask length .* != num_proposals"):
            constraint.evaluate(mask=[True, False, True, True])

        # Shorter than pool
        with pytest.raises(ValueError, match=r"Mask length .* != num_proposals"):
            constraint.evaluate(mask=[True, True])


# =============================================================================
# TESTS FOR THRESHOLD-BASED FILTERING
# =============================================================================


class TestConstraintThreshold:
    """Tests for threshold-based filtering functionality."""

    def test_threshold_converts_scores_to_boolean(self):
        """Test that threshold converts float scores to boolean filters."""

        def mock_scoring(input_sequences, config=None):
            return [ConstraintOutput(score=len(seq_tuple[0].sequence) / 10.0) for seq_tuple in input_sequences]

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
            return [ConstraintOutput(score=s) for s in (0.4, 0.8)]

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
            return [ConstraintOutput(score=s) for s in (0.2, 0.5)]

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

    def test_metadata_keys_matching_infrastructure_fields_are_allowed(self):
        """Keys like "score"/"weight" in result.metadata land under ["data"]; no collision."""

        def scoring_with_score_key(input_sequences, config):
            return [ConstraintOutput(score=0.1, metadata={"score": 0.5, "weight": 9.0}) for _ in input_sequences]

        scoring_with_score_key._constraint_supported_sequence_types = {"dna"}

        segment = _make_segment_with_proposals(["ATCG"], "dna")
        Constraint(
            inputs=[segment],
            function=scoring_with_score_key,
            function_config=MockConfig(),
        ).evaluate()

        cdata = segment.proposal_sequences[0]._constraints_metadata["scoring_with_score_key"]
        assert cdata["score"] == pytest.approx(0.1)  # infrastructure field (raw score)
        assert cdata["data"]["score"] == 0.5  # user metadata, nested under "data"
        assert cdata["data"]["weight"] == 9.0

    def test_out_of_range_scores_raise(self):
        """Constraint raw scores must be finite and within [0, 1]."""

        def negative_scoring(input_sequences, config=None):
            return [ConstraintOutput(score=s) for s in (-0.5, 1.5, 0.5)]

        negative_scoring._constraint_config_class = MockConfig
        negative_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["ATCG", "GGGG", "TTTT"]
        segment = _make_segment_with_proposals(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=negative_scoring,
            function_config=MockConfig(),
        )

        with pytest.raises(ValueError, match=r"score -0\.5 not in \[0\.0, 1\.0\]"):
            constraint.evaluate()

    def test_allow_raw_scores_permits_non_finite(self):
        """Constraints with _constraint_allow_raw_scores=True may return inf scores."""

        def inf_scoring(input_sequences, config=None):
            return [ConstraintOutput(score=float("inf")), ConstraintOutput(score=0.0)]

        inf_scoring._constraint_config_class = MockConfig
        inf_scoring._constraint_supported_sequence_types = ["dna"]
        inf_scoring._constraint_allow_raw_scores = True

        segment = _make_segment_with_proposals(["ATCG", "GGGG"], "dna")
        constraint = Constraint(
            inputs=[segment],
            function=inf_scoring,
            function_config=MockConfig(),
        )
        scores = constraint.evaluate()
        assert scores[0] == float("inf")
        assert scores[1] == 0.0

    def test_custom_metadata_is_stored_under_data(self):
        """Metadata on ConstraintOutput populates _constraints_metadata[label]["data"]."""

        def safe_scoring_function(input_sequences, config):
            return [
                ConstraintOutput(score=0.1, metadata={"gc_content": 50.0, "my_custom_metric": 42})
                for _ in input_sequences
            ]

        safe_scoring_function._constraint_supported_sequence_types = {"dna"}

        segment = _make_segment_with_proposals(["ATCG"], "dna")
        constraint = Constraint(
            inputs=[segment],
            function=safe_scoring_function,
            function_config=MockConfig(),
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        cdata = segment.proposal_sequences[0]._constraints_metadata["safe_scoring_function"]
        assert cdata["data"]["gc_content"] == 50.0
        assert cdata["data"]["my_custom_metric"] == 42

    def test_wrong_return_type_raises(self):
        """evaluate() requires functions to return ConstraintOutput instances."""

        def returns_floats(input_sequences, config=None):
            return [0.1 for _ in input_sequences]

        returns_floats._constraint_supported_sequence_types = ["dna"]

        segment = _make_segment_with_proposals(["ATCG"], "dna")
        constraint = Constraint(inputs=[segment], function=returns_floats, function_config=MockConfig())
        with pytest.raises(TypeError, match="expected ConstraintOutput"):
            constraint.evaluate()

    def test_structures_tuple_wrong_arity_raises(self):
        """Structures on ConstraintOutput must match the number of input segments when non-empty."""
        seg_a = _make_segment_with_proposals(["AAAA"], "dna")
        seg_b = _make_segment_with_proposals(["TTTT"], "dna")

        def wrong_arity(input_sequences, config=None):
            return [ConstraintOutput(score=0.0, structures=(None,)) for _ in input_sequences]

        wrong_arity._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(inputs=[seg_a, seg_b], function=wrong_arity, function_config=MockConfig())
        with pytest.raises(ValueError, match=r"1 structures, expected 2"):
            constraint.evaluate()

    def test_logits_tuple_wrong_arity_raises(self):
        """Logits on ConstraintOutput must match the number of input segments when non-empty."""
        seg_a = _make_segment_with_proposals(["AAAA"], "dna")
        seg_b = _make_segment_with_proposals(["TTTT"], "dna")

        def wrong_arity(input_sequences, config=None):
            return [ConstraintOutput(score=0.0, logits=(np.zeros((4, 4)),)) for _ in input_sequences]

        wrong_arity._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(inputs=[seg_a, seg_b], function=wrong_arity, function_config=MockConfig())
        with pytest.raises(ValueError, match=r"1 logits, expected 2"):
            constraint.evaluate()

    def test_metadata_recipient_ambiguous_label_raises(self):
        """metadata_recipient labels must be unique."""
        seg_a = _make_segment_with_proposals(["AAAA"], "dna")
        seg_b = _make_segment_with_proposals(["TTTT"], "dna")

        def ambiguous_label(input_sequences, config=None):
            return [ConstraintOutput(score=0.0, metadata={"metric": 1}, metadata_recipient="Target")]

        ambiguous_label._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[seg_a, seg_b],
            function=ambiguous_label,
            function_config=MockConfig(),
            input_slots=[InputSlot(label="Target"), InputSlot(label="Target")],
        )
        with pytest.raises(ValueError, match=r"metadata_recipient 'Target' must match exactly one input label"):
            constraint.evaluate()


# =============================================================================
# TESTS FOR GRADIENT RESULT
# =============================================================================


class TestGradientConstraintOutput:
    @pytest.mark.parametrize(
        "shapes",
        [
            pytest.param([(2, 2)], id="single-segment"),
            pytest.param([(4, 20), (3, 20)], id="multi-segment"),
        ],
    )
    def test_construction_preserves_gradient_shapes_and_default_metrics(self, shapes: list[tuple[int, int]]) -> None:
        result = GradientConstraintOutput(gradient=tuple(np.zeros(s) for s in shapes), loss=0.5)
        assert result.loss == 0.5
        assert result.metrics == {}
        assert result.structures == ()  # default: empty tuple, backward-compat for producers that omit it
        assert [g.shape for g in result.gradient] == shapes

    def test_custom_metrics_stored_and_repr_shows_shape_not_array(self) -> None:
        result = GradientConstraintOutput(gradient=(np.zeros((5, 20)),), loss=0.5, metrics={"plddt": 0.85})
        assert result.metrics["plddt"] == pytest.approx(0.85)
        # repr must elide the array (huge) and surface the shape + loss for debugging.
        r = repr(result)
        assert "(5, 20)" in r
        assert "loss=0.5" in r

    def test_frozen(self) -> None:
        from pydantic import ValidationError

        result = GradientConstraintOutput(gradient=(np.zeros((5, 20)),), loss=1.0)
        with pytest.raises(ValidationError, match="frozen"):
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
        (raw,) = _mock_backward([(segment.proposal_sequences[0],)], config=MockConfig(), temperature=1.0)
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
        assert isinstance(results[0], GradientConstraintOutput)
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

        def capturing_backward(
            input_sequences: list[tuple[Sequence, ...]], *, config: BaseModel, **kwargs: Any
        ) -> list[GradientConstraintOutput]:
            received.append(config)
            return [
                GradientConstraintOutput(gradient=(np.zeros_like(seq.logits),), loss=0.0) for (seq,) in input_sequences
            ]

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
        with pytest.raises(RuntimeError, match=r"no input has logits"):
            c.compute_gradient(temperature=1.0)

    @pytest.mark.parametrize(
        ("slots", "prep_binder_logits", "prep_target_structure", "match"),
        [
            (
                [
                    InputSlot(label="Binder Chain", requires_logits=True),
                    InputSlot(label="Target", requires_structure=True),
                ],
                False,
                True,
                r"'.*' proposal .* slot 0 'Binder Chain': missing logits",
            ),
            (
                [
                    InputSlot(label="Binder Chain", requires_logits=True),
                    InputSlot(label="Target", requires_structure=True),
                ],
                True,
                False,
                r"'.*' proposal .* slot 1 'Target': missing structure",
            ),
        ],
    )
    def test_slot_requirements_raise_on_swap(
        self, slots: list[InputSlot], prep_binder_logits: bool, prep_target_structure: bool, match: str
    ) -> None:
        """Slot ``requires_logits`` / ``requires_structure`` fire a swap-detection error when unmet."""

        def backward(
            input_sequences: list[tuple[Sequence, ...]], *, config: BaseModel, **kwargs: Any
        ) -> list[GradientConstraintOutput]:
            return [
                GradientConstraintOutput(gradient=(np.zeros((3, 20)), np.zeros((3, 20))), loss=0.0)
                for _ in input_sequences
            ]

        binder = _make_segment_with_proposals(["ACD"], seq_type="protein")
        target = _make_segment_with_proposals(["GHI"], seq_type="protein")
        if prep_binder_logits:
            binder.proposal_sequences[0].logits = np.zeros((3, 20))
        if prep_target_structure:
            target.proposal_sequences[0].structure = MockStructure.with_plddt([0.5] * 3)
        c = Constraint(
            inputs=[binder, target],
            function=mock_multi_input_scoring_function,
            function_config=MockConfig(),
            backward=backward,
            backward_config=MockConfig(),
            input_slots=slots,
        )
        with pytest.raises(RuntimeError, match=match):
            c.compute_gradient(temperature=1.0)

    @pytest.mark.parametrize(
        ("backward", "expected_exception", "match"),
        [
            (
                lambda input_sequences, *, config, **kwargs: [],
                ValueError,
                r"returned 0 gradient results, expected 1",
            ),
            (
                lambda input_sequences, *, config, **kwargs: [
                    {"gradient": np.zeros_like(s[0].logits), "loss": 0.0} for s in input_sequences
                ],
                TypeError,
                r"expected GradientConstraintOutput",
            ),
            (
                lambda input_sequences, *, config, **kwargs: [
                    GradientConstraintOutput(gradient=(np.zeros((4, 8)),), loss=0.0) for _ in input_sequences
                ],
                ValueError,
                r"segment 0: gradient shape",
            ),
            (
                lambda input_sequences, *, config, **kwargs: [
                    GradientConstraintOutput(gradient=(np.zeros((8, 4)), np.zeros((8, 4))), loss=0.0)
                    for _ in input_sequences
                ],
                ValueError,
                r"2 gradients, expected 1",
            ),
            (
                lambda input_sequences, *, config, **kwargs: [
                    GradientConstraintOutput(
                        gradient=(np.zeros_like(s[0].logits),),
                        loss=0.0,
                        structures=(MockStructure.with_plddt([0.5] * 8), MockStructure.with_plddt([0.5] * 8)),
                    )
                    for s in input_sequences
                ],
                ValueError,
                r"2 structures, expected 1",
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

    def test_compute_gradient_structures_assignment_and_none_preservation(self) -> None:
        """Non-None entries land on the real proposal; None entries leave existing structure untouched."""
        existing = MockStructure.with_plddt([0.7, 0.7, 0.7])
        new_struct = MockStructure.with_plddt([0.3, 0.3, 0.3])
        seg_a = _make_segment_with_proposals(["ACD"], seq_type="protein")
        seg_b = _make_segment_with_proposals(["GHI"], seq_type="protein")
        seg_a.proposal_sequences[0].logits = np.zeros((3, 20))
        seg_b.proposal_sequences[0].structure = existing

        def backward(
            input_sequences: list[tuple[Sequence, ...]], *, config: BaseModel, **kwargs: Any
        ) -> list[GradientConstraintOutput]:
            return [
                GradientConstraintOutput(
                    gradient=(np.zeros_like(s[0].logits), np.zeros((3, 20))),
                    loss=0.0,
                    structures=(new_struct, None),
                )
                for s in input_sequences
            ]

        _make_gradient_constraint(
            inputs=[seg_a, seg_b], function=mock_multi_input_scoring_function, backward=backward
        ).compute_gradient(temperature=1.0)

        assert seg_a.proposal_sequences[0].structure is new_struct
        assert seg_b.proposal_sequences[0].structure is existing


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
        assert isinstance(results[0], GradientConstraintOutput)
        assert results[0].gradient[0].shape == (8, 4)

        with pytest.raises(RuntimeError, match=r"has no scoring function"):
            c.evaluate()

    def test_function_only_gradient_raises(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        segment.proposal_sequences[0].logits = np.zeros((8, 4))
        c = Constraint(inputs=[segment], function=mock_single_input_scoring_function, function_config=MockConfig())
        with pytest.raises(RuntimeError, match=r"has no backward callable"):
            c.compute_gradient(temperature=1.0)

    def test_neither_callable_raises(self) -> None:
        segment = _make_segment_with_proposals(["ACTGACTG"])
        with pytest.raises(ValueError, match="At least one of"):
            Constraint(inputs=[segment])
