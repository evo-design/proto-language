from __future__ import annotations
import pytest
import copy
import math

from pydantic import BaseModel

from proto_language.language.core import (
    Segment,
    Constraint,
)
from .utils import (
    mock_single_input_scoring_function,
    mock_multi_input_scoring_function,
    mock_multi_input_scoring_function_disjoint,
    mock_dna_only_scoring_function,
    mock_protein_only_scoring_function,
)


# Empty config model for mock constraint functions
class MockConstraintConfig(BaseModel):
    """Empty config for mock constraints that don't need parameters."""
    pass


def _make_segment_with_candidates(sequences: list[str], seq_type: str = "dna") -> Segment:
    """Helper to create a segment with multiple candidate sequences for testing."""
    segment = Segment(sequence=sequences[0], sequence_type=seq_type)
    segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(sequences))]
    for i, seq_str in enumerate(sequences):
        segment.candidate_sequences[i].sequence = seq_str
    return segment


# =============================================================================
# TESTS FOR CONSTRAINT EVALUATION MODES
# =============================================================================

class TestConstraintEvaluation:
    """Tests for constraint evaluation with different input configurations."""

    @pytest.mark.parametrize("sequences,expected_scores", [
        (["ACTGACTG"], [0.25]),  # Single sequence: 2 T's out of 8
        (["ACTGACTG", "TCTGTCTG", "TTTGTTTG", "TTTTTTTT"], [0.25, 0.5, 0.75, 1.0]),  # Batch
    ])
    def test_constraint_evaluation_contiguous(self, sequences, expected_scores):
        """Tests constraint evaluation with single and batched sequences."""
        segment = _make_segment_with_candidates(sequences, "dna")
        config = MockConstraintConfig()

        # Test sequential (non-batched) mode
        constraint_seq = Constraint(
            inputs=[segment],
            function=mock_single_input_scoring_function,
            function_config=config,
        )
        scores_seq = constraint_seq.evaluate()
        assert scores_seq == expected_scores

        # Reset segment for next test
        segment = _make_segment_with_candidates(sequences, "dna")

        # Test batched mode
        constraint_batch = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=config,
        )
        scores_batch = constraint_batch.evaluate()
        assert scores_batch == expected_scores

    def test_constraint_metadata_propagation(self):
        """Tests that metadata is correctly propagated back to sequences."""
        sequences = ["ACTGACTG", "TTTTTTTT"]
        segment = _make_segment_with_candidates(sequences, "dna")
        config = MockConstraintConfig()

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=config,
        )
        constraint.evaluate()

        # Check metadata was propagated with proper prefixes
        prefix = "segment_0.mock_multi_input_scoring_function"
        for i, seq in enumerate(segment.candidate_sequences):
            assert f"{prefix}.t_count" in seq._metadata
            assert f"{prefix}.total_length" in seq._metadata
            assert f"{prefix}.t_fraction" in seq._metadata

    def test_single_input_constraint_rejects_multiple_segments(self):
        """Tests that single-input constraints reject multiple segments."""
        sequences_a = ["ACTG", "TCTG", "TTTG", "TTTT"]
        sequences_b = ["ACTG", "TCTG", "TTTG", "TTTT"]

        seg_a = _make_segment_with_candidates(sequences_a, "dna")
        seg_b = _make_segment_with_candidates(sequences_b, "dna")
        config = MockConstraintConfig()

        # Single-input constraint should reject multiple segments
        with pytest.raises(ValueError, match="single-input.*but received.*segments"):
            Constraint(
                inputs=[seg_a, seg_b],
                function=mock_multi_input_scoring_function,  # multi_input=False
                function_config=config,
            )

    def test_disjoint_mode(self):
        """Tests constraint evaluation in disjoint mode (separate sequences)."""
        sequences_a = ["AAAA", "AAAT", "AATT", "ATTT", "TTTT"]
        sequences_b = ["AAAA", "AAAC", "AACC", "ACCC", "CCCC"]

        seg_a = _make_segment_with_candidates(sequences_a, "dna")
        seg_b = _make_segment_with_candidates(sequences_b, "dna")
        config = MockConstraintConfig()

        constraint = Constraint(
            inputs=[seg_a, seg_b],
            function=mock_multi_input_scoring_function_disjoint,
            function_config=config,
        )
        scores = constraint.evaluate()

        # Score: (T% in first + C% in second) / 2
        expected_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert scores == expected_scores

        # Each segment should have its own metadata prefix
        prefix_a = "segment_0.mock_multi_input_scoring_function_disjoint"
        prefix_b = "segment_1.mock_multi_input_scoring_function_disjoint"

        for i in range(len(sequences_a)):
            assert f"{prefix_a}.t_percent" in seg_a.candidate_sequences[i]._metadata
            assert f"{prefix_b}.c_percent" in seg_b.candidate_sequences[i]._metadata


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
                function_config=MockConstraintConfig(),
            )

    def test_mixed_batch_sizes_raises_error(self):
        """Test that inconsistent candidate pool sizes raise ValueError."""
        seg1 = _make_segment_with_candidates(["ATCG", "GGGG"])  # 2 candidates
        seg2 = _make_segment_with_candidates(["TTTT"])  # 1 candidate

        # Use multi-input constraint to test batch size validation
        with pytest.raises(ValueError, match="All segments must have the same number of candidate sequences"):
            Constraint(
                inputs=[seg1, seg2],
                function=mock_multi_input_scoring_function_disjoint,
                function_config=MockConstraintConfig(),
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
            function_config=MockConstraintConfig(),
        )
        assert len(constraint.inputs) == 2

    def test_unsupported_sequence_type_raises_error(self):
        """Test that unsupported sequence type raises ValueError."""
        # Create a protein segment and try to use it with DNA-only constraint
        protein_seg = Segment(sequence="MVLSPADKTN", sequence_type="protein")

        with pytest.raises(ValueError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[protein_seg],
                function=mock_dna_only_scoring_function,
                function_config=MockConstraintConfig(),
            )

    def test_supported_sequence_type_passes_validation(self):
        """Test that supported sequence types pass validation."""
        # DNA segment with DNA-only constraint should work
        dna_seg = Segment(sequence="ATCGATCG", sequence_type="dna")

        constraint = Constraint(
            inputs=[dna_seg],
            function=mock_dna_only_scoring_function,
            function_config=MockConstraintConfig(),
        )
        assert constraint is not None

    def test_sequence_type_validation_checks_all_segments(self):
        """Test that validation checks sequence types for all input segments."""
        # All segments should be validated regardless of multi_input flag
        dna_seg = Segment(sequence="ATCGATCG", sequence_type="dna")

        # protein_only constraint with DNA segment should fail
        with pytest.raises(ValueError, match="does not support sequence type 'dna'"):
            Constraint(
                inputs=[dna_seg],
                function=mock_protein_only_scoring_function,
                function_config=MockConstraintConfig(),
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
            function_config=MockConstraintConfig(),
            label="my_custom_label"
        )
        constraint.evaluate()

        # Metadata should use custom label
        assert "segment_0.my_custom_label.t_count" in segment.candidate_sequences[0]._metadata
        # Should NOT use function name
        assert not any("mock_single_input_scoring_function" in key
                       for key in segment.candidate_sequences[0]._metadata.keys())


# =============================================================================
# TESTS FOR MASK-BASED EVALUATION
# =============================================================================

class TestConstraintMask:
    """Tests for mask-based selective evaluation."""

    def test_mask_skips_unevaluated_candidates(self):
        """Test that mask correctly skips evaluation of masked candidates."""
        sequences = ["ATTTTTTT", "AAAAAAAA", "TTTTTTTT", "AAAATTTT", "ATATATAT"]
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConstraintConfig(),
        )

        # Only evaluate candidates 0, 2, 4
        mask = [True, False, True, False, True]
        scores = constraint.evaluate(mask=mask)

        assert len(scores) == 5
        assert scores[0] == pytest.approx(0.875)  # 7/8
        assert math.isnan(scores[1])  # Skipped
        assert scores[2] == pytest.approx(1.0)    # 8/8
        assert math.isnan(scores[3])  # Skipped
        assert scores[4] == pytest.approx(0.5)    # 4/8

        # Verify metadata only propagated to evaluated candidates
        prefix = "segment_0.mock_multi_input_scoring_function"
        assert any(prefix in key for key in segment.candidate_sequences[0]._metadata.keys())
        assert not any(prefix in key for key in segment.candidate_sequences[1]._metadata.keys())

    def test_mask_all_false_returns_nan(self):
        """Test that all-false mask returns NaN for all candidates."""
        sequences = ["ATCG", "GGGG", "TTTT"]
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConstraintConfig(),
        )

        scores = constraint.evaluate(mask=[False, False, False])
        assert len(scores) == 3
        assert all(math.isnan(s) for s in scores)

    def test_mask_invalid_length_raises_error(self):
        """Test that mask with incorrect length raises ValueError."""
        sequences = ["ATCG", "GGGG", "TTTT"]
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConstraintConfig(),
        )

        with pytest.raises(ValueError, match=r"Mask length .* must match"):
            constraint.evaluate(mask=[True, False])  # Wrong length


# =============================================================================
# TESTS FOR THRESHOLD-BASED FILTERING
# =============================================================================

class TestConstraintThreshold:
    """Tests for threshold-based filtering functionality."""

    def test_threshold_converts_scores_to_boolean(self):
        """Test that threshold converts float scores to boolean filters."""
        def mock_scoring(sequences, config=None):
            return [len(seq.sequence) / 10.0 for seq in sequences]
        mock_scoring._constraint_batched = True
        mock_scoring._constraint_multi_input = False
        mock_scoring._constraint_config_class = MockConstraintConfig
        mock_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["ATCG", "ATCGATCG", "AT"]  # lengths 4, 8, 2 → scores 0.4, 0.8, 0.2
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_scoring,
            function_config=MockConstraintConfig(),
            threshold=0.5,
        )
        results = constraint.evaluate()

        # Scores <= threshold pass: 0.4 <= 0.5 (True), 0.8 <= 0.5 (False), 0.2 <= 0.5 (True)
        assert results == [True, False, True]
        assert all(isinstance(r, bool) for r in results)

    def test_no_threshold_returns_float_scores(self):
        """Test that constraints without threshold return float scores."""
        def mock_scoring(sequences, config=None):
            return [0.4, 0.8]
        mock_scoring._constraint_batched = True
        mock_scoring._constraint_multi_input = False
        mock_scoring._constraint_config_class = MockConstraintConfig
        mock_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["ATCG", "GGGG"]
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_scoring,
            function_config=MockConstraintConfig(),
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
            function_config=MockConstraintConfig(),
        )
        assert constraint.weight == 1.0

    def test_weight_multiplies_scores(self):
        """Test that weight correctly multiplies raw scores."""
        def mock_scoring(sequences, config=None):
            return [0.2, 0.5]
        mock_scoring._constraint_batched = True
        mock_scoring._constraint_multi_input = False
        mock_scoring._constraint_config_class = MockConstraintConfig
        mock_scoring._constraint_supported_sequence_types = ["dna"]

        sequences = ["AT", "GC"]
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_scoring,
            function_config=MockConstraintConfig(),
            weight=2.0,
        )
        results = constraint.evaluate()

        assert results == pytest.approx([0.4, 1.0])

    def test_weight_and_threshold_mutually_exclusive(self):
        """Test that setting both weight and threshold raises ValueError."""
        segment = Segment(sequence="ATCG", sequence_type="dna")

        with pytest.raises(ValueError, match="Both threshold .* and weight .* are set"):
            Constraint(
                inputs=[segment],
                function=mock_single_input_scoring_function,
                function_config=MockConstraintConfig(),
                threshold=0.5,
                weight=2.0,
            )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestConstraintEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_large_batch_processing(self):
        """Test constraint with large batch (100+ sequences)."""
        sequences = ["ATCG"] * 100
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConstraintConfig(),
        )
        scores = constraint.evaluate()

        assert len(scores) == 100
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_single_input_constraint_rejects_three_plus_segments(self):
        """Test that single-input constraints reject 3+ segments."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna")
        seg2 = Segment(sequence="GGGG", sequence_type="dna")
        seg3 = Segment(sequence="TTTT", sequence_type="dna")

        # Single-input constraint should reject multiple segments
        with pytest.raises(ValueError, match="single-input.*but received.*3 segments"):
            Constraint(
                inputs=[seg1, seg2, seg3],
                function=mock_single_input_scoring_function,
                function_config=MockConstraintConfig(),
            )

    def test_empty_sequence_raises_error(self):
        """Test that empty sequence causes expected error (division by zero)."""
        sequences = ["ATCG", "", "GGGG"]
        segment = _make_segment_with_candidates(sequences, "dna")

        constraint = Constraint(
            inputs=[segment],
            function=mock_multi_input_scoring_function,
            function_config=MockConstraintConfig(),
        )

        with pytest.raises(ZeroDivisionError):
            constraint.evaluate()
