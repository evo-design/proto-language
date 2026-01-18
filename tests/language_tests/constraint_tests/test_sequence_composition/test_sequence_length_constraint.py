import pytest

from proto_language.language.core import Constraint, Segment
from proto_language.language.constraint import sequence_length_constraint
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import SequenceLengthConfig


# Tests for sequence_length_constraint
class TestSequenceLengthConstraint:
    def test_single_segment(self):
        target_len = 20
        seg_match = Segment(sequence="A" * target_len)
        seg_short = Segment(sequence="A" * (target_len // 2))
        seg_long = Segment(sequence="A" * (target_len * 2))

        config = SequenceLengthConfig(target_length=target_len)
        constraint_match = Constraint(
            inputs=[seg_match],
            function=sequence_length_constraint,
            function_config=config,
        )
        constraint_short = Constraint(
            inputs=[seg_short],
            function=sequence_length_constraint,
            function_config=config,
        )
        constraint_long = Constraint(
            inputs=[seg_long],
            function=sequence_length_constraint,
            function_config=config,
        )

        assert constraint_match.evaluate()[0] == 0.0
        assert abs(constraint_short.evaluate()[0] - 0.5) < 1e-9
        assert abs(constraint_long.evaluate()[0] - 1.0) < 1e-9
        assert (
            seg_match.candidate_sequences[0]._metadata[
                "segment_0.sequence_length_constraint.length"
            ]
            == target_len
        )
        assert (
            seg_short.candidate_sequences[0]._metadata[
                "segment_0.sequence_length_constraint.length"
            ]
            == target_len // 2
        )

    def test_single_input_constraint_rejects_multiple_segments(self):
        """Tests that single-input constraints reject multiple segments."""
        seg1 = Segment(sequence="A" * 10)
        seg2 = Segment(sequence="T" * 10)

        config = SequenceLengthConfig(target_length=20)
        
        # Single-input constraint should reject multiple segments
        with pytest.raises(ValueError, match="single-input.*but received.*segments"):
            Constraint(
                inputs=[seg1, seg2],
                function=sequence_length_constraint,
                function_config=config,
            )

    @pytest.mark.parametrize(
        "seq_str, target_len, expected_score",
        [
            ("", 10, 1.0),  # Empty sequence
            ("A", 1, 0.0),  # Single character match
            ("A", 2, 0.5),  # Single character mismatch
        ],
    )
    def test_edge_cases(self, seq_str, target_len, expected_score):
        """Test constraint-specific edge cases."""
        segment = Segment(sequence=seq_str)
        config = SequenceLengthConfig(target_length=target_len)
        constraint = Constraint(
            inputs=[segment],
            function=sequence_length_constraint,
            function_config=config,
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9
