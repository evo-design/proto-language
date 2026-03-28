"""tests/language_tests/constraint_tests/test_sequence_composition/test_sequence_length_constraint.py"""
import pytest

from proto_language.language.constraint import sequence_length_constraint
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import (
    SequenceLengthConfig,
)
from proto_language.language.core import Constraint, Segment


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
        assert seg_match.proposal_sequences[0]._constraints_metadata["sequence_length_constraint"]["data"]["length"] == target_len
        assert seg_short.proposal_sequences[0]._constraints_metadata["sequence_length_constraint"]["data"]["length"] == target_len // 2

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
