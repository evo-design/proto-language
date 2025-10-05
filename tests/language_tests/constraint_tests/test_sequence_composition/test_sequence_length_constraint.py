import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import sequence_length_constraint, ConstraintRegistry
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import SequenceLengthConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for sequence_length_constraint
class TestSequenceLengthConstraint:
    def test_single_segment(self):
        target_len = 20
        seg_match = create_segment("A" * target_len)
        seg_short = create_segment("A" * (target_len // 2))
        seg_long = create_segment("A" * (target_len * 2))

        config = SequenceLengthConfig(target_length=target_len)
        constraint_match = Constraint(
            inputs=[seg_match],
            scoring_function=sequence_length_constraint,
            scoring_function_config=config,
        )
        constraint_short = Constraint(
            inputs=[seg_short],
            scoring_function=sequence_length_constraint,
            scoring_function_config=config,
        )
        constraint_long = Constraint(
            inputs=[seg_long],
            scoring_function=sequence_length_constraint,
            scoring_function_config=config,
        )

        assert constraint_match.evaluate()[0] == 0.0
        assert abs(constraint_short.evaluate()[0] - 0.5) < 1e-9
        assert abs(constraint_long.evaluate()[0] - 1.0) < 1e-9
        assert (
            seg_match.batch_sequences[0]._metadata[
                "segment_0.sequence_length_constraint.length"
            ]
            == target_len
        )
        assert (
            seg_short.batch_sequences[0]._metadata[
                "segment_0.sequence_length_constraint.length"
            ]
            == target_len // 2
        )

    def test_contiguous_concatenation(self):
        """Tests length constraint on concatenated segments."""
        target_len = 20
        seg1 = create_segment("A" * 10)
        seg2 = create_segment("T" * 10)

        config = SequenceLengthConfig(target_length=target_len)
        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=sequence_length_constraint,
            scoring_function_config=config,
            concatenate=True,
        )

        assert constraint.evaluate()[0] == 0.0
        # Check metadata propagation to original segments
        assert (
            seg1.batch_sequences[0]._metadata[
                "segment_0-segment_1.sequence_length_constraint.length"
            ]
            == target_len
        )
        assert (
            seg2.batch_sequences[0]._metadata[
                "segment_0-segment_1.sequence_length_constraint.length"
            ]
            == target_len
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
        segment = create_segment(seq_str)
        config = SequenceLengthConfig(target_length=target_len)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config=config,
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9

    def test_disjoint_mode_raises_error(self):
        """Tests that sequence_length_constraint doesn't support concatenate=False mode (constraint-specific behavior)."""
        seg1 = create_segment("A" * 10)
        seg2 = create_segment("T" * 10)
        config = SequenceLengthConfig(target_length=20)
        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=sequence_length_constraint,
            scoring_function_config=config,
            concatenate=False,
        )
        # The default scoring function expects a single Sequence, not a tuple
        with pytest.raises(AttributeError):
            constraint.evaluate()