import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import protein_length_constraint, ConstraintRegistry
from proto_language.language.constraint.protein_quality.protein_length_constraint import ProteinLengthConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for protein_length_constraint
class TestProteinLengthConstraint:
    def test_protein_within_range(self):
        """Test protein length within acceptable range."""
        segment = create_segment("MVLSPADKTNVKAAWGKVGAH", SequenceType.PROTEIN)
        config = ProteinLengthConfig(min_length=20, max_length=25)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )

        assert constraint.evaluate()[0] == 0.0
        assert (
            segment[0]._metadata["segment_0.protein_length_constraint.protein_length"]
            == 21
        )

    def test_protein_too_short(self):
        """Test protein shorter than minimum."""
        segment = create_segment("MVLSP", SequenceType.PROTEIN)
        config = ProteinLengthConfig(min_length=10, max_length=50)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        assert (
            segment[0]._metadata["segment_0.protein_length_constraint.protein_length"]
            == 5
        )

    def test_protein_too_long(self):
        """Test protein longer than maximum."""
        segment = create_segment("M" * 100, SequenceType.PROTEIN)
        config = ProteinLengthConfig(min_length=10, max_length=50)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        assert (
            segment[0]._metadata["segment_0.protein_length_constraint.protein_length"]
            == 100
        )

    def test_invalid_sequence_type(self):
        """Test that DNA sequence raises assertion (constraint-specific check)."""
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        config = ProteinLengthConfig(min_length=10, max_length=50)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )

        with pytest.raises(AssertionError):
            constraint.evaluate()