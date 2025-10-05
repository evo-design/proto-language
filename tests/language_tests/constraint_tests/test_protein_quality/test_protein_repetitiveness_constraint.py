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
from proto_language.language.constraint import (
    protein_repetitiveness_constraint,
    ConstraintRegistry,
)
from proto_language.language.constraint.protein_quality.protein_repetitiveness_constraint import ProteinRepetitivenessConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for protein_repetitiveness_constraint
class TestProteinRepetitivenessConstraint:
    def test_non_repetitive_protein(self):
        """Test protein with low repetitiveness."""
        segment = create_segment(
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF", SequenceType.PROTEIN
        )
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.5, min_repeat_length=3)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_repetitiveness_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0
        assert (
            "segment_0.protein_repetitiveness_constraint.repetitiveness_score"
            in segment[0]._metadata
        )
        assert (
            "segment_0.protein_repetitiveness_constraint.max_repetitive_fraction"
            in segment[0]._metadata
        )

    def test_highly_repetitive_protein(self):
        """Test protein with high repetitiveness."""
        segment = create_segment("AAAAAAAAAAAAAA", SequenceType.PROTEIN)
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.3)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_repetitiveness_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        rep_score = segment[0]._metadata[
            "segment_0.protein_repetitiveness_constraint.repetitiveness_score"
        ]
        assert rep_score > 0.5  # Highly repetitive

    def test_repetitive_pattern(self):
        """Test protein with repetitive pattern."""
        segment = create_segment("MVKMVKMVKMVKMVK", SequenceType.PROTEIN)
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.3, min_repeat_length=3)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_repetitiveness_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        rep_score = segment[0]._metadata[
            "segment_0.protein_repetitiveness_constraint.repetitiveness_score"
        ]
        assert rep_score > 0.3
