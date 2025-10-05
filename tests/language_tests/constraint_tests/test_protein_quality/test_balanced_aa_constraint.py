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
    balanced_aa_constraint,
    ConstraintRegistry,
)
from proto_language.language.constraint.protein_quality.balanced_aa_constraint import BalancedAaConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for balanced_aa_constraint
class TestBalancedAAConstraint:
    def test_balanced_protein(self):
        """Test protein with balanced amino acid frequencies."""
        # Create a relatively balanced sequence
        segment = create_segment(
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF", SequenceType.PROTEIN
        )
        config = BalancedAaConfig(min_aa_frequency=0.02, max_underrepresented_count=10)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_score"
            in segment[0]._metadata
        )
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_amino_acids"
            in segment[0]._metadata
        )

    def test_unbalanced_protein(self):
        """Test protein with unbalanced amino acid frequencies and metadata."""
        segment = create_segment("AAAAAAGGGGLLLLMMMM", SequenceType.PROTEIN)
        config = BalancedAaConfig(min_aa_frequency=0.1, max_underrepresented_count=2)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        # With 4 amino acids, all at ~25%, and threshold of 10%, all are above threshold
        # So underrepresented_aa_count should be 0
        assert score >= 0.0
        # Check constraint-specific metadata fields
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_count"
            in segment[0]._metadata
        )
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_score"
            in segment[0]._metadata
        )
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_amino_acids"
            in segment[0]._metadata
        )

    def test_empty_sequence(self):
        """Test empty sequence handling."""
        segment = create_segment("", SequenceType.PROTEIN)
        config = BalancedAaConfig(min_aa_frequency=0.05, max_underrepresented_count=5)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 1.0
