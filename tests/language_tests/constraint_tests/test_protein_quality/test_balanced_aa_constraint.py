import pytest
import sys

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import balanced_aa_constraint
from proto_language.language.constraint.protein_quality.balanced_aa_constraint import BalancedAaConfig
from ..utils import create_segment


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
            vectorized=True,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_score"
            in segment.candidate_sequences[0]._metadata
        )
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_amino_acids"
            in segment.candidate_sequences[0]._metadata
        )

    def test_unbalanced_protein(self):
        """Test protein with unbalanced amino acid frequencies and metadata."""
        segment = create_segment("AAAAAAGGGGLLLLMMMM", SequenceType.PROTEIN)
        config = BalancedAaConfig(min_aa_frequency=0.1, max_underrepresented_count=2)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
            vectorized=True,
        )

        score = constraint.evaluate()[0]
        # With 4 amino acids, all at ~25%, and threshold of 10%, all are above threshold
        # So underrepresented_aa_count should be 0
        assert score >= 0.0
        # Check constraint-specific metadata fields
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_count"
            in segment.candidate_sequences[0]._metadata
        )
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_score"
            in segment.candidate_sequences[0]._metadata
        )
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_amino_acids"
            in segment.candidate_sequences[0]._metadata
        )

    def test_empty_sequence(self):
        """Test empty sequence handling."""
        segment = create_segment("", SequenceType.PROTEIN)
        config = BalancedAaConfig(min_aa_frequency=0.05, max_underrepresented_count=5)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
            vectorized=True,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        # Empty sequence has no underrepresented amino acids (0 count < 0 threshold is false)
        assert scores[0] == 0.0
