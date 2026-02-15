import pytest

from proto_language.language.constraint import balanced_aa_constraint
from proto_language.language.constraint.protein_quality.balanced_aa_constraint import (
    BalancedAaConfig,
)
from proto_language.language.core import Constraint, Segment


# Tests for balanced_aa_constraint
class TestBalancedAAConstraint:
    def test_balanced_protein(self):
        """Test protein with balanced amino acid frequencies."""
        # Create a relatively balanced sequence
        segment = Segment(sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF", sequence_type="protein"
        )
        config = BalancedAaConfig(min_aa_frequency=0.02, max_underrepresented_count=10)

        constraint = Constraint(
            inputs=[segment],
            function=balanced_aa_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0
        constraints = segment.candidate_sequences[0]._constraints_metadata
        assert "underrepresented_aa_score" in constraints["balanced_aa_constraint"]["data"]
        assert "underrepresented_amino_acids" in constraints["balanced_aa_constraint"]["data"]

    def test_unbalanced_protein(self):
        """Test protein with unbalanced amino acid frequencies and metadata."""
        segment = Segment(sequence="AAAAAAGGGGLLLLMMMM", sequence_type="protein")
        config = BalancedAaConfig(min_aa_frequency=0.1, max_underrepresented_count=2)

        constraint = Constraint(
            inputs=[segment],
            function=balanced_aa_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        # With 4 amino acids, all at ~25%, and threshold of 10%, all are above threshold
        # So underrepresented_aa_count should be 0
        assert score >= 0.0
        # Check constraint-specific metadata fields
        constraints = segment.candidate_sequences[0]._constraints_metadata
        assert "underrepresented_aa_count" in constraints["balanced_aa_constraint"]["data"]
        assert "underrepresented_aa_score" in constraints["balanced_aa_constraint"]["data"]
        assert "underrepresented_amino_acids" in constraints["balanced_aa_constraint"]["data"]

    def test_empty_sequence(self):
        """Test that zero-length segment raises ValueError."""
        with pytest.raises(ValueError, match="Segment length must be positive"):
            Segment(length=0, sequence_type="protein")
