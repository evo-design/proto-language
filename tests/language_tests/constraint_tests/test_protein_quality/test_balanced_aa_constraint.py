from proto_language.language.core import Constraint, Segment
from proto_language.language.constraint import balanced_aa_constraint
from proto_language.language.constraint.protein_quality.balanced_aa_constraint import BalancedAaConfig


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
        constraints = segment.candidate_sequences[0]._metadata["constraints"]
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
        constraints = segment.candidate_sequences[0]._metadata["constraints"]
        assert "underrepresented_aa_count" in constraints["balanced_aa_constraint"]["data"]
        assert "underrepresented_aa_score" in constraints["balanced_aa_constraint"]["data"]
        assert "underrepresented_amino_acids" in constraints["balanced_aa_constraint"]["data"]

    def test_empty_sequence(self):
        """Test empty sequence handling."""
        segment = Segment(length=0, sequence_type="protein")
        config = BalancedAaConfig(min_aa_frequency=0.05, max_underrepresented_count=5)

        constraint = Constraint(
            inputs=[segment],
            function=balanced_aa_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        # Empty sequence has no underrepresented amino acids (0 count < 0 threshold is false)
        assert scores[0] == 0.0
