import pytest
from proto_language.language.core import Constraint, Segment
from proto_language.language.constraint import protein_diversity_constraint
from proto_language.language.constraint.protein_quality.protein_diversity_constraint import ProteinDiversityConfig


# Tests for protein_diversity_constraint
class TestProteinDiversityConstraint:
    def test_high_diversity(self):
        """Test protein with high amino acid diversity and constraint-specific metadata."""
        segment = Segment(sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALER", sequence_type="protein"
        )
        config = ProteinDiversityConfig(min_diversity=0.5)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 0.0
        # Check constraint-specific metadata fields
        constraints = segment.candidate_sequences[0]._metadata["constraints"]
        assert "aa_diversity_score" in constraints["protein_diversity_constraint"]["data"]
        assert "unique_amino_acid_count" in constraints["protein_diversity_constraint"]["data"]
        assert constraints["protein_diversity_constraint"]["data"]["aa_diversity_score"] > 0.5

    def test_low_diversity(self):
        """Test protein with low amino acid diversity."""
        segment = Segment(sequence="AAAAAAGGGGGGLLLLLL", sequence_type="protein")
        config = ProteinDiversityConfig(min_diversity=0.5)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        constraints = segment.candidate_sequences[0]._metadata["constraints"]
        diversity = constraints["protein_diversity_constraint"]["data"]["aa_diversity_score"]
        assert diversity < 0.5
        assert constraints["protein_diversity_constraint"]["data"]["unique_amino_acid_count"] == 3

    def test_single_amino_acid(self):
        """Test protein with only one amino acid type."""
        segment = Segment(sequence="AAAAAAAAAA", sequence_type="protein")
        config = ProteinDiversityConfig(min_diversity=0.2)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        constraints = segment.candidate_sequences[0]._metadata["constraints"]
        assert constraints["protein_diversity_constraint"]["data"]["unique_amino_acid_count"] == 1
        assert constraints["protein_diversity_constraint"]["data"]["aa_diversity_score"] == 1 / 20  # 1 out of 20 standard AAs

    def test_empty_sequence(self):
        """Test that empty sequence raises error (constraint-specific edge case)."""
        segment = Segment(length=0, sequence_type="protein")
        config = ProteinDiversityConfig(min_diversity=0.3)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        with pytest.raises(ValueError, match="Sequence is non-existent"):
            constraint.evaluate()