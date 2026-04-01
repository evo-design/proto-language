"""tests/language_tests/constraint_tests/test_protein_quality/test_protein_repetitiveness_constraint.py."""

from proto_language.language.constraint import protein_repetitiveness_constraint
from proto_language.language.constraint.protein_quality.protein_repetitiveness_constraint import (
    ProteinRepetitivenessConfig,
)
from proto_language.language.core import Constraint, Segment


# Tests for protein_repetitiveness_constraint
class TestProteinRepetitivenessConstraint:
    def test_non_repetitive_protein(self):
        """Test protein with low repetitiveness."""
        segment = Segment(sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF", sequence_type="protein")
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.5, min_repeat_length=3)

        constraint = Constraint(
            inputs=[segment],
            function=protein_repetitiveness_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "repetitiveness_score" in constraints["protein_repetitiveness_constraint"]["data"]
        assert "max_repetitive_fraction" in constraints["protein_repetitiveness_constraint"]["data"]

    def test_highly_repetitive_protein(self):
        """Test protein with high repetitiveness."""
        segment = Segment(sequence="AAAAAAAAAAAAAA", sequence_type="protein")
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.3)

        constraint = Constraint(
            inputs=[segment],
            function=protein_repetitiveness_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        constraints = segment.proposal_sequences[0]._constraints_metadata
        rep_score = constraints["protein_repetitiveness_constraint"]["data"]["repetitiveness_score"]
        assert rep_score > 0.5  # Highly repetitive

    def test_repetitive_pattern(self):
        """Test protein with repetitive pattern."""
        segment = Segment(sequence="MVKMVKMVKMVKMVK", sequence_type="protein")
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.3, min_repeat_length=3)

        constraint = Constraint(
            inputs=[segment],
            function=protein_repetitiveness_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        constraints = segment.proposal_sequences[0]._constraints_metadata
        rep_score = constraints["protein_repetitiveness_constraint"]["data"]["repetitiveness_score"]
        assert rep_score > 0.3
