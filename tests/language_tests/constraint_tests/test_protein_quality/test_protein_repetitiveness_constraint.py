import pytest

from proto_language.language.core import Constraint, SequenceType, Segment
from proto_language.language.constraint import protein_repetitiveness_constraint
from proto_language.language.constraint.protein_quality.protein_repetitiveness_constraint import ProteinRepetitivenessConfig


# Tests for protein_repetitiveness_constraint
class TestProteinRepetitivenessConstraint:
    def test_non_repetitive_protein(self):
        """Test protein with low repetitiveness."""
        segment = Segment(
            starting_sequence_or_desired_length="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF", sequence_type=SequenceType.PROTEIN
        )
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.5, min_repeat_length=3)

        constraint = Constraint(
            inputs=[segment],
            function=protein_repetitiveness_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0
        assert (
            "segment_0.protein_repetitiveness_constraint.repetitiveness_score"
            in segment.candidate_sequences[0]._metadata
        )
        assert (
            "segment_0.protein_repetitiveness_constraint.max_repetitive_fraction"
            in segment.candidate_sequences[0]._metadata
        )

    def test_highly_repetitive_protein(self):
        """Test protein with high repetitiveness."""
        segment = Segment(starting_sequence_or_desired_length="AAAAAAAAAAAAAA", sequence_type=SequenceType.PROTEIN)
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.3)

        constraint = Constraint(
            inputs=[segment],
            function=protein_repetitiveness_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        rep_score = segment.candidate_sequences[0]._metadata[
            "segment_0.protein_repetitiveness_constraint.repetitiveness_score"
        ]
        assert rep_score > 0.5  # Highly repetitive

    def test_repetitive_pattern(self):
        """Test protein with repetitive pattern."""
        segment = Segment(starting_sequence_or_desired_length="MVKMVKMVKMVKMVK", sequence_type=SequenceType.PROTEIN)
        config = ProteinRepetitivenessConfig(max_repetitiveness=0.3, min_repeat_length=3)

        constraint = Constraint(
            inputs=[segment],
            function=protein_repetitiveness_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        rep_score = segment.candidate_sequences[0]._metadata[
            "segment_0.protein_repetitiveness_constraint.repetitiveness_score"
        ]
        assert rep_score > 0.3
