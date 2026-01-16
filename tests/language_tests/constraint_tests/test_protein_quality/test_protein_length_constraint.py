import pytest

from proto_language.language.core import Constraint, Segment
from proto_language.language.constraint import protein_length_constraint
from proto_language.language.constraint.protein_quality.protein_length_constraint import ProteinLengthConfig


# Tests for protein_length_constraint
class TestProteinLengthConstraint:
    def test_protein_within_range(self):
        """Test protein length within acceptable range."""
        segment = Segment(sequence="MVLSPADKTNVKAAWGKVGAH", sequence_type="protein")
        config = ProteinLengthConfig(min_length=20, max_length=25)

        constraint = Constraint(
            inputs=[segment],
            function=protein_length_constraint,
            function_config=config,
        )

        assert constraint.evaluate()[0] == 0.0
        assert (
            segment.candidate_sequences[0]._metadata["segment_0.protein_length_constraint.protein_length"]
            == 21
        )

    def test_protein_too_short(self):
        """Test protein shorter than minimum."""
        segment = Segment(sequence="MVLSP", sequence_type="protein")
        config = ProteinLengthConfig(min_length=10, max_length=50)

        constraint = Constraint(
            inputs=[segment],
            function=protein_length_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        assert (
            segment.candidate_sequences[0]._metadata["segment_0.protein_length_constraint.protein_length"]
            == 5
        )

    def test_protein_too_long(self):
        """Test protein longer than maximum."""
        segment = Segment(sequence="M" * 100, sequence_type="protein")
        config = ProteinLengthConfig(min_length=10, max_length=50)

        constraint = Constraint(
            inputs=[segment],
            function=protein_length_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        assert (
            segment.candidate_sequences[0]._metadata["segment_0.protein_length_constraint.protein_length"]
            == 100
        )

    def test_invalid_sequence_type(self):
        """Test that DNA sequence raises ValueError at construction (centralized validation)."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        config = ProteinLengthConfig(min_length=10, max_length=50)

        with pytest.raises(ValueError, match="does not support sequence type 'dna'"):
            Constraint(
                inputs=[segment],
                function=protein_length_constraint,
                function_config=config,
            )