import pytest

from proto_language.language.core import Sequence, Segment, Construct, SequenceType


class TestSequence:
    """Tests for the base Sequence class."""

    @pytest.mark.parametrize(
        "seq_type, valid_seq, invalid_char",
        [
            ("dna", "ATCG", "U"),
            ("rna", "AUCG", "T"),
            ("protein", "ACDEFGHIKLMNPQRSTVWY", "B"),
        ],
    )
    def test_sequence_validation(self, seq_type, valid_seq, invalid_char):
        """Tests character validation for each sequence type."""
        # Test valid sequence
        seq = Sequence(valid_seq, seq_type)
        assert seq.sequence == valid_seq

        # Test invalid character on creation
        with pytest.warns(UserWarning):
            Sequence(valid_seq + invalid_char, seq_type)

        # Test invalid character on setter
        with pytest.warns(UserWarning):
            seq.sequence = valid_seq + invalid_char

    def test_custom_validation(self):
        """Tests sequence validation with a custom character set."""
        custom_chars = {"0", "1"}
        seq = Sequence("0101", valid_chars=custom_chars)
        assert seq.sequence == "0101"
        with pytest.warns(UserWarning):
            seq.sequence = "01012"

    def test_metadata(self):
        """Tests automatic and custom metadata handling."""
        seq = Sequence("ATCG", "dna", metadata={"id": "test1"})
        assert seq._metadata["id"] == "test1"
        assert seq._metadata["sequence"] == "ATCG"
        assert seq._metadata["sequence_length"] == 4

        # Test metadata update on sequence change
        seq.sequence = "GATTACA"
        assert seq._metadata["id"] == "test1"  # Custom metadata preserved
        assert seq._metadata["sequence"] == "GATTACA"
        assert seq._metadata["sequence_length"] == 7
