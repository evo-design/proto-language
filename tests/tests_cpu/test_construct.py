import pytest
import sys

sys.path.append(".")
from proto_language.base import Sequence, ConstructSegment, Construct
from proto_language.utils import SequenceType


class TestSequence:
    """Tests for the base Sequence class."""

    @pytest.mark.parametrize(
        "seq_type, valid_seq, invalid_char",
        [
            (SequenceType.DNA, "ATCG", "U"),
            (SequenceType.RNA, "AUCG", "T"),
            (SequenceType.PROTEIN, "ACDEFGHIKLMNPQRSTVWY", "B"),
        ],
    )
    def test_sequence_validation(self, seq_type, valid_seq, invalid_char):
        """Tests character validation for each sequence type."""
        # Test valid sequence
        seq = Sequence(valid_seq, seq_type)
        assert seq.sequence == valid_seq

        # Test invalid character on creation
        with pytest.raises(ValueError):
            Sequence(valid_seq + invalid_char, seq_type)

        # Test invalid character on setter
        with pytest.raises(ValueError):
            seq.sequence = valid_seq + invalid_char

    def test_custom_validation(self):
        """Tests sequence validation with a custom character set."""
        custom_chars = {"0", "1"}
        seq = Sequence("0101", valid_chars=custom_chars)
        assert seq.sequence == "0101"
        with pytest.raises(ValueError):
            seq.sequence = "01012"

    def test_metadata(self):
        """Tests automatic and custom metadata handling."""
        seq = Sequence("ATCG", SequenceType.DNA, metadata={"id": "test1"})
        assert seq._metadata["id"] == "test1"
        assert seq._metadata["sequence"] == "ATCG"
        assert seq._metadata["sequence_length"] == 4

        # Test metadata update on sequence change
        seq.sequence = "GATTACA"
        assert seq._metadata["id"] == "test1"  # Custom metadata preserved
        assert seq._metadata["sequence"] == "GATTACA"
        assert seq._metadata["sequence_length"] == 7


class TestConstructSegment:
    """Tests for the ConstructSegment class."""

    def test_initialization(self):
        """Tests that a segment is initialized with a single sequence in a list."""
        segment = ConstructSegment("ATCG", SequenceType.DNA)
        assert isinstance(segment, ConstructSegment)
        assert len(segment.batch_sequences) == 1
        assert len(segment) == 1
        assert segment[0].sequence == "ATCG"
        assert segment.sequence_type == SequenceType.DNA

    def test_create_batch(self):
        """Tests that create_batch replicates the initial sequence."""
        segment = ConstructSegment("ATCG", SequenceType.DNA, metadata={"source": "original"})
        segment.create_batch(5)
        assert len(segment) == 5
        for i in range(5):
            assert segment[i].sequence == "ATCG"
            assert segment[i]._metadata["source"] == "original"

        # Check that batched sequences are deep copies
        segment[0].sequence = "GGGG"
        segment[1]._metadata["source"] = "modified"
        assert segment[0].sequence == "GGGG"
        assert segment[1].sequence == "ATCG"
        assert segment[0]._metadata["source"] == "original"
        assert segment[1]._metadata["source"] == "modified"

    def test_iteration(self):
        """Tests iteration over the sequences in a segment."""
        segment = ConstructSegment("A")
        segment.create_batch(3)
        segment[1].sequence = "T"
        segment[2].sequence = "C"
        sequences = [s.sequence for s in segment]
        assert sequences == ["A", "T", "C"]


class TestConstruct:
    """Tests for the Construct class that combines segments."""

    def test_concatenation(self):
        """Tests concatenation of single-sequence segments."""
        seg1 = ConstructSegment("ATG", SequenceType.DNA)
        seg2 = ConstructSegment("CGC", SequenceType.DNA)
        seg3 = ConstructSegment("TAA", SequenceType.DNA)
        construct = Construct([seg1, seg2, seg3])
        
        final_sequences = construct.batch_sequences
        assert len(final_sequences) == 1
        assert final_sequences[0].sequence == "ATGCGC" + "TAA"

    def test_batched_concatenation(self):
        """Tests concatenation of batched segments."""
        seg1 = ConstructSegment("A")
        seg1.create_batch(2)
        seg1[1].sequence = "G"
        
        seg2 = ConstructSegment("C")
        seg2.create_batch(2)
        seg2[1].sequence = "T"

        construct = Construct([seg1, seg2])
        final_sequences = construct.batch_sequences
        assert len(final_sequences) == 2
        assert final_sequences[0].sequence == "AC"
        assert final_sequences[1].sequence == "GT"

    def test_validation(self):
        """Tests validation rules for creating a Construct."""
        # Empty segments list
        with pytest.raises(ValueError, match="must contain at least one segment"):
            Construct([])

        # Inconsistent sequence types
        seg_dna = ConstructSegment("A", SequenceType.DNA)
        seg_rna = ConstructSegment("U", SequenceType.RNA)
        with pytest.raises(ValueError, match="must have the same sequence_type"):
            Construct([seg_dna, seg_rna])

    def test_metadata_concatenation(self):
        """Tests how metadata is merged during concatenation."""
        seg1 = ConstructSegment("A", metadata={"id": 1, "source": "seg1"})
        seg2 = ConstructSegment("C", metadata={"id": 2, "status": "new"})

        construct = Construct([seg1, seg2])
        final_meta = construct.batch_sequences[0]._metadata

        # Metadata from later segments overwrites earlier ones on collision
        assert final_meta["id"] == 2
        assert final_meta["source"] == "seg1"
        assert final_meta["status"] == "new"
        # The sequence metadata should reflect the concatenated sequence
        assert final_meta["sequence"] == "AC"
        assert final_meta["sequence_length"] == 2

    def test_validation_inconsistent_valid_chars(self):
        """Tests that inconsistent valid_chars sets raise a ValueError."""
        seg1 = ConstructSegment("A", valid_chars={"A", "B"})
        seg2 = ConstructSegment("C", valid_chars={"C", "D"})

        with pytest.raises(ValueError, match="must have the same valid_chars"):
            Construct([seg1, seg2])
