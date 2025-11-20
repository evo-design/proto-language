import pytest

from proto_language.language.core import Sequence, Segment, Construct, SequenceType


class TestConstruct:
    """Tests for the Construct class that combines segments."""

    def test_concatenation(self):
        """Tests concatenation of single-sequence segments."""
        seg1 = Segment(starting_sequence_or_desired_length="ATG", sequence_type=SequenceType.DNA)
        seg2 = Segment(starting_sequence_or_desired_length="CGC", sequence_type=SequenceType.DNA)
        seg3 = Segment(starting_sequence_or_desired_length="TAA", sequence_type=SequenceType.DNA)
        construct = Construct([seg1, seg2, seg3])

        final_sequences = construct.joined_sequences
        assert len(final_sequences) == 1
        assert final_sequences[0].sequence == "ATGCGC" + "TAA"

    def test_batched_concatenation(self):
        """Tests concatenation of segments with multiple selected sequences."""
        seg1 = Segment(starting_sequence_or_desired_length="A")
        seg1.selected_sequences.append(Sequence(sequence="G", sequence_type=SequenceType.DNA))

        seg2 = Segment(starting_sequence_or_desired_length="C")
        seg2.selected_sequences.append(Sequence(sequence="T", sequence_type=SequenceType.DNA))

        construct = Construct([seg1, seg2])
        final_sequences = construct.joined_sequences
        assert len(final_sequences) == 2
        assert final_sequences[0].sequence == "AC"
        assert final_sequences[1].sequence == "GT"

    def test_validation(self):
        """Tests validation rules for creating a Construct."""
        # Empty segments list
        with pytest.raises(ValueError, match="must contain at least one segment"):
            Construct([])

        # Inconsistent sequence types
        seg_dna = Segment(starting_sequence_or_desired_length="A", sequence_type=SequenceType.DNA)
        seg_rna = Segment(starting_sequence_or_desired_length="U", sequence_type=SequenceType.RNA)
        with pytest.raises(ValueError, match="must have the same sequence_type"):
            Construct([seg_dna, seg_rna])

    def test_metadata_concatenation(self):
        """Tests how metadata is merged during concatenation."""
        seg1 = Segment(starting_sequence_or_desired_length="A", metadata={"id": 1, "source": "seg1"})
        seg2 = Segment(starting_sequence_or_desired_length="C", metadata={"id": 2, "status": "new"})

        construct = Construct([seg1, seg2])
        final_meta = construct.joined_sequences[0]._metadata

        # Metadata from later segments overwrites earlier ones on collision
        assert final_meta["id"] == 2
        assert final_meta["source"] == "seg1"
        assert final_meta["status"] == "new"
        # The sequence metadata should reflect the concatenated sequence
        assert final_meta["sequence"] == "AC"
        assert final_meta["sequence_length"] == 2

    def test_validation_inconsistent_valid_chars(self):
        """Tests that inconsistent valid_chars sets raise a ValueError."""
        seg1 = Segment(starting_sequence_or_desired_length="A", valid_chars={"A", "B"})
        seg2 = Segment(starting_sequence_or_desired_length="C", valid_chars={"C", "D"})

        with pytest.raises(ValueError, match="must have the same valid_chars"):
            Construct([seg1, seg2])
