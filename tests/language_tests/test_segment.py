import pytest

from proto_language.language.core import Sequence, Segment


class TestSegment:
    """Tests for the Segment class."""

    def test_initialization(self):
        """Tests that a segment is initialized with a single selected sequence."""
        segment = Segment(sequence="ATCG", sequence_type="dna")
        assert isinstance(segment, Segment)
        assert len(segment.selected_sequences) == 1
        assert segment.num_selected == 1
        assert segment[0].sequence == "ATCG"
        assert segment.sequence_type == "dna"
        assert segment.sequence_length == 4

    def test_candidate_sequences_manipulation(self):
        """Tests that candidate_sequences can be directly manipulated."""
        import copy
        segment = Segment(sequence="ATCG", sequence_type="dna", metadata={"source": "original"})

        # Directly set candidate sequences (like optimizer does)
        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        assert segment.num_candidates == 5
        for i in range(5):
            assert segment.candidate_sequences[i].sequence == "ATCG"
            assert segment.candidate_sequences[i]._metadata["source"] == "original"

        # Check that candidate sequences are independent copies
        segment.candidate_sequences[0].sequence = "GGGG"
        segment.candidate_sequences[1]._metadata["source"] = "modified"
        assert segment.candidate_sequences[0].sequence == "GGGG"
        assert segment.candidate_sequences[1].sequence == "ATCG"
        assert segment.candidate_sequences[0]._metadata["source"] == "original"
        assert segment.candidate_sequences[1]._metadata["source"] == "modified"

    def test_iteration(self):
        """Tests iteration over the selected sequences in a segment."""
        segment = Segment(sequence="A")
        # Iteration is over selected_sequences, not candidates
        segment.selected_sequences.append(Sequence(sequence="T", sequence_type="dna"))
        segment.selected_sequences.append(Sequence(sequence="C", sequence_type="dna"))
        sequences = [s.sequence for s in segment]
        assert sequences == ["A", "T", "C"]

    def test_constant_segment_initialization(self):
        """Tests that constant segments are initialized correctly."""
        # Regular segment
        regular_segment = Segment(sequence="ATCG", sequence_type="dna")
        assert regular_segment.constant is False

        # Constant segment
        constant_segment = Segment(sequence="ATCG", sequence_type="dna", constant=True)
        assert constant_segment.constant is True
        assert constant_segment.selected_sequences[0].sequence == "ATCG"

    def test_empty_constant_segment_allowed(self):
        """Tests that constant segments can be created with length only (no sequence)."""
        # Empty constant segment (for multi-step optimization where it will be filled later)
        empty_constant = Segment(length=50, sequence_type="dna", constant=True)
        assert empty_constant.constant is True
        assert empty_constant.sequence_length == 50
        assert empty_constant.original_sequence.sequence == ""

    def test_ligand_constant(self):
        """Tests that ligand segments are automatically marked as constant."""
        with pytest.warns(UserWarning):  # Just check for any warning when constant is false.
            ligand_segment = Segment(sequence="CCC", sequence_type="ligand", constant=False)
        assert ligand_segment.constant is True
