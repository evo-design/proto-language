import pytest

from proto_language.language.core import Sequence, Segment, Construct, SequenceType


class TestSegment:
    """Tests for the Segment class."""

    def test_initialization(self):
        """Tests that a segment is initialized with a single selected sequence."""
        segment = Segment(starting_sequence_or_desired_length="ATCG", sequence_type=SequenceType.DNA)
        assert isinstance(segment, Segment)
        assert len(segment.selected_sequences) == 1
        assert segment.num_selected == 1
        assert segment[0].sequence == "ATCG"
        assert segment.sequence_type == SequenceType.DNA
        assert segment.sequence_length == 4

    def test_create_batch(self):
        """Tests that create_candidates replicates the initial sequence."""
        segment = Segment(starting_sequence_or_desired_length="ATCG", sequence_type=SequenceType.DNA, metadata={"source": "original"})
        segment.create_candidates(5)
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
        segment = Segment(starting_sequence_or_desired_length="A")
        # Iteration is over selected_sequences, not candidates
        segment.selected_sequences.append(Sequence(sequence="T", sequence_type=SequenceType.DNA))
        segment.selected_sequences.append(Sequence(sequence="C", sequence_type=SequenceType.DNA))
        sequences = [s.sequence for s in segment]
        assert sequences == ["A", "T", "C"]

    def test_constant_segment_initialization(self):
        """Tests that constant segments are initialized correctly."""
        # Regular segment
        regular_segment = Segment(starting_sequence_or_desired_length="ATCG", sequence_type=SequenceType.DNA)
        assert regular_segment.constant is False
        assert regular_segment._is_assigned is False
        
        # Constant segment
        constant_segment = Segment(starting_sequence_or_desired_length="ATCG", sequence_type=SequenceType.DNA, constant=True)
        assert constant_segment.constant is True
        assert constant_segment._is_assigned is True  # Constant segments should be pre-assigned
        assert constant_segment.selected_sequences[0].sequence == "ATCG"
