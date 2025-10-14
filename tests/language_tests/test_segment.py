import pytest
import sys

sys.path.append(".")
from proto_language.language.core import Sequence, Segment, Construct, SequenceType


class TestSegment:
    """Tests for the Segment class."""

    def test_initialization(self):
        """Tests that a segment is initialized with a single sequence in a list."""
        segment = Segment("ATCG", SequenceType.DNA)
        assert isinstance(segment, Segment)
        assert len(segment.batch_sequences) == 1
        assert segment.batch_size == 1
        assert segment[0].sequence == "ATCG"
        assert segment.sequence_type == SequenceType.DNA

    def test_create_batch(self):
        """Tests that create_batch replicates the initial sequence."""
        segment = Segment("ATCG", SequenceType.DNA, metadata={"source": "original"})
        segment.create_batch(5)
        assert segment.batch_size == 5
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
        segment = Segment("A")
        segment.create_batch(3)
        segment[1].sequence = "T"
        segment[2].sequence = "C"
        sequences = [s.sequence for s in segment]
        assert sequences == ["A", "T", "C"]
