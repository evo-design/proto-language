"""tests/language_tests/test_segment.py."""

import pytest

from proto_language.language.core import Segment, Sequence


class TestSegment:
    """Tests for the Segment class."""

    def test_initialization(self):
        """Tests that a segment is initialized with a single result sequence."""
        segment = Segment(sequence="ATCG", sequence_type="dna")
        assert isinstance(segment, Segment)
        assert len(segment.result_sequences) == 1
        assert segment.num_results == 1
        assert segment[0].sequence == "ATCG"
        assert segment.sequence_type == "dna"
        assert segment.sequence_length == 4

    def test_proposal_sequences_manipulation(self):
        """Tests that proposal_sequences can be directly manipulated."""
        import copy

        segment = Segment(sequence="ATCG", sequence_type="dna", metadata={"source": "original"})

        # Directly set proposal sequences (like optimizer does)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        assert segment.num_proposals == 5
        for i in range(5):
            assert segment.proposal_sequences[i].sequence == "ATCG"
            assert segment.proposal_sequences[i]._metadata["source"] == "original"

        # Check that proposal sequences are independent copies
        segment.proposal_sequences[0].sequence = "GGGG"
        segment.proposal_sequences[1]._metadata["source"] = "modified"
        assert segment.proposal_sequences[0].sequence == "GGGG"
        assert segment.proposal_sequences[1].sequence == "ATCG"
        assert segment.proposal_sequences[0]._metadata["source"] == "original"
        assert segment.proposal_sequences[1]._metadata["source"] == "modified"

    def test_iteration(self):
        """Tests iteration over the result sequences in a segment."""
        segment = Segment(sequence="A")
        # Iteration is over result_sequences, not proposals
        segment.result_sequences.append(Sequence(sequence="T", sequence_type="dna"))
        segment.result_sequences.append(Sequence(sequence="C", sequence_type="dna"))
        sequences = [s.sequence for s in segment]
        assert sequences == ["A", "T", "C"]

    def test_has_sequence_property(self):
        """Tests that has_sequence correctly identifies segments with input sequences."""
        # Segment with a sequence
        segment_with_seq = Segment(sequence="ATCG", sequence_type="dna")
        assert segment_with_seq.has_original_sequence is True

        # Segment with just length (no sequence)
        segment_without_seq = Segment(length=50, sequence_type="dna")
        assert segment_without_seq.has_original_sequence is False
        assert segment_without_seq.sequence_length == 50
        assert segment_without_seq.original_sequence.sequence == ""

    def test_proposals_populated_checks_all(self):
        """Regression: proposals_populated must check ALL proposals (Bug 5)."""
        segment = Segment(sequence="ATCG", sequence_type="dna")

        # One populated, one empty: should be False
        segment.proposal_sequences = [
            Sequence(sequence="ATCG", sequence_type="dna"),
            Sequence(sequence="", sequence_type="dna"),
        ]
        assert not segment.proposals_populated

        # All populated: should be True
        segment.proposal_sequences = [
            Sequence(sequence="ATCG", sequence_type="dna"),
            Sequence(sequence="GCTA", sequence_type="dna"),
        ]
        assert segment.proposals_populated

        # All empty: should be False
        segment.proposal_sequences = [
            Sequence(sequence="", sequence_type="dna"),
            Sequence(sequence="", sequence_type="dna"),
        ]
        assert not segment.proposals_populated

    def test_zero_length_raises_error(self):
        """Test that Segment(length=0) raises ValueError."""
        with pytest.raises(ValueError, match="Segment length must be positive"):
            Segment(length=0, sequence_type="dna")

    def test_negative_length_raises_error(self):
        """Test that Segment(length=-1) raises ValueError."""
        with pytest.raises(ValueError, match="Segment length must be positive"):
            Segment(length=-1, sequence_type="dna")

    def test_is_ligand_property(self):
        """Tests that is_ligand correctly identifies ligand segments."""
        # DNA segment
        dna_segment = Segment(sequence="ATCG", sequence_type="dna")
        assert dna_segment.is_ligand is False

        # Ligand segment
        ligand_segment = Segment(sequence="CCC", sequence_type="ligand")
        assert ligand_segment.is_ligand is True
