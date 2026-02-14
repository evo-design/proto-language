from proto_language.language.core import Segment, Sequence


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

        segment = Segment(
            sequence="ATCG", sequence_type="dna", metadata={"source": "original"}
        )

        # Directly set candidate sequences (like optimizer does)
        segment.candidate_sequences = [
            copy.deepcopy(segment.original_sequence) for _ in range(5)
        ]
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

    def test_candidates_populated_checks_all(self):
        """Regression: candidates_populated must check ALL candidates (Bug 5)."""
        segment = Segment(sequence="ATCG", sequence_type="dna")

        # One populated, one empty — should be False
        segment.candidate_sequences = [
            Sequence(sequence="ATCG", sequence_type="dna"),
            Sequence(sequence="", sequence_type="dna"),
        ]
        assert not segment.candidates_populated

        # All populated — should be True
        segment.candidate_sequences = [
            Sequence(sequence="ATCG", sequence_type="dna"),
            Sequence(sequence="GCTA", sequence_type="dna"),
        ]
        assert segment.candidates_populated

        # All empty — should be False
        segment.candidate_sequences = [
            Sequence(sequence="", sequence_type="dna"),
            Sequence(sequence="", sequence_type="dna"),
        ]
        assert not segment.candidates_populated

    def test_is_ligand_property(self):
        """Tests that is_ligand correctly identifies ligand segments."""
        # DNA segment
        dna_segment = Segment(sequence="ATCG", sequence_type="dna")
        assert dna_segment.is_ligand is False

        # Ligand segment
        ligand_segment = Segment(sequence="CCC", sequence_type="ligand")
        assert ligand_segment.is_ligand is True
