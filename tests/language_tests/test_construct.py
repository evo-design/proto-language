import pytest

from proto_language.language.core import Sequence, Segment, Construct


class TestConstruct:
    """Tests for the Construct class that combines segments."""

    def test_concatenation(self):
        """Tests concatenation of single-sequence segments."""
        seg1 = Segment(sequence="ATG", sequence_type="dna")
        seg2 = Segment(sequence="CGC", sequence_type="dna")
        seg3 = Segment(sequence="TAA", sequence_type="dna")
        construct = Construct([seg1, seg2, seg3])

        final_sequences = construct.joined_sequences
        assert len(final_sequences) == 1
        assert final_sequences[0].sequence == "ATGCGC" + "TAA"

    def test_batched_concatenation(self):
        """Tests concatenation of segments with multiple selected sequences."""
        seg1 = Segment(sequence="A")
        seg1.selected_sequences.append(Sequence(sequence="G", sequence_type="dna"))

        seg2 = Segment(sequence="C")
        seg2.selected_sequences.append(Sequence(sequence="T", sequence_type="dna"))

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
        seg_dna = Segment(sequence="A", sequence_type="dna")
        seg_rna = Segment(sequence="U", sequence_type="rna")
        with pytest.raises(ValueError, match="must have the same sequence_type"):
            Construct([seg_dna, seg_rna])

    def test_metadata_concatenation(self):
        """Tests how metadata is merged during concatenation."""
        seg1 = Segment(sequence="A", metadata={"id": 1, "source": "seg1"})
        seg2 = Segment(sequence="C", metadata={"id": 2, "status": "new"})

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
        seg1 = Segment(sequence="A", valid_chars={"A", "B"})
        seg2 = Segment(sequence="C", valid_chars={"C", "D"})

        with pytest.raises(ValueError, match="must have the same valid_chars"):
            Construct([seg1, seg2])


class TestConstructValidation:
    """Tests for Construct._validate_construct checks."""

    # 1. Non-empty
    def test_empty_segments_raises(self):
        """Tests that empty segments list raises ValueError."""
        with pytest.raises(ValueError, match="must contain at least one segment"):
            Construct([])

    # 2. Homogeneous sequence types
    def test_mixed_sequence_types_raises(self):
        """Tests that mixed sequence types raise ValueError."""
        seg_dna = Segment(sequence="ATCG", sequence_type="dna")
        seg_rna = Segment(sequence="AUCG", sequence_type="rna")
        with pytest.raises(ValueError, match="must have the same sequence_type"):
            Construct([seg_dna, seg_rna])

    def test_mixed_sequence_types_shows_all_types(self):
        """Tests that error message shows all found types."""
        seg_dna = Segment(sequence="ATCG", sequence_type="dna")
        seg_protein = Segment(sequence="MAKT", sequence_type="protein")
        with pytest.raises(ValueError, match="dna.*protein|protein.*dna"):
            Construct([seg_dna, seg_protein])

    # 3. Homogeneous valid chars
    def test_mixed_valid_chars_raises(self):
        """Tests that mixed valid_chars raise ValueError."""
        seg1 = Segment(sequence="AB", valid_chars={"A", "B"})
        seg2 = Segment(sequence="CD", valid_chars={"C", "D"})
        with pytest.raises(ValueError, match="must have the same valid_chars"):
            Construct([seg1, seg2])

    # 4. Unique segment labels
    def test_duplicate_segment_labels_raises(self):
        """Tests that duplicate segment labels raise ValueError."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna", label="promoter")
        seg2 = Segment(sequence="GGGG", sequence_type="dna", label="promoter")
        with pytest.raises(ValueError, match="Segment labels must be unique.*promoter"):
            Construct([seg1, seg2])

    def test_multiple_duplicate_labels_shows_all(self):
        """Tests that error shows all duplicate labels."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna", label="dup1")
        seg2 = Segment(sequence="GGGG", sequence_type="dna", label="dup1")
        seg3 = Segment(sequence="CCCC", sequence_type="dna", label="dup2")
        seg4 = Segment(sequence="TTTT", sequence_type="dna", label="dup2")
        with pytest.raises(ValueError, match="Duplicates:.*dup"):
            Construct([seg1, seg2, seg3, seg4])

    def test_unlabeled_segments_get_auto_labels(self):
        """Tests that unlabeled segments get auto-assigned unique labels."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna")
        seg2 = Segment(sequence="GGGG", sequence_type="dna")
        Construct([seg1, seg2])
        # Auto-labeled as segment_0, segment_1
        assert seg1.label == "segment_0"
        assert seg2.label == "segment_1"

    def test_mixed_labeled_unlabeled_segments(self):
        """Tests that mix of labeled and unlabeled segments works."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna", label="promoter")
        seg2 = Segment(sequence="GGGG", sequence_type="dna")  # Will be auto-labeled
        Construct([seg1, seg2])  # Auto-labeling happens during construct creation
        assert seg1.label == "promoter"
        assert seg2.label == "segment_1"
