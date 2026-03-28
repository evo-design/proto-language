"""
tests/language_tests/test_serialization.py

Tests roundtrip serialization (to_dict -> from_dict) for Sequence, Segment, and Construct.
"""

# import warnings

from proto_language.language.core import Construct, Segment, Sequence
from proto_language.language.core.sequence import _DEFAULT_DNA_CHARS


class TestSequenceSerialization:
    """Test Sequence serialization and deserialization."""

    def test_basic_sequence_roundtrip(self):
        """Test basic DNA sequence serialization roundtrip."""
        # Create sequence
        seq = Sequence(sequence="ATCGATCG", sequence_type="dna")

        # Serialize
        seq_dict = seq.to_dict()

        # Verify dict structure
        assert "sequence" in seq_dict
        assert "sequence_type" in seq_dict
        assert "metadata" in seq_dict
        assert seq_dict["sequence"] == "ATCGATCG"
        assert seq_dict["sequence_type"] == "dna"

        # Deserialize
        seq_restored = Sequence.from_dict(seq_dict)

        # Verify roundtrip
        assert str(seq_restored) == str(seq)
        assert seq_restored.sequence_type == seq.sequence_type
        assert len(seq_restored) == len(seq)

    def test_sequence_with_metadata_roundtrip(self):
        """Test sequence with metadata serialization."""
        # Create sequence with metadata
        metadata = {"gc_content": 0.5, "custom_field": "test_value"}
        seq = Sequence(sequence="ATCGATCG", sequence_type="dna", metadata=metadata)

        # Serialize and deserialize
        seq_dict = seq.to_dict()
        seq_restored = Sequence.from_dict(seq_dict)

        # Verify metadata (excluding system fields)
        assert "gc_content" in seq_restored.metadata
        assert seq_restored.metadata["gc_content"] == 0.5
        assert "custom_field" in seq_restored.metadata
        assert seq_restored.metadata["custom_field"] == "test_value"

    def test_protein_sequence_roundtrip(self):
        """Test protein sequence serialization."""
        seq = Sequence(sequence="ACDEFGHIKLMNPQRSTVWY", sequence_type="protein")

        seq_dict = seq.to_dict()
        seq_restored = Sequence.from_dict(seq_dict)

        assert str(seq_restored) == str(seq)
        assert seq_restored.sequence_type == "protein"

    def test_rna_sequence_roundtrip(self):
        """Test RNA sequence serialization."""
        seq = Sequence(sequence="AUCGAUCG", sequence_type="rna")

        seq_dict = seq.to_dict()
        seq_restored = Sequence.from_dict(seq_dict)

        assert str(seq_restored) == str(seq)
        assert seq_restored.sequence_type == "rna"


class TestSegmentSerialization:
    """Test Segment serialization and deserialization."""

    def test_basic_segment_roundtrip(self):
        """Test basic segment serialization."""
        seg = Segment(sequence="ATCGATCG", sequence_type="dna", label="promoter")

        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        assert seg_restored.label == seg.label
        assert str(seg_restored.original_sequence) == str(seg.original_sequence)
        assert seg_restored.sequence_type == seg.sequence_type
        assert seg_restored.has_original_sequence == seg.has_original_sequence

    def test_segment_with_pools_roundtrip(self):
        """Test segment with proposal and result pools."""
        seg = Segment(sequence="ATCGATCG", sequence_type="dna", label="cds")

        # Modify pools
        seg.proposal_sequences = [
            Sequence("AAAAAAAA", sequence_type="dna"),
            Sequence("TTTTTTTT", sequence_type="dna"),
        ]
        seg.result_sequences = [
            Sequence("GGGGGGGG", sequence_type="dna"),
        ]

        # Serialize and deserialize
        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        # Verify pools
        assert len(seg_restored.proposal_sequences) == 2
        assert len(seg_restored.result_sequences) == 1
        assert str(seg_restored.proposal_sequences[0]) == "AAAAAAAA"
        assert str(seg_restored.proposal_sequences[1]) == "TTTTTTTT"
        assert str(seg_restored.result_sequences[0]) == "GGGGGGGG"

    def test_segment_with_sequence_roundtrip(self):
        """Test segment with sequence has has_sequence=True after roundtrip."""
        seg = Segment(
            sequence="ATATCG",
            sequence_type="dna",
            label="promoter",
        )

        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        assert seg_restored.has_original_sequence is True
        assert str(seg_restored.original_sequence) == "ATATCG"

    def test_segment_with_length_only_roundtrip(self):
        """Test segment with length only has has_sequence=False after roundtrip."""
        seg = Segment(length=50, sequence_type="dna", label="variable")

        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        assert seg_restored.has_original_sequence is False
        assert seg_restored.sequence_length == 50

    def test_segment_with_metadata_roundtrip(self):
        """Test segment with sequence metadata."""
        metadata = {"annotation": "strong_promoter"}
        seg = Segment(
            sequence="ATATCG", sequence_type="dna", label="promoter", metadata=metadata
        )

        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        assert "annotation" in seg_restored.original_sequence.metadata
        assert (
            seg_restored.original_sequence.metadata["annotation"] == "strong_promoter"
        )

    def test_ligand_segment_roundtrip(self):
        """Test ligand segment serialization doesn't crash on valid_chars=None (B1)."""
        seg = Segment(sequence="CCO", sequence_type="ligand", label="ethanol")

        seg_dict = seg.to_dict()
        # This would crash with set(None) TypeError before the fix
        seg_restored = Segment.from_dict(seg_dict)

        assert str(seg_restored.original_sequence) == "CCO"
        assert seg_restored.sequence_type == "ligand"
        assert seg_restored.valid_chars is None
        assert seg_restored.label == "ethanol"
        assert len(seg_restored.proposal_sequences) == 1
        assert len(seg_restored.result_sequences) == 1

    def test_ligand_sequence_roundtrip(self):
        """Test ligand Sequence roundtrip at the Sequence level (B1)."""
        seq = Sequence(sequence="CCO", sequence_type="ligand")

        seq_dict = seq.to_dict()
        seq_restored = Sequence.from_dict(seq_dict)

        assert seq_restored.sequence == "CCO"
        assert seq_restored.sequence_type == "ligand"
        assert seq_restored.valid_chars is None

    # def test_segment_roundtrip_no_warnings(self):
    #     """Test that Segment.from_dict doesn't emit spurious UserWarnings (B2)."""
    #     seg = Segment(
    #         sequence="ATATCG", sequence_type="dna", label="promoter",
    #         metadata={"custom_key": "custom_value"},
    #     )
    #     seg_dict = seg.to_dict()

    #     with warnings.catch_warnings(record=True) as caught:
    #         warnings.simplefilter("always")
    #         Segment.from_dict(seg_dict)

    #     user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    #     assert len(user_warnings) == 0, (
    #         f"Unexpected UserWarnings during deserialization: "
    #         f"{[str(w.message) for w in user_warnings]}"
    #     )

    def test_valid_chars_preserved_as_frozenset_after_roundtrip(self):
        """Test that valid_chars reuses shared frozenset defaults after roundtrip (B3)."""
        seq = Sequence(sequence="ATCG", sequence_type="dna")

        seq_dict = seq.to_dict()
        seq_restored = Sequence.from_dict(seq_dict)

        # Should reuse the shared module-level frozenset, not allocate a new set
        assert seq_restored._valid_chars is _DEFAULT_DNA_CHARS


class TestConstructSerialization:
    """Test Construct serialization and deserialization."""

    def test_basic_construct_roundtrip(self):
        """Test basic construct serialization."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna", label="promoter")
        seg2 = Segment(sequence="GGGG", sequence_type="dna", label="cds")

        construct = Construct([seg1, seg2])

        # Serialize and deserialize
        construct_dict = construct.to_dict()
        construct_restored = Construct.from_dict(construct_dict)

        # Verify structure
        assert len(construct_restored.segments) == 2
        assert construct_restored.segments[0].label == "promoter"
        assert construct_restored.segments[1].label == "cds"
        assert str(construct_restored.segments[0].original_sequence) == "ATCG"
        assert str(construct_restored.segments[1].original_sequence) == "GGGG"

    def test_construct_with_multiple_segments_roundtrip(self):
        """Test construct with multiple segments."""
        segments = [
            Segment(sequence="AAAA", sequence_type="dna", label="promoter"),
            Segment(sequence="TTTT", sequence_type="dna", label="five_utr"),
            Segment(sequence="GGGG", sequence_type="dna", label="cds"),
            Segment(sequence="CCCC", sequence_type="dna", label="terminator"),
        ]

        construct = Construct(segments)

        construct_dict = construct.to_dict()
        construct_restored = Construct.from_dict(construct_dict)

        assert len(construct_restored.segments) == 4
        assert [seg.label for seg in construct_restored.segments] == [
            "promoter",
            "five_utr",
            "cds",
            "terminator",
        ]

    def test_construct_joined_sequences_after_roundtrip(self):
        """Test that joined_sequences works after deserialization."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna", label="seg1")
        seg2 = Segment(sequence="GGGG", sequence_type="dna", label="seg2")

        construct = Construct([seg1, seg2])

        # Serialize and deserialize
        construct_dict = construct.to_dict()
        construct_restored = Construct.from_dict(construct_dict)

        # Verify joined_sequences works
        joined = construct_restored.joined_sequences
        assert len(joined) == 1
        assert str(joined[0]) == "ATCGGGGG"

    def test_construct_with_mixed_segments_roundtrip(self):
        """Test construct with segments that have/don't have sequences."""
        seg1 = Segment(sequence="ATATCG", sequence_type="dna", label="promoter")
        seg2 = Segment(length=50, sequence_type="dna", label="variable")

        construct = Construct([seg1, seg2])

        construct_dict = construct.to_dict()
        construct_restored = Construct.from_dict(construct_dict)

        assert construct_restored.segments[0].has_original_sequence is True
        assert construct_restored.segments[1].has_original_sequence is False

    def test_construct_label_roundtrip(self):
        """Test that construct labels are preserved during serialization."""
        seg1 = Segment(sequence="ATCG", sequence_type="dna", label="promoter")
        seg2 = Segment(sequence="GGGG", sequence_type="dna", label="cds")

        # Test with explicit label
        construct_with_label = Construct([seg1, seg2], label="plasmid")
        construct_dict = construct_with_label.to_dict()
        construct_restored = Construct.from_dict(construct_dict)
        assert construct_restored.label == "plasmid"

        # Test without label (should be None)
        construct_without_label = Construct([seg1, seg2])
        construct_dict = construct_without_label.to_dict()
        construct_restored = Construct.from_dict(construct_dict)
        assert construct_restored.label is None
