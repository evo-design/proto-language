"""
Tests for serialization/deserialization of core language objects.

Tests roundtrip serialization (to_dict -> from_dict) for Sequence, Segment, and Construct.
"""

import pytest
from proto_language.language.core import Sequence, Segment, Construct, SequenceType


class TestSequenceSerialization:
    """Test Sequence serialization and deserialization."""

    def test_basic_sequence_roundtrip(self):
        """Test basic DNA sequence serialization roundtrip."""
        # Create sequence
        seq = Sequence(sequence="ATCGATCG", sequence_type=SequenceType.DNA)

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
        seq = Sequence(
            sequence="ATCGATCG",
            sequence_type=SequenceType.DNA,
            metadata=metadata
        )

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
        seq = Sequence(sequence="ACDEFGHIKLMNPQRSTVWY", sequence_type=SequenceType.PROTEIN)

        seq_dict = seq.to_dict()
        seq_restored = Sequence.from_dict(seq_dict)

        assert str(seq_restored) == str(seq)
        assert seq_restored.sequence_type == SequenceType.PROTEIN

    def test_rna_sequence_roundtrip(self):
        """Test RNA sequence serialization."""
        seq = Sequence(sequence="AUCGAUCG", sequence_type=SequenceType.RNA)

        seq_dict = seq.to_dict()
        seq_restored = Sequence.from_dict(seq_dict)

        assert str(seq_restored) == str(seq)
        assert seq_restored.sequence_type == SequenceType.RNA


class TestSegmentSerialization:
    """Test Segment serialization and deserialization."""

    def test_basic_segment_roundtrip(self):
        """Test basic segment serialization."""
        seg = Segment(starting_sequence_or_desired_length="ATCGATCG", sequence_type=SequenceType.DNA, label="promoter")

        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        assert seg_restored.label == seg.label
        assert str(seg_restored.original_sequence) == str(seg.original_sequence)
        assert seg_restored.sequence_type == seg.sequence_type
        assert seg_restored.constant == seg.constant

    def test_segment_with_pools_roundtrip(self):
        """Test segment with candidate and selected pools."""
        seg = Segment(starting_sequence_or_desired_length="ATCGATCG", sequence_type=SequenceType.DNA, label="cds")

        # Modify pools
        seg.candidate_sequences = [
            Sequence("AAAAAAAA", sequence_type=SequenceType.DNA),
            Sequence("TTTTTTTT", sequence_type=SequenceType.DNA),
        ]
        seg.selected_sequences = [
            Sequence("GGGGGGGG", sequence_type=SequenceType.DNA),
        ]

        # Serialize and deserialize
        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        # Verify pools
        assert len(seg_restored.candidate_sequences) == 2
        assert len(seg_restored.selected_sequences) == 1
        assert str(seg_restored.candidate_sequences[0]) == "AAAAAAAA"
        assert str(seg_restored.candidate_sequences[1]) == "TTTTTTTT"
        assert str(seg_restored.selected_sequences[0]) == "GGGGGGGG"

    def test_constant_segment_roundtrip(self):
        """Test constant segment serialization."""
        seg = Segment(
            starting_sequence_or_desired_length="ATATCG",
            sequence_type=SequenceType.DNA,
            label="promoter",
            constant=True
        )

        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        assert seg_restored.constant == True
        assert seg_restored._is_assigned == True
        assert str(seg_restored.original_sequence) == "ATATCG"

    def test_segment_with_metadata_roundtrip(self):
        """Test segment with sequence metadata."""
        metadata = {"annotation": "strong_promoter"}
        seg = Segment(
            starting_sequence_or_desired_length="ATATCG",
            sequence_type=SequenceType.DNA,
            label="promoter",
            metadata=metadata
        )

        seg_dict = seg.to_dict()
        seg_restored = Segment.from_dict(seg_dict)

        assert "annotation" in seg_restored.original_sequence.metadata
        assert seg_restored.original_sequence.metadata["annotation"] == "strong_promoter"


class TestConstructSerialization:
    """Test Construct serialization and deserialization."""

    def test_basic_construct_roundtrip(self):
        """Test basic construct serialization."""
        seg1 = Segment(starting_sequence_or_desired_length="ATCG", sequence_type=SequenceType.DNA, label="promoter")
        seg2 = Segment(starting_sequence_or_desired_length="GGGG", sequence_type=SequenceType.DNA, label="cds")

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
            Segment(starting_sequence_or_desired_length="AAAA", sequence_type=SequenceType.DNA, label="promoter"),
            Segment(starting_sequence_or_desired_length="TTTT", sequence_type=SequenceType.DNA, label="five_utr"),
            Segment(starting_sequence_or_desired_length="GGGG", sequence_type=SequenceType.DNA, label="cds"),
            Segment(starting_sequence_or_desired_length="CCCC", sequence_type=SequenceType.DNA, label="terminator"),
        ]

        construct = Construct(segments)

        construct_dict = construct.to_dict()
        construct_restored = Construct.from_dict(construct_dict)

        assert len(construct_restored.segments) == 4
        assert [seg.label for seg in construct_restored.segments] == \
               ["promoter", "five_utr", "cds", "terminator"]

    def test_construct_joined_sequences_after_roundtrip(self):
        """Test that joined_sequences works after deserialization."""
        seg1 = Segment(starting_sequence_or_desired_length="ATCG", sequence_type=SequenceType.DNA, label="seg1")
        seg2 = Segment(starting_sequence_or_desired_length="GGGG", sequence_type=SequenceType.DNA, label="seg2")

        construct = Construct([seg1, seg2])

        # Serialize and deserialize
        construct_dict = construct.to_dict()
        construct_restored = Construct.from_dict(construct_dict)

        # Verify joined_sequences works
        joined = construct_restored.joined_sequences
        assert len(joined) == 1
        assert str(joined[0]) == "ATCGGGGG"


    def test_construct_with_constant_segment_roundtrip(self):
        """Test construct with constant segment."""
        seg1 = Segment(starting_sequence_or_desired_length="ATATCG", sequence_type=SequenceType.DNA, label="promoter", constant=True)
        seg2 = Segment(starting_sequence_or_desired_length="ATATCG", sequence_type=SequenceType.DNA, label="cds", constant=False)

        construct = Construct([seg1, seg2])

        construct_dict = construct.to_dict()
        construct_restored = Construct.from_dict(construct_dict)

        assert construct_restored.segments[0].constant == True
        assert construct_restored.segments[1].constant == False
