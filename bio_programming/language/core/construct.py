"""
Construct class for the biological programming language.

Represents a full biological construct composed of multiple segments.
"""
from __future__ import annotations
from typing import List, Iterable, Optional

from . import Sequence, Segment
from .sequence import create_concatenated_sequence


class Construct:
    """
    External class that represents a full biological construct. 
    Consists of multiple Segment objects that are concatenated together.

    Examples:
        Creating a construct from labeled segments:
        >>> promoter = Segment(sequence="TATA", sequence_type="dna", label="promoter")
        >>> cds = Segment(sequence="ATGCCC", sequence_type="dna", label="coding_region")
        >>> gene = Construct([promoter, cds], label="my_gene")
        >>> gene.joined_sequences  # [Sequence("TATAATGCCC", "dna")]
    """

    def __init__(self, segments: Iterable[Segment], label: Optional[str] = None) -> None:
        """
        Initialize a Construct with Segment objects.

        Args:
            segments: An iterable of Segment objects in order.
            label: Optional label for this construct (e.g., "plasmid", "insert").
        """
        # Convert to tuple for validation and storage
        self.segments = tuple(segments)
        self._validate_construct()

        self.label = label

        # Any unlabeled segments will be labeled as segment_i
        for i, segment in enumerate(self.segments):
            if segment.label is None:
                segment.label = f"segment_{i}"

    @property
    def sequence_type(self):
        """Sequence type derived from segments (read-only)."""
        return self.segments[0].sequence_type

    @property
    def valid_chars(self):
        """Valid characters derived from segments (read-only)."""
        return self.segments[0].valid_chars

    @property
    def joined_sequences(self) -> List[Sequence]:
        """
        Get the joined Sequence objects from selected pools (user-facing results).
        Joins corresponding sequences from each segment's selected_sequences.

        Example:
            >>> construct.segment1.selected_sequences = [Seq("AAA"), Seq("TTT")]
            >>> construct.segment2.selected_sequences = [Seq("CCC"), Seq("GGG")]
            >>> construct.joined_sequences  # [Sequence("AAACCC"), Sequence("TTTGGG")]
        """
        joined_sequences = []

        for sequences_to_combine in zip(*[segment.selected_sequences for segment in self.segments]):
            joined_seq = create_concatenated_sequence(sequences_to_combine, merge_metadata=True)
            joined_sequences.append(joined_seq)

        return joined_sequences

    def _validate_construct(self) -> None:
        """
        Validate that all segments in the construct are compatible.

        Raises:
            ValueError: If construct contains no segments, segments have different
                sequence types, segments have different valid characters, or segments
                have inconsistent selected pool sizes.
        """
        if not self.segments:
            raise ValueError("Construct must contain at least one segment")
        
        if not all(segment.sequence_type == self.segments[0].sequence_type for segment in self.segments):
            all_types = set(segment.sequence_type for segment in self.segments)
            raise ValueError(f"All segments in a construct must have the same sequence_type. Found: {all_types}")
        
        if not all(segment.valid_chars == self.segments[0].valid_chars for segment in self.segments):
            raise ValueError("All segments in a construct must have the same valid_chars.")

    def to_dict(self) -> dict:
        """Serialize Construct to dictionary for cloud/API communication."""
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "sequence_type": self.sequence_type,
            "valid_chars": list(self.valid_chars) if self.valid_chars else None,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data) -> "Construct":
        """Deserialize Construct from dictionary."""
        segments = [Segment.from_dict(seg_data) for seg_data in data["segments"]]
        return cls(segments=segments, label=data.get("label"))
