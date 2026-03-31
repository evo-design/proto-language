"""
proto_language/language/core/construct.py

Represents a full biological construct composed of multiple segments.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from . import Segment, Sequence

logger = logging.getLogger(__name__)
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
            segments (Iterable[Segment]): An iterable of Segment objects in order.
            label (str | None): Optional label for this construct (e.g., "plasmid", "insert").
        """
        # Convert to tuple for validation and storage
        self.segments = tuple(segments)
        self.label = label

        # Any unlabeled segments will be labeled as segment_i
        for i, segment in enumerate(self.segments):
            if segment.label is None:
                segment.label = f"segment_{i}"
        self._validate_construct()
        segment_labels = [s.label for s in self.segments]
        logger.debug(f"Created Construct: label={label}, segments={segment_labels}")

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
        Get the joined Sequence objects from result pools (user-facing results).
        Joins corresponding sequences from each segment's result_sequences.
        Includes segment metadata nested under _metadata["segments"][segment_label].

        Example:
            >>> construct.segment1.result_sequences = [Seq("AAA"), Seq("TTT")]
            >>> construct.segment2.result_sequences = [Seq("CCC"), Seq("GGG")]
            >>> construct.joined_sequences  # [Sequence("AAACCC"), Sequence("TTTGGG")]
        """
        joined_sequences = []
        segment_labels = [seg.label for seg in self.segments]

        pool_sizes = [len(seg.result_sequences) for seg in self.segments]
        if len(set(pool_sizes)) > 1:
            raise RuntimeError(
                f"Cannot join sequences: segments have mismatched result_sequences lengths: "
                f"{dict(zip(segment_labels, pool_sizes))}"
            )

        for sequences_to_combine in zip(*[segment.result_sequences for segment in self.segments]):
            joined_seq = create_concatenated_sequence(sequences_to_combine, segment_labels)
            joined_sequences.append(joined_seq)

        return joined_sequences

    def _validate_construct(self) -> None:
        """
        Validate construct configuration.

        Checks:
            1. Non-empty: Construct must contain at least one segment.
            2. Homogeneous types: All segments must share the same sequence_type.
            3. Homogeneous chars: All segments must share the same valid_chars.
            4. Unique labels: Segment labels must be unique within this construct.

        Raises:
            ValueError: If any validation check fails.
        """
        # 1. Non-empty
        if not self.segments:
            raise ValueError("Construct must contain at least one segment")

        # 2. Homogeneous sequence types
        types = {seg.sequence_type for seg in self.segments}
        if len(types) > 1:
            raise ValueError(f"All segments must have the same sequence_type. Found: {types}")

        # 3. Homogeneous valid chars
        if not all(seg.valid_chars == self.segments[0].valid_chars for seg in self.segments):
            raise ValueError("All segments must have the same valid_chars")

        # Validate segment labels are unique within this construct
        segment_labels = [s.label for s in self.segments]
        if len(segment_labels) != len(set(segment_labels)):
            duplicates = [l for l in segment_labels if segment_labels.count(l) > 1]
            raise ValueError(f"Segment labels must be unique within a construct. Duplicates: {set(duplicates)}")

    def to_dict(self) -> dict:
        """Serialize Construct to dictionary for cloud/API communication."""
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "sequence_type": self.sequence_type,
            "valid_chars": list(self.valid_chars) if self.valid_chars else None,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data) -> Construct:
        """Deserialize Construct from dictionary."""
        segments = [Segment.from_dict(seg_data) for seg_data in data["segments"]]
        return cls(segments=segments, label=data.get("label"))
