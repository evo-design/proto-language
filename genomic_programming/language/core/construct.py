"""
Construct class for the biological programming language.

Represents a full biological construct composed of multiple segments.
"""

from typing import List, Iterable

from . import Sequence, Segment


class Construct:
    """
    External class that represents a full biological construct. 
    Consists of multiple Segment objects that are concatenated together.

    Examples:
        Creating a construct from labeled segments:
        >>> promoter = Segment("TATA", SequenceType.DNA, label="promoter")
        >>> cds = Segment("ATGCCC", SequenceType.DNA, label="coding_region")
        >>> gene = Construct([promoter, cds])
        >>> gene.joined_sequences  # [Sequence("TATAATGCCC", SequenceType.DNA)]
    """

    def __init__(self, segments: Iterable[Segment]) -> None:
        """
        Initialize a Construct with Segment objects.

        Args:
            segments: An iterable of Segment objects in order.
        """
        # Convert to tuple for validation and storage
        self.segments = tuple(segments)
        self._validate_construct()

        self.sequence_type = self.segments[0].sequence_type
        self._valid_chars = self.segments[0]._valid_chars

        # Any unlabeled segments will be labeled as segment_i
        for i, segment in enumerate(self.segments):
            if segment.label is None:
                segment.label = f"segment_{i}"

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
            joined_seq = Sequence.from_sequences(
                subsequences=sequences_to_combine,
                merge_metadata=True
            )
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
        
        if not all(segment._valid_chars == self.segments[0]._valid_chars for segment in self.segments):
            raise ValueError("All segments in a construct must have the same valid_chars.")