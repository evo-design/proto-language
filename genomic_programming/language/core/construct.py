"""
Construct class for the proto-language.

Represents a full biological construct composed of multiple segments.
"""

from typing import Any, Iterable, Optional, Set, Tuple

from . import Sequence, Segment, SequenceType


class Construct:
    """
    External class that represents a full biological construct.

    Consists of multiple Segment objects that are concatenated together.

    Examples:
        Creating a construct from labeled segments:
        >>> promoter = Segment("TATA", SequenceType.DNA, label="promoter")
        >>> coding = Segment("ATGCCC", SequenceType.DNA, label="coding_region")
        >>> terminator = Segment("TTTT", SequenceType.DNA, label="terminator")
        >>> gene = Construct([promoter, coding, terminator])
        >>> gene.batch_sequences  # [Sequence("TATAATGCCCTTTT", SequenceType.DNA)]
    """

    def __init__(
        self,
        segments: Iterable[Segment],
    ) -> None:
        """
        Initialize a Construct with Segment objects.

        Args:
            segments: An iterable of Segment objects in order.

        Raises:
            ValueError: If construct contains no segments, segments have different
                sequence types, segments have different valid characters, or segments have different batch sizes.
        """
        # Convert to tuple for validation and storage
        self.segments: Tuple[Segment, ...] = tuple(segments)

        # Any unlabeled segments will be labeled as segment_i
        for i, segment in enumerate(self.segments):
            if segment.label is None:
                segment.label = f"segment_{i}"

        # Ensure segments are valid
        if not self.segments:
            raise ValueError("Construct must contain at least one segment")
        if not all(
            segment.sequence_type == self.segments[0].sequence_type
            for segment in self.segments
        ):
            all_types = set(segment.sequence_type for segment in self.segments)
            raise ValueError(
                f"All segments in a construct must have the same sequence_type. Found: {all_types}"
            )
        if not all(
            segment._valid_chars == self.segments[0]._valid_chars
            for segment in self.segments
        ):
            raise ValueError(
                "All segments in a construct must have the same valid_chars."
            )
        
        # Ensure consistent batch sizes across all segments
        batch_sizes = [len(segment.batch_sequences) for segment in self.segments]
        if not all(size == batch_sizes[0] for size in batch_sizes):
            raise ValueError(
                f"Inconsistent batch sizes across construct segments. Found: {batch_sizes}. "
                f"All segments must have the same batch size."
            )

        self.sequence_type: SequenceType = self.segments[0].sequence_type
        self._valid_chars: Optional[Set[str]] = self.segments[0]._valid_chars

    @property
    def batch_sequences(self) -> Tuple[Sequence, ...]:
        """
        Get the joined Sequence objects batch that represent one user-facing Construct.

        Returns:
            Tuple of joined Sequence objects where each element represents
            the joining of the i-th sequence from each segment in order.
        """
        # Join corresponding sequences from each segment with metadata propagation
        # Example: [Seq("AAA"), Seq("TTT"), Seq("GGG")] → [Sequence("AAATTTGGG")]
        joined_sequences = []
        batch_size = self.segments[0].batch_size
        
        for batch_position in range(batch_size):
            sequences_to_combine = [segment.batch_sequences[batch_position] for segment in self.segments]
            joined_seq = Sequence.from_sequences(
                subsequences=sequences_to_combine,
                merge_metadata=True
            )
                
            joined_sequences.append(joined_seq)
            
        return tuple(joined_sequences)

