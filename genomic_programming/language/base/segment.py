"""
Segment class for the proto-language.

Represents building blocks for biological constructs.
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Union
import copy

from .sequence import Sequence, SequenceType


class Segment:
    """
    External class that represents the building blocks for a Construct.

    This is the most modular user-facing unit for the programming language.

    Examples:
        Creating a Segment:
        >>> promoter = Segment(sequence="TATA", sequence_type=SequenceType.DNA, label="promoter")
        >>> promoter.label  # "promoter"
    """

    def __init__(
        self,
        sequence: str = "",
        sequence_type: Optional[Union[SequenceType, str]] = SequenceType.DNA,
        valid_chars: Optional[Set[str]] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a Segment with a single sequence.

        Args:
            sequence: The biological sequence string. Defaults to empty string.
            sequence_type: Type of biological sequence (DNA, RNA, or PROTEIN). Defaults to DNA.
            valid_chars: Optional custom set of valid characters for sequence validation.
            metadata: Additional data associated with this sequence.
            label: Optional label for this segment (e.g., "promoter", "coding_region").
        """
        seq = Sequence(
            sequence=sequence,
            sequence_type=sequence_type,
            metadata=metadata,
            valid_chars=valid_chars,
        )
        self.batch_sequences: List[Sequence] = [seq]
        self.sequence_type: SequenceType = SequenceType(seq.sequence_type)
        self._valid_chars: Optional[Set[str]] = seq._valid_chars
        self._is_assigned: bool = False
        self.label: Optional[str] = label

    def create_batch(self, batch_size: int) -> None:
        """
        Set the batch size by replicating the first sequence across the batch.

        Args:
            batch_size: The desired batch size.
        """
        self.batch_sequences = [
            copy.deepcopy(self.batch_sequences[0]) for _ in range(batch_size)
        ]

    @property
    def batch_size(self) -> int:
        """
        Get the batch size of the Segment.

        Returns:
            Number of Sequence objects in the Segment.
        """
        return len(self.batch_sequences)

    def __iter__(self) -> Iterator[Sequence]:
        """
        Iterate over all Sequence objects in the Segment.

        Returns:
            Iterator over Sequence objects.
        """
        return iter(self.batch_sequences)

    def __getitem__(self, index: int) -> Sequence:
        """
        Get a specific sequence from the batch by index.

        Args:
            index: The index of the sequence to retrieve.

        Returns:
            The Sequence object at the specified index.
        """
        return self.batch_sequences[index]
