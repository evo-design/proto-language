"""
Segment class for the proto-language.

Represents building blocks for biological constructs.
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Union
import copy
from .sequence import Sequence, SequenceType


class Segment:
    """
    Building block for biological constructs with two sequence pools: candidate (work space) and selected (results space):
    - candidate_sequences: Working space for optimizer proposals (mutations, offspring, rollouts)
    - selected_sequences: Results space containing current best sequences (user-facing)

    Examples:
        Creating a Segment:
        >>> promoter = Segment(sequence="TATA", sequence_type=SequenceType.DNA, label="promoter")
        >>> promoter.label  # "promoter"
        >>> promoter.selected_sequences  # [Sequence("TATA")]
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
        Initialize a Segment with dual sequence pools.

        Args:
            sequence: The biological sequence string. Defaults to empty string.
            sequence_type: Type of biological sequence (DNA, RNA, or PROTEIN). Defaults to DNA.
            valid_chars: Optional custom set of valid characters for sequence validation.
            label: Optional label for this segment (e.g., "promoter", "coding_region").
            metadata: Additional data associated with this sequence.
        """
        seq = Sequence(
            sequence=sequence,
            sequence_type=sequence_type,
            metadata=metadata,
            valid_chars=valid_chars,
        )
        self.original_sequence = seq
        # Dual pools: candidates (work space) and selected (results space)
        self.candidate_sequences: List[Sequence] = [seq]
        self.selected_sequences: List[Sequence] = [seq]

        self.sequence_type: SequenceType = SequenceType(seq.sequence_type)
        self._valid_chars: Optional[Set[str]] = seq._valid_chars
        self._is_assigned: bool = False
        self.label: Optional[str] = label

    @property
    def num_selected(self) -> int:
        """Number of sequences in selected pool (solution space)."""
        return len(self.selected_sequences)

    @property
    def num_candidates(self) -> int:
        """Number of sequences in candidate pool (proposal space)."""
        return len(self.candidate_sequences)

    def __iter__(self) -> Iterator[Sequence]:
        """Iterate over selected sequences (user-facing results)."""
        return iter(self.selected_sequences)

    def __getitem__(self, index: int) -> Sequence:
        """Index into selected sequences (user-facing results)."""
        return self.selected_sequences[index]

    def create_candidates(self, num_candidates: int) -> None:
        """Create a new candidate pool of the given size."""
        self.candidate_sequences = [copy.deepcopy(self.original_sequence) for _ in range(num_candidates)]