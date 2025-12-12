"""
Segment class for the proto-language.

Represents building blocks for biological constructs.
"""
from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Set, Union
import copy
from .sequence import Sequence, SequenceType


class Segment:
    """
    Building block for biological constructs with two sequence pools: candidate (work space) and selected (results space):
    - candidate_sequences: Working space for optimizer proposals (mutations, offspring, rollouts)
    - selected_sequences: Results space containing current best sequences (user-facing)

    Examples:
        Creating a Segment with a sequence:
        >>> promoter = Segment(sequence="TATA", sequence_type="dna", label="promoter")
        >>> promoter.label  # "promoter"
        >>> promoter.sequence_length  # 4 (inferred from sequence)
        >>> promoter.selected_sequences  # [Sequence("TATA")]

        Creating a Segment with just a length:
        >>> variable_region = Segment(length=100, sequence_type="dna", label="variable")
        >>> variable_region.sequence_length  # 100
        >>> variable_region.selected_sequences  # [Sequence("")]
    """

    def __init__(
        self,
        sequence: Optional[str] = None,
        length: Optional[int] = None,
        sequence_type: SequenceType = "dna",
        valid_chars: Optional[Set[str]] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        constant: bool = False,
    ) -> None:
        """
        Initialize a Segment with dual sequence pools.

        Args:
            sequence: Optional biological sequence string. If provided, length is inferred.
            length: Optional desired length for sequences. Required if sequence not provided.
            sequence_type: Type of biological sequence ("dna", "rna", "protein", or "ligand"). Defaults to "dna".
            valid_chars: Optional custom set of valid characters for sequence validation.
            label: Optional label for this segment (e.g., "promoter", "coding_region").
            metadata: Additional data associated with this sequence.
            constant: If True, the sequence is constant and cannot be mutated.

        Raises:
            ValueError: If both sequence and length are provided, or if neither is provided.
        """
        # Exactly one of sequence or length must be provided
        if sequence is None and length is None:
            raise ValueError("Must provide either 'sequence' or 'length'")
        elif sequence is not None and length is not None:
            raise ValueError("Cannot provide both 'sequence' and 'length' - choose one")

        # If sequence is provided - set sequence_length and initial_sequence
        elif sequence is not None:
            initial_sequence = sequence
            self.sequence_length = len(sequence)

        # If length is provided - set sequence_length and initial_sequence to empty
        else:
            initial_sequence = ""
            self.sequence_length = length

        seq = Sequence(
            sequence=initial_sequence,
            sequence_type=sequence_type,
            metadata=metadata,
            valid_chars=valid_chars,
        )
        self.original_sequence: Sequence = seq
        # Dual pools: candidates (work space) and selected (results space)
        self.candidate_sequences: List[Sequence] = [seq]
        self.selected_sequences: List[Sequence] = [seq]

        self.sequence_type: SequenceType = seq.sequence_type
        self._valid_chars: Optional[Set[str]] = seq._valid_chars
        self.label: Optional[str] = label

        if constant and not self.original_sequence.sequence:
            raise ValueError("Constant segment must be initialized with a non-empty sequence.")

        # Constant segment is assigned by default
        self.constant = constant
        self._is_assigned: bool = True if constant else False

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Segment to dictionary for cloud/API communication."""
        return {
            "original_sequence": self.original_sequence.to_dict(),
            "sequence_length": self.sequence_length,
            "candidate_sequences": [seq.to_dict() for seq in self.candidate_sequences],
            "selected_sequences": [seq.to_dict() for seq in self.selected_sequences],
            "sequence_type": self.sequence_type,
            "valid_chars": list(self._valid_chars) if self._valid_chars else None,
            "label": self.label,
            "constant": self.constant,
            "_is_assigned": self._is_assigned,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        """Deserialize Segment from dictionary."""
        # Reconstruct original sequence
        original_seq = Sequence.from_dict(data["original_sequence"])

        # Use sequence if available, otherwise use length
        segment = cls(
            sequence=original_seq.sequence if original_seq.sequence else None,
            length=data["sequence_length"] if not original_seq.sequence else None,
            sequence_type=data["sequence_type"],
            valid_chars=set(data["valid_chars"]) if "valid_chars" in data else None,
            label=data.get("label"),
            metadata=original_seq._metadata,
            constant=data.get("constant", False),
        )

        # Restore sequence pools
        segment.original_sequence = original_seq
        segment.candidate_sequences = [Sequence.from_dict(seq_data) for seq_data in data["candidate_sequences"]]
        segment.selected_sequences = [Sequence.from_dict(seq_data) for seq_data in data["selected_sequences"]]
        segment._is_assigned = data.get("_is_assigned", False)

        return segment

