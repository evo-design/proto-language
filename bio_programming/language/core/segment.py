"""
Segment class for the proto-language.

Represents building blocks for biological constructs.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, List, Optional, Set

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
    ) -> None:
        """
        Initialize a Segment with dual sequence pools.

        Args:
            sequence: Optional biological sequence string. If provided, length is inferred.
            length: Optional desired length for sequences. Required if sequence not provided.
            sequence_type: Type of biological sequence ("dna", "rna", or "protein"). Defaults to "dna".
            valid_chars: Optional custom set of valid characters for sequence validation.
            label: Optional label for this segment (e.g., "promoter", "coding_region").
            metadata: Additional data associated with this sequence.

        Raises:
            ValueError: If both sequence and length are provided, if neither is provided,
                or if a ligand segment is created with only a length (ligands require a sequence).
        """
        # Exactly one of sequence or length must be provided
        if sequence is None and length is None:
            raise ValueError("Must provide either 'sequence' or 'length'")
        elif sequence is not None and length is not None:
            raise ValueError("Cannot provide both 'sequence' and 'length' - choose one")

        # Ligand segments must be initialized with a sequence (SMILES string), not just a length
        if sequence_type == "ligand" and sequence is None:
            raise ValueError("Ligand segments must be initialized with a sequence (SMILES string), not just a length")

        # If sequence is provided - set sequence_length and initial_sequence
        elif sequence is not None:
            initial_sequence = sequence
            self.sequence_length = len(sequence)

        # If length is provided - set sequence_length and initial_sequence to empty
        else:
            initial_sequence = ""
            self.sequence_length = length

        # Original sequence is read-only after construction
        self._original_sequence: Sequence = Sequence(
            sequence=initial_sequence,
            sequence_type=sequence_type,
            metadata=metadata,
            valid_chars=valid_chars,
        )
        # Dual pools: candidates (work space) and selected (results space)
        # These are deep copies so modifications don't affect original_sequence
        self.candidate_sequences: List[Sequence] = [copy.deepcopy(self._original_sequence)]
        self.selected_sequences: List[Sequence] = [copy.deepcopy(self._original_sequence)]

        self.sequence_type: SequenceType = self._original_sequence.sequence_type
        self._valid_chars: Optional[Set[str]] = self._original_sequence._valid_chars
        self.label: Optional[str] = label

    @property
    def num_selected(self) -> int:
        """Number of sequences in selected pool (solution space)."""
        return len(self.selected_sequences)

    @property
    def num_candidates(self) -> int:
        """Number of sequences in candidate pool (proposal space)."""
        return len(self.candidate_sequences)

    @property
    def original_sequence(self) -> Sequence:
        """Original sequence (read-only). Preserves user intent for serialization."""
        return self._original_sequence

    @property
    def has_original_sequence(self) -> bool:
        """Whether segment was created with a sequence (vs just a length)."""
        return bool(self._original_sequence.sequence)

    @property
    def populated_sequences(self) -> bool:
        """
        Whether segment has sequences from original input or previous optimization.
        Only checks original sequence (original user input) and selected sequences (previous optimization results).
        Candidate sequences are not considered because they the staging area for optimizations.
        """
        return bool(
            self._original_sequence.sequence or
            (self.selected_sequences and self.selected_sequences[0].sequence)
        )

    @property
    def candidates_populated(self) -> bool:
        """Whether candidate sequences have actual sequences (not empty)."""
        return bool(self.candidate_sequences[0].sequence)

    @property
    def is_ligand(self) -> bool:
        """Whether this segment is a ligand (ligands cannot be mutated by generators)."""
        return self.sequence_type == "ligand"

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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        """Deserialize Segment from dictionary."""
        # Reconstruct original sequence
        original_seq = Sequence.from_dict(data["original_sequence"])

        # Use input sequence if available, otherwise use length
        segment = cls(
            sequence=original_seq.sequence if original_seq.sequence else None,
            length=data["sequence_length"] if not original_seq.sequence else None,
            sequence_type=data["sequence_type"],
            valid_chars=set(data["valid_chars"]) if "valid_chars" in data else None,
            label=data.get("label"),
            metadata=original_seq._metadata,
        )

        # Restore sequence pools
        segment.candidate_sequences = [Sequence.from_dict(seq_data) for seq_data in data["candidate_sequences"]]
        segment.selected_sequences = [Sequence.from_dict(seq_data) for seq_data in data["selected_sequences"]]

        return segment
