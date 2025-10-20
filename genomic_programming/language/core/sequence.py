"""
Sequence class for the proto-language.

Represents a single DNA, RNA, or protein sequence with validation and metadata.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import warnings

from ...utils.helpers import propagate_metadata


# Valid characters for different sequence types
DNA_NUCLEOTIDES = "ACGT"
RNA_NUCLEOTIDES = "ACGU"
PROTEIN_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
LIGAND_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'- "


class SequenceType(Enum):
    """Enumeration of supported biological sequence types."""

    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"
    LIGAND = "ligand"


class Sequence:
    """
    Internal data structure for the basic unit of the programming language.

    Represents a single DNA, RNA, or protein sequence. The class enforces sequence type
    constraints and maintains metadata that gets updated when the sequence changes.
    """

    def __init__(
        self,
        sequence: str = "",
        sequence_type: Optional[Union[SequenceType, str]] = SequenceType.DNA,
        valid_chars: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a Sequence with sequence data and metadata.

        Args:
            sequence: The biological sequence string. Defaults to empty string.
            sequence_type: Type of biological sequence (SequenceType.DNA, SequenceType.RNA, or SequenceType.PROTEIN). Defaults to DNA.
            valid_chars: Optional custom set of valid characters for sequence validation.
                If provided, overrides the default character set for the sequence_type.
            metadata: Additional data associated with this sequence.
        """
        self.sequence_type: SequenceType = SequenceType(sequence_type)
        # Set up character validation based on sequence type or custom valid_chars
        if valid_chars:
            self._valid_chars: Optional[Set[str]] = valid_chars
        elif self.sequence_type == SequenceType.DNA:
            self._valid_chars = set(DNA_NUCLEOTIDES)
        elif self.sequence_type == SequenceType.RNA:
            self._valid_chars = set(RNA_NUCLEOTIDES)
        elif self.sequence_type == SequenceType.PROTEIN:
            self._valid_chars = set(PROTEIN_AMINO_ACIDS)
        elif self.sequence_type == SequenceType.LIGAND:
            self._valid_chars = set(LIGAND_CHARS)
        else:
            raise ValueError(f"Unsupported sequence_type: {self.sequence_type}")

        self._validate_sequence(sequence)
        self._sequence: str = sequence
        self._metadata = {}
        protected_metadata = {
            "sequence": sequence,
            "sequence_length": len(sequence),
        }

        # Add user metadata, warning if they try to override protected keys
        if metadata:
            conflicting_keys = [key for key in metadata if key in protected_metadata]
            if conflicting_keys:
                warnings.warn(
                    f"System-managed metadata for {conflicting_keys} cannot be manually set and will be silently overridden",
                    UserWarning,
                    stacklevel=2
                )
            self._metadata.update(metadata)
        self._metadata.update(protected_metadata)

    def _validate_sequence(self, sequence: str) -> None:
        """
        Validate that sequence contains only allowed characters for its type.

        Args:
            sequence: The sequence string to validate.

        Raises:
            ValueError: If sequence contains invalid characters for this sequence type.
        """
        invalid_chars = set(sequence) - self._valid_chars
        if invalid_chars:
            raise ValueError(
                f"Invalid characters found: {', '.join(invalid_chars)}. "
                f"Valid characters are: {', '.join(sorted(self._valid_chars))}"
            )

    @property 
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata dictionary with consistent ordering.
        
        Returns:
            Dict with system keys first, then constraint keys in chronological order.
        """
        system_keys = ["sequence", "sequence_length"]

        return {
            **{k: self._metadata[k] for k in system_keys if k in self._metadata},  # System keys first
            **{k: v for k, v in self._metadata.items() if k not in set(system_keys)}    # Constraint keys
        }

    @property
    def sequence(self) -> str:
        """
        Get the current sequence string.

        Returns:
            The sequence string.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, new_sequence: str) -> None:
        """
        Set a new sequence string with validation and metadata updates.

        Args:
            new_sequence: The new sequence string to set.

        Raises:
            ValueError: If the new sequence contains invalid characters.
        """
        # TODO: REVISIT THIS TRUNCATION
        # # Truncate at the first space character (EOS/space token)
        # space_index = new_sequence.find(' ')
        # if space_index != -1:
        #     new_sequence = new_sequence[:space_index]

        self._validate_sequence(new_sequence)
        self._sequence = new_sequence
        self._metadata["sequence"] = new_sequence
        self._metadata["sequence_length"] = len(new_sequence)

    def __len__(self) -> int:
        """
        Get the length of the sequence.

        Returns:
            Number of characters in the sequence.
        """
        return len(self._sequence)

    def __str__(self) -> str:
        """
        Get the sequence as a string.

        Returns:
            The sequence string.
        """
        return self._sequence

    @staticmethod
    def from_sequences(
        subsequences: List['Sequence'],
        merge_metadata: bool = False
    ) -> 'Sequence':
        """
        Create a sequence by joining subsequences with optional metadata propagation.
        
        This alternative constructor joins subsequences and optionally merges
        their metadata with sequence label prefixing to avoid key collisions.
        
        Args:
            subsequences: List of Sequence objects to join
            merge_metadata: If True, merge non-system metadata; if False, start clean
            
        Returns:
            Single joined Sequence object with only system metadata (if merge_metadata=False)
            or with merged non-system metadata (if merge_metadata=True)
            
        Example:
            >>> sequences = [Seq("ATG"), Seq("CCC")]
            >>> clean_seq = Sequence.from_sequences(sequences, merge_metadata=False)
            >>> # Returns Seq("ATGCCC") with only system metadata
        """
        combined_sequence_string = "".join(sequence.sequence for sequence in subsequences)
        combined_metadata = {}

        if merge_metadata:
            for sequence in subsequences:
                # Only propagate non-system metadata (no prefix needed)
                propagate_metadata(sequence._metadata, combined_metadata)

        return Sequence(
            sequence=combined_sequence_string,
            sequence_type=subsequences[0].sequence_type, # assumed to be the same for all subsequences
            valid_chars=subsequences[0]._valid_chars, # assumed to be the same for all subsequences
            metadata=combined_metadata
        )
