"""
proto_language/language/core/sequence.py

Sequence class for the proto-language.

Represents a single DNA, RNA, protein, or ligand sequence with validation and metadata.
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Optional, Set

# Valid characters for different sequence types
DNA_NUCLEOTIDES = "ACGT"
RNA_NUCLEOTIDES = "ACGU"
PROTEIN_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# MEMORY OPTIMIZATION FOR DEEPCOPYING: shared default character sets (frozenset for immutability)
_DEFAULT_DNA_CHARS: FrozenSet[str] = frozenset(DNA_NUCLEOTIDES)
_DEFAULT_RNA_CHARS: FrozenSet[str] = frozenset(RNA_NUCLEOTIDES)
_DEFAULT_PROTEIN_CHARS: FrozenSet[str] = frozenset(PROTEIN_AMINO_ACIDS)


# Type alias for supported biological sequence types
SequenceType = Literal["dna", "rna", "protein", "ligand"]

# Reserved keys in the computed .metadata property — user-provided metadata
# should not use these keys as they will be overwritten by identity fields.
_RESERVED_METADATA_KEYS = frozenset({"sequence", "sequence_length", "constraints"})


class Sequence:
    """
    Internal data structure for the basic unit of the programming language.

    Represents a single DNA, RNA, protein, or ligand sequence. The class enforces
    sequence type constraints and maintains metadata that gets updated when the
    sequence changes.

    For ligands, we assume SMILES string representations. Because there is no
    authoritative set of characters, and SMILES strings have a unique syntax,
    the validation relies on RDKit.

    Validation is performed on all sequences. Invalid genetic characters or SMILES
    syntax results in a warning but does not terminate the program.
    """

    def __init__(
        self,
        sequence: str = "",
        sequence_type: SequenceType = "dna",
        valid_chars: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a Sequence with sequence data and metadata.

        Args:
            sequence (str): The biological sequence string. Defaults to empty string.
            sequence_type (SequenceType): Type of biological sequence ("dna", "rna", or "protein"). Defaults to "dna".
            valid_chars (set[str] | None): Optional custom set of valid characters for sequence validation.
                If provided, overrides the default character set for the sequence_type.
            metadata (dict[str, Any] | None): Additional data associated with this sequence.
        """
        self._sequence_type: SequenceType = sequence_type
        # Set up character validation based on sequence type or custom valid_chars
        # Default chars use shared module-level frozensets to avoid allocation
        if valid_chars:
            self._valid_chars: Optional[Set[str] | FrozenSet[str]] = valid_chars
        elif self._sequence_type == "dna":
            self._valid_chars = _DEFAULT_DNA_CHARS
        elif self._sequence_type == "rna":
            self._valid_chars = _DEFAULT_RNA_CHARS
        elif self._sequence_type == "protein":
            self._valid_chars = _DEFAULT_PROTEIN_CHARS
        elif self._sequence_type == "ligand":
            self._valid_chars = None  # Validation handled by RDKit.
        else:
            raise ValueError(f"Unsupported sequence_type: {self._sequence_type}")

        self._validate_sequence(sequence)
        self._sequence: str = sequence
        self._metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        self._constraints_metadata: Dict[str, Any] = {}

        # Warn about reserved key collisions in user-provided metadata
        if self._metadata:
            collisions = _RESERVED_METADATA_KEYS & self._metadata.keys()
            if collisions:
                warnings.warn(
                    f"Metadata contains reserved keys {collisions} that will be "
                    f"overwritten by identity fields in the .metadata property."
                )

    def _validate_sequence(self, sequence: str) -> None:
        """
        Validate that genetic sequences contain only allowed characters for their types
        or that ligand sequences follow the RDKit SMILES syntax.

        Args:
            sequence (str): The sequence string to validate.

        Raises:
            ValueError: If sequence contains invalid characters for this sequence type.
        """
        if self.sequence_type == "ligand":
            validate_smiles(sequence)
            return

        invalid_chars = _return_invalid_chars(sequence, self._valid_chars)
        if invalid_chars:
            warnings.warn(
                f"Invalid characters found: {', '.join(invalid_chars)}. "
                f"Valid characters are: {', '.join(sorted(self._valid_chars))}"
            )

    @property
    def sequence_type(self) -> SequenceType:
        """Sequence type (read-only after construction)."""
        return self._sequence_type

    @property
    def valid_chars(self) -> Optional[Set[str] | FrozenSet[str]]:
        """Valid characters for this sequence (read-only after construction)."""
        return self._valid_chars

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Computed read-only view combining identity, user/generator metadata, and constraints.

        Identity fields (sequence, sequence_length, constraints) always take
        precedence over user-provided metadata with the same keys.
        """
        result = dict(self._metadata)
        result["sequence"] = self._sequence
        result["sequence_length"] = len(self._sequence)
        result["constraints"] = self._constraints_metadata
        return result

    @property
    def sequence(self) -> str:
        """
        Get the current sequence string.

        Returns:
            str: The sequence string.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, new_sequence: str) -> None:
        """
        Set a new sequence string with validation.

        Args:
            new_sequence (str): The new sequence string to set.

        Raises:
            ValueError: If the new sequence contains invalid characters.
        """
        self._validate_sequence(new_sequence)
        self._sequence = new_sequence

    def __len__(self) -> int:
        """
        Get the length of the sequence.

        Returns:
            int: Number of characters in the sequence.
        """
        return len(self._sequence)

    def __str__(self) -> str:
        """
        Get the sequence as a string.

        Returns:
            str: The sequence string.
        """
        return self._sequence

    def __getitem__(self, key):
        """
        Support subscripting and slicing of the sequence.

        Args:
            key: Index or slice object.

        Returns:
            Character at index or substring for slice.
        """
        return self._sequence[key]

    def __deepcopy__(self, memo):
        """
        Optimized deepcopy: share stable data, only copy mutable dicts.

        - _valid_chars, _sequence_type, _sequence: Immutable, share reference
        - _metadata, _constraints: Mutable, must deep copy
        """
        new_seq = object.__new__(Sequence)
        new_seq._sequence = self._sequence
        new_seq._sequence_type = self._sequence_type
        new_seq._valid_chars = self._valid_chars
        new_seq._metadata = copy.deepcopy(self._metadata, memo)
        new_seq._constraints_metadata = copy.deepcopy(self._constraints_metadata, memo)
        memo[id(self)] = new_seq
        return new_seq

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Sequence to dictionary for cloud/API communication."""
        return {
            "sequence": self._sequence,
            "sequence_type": self.sequence_type,
            "valid_chars": list(self._valid_chars) if self._valid_chars else None,
            "metadata": copy.deepcopy(self._metadata) if self._metadata else {},
            "constraints": copy.deepcopy(self._constraints_metadata) if self._constraints_metadata else {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Sequence:
        """Deserialize Sequence from dictionary."""
        if data.get("valid_chars"):
            chars = frozenset(data["valid_chars"])
            if chars == _DEFAULT_DNA_CHARS:
                valid_chars = _DEFAULT_DNA_CHARS
            elif chars == _DEFAULT_RNA_CHARS:
                valid_chars = _DEFAULT_RNA_CHARS
            elif chars == _DEFAULT_PROTEIN_CHARS:
                valid_chars = _DEFAULT_PROTEIN_CHARS
            else:
                valid_chars = chars
        else:
            valid_chars = None
        seq = cls(
            sequence=data["sequence"],
            sequence_type=data["sequence_type"],
            valid_chars=valid_chars,
            metadata=data.get("metadata") or None,
        )
        seq._constraints_metadata = data.get("constraints", {})
        return seq

def create_concatenated_sequence(subsequences: Iterable[Sequence], segment_labels: Optional[List[str]] = None) -> Sequence:
    """
    Concatenate subsequences into a single Sequence object.

    Args:
        subsequences (Iterable[Sequence]): Iterable of Sequence objects to concatenate
        segment_labels (list[str] | None): Optional list of segment labels for metadata nesting

    Returns:
        Sequence: Single Sequence with concatenated content. If segment_labels provided,
            includes segment metadata nested under _metadata["segments"][label].
    """
    seq_list = list(subsequences)
    if not seq_list:
        raise ValueError("Cannot concatenate an empty sequence list")
    combined_sequence_string = "".join(seq.sequence for seq in seq_list)

    joined_seq = Sequence(
        sequence=combined_sequence_string,
        sequence_type=seq_list[0].sequence_type,
        valid_chars=seq_list[0].valid_chars,
    )

    # Merge segment metadata if labels provided
    if segment_labels:
        assert len(segment_labels) == len(seq_list), f"Length mismatch: {len(segment_labels)} labels provided but {len(seq_list)} sequences to concatenate"
        segments_metadata = {
            label: {
                **copy.deepcopy(seq._metadata),
                "constraints": copy.deepcopy(seq._constraints_metadata),
            }
            for label, seq in zip(segment_labels, seq_list)
        }
        joined_seq._metadata["segments"] = segments_metadata
    return joined_seq

# =============================================================================
# Sequence Validation Helpers
# =============================================================================
def _return_invalid_chars(sequence: str, valid_chars: Set[str]) -> Set[str]:
    """
    Return the invalid characters in a sequence given a set of valid characters.

    Args:
        sequence (str): The sequence string to validate.
        valid_chars (set[str]): The set of valid characters.

    Returns:
        set[str]: The set of invalid characters.
    """
    invalid_chars = set(sequence) - valid_chars
    return invalid_chars


def return_invalid_dna_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Helper function that returns the invalid characters in a DNA sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default DNA characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = DNA_NUCLEOTIDES + additional_valid_chars

    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_rna_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Helper function that returns the invalid characters in a RNA sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default RNA characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = RNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_nucleotide_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Helper function that returns the invalid characters in a nucleotide sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default nucleotide characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = DNA_NUCLEOTIDES + RNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_protein_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Return the invalid characters in a protein sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default protein amino acids.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = PROTEIN_AMINO_ACIDS + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def validate_smiles(smiles: str, verbose: bool = True) -> bool:
    """
    Validate SMILES string using RDKit if available.

    Args:
        smiles (str): The SMILES string to validate.
        verbose (bool): Print warnings.

    Returns:
        bool: True if valid SMILES, False if invalid or RDKit unavailable.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            warnings.warn(
                f"RDKit could not parse SMILES: '{smiles}'. "
                "This may not be a valid molecule."
            )
        return False
    return True


def detect_sequence_type(sequence: str) -> str:
    """
    Attempts to determine the type of a sequence based on the characters it contains.
    Starts with more specific sequence types (less characters allowed) and works
    its way down to the least specific. Returns "unknown" if the sequence type
    cannot be determined.

    Note that there are ambiguous cases (e.g., "CCCCCC" could be DNA, RNA, protein, or
    ligand SMILES). Prority is: DNA, RNA, protein, ligand.

    Args:
        sequence (str): The sequence string to detect the type of.

    Returns:
       str: The type of the sequence ("dna", "rna", "protein", "ligand", or "unknown").
    """

    # DNA ================================================================
    invalid_chars = return_invalid_dna_chars(sequence, additional_valid_chars="N")
    if not invalid_chars:
        return "dna"

    # RNA ================================================================
    invalid_chars = return_invalid_rna_chars(sequence, additional_valid_chars="TN")
    if not invalid_chars:
        return "rna"

    # Protein =============================================================
    invalid_chars = return_invalid_protein_chars(sequence, additional_valid_chars="X*")
    if not invalid_chars:
        return "protein"

    # Ligand/SMILES ========================================================
    if validate_smiles(sequence, verbose=False):
        return "ligand"

    # Otherwise, return unknown
    return "unknown"
