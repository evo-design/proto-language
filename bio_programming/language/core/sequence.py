"""
sequence.py

Sequence class for the proto-language.

Represents a single DNA, RNA, protein, or ligand sequence with validation and metadata.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Literal, Optional, Set
import warnings

from proto_language.utils.helpers import propagate_metadata

# Valid characters for different sequence types
DNA_NUCLEOTIDES = "ACGT"
RNA_NUCLEOTIDES = "ACGU"
PROTEIN_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# Type alias for supported biological sequence types
SequenceType = Literal["dna", "rna", "protein", "ligand"]


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
            sequence: The biological sequence string. Defaults to empty string.
            sequence_type: Type of biological sequence ("dna", "rna", or "protein"). Defaults to "dna".
            valid_chars: Optional custom set of valid characters for sequence validation.
                If provided, overrides the default character set for the sequence_type.
            metadata: Additional data associated with this sequence.
        """
        self.sequence_type: SequenceType = sequence_type
        # Set up character validation based on sequence type or custom valid_chars
        if valid_chars:
            self._valid_chars: Optional[Set[str]] = valid_chars
        elif self.sequence_type == "dna":
            self._valid_chars = set(DNA_NUCLEOTIDES)
        elif self.sequence_type == "rna":
            self._valid_chars = set(RNA_NUCLEOTIDES)
        elif self.sequence_type == "protein":
            self._valid_chars = set(PROTEIN_AMINO_ACIDS)
        elif self.sequence_type == "ligand":
            self._valid_chars = None  # Validation handled by RDKit.
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
                    stacklevel=2,
                )
            self._metadata.update(metadata)
        self._metadata.update(protected_metadata)

    def _validate_sequence(self, sequence: str) -> None:
        """
        Validate that genetic sequences contain only allowed characters for their types
        or that ligand sequences follow the RDKit SMILES syntax.

        Args:
            sequence: The sequence string to validate.

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
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata dictionary with consistent ordering.

        Returns:
            Dict with system keys first, then constraint keys in chronological order.
        """
        system_keys = ["sequence", "sequence_length"]

        return {**{k: self._metadata[k] for k in system_keys if k in self._metadata},  # System keys first
            **{k: v for k, v in self._metadata.items() if k not in set(system_keys)},  # Constraint keys
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

    def __getitem__(self, key):
        """
        Support subscripting and slicing of the sequence.

        Args:
            key: Index or slice object.

        Returns:
            Character at index or substring for slice.
        """
        return self._sequence[key]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Sequence to dictionary for cloud/API communication."""
        return {
            "sequence": self._sequence,
            "sequence_type": self.sequence_type,
            "valid_chars": list(self._valid_chars) if self._valid_chars else None,
            "metadata": {k: v for k, v in self._metadata.items() if k not in ["sequence", "sequence_length"]},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sequence":
        """Deserialize Sequence from dictionary."""
        valid_chars = set(data["valid_chars"]) if data.get("valid_chars") else None
        return cls(
            sequence=data["sequence"],
            sequence_type=data["sequence_type"],
            valid_chars=valid_chars,
            metadata=data.get("metadata", {}),
        )

def create_concatenated_sequence(
    subsequences: Iterable[Sequence],
    merge_metadata: bool = False,
) -> Sequence:
    """
    Concatenate subsequences into a single Sequence object.

    Args:
        sequences: Iterable of Sequence objects to concatenate
        merge_metadata: If True, merge non-system metadata; if False, start clean

    Returns:
        Single Sequence with concatenated content
    """
    seq_list = list(subsequences)
    combined_sequence_string = "".join(seq.sequence for seq in seq_list)
    combined_metadata = {}

    if merge_metadata:
        for seq in seq_list:
            propagate_metadata(seq._metadata, combined_metadata)

    return Sequence(
        sequence=combined_sequence_string,
        sequence_type=seq_list[0].sequence_type,
        valid_chars=seq_list[0]._valid_chars,
        metadata=combined_metadata,
    )


# =============================================================================
# Sequence Validation Helpers
# =============================================================================
def _return_invalid_chars(sequence: str, valid_chars: Set[str]) -> Set[str]:
    """
    Return the invalid characters in a sequence given a set of valid characters.

    Args:
        sequence: The sequence string to validate.
        valid_chars: The set of valid characters.

    Returns:
        The set of invalid characters.
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
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default DNA characters.

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
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default RNA characters.

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
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default nucleotide characters.

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
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default protein amino acids.

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
        smiles: The SMILES string to validate.
        verbose: Print warnings.

    Returns:
        True if valid SMILES, False if invalid or RDKit unavailable.
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
       string: The type of the sequence ("dna", "rna", "protein", "ligand", or "unknown").
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
