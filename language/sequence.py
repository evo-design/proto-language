from abc import ABC, abstractmethod
from typing import Any, List, Dict, Set
from Bio.Alphabet.IUPAC import ExtendedIUPACDNA, IUPACAmbiguousRNA, ExtendedIUPACProtein


class ProgramSequence:
    def __init__(self, sequence: str) -> None:
        """
        Initializes the ProgramSequence object.

        Args:
            sequence (str): The biological sequence as a string.
        """
        self._sequence: str = sequence.upper()
        self._metadata: Dict[str, Any] = {}
    
    def _validate_sequence(self, sequence: str, valid_chars: Set[str]) -> None:
        """
        Checks if the sequence consists of valid characters.

        Args:
            sequence (str): The biological sequence.
            valid_chars (Set[str]): A set of valid characters.
        """
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid characters found: {', '.join(invalid_chars)}. "
                            f"Valid characters are: {', '.join(sorted(valid_chars))}")

    @property
    def sequence(self) -> str:
        """
        Returns the underlying sequence string.

        Returns:
            str: The sequence.
        """
        return self._sequence

    def __len__(self) -> int:
        """
        Returns the length of the sequence.

        Returns:
            int: The length of the sequence string.
        """
        return len(self._sequence)

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation (the sequence itself).

        Returns:
            str: The sequence string.
        """
        return self._sequence

    def __eq__(self, other: object) -> bool:
        """
        Checks equality based on the sequence string.

        Args:
            other (object): The object to compare against.

        Returns:
            bool: True if the other object is a ProgramSequence with the same
                  sequence string, False otherwise.
        """
        if not isinstance(other, ProgramSequence):
            return NotImplemented
        return self._sequence == other._sequence

    def __hash__(self) -> int:
        """
        Computes hash based on the sequence string for use in sets/dicts.

        Returns:
            int: The hash value.
        """
        return hash(self._sequence)


class ProgramDNASequence(ProgramSequence):
    """
    A version of ProgramSequence for DNA sequences.
    """
    def __init__(self, sequence: str) -> None:
        """
        Initializes the ProgramDNASequence object.

        Args:
            sequence (str): The DNA sequence as a string.
        """
        self._validate_sequence(sequence, set(ExtendedIUPACDNA))
        super().__init__(sequence)


class ProgramRNASequence(ProgramSequence):
    """
    A version of ProgramSequence for RNA sequences.
    """
    def __init__(self, sequence: str) -> None:
        """
        Initializes the ProgramRNASequence object.

        Args:
            sequence (str): The RNA sequence as a string.
        """
        self._validate_sequence(sequence, set(IUPACAmbiguousRNA))
        super().__init__(sequence)


class ProgramProteinSequence(ProgramSequence):
    """
    A version of ProgramSequence for protein sequences.
    """
    def __init__(self, sequence: str) -> None:
        """
        Initializes the ProgramProteinSequence object.

        Args:
            sequence (str): The protein sequence as a string.
        """
        self._validate_sequence(sequence, set(ExtendedIUPACProtein))
        super().__init__(sequence)
