from Bio.Data import IUPACData
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Set

from .base import ProgramSequence, ProgramGenerator


class ProgramDNASequence(ProgramSequence):
    """
    A version of ProgramSequence for DNA sequences.
    """
    def __init__(
        self,
        generator: ProgramGenerator,
        generator_output_idx: int,
        sequence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_chars: Optional[Set[str]] = None
    ) -> None:
        """
        Initializes the ProgramDNASequence object.

        Args:
            generator (ProgramGenerator): The generator that updates `sequence.`
            generator_output_idx (int): The index into the generator's output list.
            sequence (Optional[str]): The value of the DNA sequence string.
            metadata (Optional[Dict[str, Any]]): Metadata for the sequence.
            valid_chars (Optional[Set[str]]): A set of valid characters that the sequence
                                              can take on.
        """
        if valid_chars is None:
            valid_chars = set(IUPACData.ambiguous_dna_letters + '-')
        super().__init__(
            generator,
            generator_output_idx,
            sequence,
            metadata,
            valid_chars,
        )


class ProgramRNASequence(ProgramSequence):
    """
    A version of ProgramSequence for RNA sequences.
    """
    def __init__(
        self,
        generator: ProgramGenerator,
        generator_output_idx: int,
        sequence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_chars: Optional[Set[str]] = None
    ) -> None:
        """
        Initializes the ProgramRNASequence object.

        Args:
            generator (ProgramGenerator): The generator that updates `sequence.`
            generator_output_idx (int): The index into the generator's output list.
            sequence (Optional[str]): The value of the RNA sequence string.
            valid_chars (Optional[Set[str]]): A set of valid characters that the sequence
                                              can take on.
        """
        if valid_chars is None:
            valid_chars = set(IUPACData.ambiguous_rna_letters + '-')
        super().__init__(
            generator,
            generator_output_idx,
            sequence,
            metadata,
            valid_chars,
        )


class ProgramProteinSequence(ProgramSequence):
    """
    A version of ProgramSequence for protein sequences.
    """
    def __init__(
        self,
        generator: ProgramGenerator,
        generator_output_idx: int,
        sequence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_chars: Optional[Set[str]] = None
    ) -> None:
        """
        Initializes the ProgramProteinSequence object.

        Args:
            generator (ProgramGenerator): The generator that updates `sequence.`
            generator_output_idx (int): The index into the generator's output list.
            sequence (Optional[str]): The value of the protein sequence string.
            valid_chars (Optional[Set[str]]): A set of valid characters that the sequence
                                              can take on.
        """
        if valid_chars is None:
            valid_chars = set(IUPACData.protein_letters_1to3.keys() + '*-')
        super().__init__(
            generator,
            generator_output_idx,
            sequence,
            metadata,
            valid_chars,
        )
