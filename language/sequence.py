from abc import ABC, abstractmethod
from typing import Any, List, Dict, Set
from Bio.Alphabet.IUPAC import ExtendedIUPACDNA, IUPACAmbiguousRNA, ExtendedIUPACProtein

class ProgramSequence:
    def __init__(self, sequence: str) -> None:
        self._sequence: str = sequence.upper()
        self._metadata: Dict[str, Any] = {}
    
    def _validate_sequence(self, sequence: str, valid_chars: Set[str]) -> None:
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid characters found: {', '.join(invalid_chars)}. "
                            f"Valid characters are: {', '.join(sorted(valid_chars))}")
    
    def __len__(self) -> int:
        return self.length
    
    def __str__(self) -> str:
        return self._sequence

class ProgramDNASequence(ProgramSequence):
    def __init__(self, sequence: str) -> None:
        self._validate_sequence(sequence, set(ExtendedIUPACDNA))
        super().__init__(sequence)

class ProgramRNASequence(ProgramSequence):
    def __init__(self, sequence: str) -> None:
        self._validate_sequence(sequence, set(IUPACAmbiguousRNA))
        super().__init__(sequence)

class ProgramProteinSequence(ProgramSequence):
    def __init__(self, sequence: str) -> None:
        self._validate_sequence(sequence, set(ExtendedIUPACProtein))
        super().__init__(sequence)