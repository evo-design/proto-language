"""
Generator base class for the biological programming language.

Provides the abstract interface for sequence generation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum

from .segment import Segment

class GeneratorType(Enum):
    """Enumeration of supported generator types."""
    AUTOREGRESSIVE = "autoregressive"
    MUTATION = "mutation"


class Generator(ABC):
    """
    Generator base class that modify candidate_sequences of assigned segments during optimization.

    Subclasses must implement `__init__()`, `assign()`, and `sample()`
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize the generator with configuration parameters.
        """
        # TODO: add logic to handle multiple assigned segments (if necessary)
        self._assigned_segment: Optional[Segment] = None

    @abstractmethod
    def assign(
        self, assigned_segment: Segment
    ) -> None:
        """
        Assign a Segment to the generator and initialize the generator.
        The generator will modify the Segment's candidate_sequences internally during sampling.
        """
        if assigned_segment.constant:
            raise ValueError(f"Cannot assign constant segment '{assigned_segment.label}' to generator. Constant segments should not be mutated during optimization.")
        
        if assigned_segment.original_sequence and len(assigned_segment.original_sequence.sequence) != self.sequence_length:
            raise ValueError(f"Provided sequence length ({len(assigned_segment.original_sequence.sequence)}) must match configured sequence_length ({self.sequence_length})")

    @abstractmethod
    def sample(self) -> None:
        """
        Sample new sequences by modifying the assigned Segment's candidate_sequences in-place.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the sample() method.")

    def _validate_generator(self) -> None:
        """
        Validate that the generator has been assigned.
        Raises:
            RuntimeError: If generator hasn't been assigned.
        """
        if self._assigned_segment is None:
            raise RuntimeError(f"Generator {self.__class__.__name__} has not been assigned a Segment via assign().")