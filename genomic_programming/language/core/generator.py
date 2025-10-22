"""
Generator base class for the biological programming language.

Provides the abstract interface for sequence generation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .segment import Segment


class Generator(ABC):
    """
    Generator base class that modify candidate_sequences of assigned segments during optimization.

    Subclasses must implement `assign()` (Assign segments to the generator) and `sample()`
    """

    def __init__(self) -> None:
        """
        Initialize the generator with configuration parameters.
        """
        # TODO: add logic to handle multiple assigned segments (if necessary)
        self._assigned_segment: Optional[Segment] = None

    def _validate_generator(self) -> None:
        """
        Validate that the generator has been assigned.

        Raises:
            RuntimeError: If generator hasn't been assigned.
        """
        if self._assigned_segment is None:
            raise RuntimeError(f"Generator {self.__class__.__name__} has not been assigned a Segment via assign().")

    @abstractmethod
    def assign(
        self, assigned_segment: Segment
    ) -> None:
        """
        Assign a Segment to the generator and initialize the generator.
        The generator will modify the Segment's candidate_sequences internally during sampling.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the assign() method.")

    @abstractmethod
    def sample(self) -> None:
        """
        Sample new sequences by modifying the assigned Segment's candidate_sequences in-place.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the sample() method.")
