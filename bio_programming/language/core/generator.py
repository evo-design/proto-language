"""
Generator base class for the biological programming language.

Provides the abstract interface for sequence generation algorithms.
"""

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from typing import Optional

from .segment import Segment


class Generator(ABC):
    """
    Generator base class that modifies candidate_sequences of assigned segments during optimization.

    Subclasses must implement `__init__()` and `sample()`. Override `assign()` only if
    additional validation or initialization is needed (call super().assign() first).
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize the generator with configuration parameters.
        """
        # TODO: add logic to handle multiple assigned segments (if necessary)
        self._assigned_segment: Optional[Segment] = None
        self.__spec: Optional["GeneratorSpec"] = None  # Lazy-loaded via property

    # Required lazy loading for mock generators to function in tests.
    @property
    def _spec(self) -> "GeneratorSpec":
        """Lazy-load the generator spec from the registry."""
        if self.__spec is None:
            from proto_language.language.generator.generator_registry import GeneratorRegistry
            self.__spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self))
        return self.__spec
        

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a Segment to the generator.

        Subclasses should call super().assign() first, then perform any
        additional validation/initialization as necessary.

        Raises:
            ValueError: If segment is a ligand or has incompatible sequence type.
        """
        # Ligand segments cannot be mutated by generators
        if assigned_segment.is_ligand:
            raise ValueError(f"Cannot assign generator to ligand segment '{assigned_segment.label}'. Ligand segments cannot be mutated.")

        # Validate sequence type compatibility from registry
        supported_types = self._spec.supported_sequence_types

        if supported_types and assigned_segment.sequence_type not in supported_types:
            supported_types_str = ", ".join(supported_types)
            raise ValueError(f"Generator {self.__class__.__name__} does not support sequence type '{assigned_segment.sequence_type}'. Supported types: [{supported_types_str}]")
        self._assigned_segment = assigned_segment


    @abstractmethod
    def sample(self) -> None:
        """
        Sample new sequences by modifying the assigned Segment's candidate_sequences in-place.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the sample() method.")

    def _validate_generator(self) -> None:
        """Validate the generator."""
        if self._assigned_segment is None:
            raise RuntimeError(f"Generator {self.__class__.__name__} has no segment assigned.")

        # Warn if segment already has populated sequences that will be overwritten (autoregressive only)
        if self._spec.category == "autoregressive" and self._assigned_segment.candidates_populated:
            warnings.warn(f"Segment '{self._assigned_segment.label or 'unlabeled'}' has an input sequence that will be overwritten by {self.__class__.__name__}.")

        # Initialize random sequences for mutation generators if no input template sequence provided.
        if self._spec.category == "mutation":
            if not self._assigned_segment.candidates_populated:
                warnings.warn(f"Generator {self.__class__.__name__} is a mutation generator, but candidates have no sequences. Initializing random starting sequences.")
                valid_chars = list(self._assigned_segment._valid_chars - set(" "))
                random_sequence = "".join(random.choice(valid_chars) for _ in range(self._assigned_segment.sequence_length))
                for sequence in self._assigned_segment.candidate_sequences:
                    sequence.sequence = random_sequence

        # Initialize unknown (X) sequences for inverse folding generators if no input sequence provided.
        if self._spec.category == "inverse_folding":
            if not self._assigned_segment.candidates_populated:
                unknown_sequence = "X" * self._assigned_segment.sequence_length
                for sequence in self._assigned_segment.candidate_sequences:
                    sequence.sequence = unknown_sequence
