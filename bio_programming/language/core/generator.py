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

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a Segment to the generator.

        Subclasses should call super().assign() first, then perform any
        additional validation/initialization as necessary.

        Raises:
            ValueError: If segment is a ligand or has incompatible sequence type.
        """
        from proto_language.language.generator.generator_registry import GeneratorRegistry

        # Ligand segments cannot be mutated by generators
        if assigned_segment.is_ligand:
            raise ValueError(
                f"Cannot assign generator to ligand segment '{assigned_segment.label}'. "
                "Ligand segments cannot be mutated."
            )

        # Validate sequence type compatibility from registry
        spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self))
        supported_types = spec.supported_sequence_types

        if supported_types and assigned_segment.sequence_type not in supported_types:
            supported_types_str = ", ".join(supported_types)
            raise ValueError(f"Generator {self.__class__.__name__} does not support sequence type '{assigned_segment.sequence_type}'. Supported types: [{supported_types_str}]")

        self._assigned_segment = assigned_segment

        # Warn if segment already has candidate sequences that will be overwritten (autoregressive only)
        if spec.category == "autoregressive" and assigned_segment.candidate_sequences:
            warnings.warn(f"Segment '{assigned_segment.label or 'unlabeled'}' has populated candidate sequence(s) that will be overwritten by {self.__class__.__name__}.")

        # For mutation generators, initialize a random starting sequence if not provided
        if spec.category == "mutation" and not assigned_segment.original_sequence.sequence:
            warnings.warn(f"No starting sequence provided for generator {self.__class__.__name__}. Initializing a random starting sequence.")
            valid_chars = list(assigned_segment._valid_chars - set(" "))
            assigned_segment.original_sequence.sequence = "".join(random.choice(valid_chars) for _ in range(assigned_segment.sequence_length))

    @abstractmethod
    def sample(self) -> None:
        """
        Sample new sequences by modifying the assigned Segment's candidate_sequences in-place.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the sample() method.")
