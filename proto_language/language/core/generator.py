"""Provides the abstract interface for sequence generation algorithms."""

from __future__ import annotations

import logging
import random
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from proto_language.language.core.segment import Segment

if TYPE_CHECKING:
    from proto_language.language.generator.generator_registry import GeneratorSpec

logger = logging.getLogger(__name__)


class Generator(ABC):
    """Generator base class that modifies proposal_sequences of assigned segments during optimization.

    Subclasses must implement `__init__()` and `sample()`. Override `assign()` only if
    additional validation or initialization is needed (call super().assign() first).

    Attributes:
        batch_size (int): Number of sequences to generate per batch.
    """

    batch_size: int = 1  # GPU generators override to higher values

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the generator with configuration parameters."""
        self._assigned_segment: Segment | None = None
        self.__spec: GeneratorSpec | None = None  # Lazy-loaded via property

    # Required lazy loading for mock generators to function in tests.
    @property
    def _spec(self) -> GeneratorSpec:
        """Lazy-load the generator spec from the registry."""
        if self.__spec is None:
            from proto_language.language.generator.generator_registry import (
                GeneratorRegistry,
            )
            self.__spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self))
        return self.__spec


    def assign(self, assigned_segment: Segment) -> None:
        """Assign a Segment to the generator.

        Subclasses should call super().assign() first, then perform any
        additional validation/initialization as necessary.

        Args:
            assigned_segment (Segment): Segment to assign generated sequences to.

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
        logger.debug(f"Generator.assign: {self.__class__.__name__} -> segment={assigned_segment.label}")


    @abstractmethod
    def sample(self) -> None:
        """Sample new sequences by modifying the assigned Segment's proposal_sequences in-place."""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the sample() method.")

    def _validate_generator(self) -> None:
        """Validate the generator."""
        if self._assigned_segment is None:
            raise RuntimeError(f"Generator {self.__class__.__name__} has no segment assigned.")

        segment = self._assigned_segment

        if not segment.proposal_sequences:
            raise RuntimeError(
                f"Segment '{segment.label or 'unlabeled'}' has an empty proposal_sequences pool."
            )

        # Warn if segment already has populated sequences that will be overwritten (autoregressive only)
        if self._spec.category == "autoregressive" and segment.proposals_populated:
            warnings.warn(f"Segment '{segment.label or 'unlabeled'}' has an input sequence that will be overwritten by {self.__class__.__name__}.", stacklevel=2)

        # Initialize random sequences for mutation generators if no input template sequence provided.
        if self._spec.category == "mutation" and not segment.proposals_populated:
            warnings.warn(f"Generator {self.__class__.__name__} is a mutation generator, but proposals have no sequences. Initializing random starting sequences.", stacklevel=2)
            assert segment.valid_chars is not None  # noqa: S101 -- mypy type narrowing
            valid_chars = list(segment.valid_chars - set(" "))
            for sequence in segment.proposal_sequences:
                random_sequence = "".join(random.choice(valid_chars) for _ in range(segment.sequence_length))  # noqa: S311 -- non-cryptographic, used for random sequence initialization
                sequence.sequence = random_sequence

        # Initialize unknown (X) sequences for inverse folding generators if no input sequence provided.
        if self._spec.category == "inverse_folding" and not segment.proposals_populated:
                unknown_sequence = "X" * segment.sequence_length
                for sequence in segment.proposal_sequences:
                    sequence.sequence = unknown_sequence

        logger.debug(f"Generator validated: {self.__class__.__name__}, category={self._spec.category}")
