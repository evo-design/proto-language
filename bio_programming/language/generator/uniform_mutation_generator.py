"""
UniformMutationGenerator for random point mutations.
"""

from __future__ import annotations

import random
import time
from typing import Optional, final

from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator, Segment
from proto_language.language.generator.generator_registry import generator


class MutationWindow(BaseConfig):
    """Configuration for mutation window specifying the range to mutate.

    Attributes:
        start (Optional[int]): Start index (0-based, inclusive). Must be non-negative.
        end (Optional[int]): End index (0-based, exclusive). Must be greater than start.
    """

    start: Optional[int] = ConfigField(
        ge=0,
        title="Start Index",
        description="Start index of mutation window (0-based, inclusive)",
    )
    end: Optional[int] = ConfigField(
        ge=0,
        title="End Index",
        description="End index of mutation window (0-based, exclusive)",
    )

    @model_validator(mode="after")
    def validate_mutation_window(self):
        """Validate mutation window constraints."""
        # Both start and end must be provided together
        if (self.start is None) != (self.end is None):
            raise ValueError(
                "Both start and end must be provided together for mutation window"
            )

        # If both are provided, validate that end > start
        if self.start is not None and self.end <= self.start:
            raise ValueError(
                f"Mutation window end ({self.end}) must be greater than start ({self.start})"
            )

        return self


class UniformMutationGeneratorConfig(BaseConfig):
    """Configuration object for UniformMutationGenerator.

    This class defines configuration parameters for the uniform mutation generator,
    which introduces random point mutations into sequences for diversity exploration.

    Attributes:
        num_mutations (int): Number of positions to randomly mutate per sample.
            Each mutation replaces a character with a different random character
            from the valid alphabet. If this exceeds the sequence length or mutation
            window size, it is automatically capped at the available positions.
            Must be at least 0. Default: 1.

        mutation_window (Optional[MutationWindow]): Optional window specifying which
            region of the sequence to mutate. Accepted formats:

            - A tuple/list: ``(start, end)`` using Python indexing (0-based, end-exclusive)
            - A dict: ``{"start": 0, "end": 100}``
            - A MutationWindow instance

            Examples:
            - ``(0, 100)``: Mutate only first 100 positions
            - ``{"start": 5, "end": 10}``: Mutate only positions 5-9
            - ``[50, 150]``: Mutate only positions 50-149
            - ``MutationWindow(start=50, end=150)``: Mutate only positions 50-149
            - ``None``: Mutate entire sequence (default)

            Both values must be within ``[0, sequence_length]``. Default: ``None``.

        debug_with_sleep_calls (bool): Enable debug mode with 1-second sleep calls
            during sampling. Only use for testing parallel execution or profiling.
            Default: ``False``.
    """

    # Advanced parameters (have default values)
    num_mutations: int = ConfigField(
        default=1,
        ge=0,
        title="Num Mutations",
        description="Number of positions to mutate per sample",
        advanced=True,
    )
    mutation_window: Optional[MutationWindow] = ConfigField(
        default=None,
        title="Mutation Window",
        description="Only mutate the sequence within this range. (start, end) using Python index conventions.",
        advanced=True,
    )
    debug_with_sleep_calls: bool = ConfigField(
        default=False,
        title="Debug with Sleep Calls",
        description="Enable debug mode with sleep calls (for testing purposes only)",
        advanced=True,
    )

    @field_validator("mutation_window", mode="before")
    @classmethod
    def allow_tuple_for_window(cls, v):
        """Convert tuple/list input to dict format for Pydantic model parsing."""
        if v is None:
            return v

        if isinstance(v, (tuple, list)):
            if len(v) != 2:
                raise ValueError(f"Mutation window tuple must have exactly 2 elements, got {len(v)}")
            return {"start": v[0], "end": v[1]}

        return v


@generator(
    key="uniform-mutation",
    label="Uniform Mutation Generator",
    config=UniformMutationGeneratorConfig,
    description="Random point mutations for sequence diversity",
    requires_gpu=False,
    tools_called=[],
    category="mutation",
    supported_sequence_types=[],
)
@final
class UniformMutationGenerator(Generator):
    """Sequence generator that introduces random point mutations.

    This generator creates sequence diversity by randomly mutating specified positions
    in DNA, RNA, or protein sequences. Can start from a provided sequence or generate
    a random initial sequence. Useful for exploring local sequence space around a
    starting point.

    The generator category is ``"mutation"``, indicating it modifies existing
    sequences rather than generating from scratch.

    Attributes:
        num_mutations (int): Number of positions to mutate per sample.
        mutation_window (Optional[MutationWindow]): Optional region to restrict mutations.
        debug_with_sleep_calls (bool): Whether to add sleep delays for testing.

    Example:
        >>> from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig
        >>> from proto_language.language.core import Segment, SequenceType
        >>> config = UniformMutationGeneratorConfig(num_mutations=2)
        >>> gen = UniformMutationGenerator(config)
        >>> segment = Segment(length=100, sequence_type="dna")
        >>> gen.assign(segment)
        >>> gen.sample()  # Introduces 2 random mutations
    """

    def __init__(self, config: UniformMutationGeneratorConfig) -> None:
        """
        Initialize the uniform mutation generator.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.num_mutations = config.num_mutations
        self.debug_with_sleep_calls = config.debug_with_sleep_calls
        self.mutation_window = config.mutation_window

    def assign(self, assigned_segment: Segment) -> None:
        """
        Assign a Segment to this generator.

        Validates mutation_window against segment's sequence_length if specified.
        Random starting sequence initialization is handled by the base class.
        """
        super().assign(assigned_segment)

        # Validate mutation window against segment's sequence_length
        if (
            self.mutation_window is not None
            and self.mutation_window.start is not None
            and (
                self.mutation_window.start >= assigned_segment.sequence_length
                or self.mutation_window.end > assigned_segment.sequence_length
            )
        ):
            raise ValueError(f"Mutation window ({self.mutation_window.start}, {self.mutation_window.end}) incompatible with segment length {assigned_segment.sequence_length}.")

    def sample(self) -> None:
        """
        Introduce random point mutations in candidate sequences.

        Mutates each sequence in the candidate pool by selecting random positions
        and replacing characters with different random characters from the vocabulary.

        Raises:
            RuntimeError: If called before assign().
            ValueError: If candidate pool is empty.
        """
        self._validate_generator()
        # Sleep for testing purposes if debug_with_sleep_calls is enabled
        if self.debug_with_sleep_calls:
            time.sleep(1.0)

        # Mutate each candidate sequence
        for sequence in self._assigned_segment.candidate_sequences:
            current_sequence = sequence.sequence
            sequence_length = len(current_sequence)

            # Define the positions to mutate
            if self.mutation_window is None or self.mutation_window.start is None:
                window_range = range(sequence_length)
            else:
                window_range = range(self.mutation_window.start, self.mutation_window.end)

            # Cap mutations to both sequence length and available window positions
            actual_mutations = min(self.num_mutations, sequence_length, len(window_range))

            # Select random positions to mutate (without replacement)
            positions_to_mutate = random.sample(window_range, actual_mutations)

            # Apply mutations
            for pos in positions_to_mutate:
                current_char = current_sequence[pos]

                # Make sure the mutated character is different from the current one
                possible_mutations = [
                    c for c in self._assigned_segment.valid_chars if c != current_char
                ]
                mutated_char = random.choice(possible_mutations)
                current_sequence = (
                    current_sequence[:pos] +
                    mutated_char +
                    current_sequence[pos + 1:]
                )

            sequence.sequence = current_sequence
