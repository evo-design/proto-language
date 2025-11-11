"""
UniformMutationGenerator for random point mutations.
"""

from typing import final, Optional, Tuple
import random
import time

from pydantic import Field, model_validator

from ..core import Generator, GeneratorType, Segment
from proto_language.base_config import BaseConfig
from .generator_registry import GeneratorRegistry


class UniformMutationGeneratorConfig(BaseConfig):
    """Configuration for UniformMutationGenerator."""
    # Required parameters
    sequence_length: int = Field(
        ge=1,
        title="Sequence length",
        description="Target length for generated sequences"
    )

    # Optional parameters (have defaults)
    num_mutations: int = Field(
        default=1,
        ge=0,
        title="Num mutations",
        description="Number of positions to mutate per sample"
    )
    mutation_window: Optional[Tuple[int, int]] = Field(
        default=None,
        title="Mutation window",
        description="Only mutate the sequence within this range. Uses Python conventions for defining the range (start:end)."
    )
    debug_with_sleep_calls: bool = Field(
        default=False,
        title="Debug with sleep calls",
        description="Enable debug mode with sleep calls (for testing purposes only)"
    )

    @model_validator(mode='after')
    def validate_mutation_window(self):
        """Validate that the mutation window has reasonable values."""
        if self.mutation_window is not None:
            if len(self.mutation_window) != 2:
                raise ValueError(
                    f"Mutation window must have two entries, found: {self.mutation_window}"
                )
            if self.mutation_window[0] >= self.sequence_length or \
               self.mutation_window[1] > self.sequence_length:
                raise ValueError(
                    f"Mutation window incompatible with a sequence length of {self.sequence_length}, "
                    f"found: {self.mutation_window}"
           )
        return self


@GeneratorRegistry.register(
    key="uniform-mutation",
    label="Uniform Mutation Generator",
    config=UniformMutationGeneratorConfig,
    description="Random point mutations for sequence diversity",
    type=GeneratorType.MUTATION,
    requires_gpu=False,
)
@final
class UniformMutationGenerator(Generator):
    """
    A sequence generator that proposes random point mutations.

    This generator initializes with a random sequence and samples single-nucleotide
    or amino acid mutations on each call to sample().

    Examples:
        Creating a DNA mutation generator with config:
        >>> from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig
        >>> config = UniformMutationGeneratorConfig(
        ...     sequence_length=100,
        ...     num_mutations=2
        ... )
        >>> gen = UniformMutationGenerator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.DNA)
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
        self.sequence_length = config.sequence_length
        self.num_mutations = config.num_mutations
        self.debug_with_sleep_calls = config.debug_with_sleep_calls
        self.mutation_window = config.mutation_window
        self.type = GeneratorType.MUTATION

    def assign(self, assigned_segment: Segment) -> None:
        """
        Assign a Segment to this generator.

        - If no starting sequence, initialize a uniformly random sequence of configured length.
        - If starting sequence is provided, validates that the sequence length matches the configured length.
        """
        super().assign(assigned_segment)

        valid_chars = assigned_segment._valid_chars - set(" ")
        valid_chars_list = list(valid_chars)

        if not assigned_segment.original_sequence:
            # Generate random sequence
            assigned_segment.original_sequence.sequence = "".join(
                random.choice(valid_chars_list) for _ in range(self.sequence_length)
            )
        else:
            # Validate provided sequence
            if len(assigned_segment.original_sequence.sequence) != self.sequence_length:
                raise ValueError(f"Provided sequence length ({len(assigned_segment.original_sequence.sequence)}) must match generator's configured sequence_length ({self.sequence_length}).")

        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True

    def sample(self) -> None:
        """
        Introduce random point mutations in candidate sequences.

        Mutates each sequence in the candidate pool by selecting random positions
        and replacing characters with different random characters from the vocabulary.

        Raises:
            RuntimeError: If called before assign().
            ValueError: If candidate pool is empty.
        """
        # Sleep for testing purposes if debug_with_sleep_calls is enabled
        if self.debug_with_sleep_calls:
            time.sleep(1.0)

        # Mutate each candidate sequence
        for sequence in self._assigned_segment.candidate_sequences:
            current_sequence = sequence.sequence
            sequence_length = len(current_sequence)

            # Ensure we don't try to mutate more positions than available
            actual_mutations = min(self.num_mutations, sequence_length)

            # Define the positions to mutate
            if self.mutation_window is None:
                window_range = range(sequence_length)
            else:
                window_range = range(self.mutation_window[0], self.mutation_window[1])

            # Select random positions to mutate (without replacement)
            positions_to_mutate = random.sample(window_range, actual_mutations)

            # Apply mutations
            for pos in positions_to_mutate:
                current_char = current_sequence[pos]

                # Make sure the mutated character is different from the current one
                possible_mutations = [
                    c for c in self._assigned_segment._valid_chars if c != current_char
                ]
                mutated_char = random.choice(possible_mutations)
                current_sequence = (
                    current_sequence[:pos]
                    + mutated_char
                    + current_sequence[pos + 1:]
                )

            sequence.sequence = current_sequence
