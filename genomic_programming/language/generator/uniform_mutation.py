"""
UniformMutationGenerator for random point mutations.

A sequence generator that proposes random point mutations.
"""

from typing import final
import random
import time

from pydantic import Field

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from .generator_registry import GeneratorRegistry


class UniformMutationGeneratorConfig(BaseConfig):
    """Configuration for UniformMutationGenerator."""
    sequence_length: int = Field(default=100, ge=1, description="Length of sequences to generate")
    num_mutations: int = Field(default=1, ge=0, description="Number of mutations per sequence per sample")
    debug_with_sleep_calls: bool = Field(default=False, description="Enable debug mode with sleep calls (for testing purposes only)")


@GeneratorRegistry.register(
    key="uniform-mutation",
    label="Uniform Mutation Generator",
    config=UniformMutationGeneratorConfig,
    description="Random point mutations for sequence diversity",
    category="mutation",
    requires_gpu=False,
    autoregressive=False,
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

        Using a mutation scheduler:
        >>> def mutation_scheduler(iteration: int) -> int:
        ...     return max(1, 10 - iteration // 10)  # Decrease mutations over time
        >>> config = UniformMutationGeneratorConfig(
        ...     sequence_length=100,
        ...     mutation_scheduler=mutation_scheduler
        ... )
        >>> gen = UniformMutationGenerator(config)
    """

    def __init__(self, config: UniformMutationGeneratorConfig) -> None:
        """
        Initialize the uniform mutation generator.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.sequence_length = config.sequence_length
        self.num_mutations = config.num_mutations
        self.debug_with_sleep_calls = config.debug_with_sleep_calls
        self.mutation_scheduler = None  # Can be set after initialization if needed
        self.autoregressive = False

    def assign(self, assigned_segment: Segment) -> None:
        """
        Initializes the segment's selected pool with a sequence of the configured length.
        Generator will write to candidate_sequences during sample() calls.

        Args:
            assigned_segment: A single Segment to be assigned to this generator.
        """
        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True

        initial_sequence = self._assigned_segment.selected_sequences[0].sequence
        valid_chars = self._assigned_segment._valid_chars - set(" ")
        valid_chars_list = list(valid_chars)

        if initial_sequence == "":
            # Generate random sequence
            self._assigned_segment.selected_sequences[0].sequence = "".join(
                random.choice(valid_chars_list) for _ in range(self.sequence_length)
            )
        else:
            # Validate provided sequence
            if len(initial_sequence) != self.sequence_length:
                raise ValueError(
                    f"Provided sequence length ({len(initial_sequence)}) must match "
                    f"configured sequence_length ({self.sequence_length})"
                )

        self._is_initialized = True

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

        # Determine number of mutations for this iteration
        if self.mutation_scheduler is not None:
            current_mutations = self.mutation_scheduler(self.iteration_count)
        else:
            current_mutations = self.num_mutations

        # Mutate each candidate sequence
        for sequence in self._assigned_segment.candidate_sequences:
            current_sequence = sequence.sequence
            sequence_length = len(current_sequence)

            # Ensure we don't try to mutate more positions than available
            actual_mutations = min(current_mutations, sequence_length)

            # Select random positions to mutate (without replacement)
            positions_to_mutate = random.sample(range(sequence_length), actual_mutations)

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
