"""
UniformMutationGenerator for random point mutations.
"""
from __future__ import annotations
from typing import final, Optional, Tuple
import random
import time

from proto_language.language.core import Generator, Segment
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.generator.generator_registry import GeneratorRegistry


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

        mutation_window (Optional[Tuple[int, int]]): Optional window specifying which
            region of the sequence to mutate. Format: ``(start, end)`` using Python
            indexing (0-based, end-exclusive). For example:

            - ``(0, 100)``: Mutate only first 100 positions
            - ``(50, 150)``: Mutate only positions 50-149
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
    mutation_window: Optional[Tuple[int, int]] = ConfigField(
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


@GeneratorRegistry.register(
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
        mutation_window (Optional[Tuple[int, int]]): Optional region to restrict mutations.
        debug_with_sleep_calls (bool): Whether to add sleep delays for testing.
        category (str): Set to ``"mutation"``.
        supported_sequence_types (List[SequenceType]): Empty list indicates support for all sequence types.

    Example:
        >>> from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig
        >>> from proto_language.language.core import Segment, SequenceType
        >>> config = UniformMutationGeneratorConfig(num_mutations=2)
        >>> gen = UniformMutationGenerator(config)
        >>> segment = Segment(length=100, sequence_type="dna")
        >>> gen.assign(segment)
        >>> gen.sample()  # Introduces 2 random mutations
    """

    supported_sequence_types = []  # Empty list means supports all sequence types

    def __init__(self, config: UniformMutationGeneratorConfig) -> None:
        """
        Initialize the uniform mutation generator.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()
        self.num_mutations = config.num_mutations
        self.debug_with_sleep_calls = config.debug_with_sleep_calls
        self.mutation_window = config.mutation_window
        self.category = "mutation"

    def assign(self, assigned_segment: Segment) -> None:
        """
        Assign a Segment to this generator.

        - If no starting sequence, initialize a uniformly random sequence matching segment's length.
        - Validates mutation_window against segment's sequence_length if specified.
        """
        super().assign(assigned_segment)

        # Validate mutation window against segment's sequence_length
        if self.mutation_window is not None:
            if len(self.mutation_window) != 2:
                raise ValueError(f"Mutation window must have two entries (got {self.mutation_window})")
            if self.mutation_window[0] >= assigned_segment.sequence_length or self.mutation_window[1] > assigned_segment.sequence_length:
                raise ValueError(f"Mutation window {self.mutation_window} incompatible with segment length {assigned_segment.sequence_length}")

        valid_chars = assigned_segment._valid_chars - set(" ")
        valid_chars_list = list(valid_chars)

        # Generate random sequence matching segment's length
        if not assigned_segment.original_sequence.sequence:
            assigned_segment.original_sequence.sequence = "".join(random.choice(valid_chars_list) for _ in range(assigned_segment.sequence_length))

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
