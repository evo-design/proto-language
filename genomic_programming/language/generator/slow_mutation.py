"""
SlowMutationGenerator for testing and demonstration.

A generator that introduces mutations slowly with configurable delays.
"""

from typing import final
import random
import time

from pydantic import Field

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from .generator_registry import GeneratorRegistry


class SlowMutationGeneratorConfig(BaseConfig):
    """Configuration for SlowMutationGenerator (for testing)."""
    batch_size: int = Field(default=1, ge=1, description="Number of sequence variants to generate")
    sequence_length: int = Field(default=20, ge=1, description="Length of sequences to generate")
    sleep_time: float = Field(default=2.0, ge=0.0, description="Sleep time in seconds per sample call")


@GeneratorRegistry.register(
    key="slow-mutation",
    config=SlowMutationGeneratorConfig,
    description="Slow mutations for testing and demonstration",
    category="mutation",
    requires_gpu=False,
    supports_batch=True
)
@final
class SlowMutationGenerator(Generator):
    """A generator that introduces mutations slowly for testing and demonstration purposes."""
    
    def __init__(self, config: SlowMutationGeneratorConfig):
        super().__init__(batch_size=config.batch_size)
        self.config = config
        self.sequence_length = config.sequence_length
        self.sleep_time = config.sleep_time
        
    def assign(self, assigned_segments: Segment) -> None:
        """Assign a Segment to this generator.
        
        Args:
            assigned_segments: A single Segment to be assigned to this generator.
        """
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        
        # Initialize with random sequence if empty
        if self._generator_output.batch_sequences[0].sequence == "":
            valid_chars = list(self._generator_output._valid_chars - set(" "))
            initial_sequence = "".join(
                random.choice(valid_chars) for _ in range(self.sequence_length)
            )
            self._generator_output.batch_sequences[0].sequence = initial_sequence
            
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True
        
    def sample(self):
        """Sample with a sleep to simulate slow processing."""
        self._validate_generator()
        
        time.sleep(self.sleep_time)
        
        # Simple mutation logic
        for sequence in self._generator_output.batch_sequences:
            if len(sequence.sequence) > 0:
                # Mutate a random position
                mutated_index = random.randint(0, len(sequence.sequence) - 1)
                current_sequence = sequence.sequence
                current_char = current_sequence[mutated_index]
                
                # Get valid mutations
                possible_mutations = [
                    c for c in self._generator_output._valid_chars 
                    if c != current_char and c != " "
                ]
                
                if possible_mutations:
                    mutated_char = random.choice(possible_mutations)
                    new_sequence = (
                        current_sequence[:mutated_index] + 
                        mutated_char + 
                        current_sequence[mutated_index + 1:]
                    )
                    sequence.sequence = new_sequence

