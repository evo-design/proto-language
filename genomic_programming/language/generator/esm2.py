"""
Esm2 Generator

Extracted from generator.py for better code organization.
"""

from typing import List, final, Optional

from pydantic import Field, field_validator

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from .generator_registry import GeneratorRegistry


class ESM2GeneratorConfig(BaseConfig):
    """Configuration for ESM2Generator."""
    esm2_type: str = Field(default="esm2_t33_650M_UR50D", description="ESM2 model variant")
    esm2_local_path: Optional[str] = Field(default=None, description="Optional path to local model weights")
    sequence_length: int = Field(default=100, ge=1, description="Length of protein sequences to generate")
    temperature: float = Field(default=1.0, gt=0.0, description="Sampling temperature")
    decoding_method: str = Field(
        default="entropy",
        description="Position selection strategy: 'entropy', 'max_logit', or 'random'"
    )
    top_k: int = Field(default=5, ge=1, description="Number of positions to sample per iteration")
    batch_size: int = Field(default=1, ge=1, description="Number of sequences to generate")
    prepend_prompt: bool = Field(default=False, description="Whether to prepend prompt to generated sequences")
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v, info):
        if 'sequence_length' in info.data and v > info.data['sequence_length']:
            raise ValueError(f"top_k ({v}) cannot exceed sequence_length ({info.data['sequence_length']})")
        return v


@GeneratorRegistry.register(
    key="esm2",
    config=ESM2GeneratorConfig,
    description="ESM-2 protein language model for protein sequence generation",
    category="language_model",
    requires_gpu=True,
    supports_batch=True
)
@final
class ESM2Generator(Generator):
    """
    A protein sequence generator using the ESM-2 protein language model.

    This generator uses the ESM-2 protein language model to propose sequences and
    mutations based on the model's logits. It supports various decoding strategies
    for selecting positions to mutate and uses temperature-controlled sampling
    for amino acid selection.

    Examples:
        Basic protein generation:
        >>> from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig
        >>> config = ESM2GeneratorConfig(
        ...     esm2_type="esm2_t33_650M_UR50D",
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5,
        ...     batch_size=3
        ... )
        >>> gen = ESM2Generator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen.assign(segment)  # Creates random initial sequences from mask tokens
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(self, config: ESM2GeneratorConfig) -> None:
        """
        Initialize the ESM-2 generator with model and sampling configuration.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__(batch_size=config.batch_size)
        self.config = config
        self.esm2_type = config.esm2_type
        self.esm2_local_path = config.esm2_local_path
        self.sequence_length = config.sequence_length
        self.temperature = config.temperature
        self.decoding_method = config.decoding_method
        self.top_k = config.top_k
        self.batch_size = config.batch_size
        self.prepend_prompt = config.prepend_prompt

    def assign(self, assigned_segments: Segment) -> None:
        """
        Assign a Segment to this generator.

        Creates initial sequences by running ESM-2 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If the segment already contains sequences, they will be used as starting points.

        Args:
            assigned_segments: A single Segment to be assigned to this generator.

        Raises:
            ValueError: If assigned_segments is not a single Segment object.
            AssertionError: If provided sequence length doesn't match configured length.
        """
        # Validate that we received a single Segment, not a list or other type
        if not isinstance(assigned_segments, Segment):
            raise ValueError(
                f"ESM2Generator.assign() expects a single Segment object, "
                f"got {type(assigned_segments).__name__}. If you have multiple segments, "
                f"assign them to separate generator instances."
            )

        # Validate provided sequence length if not empty
        initial_sequence = assigned_segments.batch_sequences[0].sequence
        if initial_sequence != "":
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )

        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM-2 logits,
        selects top-k positions based on the decoding method, and samples new amino acids
        at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()
        sequences = [self._generator_output.batch_sequences[i].sequence for i in range(self.batch_size)]

        # Use ESM2 sampling tool
        from ...tools.models.language_models.esm2.esm2 import run_esm2_sample, ESM2SampleConfig
        
        config = ESM2SampleConfig(
            sequences=sequences,
            model_name=self.esm2_type,
            sequence_length=self.sequence_length,
            temperature=self.temperature,
            decoding_method=self.decoding_method,
            top_k=self.top_k,
            keep_on_device=True,  # Keep for repeated calls
            verbose=False
        )
        
        result = run_esm2_sample(config)
        mutated_sequences = result.sequences

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._generator_output.batch_sequences[i].sequence = sequence
