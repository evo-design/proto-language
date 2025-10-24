"""
ESM3 Generator for protein sequence generation
"""

from typing import List, final, Optional

from pydantic import Field, field_validator

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from .generator_registry import GeneratorRegistry


class ESM3GeneratorConfig(BaseConfig):
    """Configuration for ESM3Generator."""
    esm3_type: str = Field(default="esm3_sm_open_v1", description="ESM3 model variant")
    esm3_local_path: Optional[str] = Field(default=None, description="Optional path to local model weights")
    sequence_length: int = Field(default=100, ge=1, description="Length of protein sequences to generate")
    temperature: float = Field(default=1.0, gt=0.0, description="Sampling temperature")
    decoding_method: str = Field(
        default="entropy",
        description="Position selection strategy: 'entropy', 'max_logit', or 'random'"
    )
    top_k: int = Field(default=5, ge=1, description="Number of positions to sample per iteration")
    prepend_prompt: bool = Field(default=False, description="Whether to prepend prompt to generated sequences")
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v, info):
        if 'sequence_length' in info.data and v > info.data['sequence_length']:
            raise ValueError(f"top_k ({v}) cannot exceed sequence_length ({info.data['sequence_length']})")
        return v


@GeneratorRegistry.register(
    key="esm3",
    label="ESM3 Protein Language Model",
    config=ESM3GeneratorConfig,
    description="ESM-3 open protein language model for protein sequence generation",
    category="language_model",
    requires_gpu=True,
    autoregressive=False,
)
@final
class ESM3Generator(Generator):
    """
    A protein sequence generator using the ESM-3 open protein language model.

    This generator uses the (open) ESM-3 protein language model to propose sequences and
    mutations based on the model's logits. It supports various decoding strategies
    for selecting positions to mutate and uses temperature-controlled sampling
    for amino acid selection.

    Examples:
        Basic protein generation:
        >>> from proto_language.language.generator import ESM3Generator, ESM3GeneratorConfig
        >>> config = ESM3GeneratorConfig(
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5
        ... )
        >>> gen = ESM3Generator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen.assign(segment)  # Creates random initial sequences from mask tokens
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(self, config: ESM3GeneratorConfig) -> None:
        """
        Initialize the ESM3 generator with model and sampling configuration.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()
        self.esm3_type = config.esm3_type
        self.esm3_local_path = config.esm3_local_path
        self.sequence_length = config.sequence_length
        self.temperature = config.temperature
        self.decoding_method = config.decoding_method
        self.top_k = config.top_k
        self.prepend_prompt = config.prepend_prompt
        self.autoregressive = False
        
    def assign(
        self, assigned_segment: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        Creates initial sequences by running ESM3 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If the segment already contains sequences, they will be used as starting points.

        Args:
            assigned_segment: A single Segment to be assigned to this generator.

        Raises:
            ValueError: If assigned_segment is not a single Segment object.
            AssertionError: If provided sequence length doesn't match configured length.
        """
        initial_sequence = assigned_segment.selected_sequences[0].sequence
        if initial_sequence != "" and len(initial_sequence) != self.sequence_length:
            raise ValueError(f"Provided sequence length ({len(initial_sequence)}) must match configured sequence_length ({self.sequence_length})")

        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM3 logits,
        selects top-k positions based on the decoding method, and samples new amino acids
        at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()
        sequences = [seq.sequence for seq in self._assigned_segment.candidate_sequences]

        # Use ESM3 sampling tool
        from ...tools.models.language_models.esm3.esm3 import run_esm3_sample, ESM3SampleConfig
        from ...tools.models.language_models.schemas import LanguageModelInput

        # Create input and config objects
        esm3_input = LanguageModelInput(sequences=sequences)
        config = ESM3SampleConfig(
            model_name=self.esm3_type,
            sequence_length=self.sequence_length,
            temperature=self.temperature,
            decoding_method=self.decoding_method,
            top_k=self.top_k,
            keep_on_device=True,  # Keep for repeated calls
            verbose=False
        )

        result = run_esm3_sample(inputs=esm3_input, config=config)
        mutated_sequences = result.sequences

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._assigned_segment.candidate_sequences[i].sequence = sequence
