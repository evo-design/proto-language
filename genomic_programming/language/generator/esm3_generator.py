"""
ESM3 Generator for protein sequence generation
"""

from typing import final, Literal

from pydantic import Field, field_validator

from ..core import Generator, GeneratorType, Segment
from proto_language.base_config import BaseConfig
from proto_language.tools.models.language_models.esm3.esm3 import run_esm3_sample, ESM3SampleConfig, LanguageModelInput
from .generator_registry import GeneratorRegistry


class ESM3GeneratorConfig(BaseConfig):
    """Configuration for ESM3Generator."""
    # Required parameters
    sequence_length: int = Field(
        ge=1,
        title="Sequence length",
        description="Target length for generated sequences"
    )

    # Optional parameters (have defaults)
    model_checkpoint: Literal["esm3_sm_open_v1"] = Field(
        default="esm3_sm_open_v1",
        title="Model type",
        description="ESM3 model checkpoint to use"
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling by adjusting probability distribution sharpness. Lower values (<1) make outputs more deterministic; higher values (>1) produce more varied and creative generations."
    )
    decoding_method: Literal["entropy", "max_logit", "random"] = Field(
        default="entropy",
        title="Decoding method",
        description="Position selection strategy for sampling: entropy, max_logit, or random"
    )
    num_mutations: int = Field(
        default=1,
        ge=1,
        title="Num mutations",
        description="Number of positions to mutate per sampling iteration"
    )
    
    @field_validator('num_mutations')
    @classmethod
    def validate_num_mutations(cls, v, info):
        if 'sequence_length' in info.data and v > info.data['sequence_length']:
            raise ValueError(f"num_mutations ({v}) cannot exceed sequence_length ({info.data['sequence_length']})")
        return v


@GeneratorRegistry.register(
    key="esm3",
    label="ESM3 Protein Language Model",
    config=ESM3GeneratorConfig,
    description="ESM-3 open protein language model for protein sequence generation",
    type=GeneratorType.MUTATION,
    requires_gpu=True,
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
        ...     num_mutations=5
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
        self.model_checkpoint = config.model_checkpoint
        self.sequence_length = config.sequence_length
        self.temperature = config.temperature
        self.decoding_method = config.decoding_method
        self.num_mutations = config.num_mutations
        self.type = GeneratorType.MUTATION
        
    def assign(
        self, assigned_segment: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        - If starting sequence is provided, validates that the sequence length matches the configured length.
        """
        super().assign(assigned_segment)
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
        # Create input and config objects
        sequences = [seq.sequence for seq in self._assigned_segment.candidate_sequences]
        esm3_input = LanguageModelInput(sequences=sequences)
        config = ESM3SampleConfig(
            model_checkpoint=self.model_checkpoint,
            sequence_length=self.sequence_length,
            temperature=self.temperature,
            decoding_method=self.decoding_method,
            num_mutations=self.num_mutations,
            keep_on_gpu=True,  # Keep for repeated calls
            verbose=False
        )

        result = run_esm3_sample(inputs=esm3_input, config=config)
        mutated_sequences = result.sequences

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._assigned_segment.candidate_sequences[i].sequence = sequence
