"""
Evo2 Generator for DNA sequence generation
"""

from typing import List, Optional, Dict, final
import torch

from pydantic import Field, model_validator

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from proto_language.tools.models.language_models.evo2 import run_evo2_sample, Evo2SampleConfig
from .generator_registry import GeneratorRegistry


class Evo2GeneratorConfig(BaseConfig):
    """Configuration for Evo2Generator."""
    prompts: List[str] = Field(description="Prompt sequences for generation (single prompt or multiple)")
    # TODO: num_replications to replicate prompts and kv caches for beam search
    batch_size: Optional[int] = Field(default=None, ge=1, description="Number of sequences to generate in parallel")
    model_name: str = Field(default="evo2_7b", description="Evo2 model variant to use")
    local_path: Optional[str] = Field(default=None, description="Optional path to local model weights")
    top_k: int = Field(default=4, ge=1, description="Top-k sampling parameter")
    top_p: float = Field(default=1, gt=0.0, le=1.0, description="Top-p sampling parameter")
    temperature: float = Field(default=1.0, gt=0.0, description="Sampling temperature")
    num_tokens: int = Field(default=32, ge=1, description="Number of tokens to generate after prompt")
    cached_generation: bool = Field(default=False, description="Whether to cache model states (for beam search)")
    force_prompt_threshold: Optional[int] = Field(default=None, description="Optional number of tokens to prefill in parallel before switching to prompt forcing. Used to reduce peak memory usage and support longer prompts")
    max_seqlen: Optional[int] = Field(default=None, description="Optional maximum sequence length to generate. Determines the max size of the cache if larger. Otherwise automatically determined using prompt length + max_tokens")
    stop_at_eos: bool = Field(default=True, description="Whether to stop at end-of-sequence token")
    verbose: bool = Field(default=False, description="Whether to print verbose output")
    prepend_prompt: bool = Field(default=False, description="Whether to prepend prompt to generated sequences")

    
    @model_validator(mode='after')
    def validate_prompts_length(self):
        """Validate that all prompts have the same length."""
        if len(set(len(seq) for seq in self.prompts)) != 1:
            raise ValueError(f"All prompts must have same length, got: {[len(seq) for seq in self.prompts]}")
        
        return self


@GeneratorRegistry.register(
    key="evo2",
    label="Evo2 DNA Language Model",
    config=Evo2GeneratorConfig,
    description="Evo2 genome language model for DNA sequence generation",
    category="language_model",
    requires_gpu=True,
    autoregressive=True,
)
@final
class Evo2Generator(Generator):
    """
    A sequence generator that uses the Evo2 genome language model for DNA sequence generation.

    Examples:
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> config = Evo2GeneratorConfig(
        ...     prompts=["ATG"],
        ...     evo2_type="evo2_7b",
        ...     sequence_length=1000,
        ...     temperature=0.8
        ... )
        >>> gen = Evo2Generator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.DNA)
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates sequences from prompts
    """

    def __init__(self, config: Evo2GeneratorConfig) -> None:
        """
        Initialize the Evo2 generator with model configuration and sampling parameters.

        For detailed documentation of Evo2 sampling parameters, refer to:
        https://github.com/arcinstitute/evo2 and https://github.com/Zymrael/vortex

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()
        self.prompts = config.prompts
        self.model_name = config.model_name
        self.local_path = config.local_path
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = config.temperature
        self.num_tokens = config.num_tokens
        self.cached_generation = config.cached_generation
        self.force_prompt_threshold = config.force_prompt_threshold
        self.max_seqlen = config.max_seqlen
        self.stop_at_eos = config.stop_at_eos
        self.verbose = config.verbose
        self.prepend_prompt = config.prepend_prompt

        # Store KV caches for each candidate sequence (List of cache dicts)
        self.kv_caches = None

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a Segment to this generator"""
        # Warn user if existing candidate sequences will be overwritten
        if assigned_segment.original_sequence:
            print(f"Warning: Evo2Generator will overwrite {assigned_segment.original_sequence.sequence} when sample() is called due to autoregressive generation.")

        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True
        self.autoregressive = True

    def sample(self, prompts: Optional[List[str]] = None, batch_size: int = 1) -> None:
        """
        Generate sequences using the Evo2 model and update generator output.

        Args:
            prompts: Optional list of prompt sequences to use instead of self.prompts.
            batch_size: Number of sequences to generate in parallel.
        """
        self._validate_generator()

        # Use provided prompts or fall back to the default prompt
        sampling_prompts = prompts if prompts is not None else self.prompts

        # Validate number of prompts matches candidate pool size
        if len(sampling_prompts) != len(self._assigned_segment.candidate_sequences):
            raise ValueError(f"Number of prompts ({len(sampling_prompts)}) must match candidate pool size ({len(self._assigned_segment.candidate_sequences)})")

        # Prepare KV cache for cached generation
        old_kv_cache = None
        if self.cached_generation and self.kv_caches is not None:
            # Combine per-sample caches back into batched format
            old_kv_cache = self.kv_caches

        # Create config for the tool
        sample_config = Evo2SampleConfig(
            prompts=sampling_prompts,
            model_name=self.model_name,
            local_path=self.local_path,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            num_tokens=self.num_tokens,
            cached_generation=self.cached_generation,
            force_prompt_threshold=self.force_prompt_threshold,
            max_seqlen=self.max_seqlen,
            verbose=self.verbose,
            stop_at_eos=self.stop_at_eos,
            old_kv_cache=old_kv_cache,
        )

        # Run the sampling tool
        result = run_evo2_sample(sample_config)
        generated_sequences = result.sequences

        # Store KV cache for next generation if cached_generation is enabled
        if self.cached_generation and result.kv_caches is not None:
            self.kv_caches = result.kv_caches

        if self.prepend_prompt:
            for i in range(len(generated_sequences)):
                generated_sequences[i] = self.prompts[i] + generated_sequences[i]

        # Update candidate sequences
        for idx, sequence in enumerate(generated_sequences):
            self._assigned_segment.candidate_sequences[idx].sequence = sequence
