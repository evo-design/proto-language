"""
Evo2 Generator for DNA sequence generation
"""

from typing import List, Optional, Dict, final, Union
import warnings

from pydantic import Field, model_validator

from ..core import Generator, GeneratorType, Segment
from proto_language.base_config import BaseConfig
from proto_language.tools.models.language_models.evo2 import run_evo2_sample, Evo2SampleInput, Evo2SampleConfig
from proto_language.tools.models.language_models.evo2.inference import EVO2_MODEL_CHECKPOINTS
from .generator_registry import GeneratorRegistry


class Evo2GeneratorConfig(BaseConfig):
    """Configuration for Evo2Generator."""
    # Required parameters
    prompts: Union[str, List[str]] = Field(
        title="Prompts",
        description="Prompt sequences for DNA sequence generation (single prompt or multiple)"
    )
    num_tokens: int = Field(
        ge=1,
        title="Num tokens",
        description="Number of tokens to generate after prompt"
    )

    # Optional parameters (have defaults)
    model_checkpoint: EVO2_MODEL_CHECKPOINTS = Field(
        default="evo2_7b",
        title="Model name",
        description="Evo2 model checkpoint to use"
    )
    local_path: Optional[str] = Field(
        default=None,
        title="Local checkpoint path",
        description="Path to local checkpoint weights for custom or finetuned models"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        title="Top-k",
        description="Limits sampling to the top-k most probable tokens at each generation step. Lower values make output more deterministic, higher values increase diversity."
    )
    top_p: float = Field(
        default=1,
        gt=0.0,
        le=1.0,
        title="Top-p",
        description="Nucleus sampling threshold. Chooses the smallest set of tokens whose cumulative probability mass ≥ top-p. Lower values yield more conservative, focused output; higher values allow more randomness."
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling by adjusting probability distribution sharpness. Lower values (<1) make outputs more deterministic; higher values (>1) produce more varied and creative generations."
    )
    force_prompt_threshold: Optional[int] = Field(
        default=None,
        title="Force prompt threshold",
        description="Optional number of tokens to prefill in parallel before switching to prompt forcing. Used to reduce peak memory usage and support longer prompts"
    )
    max_seqlen: Optional[int] = Field(
        default=None,
        title="Max sequence length",
        description="Optional maximum sequence length to generate. Determines the max size of the cache if larger. Otherwise automatically determined using prompt length + max_tokens"
    )
    stop_at_eos: bool = Field(
        default=True,
        title="Stop at EOS",
        description="Whether to stop at end-of-sequence token"
    )
    batched: bool = Field(
        default=True,
        title="Batched",
        description="Whether to use batched generation, set to true if # of prompts > 1."
    )
    cached_generation: bool = Field(
        default=True,
        title="Cached generation",
        description="Whether to use cached generation"
    )
    store_kv_cache: bool = Field(
        default=False,
        title="KV caching",
        description="Whether to store and reuse kv cache"
    )
    prepend_prompt: bool = Field(
        default=False,
        title="Prepend prompt",
        description="Whether to prepend prompt to generation"
    )
    verbose: bool = Field(
        default=False,
        title="Verbose",
        description="Whether to print verbose output"
    )
    
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
    type=GeneratorType.AUTOREGRESSIVE,
    requires_gpu=True,
)
@final
class Evo2Generator(Generator):
    """
    A sequence generator that uses the Evo2 genome language model for DNA sequence generation.

    Examples:
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> config = Evo2GeneratorConfig(
        ...     prompts="ATG",
        ...     num_tokens=1000,
        ...     model_checkpoint="evo2_7b",
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
        self.model_checkpoint = config.model_checkpoint
        self.local_path = config.local_path
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = config.temperature
        self.num_tokens = config.num_tokens
        self.force_prompt_threshold = config.force_prompt_threshold
        self.max_seqlen = config.max_seqlen
        self.stop_at_eos = config.stop_at_eos
        self.verbose = config.verbose
        self.cached_generation = config.cached_generation
        self.batched = config.batched
        self.store_kv_cache = config.store_kv_cache
        self.prepend_prompt = config.prepend_prompt
        self.type = GeneratorType.AUTOREGRESSIVE

        # store old KV caches for cached generation
        self.kv_caches: List[Dict] = []

    def assign(self, assigned_segment: Segment) -> None:
        """
        Assign a Segment to this generator.

        If starting sequence is provided, warn user it will be overwritten by sample().
        """
        super().assign(assigned_segment)
        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True

    def sample(self, prompts: Optional[List[str]] = None, prepend_prompt: Optional[bool] = None, old_kv_cache: Optional[Dict] = None) -> None:
        """
        Generate sequences using the Evo2 model and update generator output.

        When store_kv_cache=True, stores KV caches in self.kv_caches for access
        by beam search optimizer. The caches are overwritten on each call to sample().

        Args:
            prompts: Optional list of prompt sequences to use instead of self.prompts.
            prepend_prompt: Optional boolean to prepend prompts to generated sequences.
            old_kv_cache: Optional cache state to continue from (batched format). 
        """
        # Use provided prompts or fall back to the default prompt
        sampling_prompts = prompts if prompts is not None else self.prompts

        # Warn if number of prompts does not match candidate pool size
        if len(sampling_prompts) != len(self._assigned_segment.candidate_sequences):
            warnings.warn(f"Number of prompts ({len(sampling_prompts)}) does not match candidate pool size ({len(self._assigned_segment.candidate_sequences)})")

        # Create config for the tool
        inputs = Evo2SampleInput(prompts=sampling_prompts)
        sample_config = Evo2SampleConfig(
            prompts=sampling_prompts,
            prepend_prompt=prepend_prompt if prepend_prompt is not None else self.prepend_prompt,
            model_checkpoint=self.model_checkpoint,
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
            keep_on_gpu=True, # Keep for repeated calls
            batched=True
        )

        # Run the sampling tool
        evo2_output = run_evo2_sample(inputs=inputs, config=sample_config)
        generated_sequences = evo2_output.sequences
        self.kv_caches = evo2_output.kv_caches if self.store_kv_cache else None

        # Update candidate sequences
        for idx, sequence in enumerate(generated_sequences):
            self._assigned_segment.candidate_sequences[idx].sequence = sequence

    def replicate_cache(self, cache: Dict, n_replicates: int) -> Dict:
        """Replicate cache N times for beam branching."""
        from vortex.model.cache import InferenceParams, HyenaCascadeIIRInferenceParams, HyenaCascadeFIRInferenceParams
        if not cache:
            return cache

        if n_replicates < 1:
            raise ValueError(f'n_replicates must be at least 1 (found {n_replicates}).')

        kv = next(iter(cache['mha'].key_value_memory_dict.values()))
        if kv.shape[0] != 1:
            raise ValueError(f'Cache must only have one cache entry to replicate (found {kv.shape[0]}).')

        mha, hcl, hcm, hcs = cache['mha'], cache['hcl'], cache['hcm'], cache['hcs']

        return {
            'mha': InferenceParams(
                max_seqlen=mha.max_seqlen,
                max_batch_size=mha.max_batch_size,
                seqlen_offset=mha.seqlen_offset,
                batch_size_offset=mha.batch_size_offset,
                key_value_memory_dict={
                    key: data.repeat(n_replicates, 1, 1, 1, 1)
                    for key, data in mha.key_value_memory_dict.items()
                },
            ),
            'hcl': HyenaCascadeIIRInferenceParams(
                fir_filter_length=hcl.fir_filter_length,
                state_dim=hcl.state_dim,
                seqlen_offset=hcl.seqlen_offset,
                fir_state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcl.fir_state_dict.items()
                },
                state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcl.state_dict.items()
                },
            ),
            'hcm': HyenaCascadeFIRInferenceParams(
                fir_filter_length=hcm.fir_filter_length,
                seqlen_offset=hcm.seqlen_offset,
                fir_inner_filter_length=hcm.fir_inner_filter_length,
                fir_state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcm.fir_state_dict.items()
                },
                fir_inner_state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcm.fir_inner_state_dict.items()
                },
                state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcm.state_dict.items()
                },
            ),
            'hcs': HyenaCascadeFIRInferenceParams(
                fir_filter_length=hcs.fir_filter_length,
                seqlen_offset=hcs.seqlen_offset,
                fir_inner_filter_length=hcs.fir_inner_filter_length,
                fir_state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcs.fir_state_dict.items()
                },
                fir_inner_state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcs.fir_inner_state_dict.items()
                },
                state_dict={
                    key: data.repeat(n_replicates, 1, 1)
                    for key, data in hcs.state_dict.items()
                },
            )
        }