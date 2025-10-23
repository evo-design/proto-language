"""
Evo2 Generator for DNA sequence generation
"""

from typing import List, Optional, Dict, final
import warnings

from pydantic import Field, model_validator

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from proto_language.tools.models.language_models.evo2 import run_evo2_sample, Evo2SampleConfig
from .generator_registry import GeneratorRegistry


class Evo2GeneratorConfig(BaseConfig):
    """Configuration for Evo2Generator."""
    prompts: List[str] = Field(description="Prompts for DNA sequence generation (single prompt or multiple)")
    model_name: str = Field(default="evo2_7b", description="Evo2 model variant to use")
    local_path: Optional[str] = Field(default=None, description="Optional path to local model weights")
    top_k: int = Field(default=4, ge=1, description="Top-k sampling parameter")
    top_p: float = Field(default=1, gt=0.0, le=1.0, description="Top-p sampling parameter")
    temperature: float = Field(default=1.0, gt=0.0, description="Sampling temperature")
    num_tokens: int = Field(default=32, ge=1, description="Number of tokens to generate after prompt")
    force_prompt_threshold: Optional[int] = Field(default=None, description="Optional number of tokens to prefill in parallel before switching to prompt forcing. Used to reduce peak memory usage and support longer prompts")
    max_seqlen: Optional[int] = Field(default=None, description="Optional maximum sequence length to generate. Determines the max size of the cache if larger. Otherwise automatically determined using prompt length + max_tokens")
    stop_at_eos: bool = Field(default=True, description="Whether to stop at end-of-sequence token")
    verbose: bool = Field(default=False, description="Whether to print verbose output")
    
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
        self.force_prompt_threshold = config.force_prompt_threshold
        self.max_seqlen = config.max_seqlen
        self.stop_at_eos = config.stop_at_eos
        self.verbose = config.verbose
        self.autoregressive = True

        # store old KV caches for cached generation
        self.kv_caches: List[Dict] = []

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a Segment to this generator"""
        # Warn user if existing candidate sequences will be overwritten
        if assigned_segment.original_sequence:
            print(f"Warning: Evo2Generator will overwrite {assigned_segment.original_sequence.sequence} when sample() is called due to autoregressive generation.")

        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True
        self.autoregressive = True

    def sample(self, prompts: Optional[List[str]] = None, prepend_prompt: bool = False, cached_generation: bool = False, old_kv_cache: Optional[Dict] = None) -> None:
        """
        Generate sequences using the Evo2 model and update generator output.

        When cached_generation=True, stores KV caches in self.kv_caches for access
        by beam search optimizer. The caches are overwritten on each call to sample().

        Args:
            prompts: Optional list of prompt sequences to use instead of self.prompts.
            prepend_prompt: Whether to prepend the prompt to the generated sequence in the output.
            cached_generation: Whether to enable KV caching. Set to True for beam search
                and other sequential generation contexts. Set to False (default) for 
                independent sampling (MCMC, TopK). When True, stores caches in self.kv_caches.
            old_kv_cache: Optional cache state to continue from (batched format). 
                Requires cached_generation=True.
        """
        self._validate_generator()

        # Use provided prompts or fall back to the default prompt
        sampling_prompts = prompts if prompts is not None else self.prompts

        # Warn if number of prompts does not match candidate pool size
        if len(sampling_prompts) != len(self._assigned_segment.candidate_sequences):
            warnings.warn(f"Number of prompts ({len(sampling_prompts)}) does not match candidate pool size ({len(self._assigned_segment.candidate_sequences)})")

        # Create config for the tool
        sample_config = Evo2SampleConfig(
            prompts=sampling_prompts,
            prepend_prompt=prepend_prompt,
            model_name=self.model_name,
            local_path=self.local_path,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            num_tokens=self.num_tokens,
            cached_generation=cached_generation,
            force_prompt_threshold=self.force_prompt_threshold,
            max_seqlen=self.max_seqlen,
            verbose=self.verbose,
            stop_at_eos=self.stop_at_eos,
            old_kv_cache=old_kv_cache,
            keep_on_device=True,
        )

        # Run the sampling tool
        evo2_output = run_evo2_sample(sample_config)
        generated_sequences = evo2_output.sequences
        self.kv_caches = evo2_output.kv_caches if cached_generation else None

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