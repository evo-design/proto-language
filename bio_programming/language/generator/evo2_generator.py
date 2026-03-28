"""
proto_language/language/generator/evo2_generator.py

Evo2 Generator for DNA sequence generation
"""

from __future__ import annotations

from typing import Dict, List, Optional, final

from proto_tools import Evo2SampleConfig, Evo2SampleInput, run_evo2_sample
from proto_tools.tools.causal_models.evo2.evo2_sample import (
    EVO2_MODEL_CHECKPOINTS,
)
from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import generator


class Evo2GeneratorConfig(BaseConfig):
    """Configuration object for Evo2Generator.

    This class defines configuration parameters for the Evo2 generator, which uses
    a 7B parameter genomic language model to generate DNA sequences autoregressively
    from prompt sequences.

    Attributes:
        prompts (list[str]): Prompt sequence(s) to start DNA generation.
            Can be a single prompt string (automatically converted to list) or list of
            prompts for batch generation. All prompts must have the same length.
            Uses Evo2's special formatting (refer to Evo2 documentation for prompt
            format details).

        model_checkpoint (EVO2_MODEL_CHECKPOINTS): Evo2 model checkpoint to use. Options:

            - ``"evo2_7b"``: 7 billion parameter Evo2 model (default)

            Default: ``"evo2_7b"``.

        local_path (str | None): Path to local model weights directory for custom
            or fine-tuned models. If ``None``, downloads from Hugging Face.
            Default: ``None``.

        top_k (int): Limits sampling to the top-k most probable tokens at each
            generation step. Lower values produce more focused sequences, higher
            values increase diversity. Must be at least 1. Default: 4.

        top_p (float): Nucleus sampling parameter. Chooses tokens whose cumulative
            probability mass is at least ``top_p``. Range: (0.0, 1.0]. Default: 1.0.

        temperature (float): Scales randomness of sampling by adjusting probability
            distribution sharpness. Lower values are more deterministic, higher
            values more diverse. Must be greater than 0. Default: 1.0.

        force_prompt_threshold (int | None): Optional number of tokens to prefill
            in parallel before switching to autoregressive generation. Can speed up
            generation for long prompts. Default: ``None``.

        max_seqlen (int | None): Optional maximum sequence length to generate.
            Determines KV cache size. If ``None``, automatically calculated.
            Default: ``None``.

        stop_at_eos (bool): Whether to stop generation when end-of-sequence token
            is encountered. If ``False``, always generates exactly ``num_tokens``.
            Default: ``True``.

        batched (bool): Whether to use batched generation when multiple prompts
            are provided. Batched generation is faster but requires all prompts
            to have the same length. Default: ``True``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.

        cached_generation (bool): Whether to use KV caching for faster generation.
            Caching stores intermediate states to avoid recomputation.
            Default: ``True``.

        store_kv_cache (bool): Whether to store and expose KV caches after generation.
            Useful for beam search optimizers or continued generation. Caches are
            stored in ``self.kv_caches`` and overwritten on each ``sample()`` call.
            Default: ``False``.

        prepend_prompt (bool): Whether to prepend the prompt to the generated
            sequence in the output. If ``False``, only newly generated tokens are
            returned. Default: ``False``.

        verbose (bool): Whether to print detailed generation progress and timing.
            Default: ``False``.

    Note:
        All prompts must have identical lengths for batched generation. For detailed
        information on Evo2 parameters, see: https://github.com/arcinstitute/evo2
    """

    # Required parameters
    prompts: List[str] = ConfigField(
        title="Prompts",
        description="Prompt sequences for DNA sequence generation (single prompt or multiple)",
    )
    model_checkpoint: EVO2_MODEL_CHECKPOINTS = ConfigField(
        default="evo2_7b",
        title="Model Checkpoint",
        description="Evo2 model checkpoint to use",
    )

    # Advanced parameters
    local_path: Optional[str] = ConfigField(
        default=None,
        title="Local Checkpoint Path",
        description="Path to local checkpoint weights for custom or finetuned models",
        hidden=True,
    )
    top_k: int = ConfigField(
        default=4,
        ge=1,
        title="Top-k",
        description="Limits sampling to the top-k most probable tokens at each generation step.",
        advanced=True,
    )
    top_p: float = ConfigField(
        title="Top-p",
        default=1,
        gt=0.0,
        le=1.0,
        description="Chooses the smallest set of tokens whose cumulative probability mass ≥ top-p.",
        advanced=True,
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling by adjusting probability distribution sharpness.",
        advanced=True,
    )
    force_prompt_threshold: Optional[int] = ConfigField(
        default=None,
        title="Force Prompt Threshold",
        description="Optional number of tokens to prefill in parallel before switching to prompt forcing.",
        advanced=True,
    )
    max_seqlen: Optional[int] = ConfigField(
        default=None,
        title="Max Sequence Length",
        description="Optional maximum sequence length to generate. Determines the max size of the cache if larger.",
        advanced=True,
    )
    stop_at_eos: bool = ConfigField(
        default=True,
        title="Stop at EOS",
        description="Whether to stop at end-of-sequence token",
        advanced=True,
    )
    # Determine how we want to handle this for the client.
    batched: bool = ConfigField(
        default=True,
        title="Batched",
        description="Whether to use batched generation, set to true if # of prompts > 1.",
        hidden=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )
    cached_generation: bool = ConfigField(
        default=True,
        title="Cached Generation",
        description="Whether to use cached generation",
        advanced=True,
    )
    store_kv_cache: bool = ConfigField(
        default=False,
        title="Store KV Cache",
        description="Whether to store and reuse Key-Value cache",
        advanced=True,
        depends_on={"field": "cached_generation"},
    )
    prepend_prompt: bool = ConfigField(
        default=False,
        title="Prepend Prompt",
        description="Whether to prepend prompt to generation",
        hidden=True,
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print verbose output",
        hidden=True,
    )

    @field_validator("prompts", mode="before")
    @classmethod
    def normalize_prompts(cls, v):
        """Convert single string to list for consistent internal handling."""
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_prompts_length(self):
        """Validate that all prompts have the same length."""
        if len(set(len(seq) for seq in self.prompts)) != 1:
            raise ValueError(f"All prompts must have same length, got: {[len(seq) for seq in self.prompts]}")

        return self


@generator(
    key="evo2",
    label="Evo2 DNA Language Model",
    config=Evo2GeneratorConfig,
    description="Evo2 genome language model for DNA sequence generation",
    uses_gpu=True,
    tools_called=["evo2-sample"],
    category="autoregressive",
    supported_sequence_types=["dna"],
)
@final
class Evo2Generator(Generator):
    """Sequence generator using Evo2 genomic language model for DNA generation.

    This generator uses the Evo2 7B parameter model to autoregressively generate
    DNA sequences from prompt sequences. Supports advanced sampling strategies,
    KV caching for efficiency, and batch generation.

    The generator category is ``"autoregressive"``, indicating sequences
    are generated token-by-token from left to right.

    The number of tokens to generate is automatically calculated based on the
    assigned segment's sequence_length, prompt length, and prepend_prompt setting.

    Attributes:
        prompts: Prompt sequences for generation.
        model_checkpoint: Evo2 model checkpoint name.
        temperature: Sampling temperature for diversity control.
        kv_caches: Stored KV caches when ``store_kv_cache=True``.
        batch_size (int): Number of sequences to generate per batch.

    Example:
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> from proto_language.language.core import Segment, SequenceType
        >>> config = Evo2GeneratorConfig(
        ...     prompts="ATG",
        ...     temperature=0.8
        ... )
        >>> gen = Evo2Generator(config)
        >>> # Segment length determines how many tokens to generate
        >>> segment = Segment(length=1003, sequence_type="dna")
        >>> gen.assign(segment)  # num_tokens = 1003 - 3 = 1000
        >>> gen.sample()  # Generates DNA sequences
    """

    def __init__(self, config: Evo2GeneratorConfig) -> None:
        """
        Initialize the Evo2 generator with model configuration and sampling parameters.

        For detailed documentation of Evo2 sampling parameters, refer to:
        https://github.com/arcinstitute/evo2 and https://github.com/Zymrael/vortex

        Args:
            config (Evo2GeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.prompts = config.prompts
        self.model_checkpoint = config.model_checkpoint
        self.local_path = config.local_path
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = config.temperature
        self.force_prompt_threshold = config.force_prompt_threshold
        self.max_seqlen = config.max_seqlen
        self.stop_at_eos = config.stop_at_eos
        self.verbose = config.verbose
        self.cached_generation = config.cached_generation
        self.batched = config.batched
        self.batch_size = config.batch_size
        self.store_kv_cache = config.store_kv_cache
        self.prepend_prompt = config.prepend_prompt
        self.kv_caches: List[Dict] = []

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prepend_prompt: Optional[bool] = None,
        num_tokens: Optional[int] = None,
        old_kv_cache: Optional[Dict] = None,
    ) -> None:
        """Generate sequences using the Evo2 model.

        Args:
            prompts (list[str] | None): Optional prompts to use instead of self.prompts.
            prepend_prompt (bool | None): Optional override for prepend_prompt setting.
            num_tokens (int | None): Optional explicit token count (used by beam search).
            old_kv_cache (dict | None): Optional cache state to continue from (batched format).
        """
        self._validate_generator()

        sampling_prompts = prompts if prompts is not None else self._replicate_prompts(self.prompts)
        prepend_prompt = prepend_prompt if prepend_prompt is not None else self.prepend_prompt
        if num_tokens is None:
            num_tokens = self._compute_num_tokens(len(sampling_prompts[0]), prepend_prompt)

        inputs = Evo2SampleInput(prompts=sampling_prompts)
        sample_config = Evo2SampleConfig(
            prepend_prompt=prepend_prompt,
            model_checkpoint=self.model_checkpoint,
            local_path=self.local_path,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            num_tokens=num_tokens,
            cached_generation=self.cached_generation,
            force_prompt_threshold=self.force_prompt_threshold,
            max_seqlen=self.max_seqlen,
            verbose=self.verbose,
            stop_at_eos=self.stop_at_eos,
            old_kv_cache=old_kv_cache,
            keep_on_gpu=True,
            batched=self.batched,
            batch_size=self.batch_size,
        )

        evo2_output = run_evo2_sample(inputs=inputs, config=sample_config)
        generated_sequences = evo2_output.sequences
        self.kv_caches = evo2_output.kv_caches if self.store_kv_cache else []

        for proposal, sequence in zip(
            self._assigned_segment.proposal_sequences, generated_sequences, strict=True
        ):
            proposal.sequence = sequence

    def replicate_cache(self, cache: Dict, n_replicates: int) -> Dict:
        """Replicate cache N times for beam branching."""
        from vortex.model.cache import (
            HyenaCascadeFIRInferenceParams,
            HyenaCascadeIIRInferenceParams,
            InferenceParams,
        )

        if not cache:
            return cache

        if n_replicates < 1:
            raise ValueError(f"n_replicates must be at least 1 (found {n_replicates}).")

        kv = next(iter(cache["mha"].key_value_memory_dict.values()))
        if kv.shape[0] != 1:
            raise ValueError(f"Cache must only have one cache entry to replicate (found {kv.shape[0]}).")

        mha, hcl, hcm, hcs = cache["mha"], cache["hcl"], cache["hcm"], cache["hcs"]

        return {
            "mha": InferenceParams(
                max_seqlen=mha.max_seqlen,
                max_batch_size=mha.max_batch_size,
                seqlen_offset=mha.seqlen_offset,
                batch_size_offset=mha.batch_size_offset,
                key_value_memory_dict={
                    key: data.repeat(n_replicates, 1, 1, 1, 1)
                    for key, data in mha.key_value_memory_dict.items()
                },
            ),
            "hcl": HyenaCascadeIIRInferenceParams(
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
            "hcm": HyenaCascadeFIRInferenceParams(
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
            "hcs": HyenaCascadeFIRInferenceParams(
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
            ),
        }

    def _replicate_prompts(self, prompts: List[str]) -> List[str]:
        """Match prompt count to proposal count, replicating single prompts."""
        num_proposals = len(self._assigned_segment.proposal_sequences)
        if len(prompts) == num_proposals:
            return prompts
        if len(prompts) == 1:
            return prompts * num_proposals
        raise ValueError(
            f"Expected 1 or {num_proposals} prompts, got {len(prompts)}"
        )

    def _compute_num_tokens(
        self, prompt_length: int, prepend_prompt: bool
    ) -> int:
        """Compute tokens to generate based on segment length and prompt settings."""
        segment_length = self._assigned_segment.sequence_length
        num_tokens = (
            (segment_length - prompt_length) if prepend_prompt else segment_length
        )
        if num_tokens < 1:
            raise ValueError(
                f"Prompt length ({prompt_length}) exceeds segment length ({segment_length})"
            )
        return num_tokens
