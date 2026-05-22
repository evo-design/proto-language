"""Evo1 Generator for DNA sequence generation."""

from typing import Any, final

from proto_tools import (
    EVO1_MODEL_CHECKPOINTS,
    Evo1SampleConfig,
    Evo1SampleInput,
    run_evo1_sample,
)
from pydantic import field_validator, model_validator

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField


class Evo1GeneratorConfig(BaseConfig):
    """Configuration object for Evo1Generator.

    Attributes:
        prompts (list[str]): Prompt sequence(s) for DNA generation. All prompts must
            have the same length.
        model_checkpoint (EVO1_MODEL_CHECKPOINTS): Evo1 model checkpoint to use.
        top_k (int): Top-k sampling parameter.
        temperature (float): Sampling temperature.
        prepend_prompt (bool): Whether to prepend the prompt to the output.
        device (str): GPU device to run Evo1 on, e.g. ``"cuda"`` or
            ``"cuda:0"``.
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors.
        verbose (bool): Whether to print generation progress.
    """

    prompts: list[str] = ConfigField(
        title="Prompts",
        description="Prompt sequences for DNA generation (single prompt or multiple)",
    )
    model_checkpoint: EVO1_MODEL_CHECKPOINTS = ConfigField(
        default="evo-1-8k-base",
        title="Model Checkpoint",
        description="Evo1 model variant to load (e.g. evo-1-8k-base).",
    )

    # Advanced parameters
    top_k: int = ConfigField(
        default=4,
        ge=1,
        title="Top-k",
        description="At each step, restrict sampling to the k most probable tokens.",
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Sharpness of the sampling distribution. Below 1 sharpens; above 1 increases diversity. Must be > 0.",
    )
    prepend_prompt: bool = ConfigField(
        default=False,
        title="Prepend Prompt",
        description="Whether to prepend prompt to generation",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="GPU device to run Evo1 on (e.g. 'cuda' or 'cuda:0').",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print verbose output",
    )

    @field_validator("prompts", mode="before")
    @classmethod
    def normalize_prompts(cls, v: Any) -> Any:
        """Convert single string to list for consistent handling."""
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_prompts_length(self) -> "Evo1GeneratorConfig":
        """Validate that all prompts have the same length."""
        if len({len(seq) for seq in self.prompts}) != 1:
            raise ValueError(f"All prompts must have same length, got: {[len(seq) for seq in self.prompts]}")
        return self


@generator(
    key="evo1",
    label="Evo1 DNA Language Model",
    config=Evo1GeneratorConfig,
    description="Evo1 genome language model for DNA sequence generation",
    uses_gpu=True,
    tools_called=["evo1"],
    supported_sequence_types=["dna"],
)
@final
class Evo1Generator(Generator):
    """Sequence generator using the Evo1 genomic language model.

    Supports multiple checkpoints including CRISPR and transposon fine-tuned
    variants. The number of tokens to generate is automatically calculated
    based on the assigned segment's sequence_length.

    Attributes:
        batch_size (int): Number of sequences to generate per batch.

    Example:
        >>> config = Evo1GeneratorConfig(
        ...     prompts="ATG",
        ...     model_checkpoint="evo-1-8k-crispr",
        ...     temperature=1.0,
        ... )
        >>> gen = Evo1Generator(config)
        >>> segment = Segment(length=1003, sequence_type="dna")
        >>> gen.assign(segment)  # max_new_tokens = 1003 - 3 = 1000
        >>> gen.sample()
    """

    input_type = GeneratorInputType.PROMPT

    def __init__(self, config: Evo1GeneratorConfig) -> None:
        """Initialize the Evo1 generator with model and sampling configuration."""
        super().__init__()
        self.config = config
        self.prompts = config.prompts
        self.model_checkpoint = config.model_checkpoint
        self.top_k = config.top_k
        self.temperature = config.temperature
        self.prepend_prompt = config.prepend_prompt
        self.device = config.device
        self.batch_size = config.batch_size
        self.verbose = config.verbose

    def _sample(
        self,
        prompts: list[str] | None = None,
        prepend_prompt: bool | None = None,
        max_new_tokens: int | None = None,
        old_kv_cache: None = None,
    ) -> None:
        """Generate sequences using the Evo1 model.

        Args:
            prompts (list[str] | None): Optional prompts overriding self.prompts.
            prepend_prompt (bool | None): Optional prepend_prompt override.
            max_new_tokens (int | None): Optional new-token count; derived from segment length if None.
            old_kv_cache (None): Must be None; Evo1 does not support KV caching.
        """
        if old_kv_cache is not None:
            raise ValueError("Evo1Generator does not support KV caching; use use_kv_caching=False")
        self._validate_generator()

        sampling_prompts = prompts if prompts is not None else self._replicate_prompts(self.prompts)
        prepend_prompt = prepend_prompt if prepend_prompt is not None else self.prepend_prompt
        if max_new_tokens is None:
            max_new_tokens = self._compute_max_new_tokens(len(sampling_prompts[0]), prepend_prompt)

        inputs = Evo1SampleInput(prompts=sampling_prompts)
        sample_config = Evo1SampleConfig(
            prepend_prompt=prepend_prompt,
            model_name=self.model_checkpoint,
            top_k=self.top_k,
            temperature=self.temperature,
            max_new_tokens=max_new_tokens,
            device=self.device,
            batch_size=self.batch_size,
            verbose=self.verbose,
            seed=self._next_seed(),
        )

        evo1_output = run_evo1_sample(inputs=inputs, config=sample_config)
        generated_sequences = evo1_output.sequences
        # Some checkpoints don't return scores; record None so consumers can subscript unconditionally.
        scores = evo1_output.scores or [None] * len(generated_sequences)
        key = self._spec.key

        for proposal, sequence, score in zip(self.segment.proposal_sequences, generated_sequences, scores, strict=True):
            proposal.sequence = sequence
            proposal._generator_metadata[key] = {"score": score}

    def _replicate_prompts(self, prompts: list[str]) -> list[str]:
        """Match prompt count to proposal count, replicating single prompts."""
        num_proposals = len(self.segment.proposal_sequences)
        if len(prompts) == num_proposals:
            return prompts
        if len(prompts) == 1:
            return prompts * num_proposals
        raise ValueError(f"Expected 1 or {num_proposals} prompts, got {len(prompts)}")

    def _compute_max_new_tokens(self, prompt_length: int, prepend_prompt: bool) -> int:
        """Max new-tokens to fill the segment: ``segment_length - prompt_length`` when prepending, else ``segment_length``."""
        segment_length = self.segment.sequence_length
        max_new_tokens = (segment_length - prompt_length) if prepend_prompt else segment_length
        if max_new_tokens < 1:
            raise ValueError(f"Prompt length ({prompt_length}) exceeds segment length ({segment_length})")
        return max_new_tokens
