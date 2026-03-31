"""
proto_language/language/generator/evo1_generator.py

Evo1 Generator for DNA sequence generation.
"""

from __future__ import annotations

from typing import List, Optional, final

from proto_tools import (
    EVO1_MODEL_CHECKPOINTS,
    Evo1SampleConfig,
    Evo1SampleInput,
    run_evo1_sample,
)
from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import generator


class Evo1GeneratorConfig(BaseConfig):
    """Configuration object for Evo1Generator.

    Attributes:
        prompts (list[str]): Prompt sequence(s) for DNA generation. All prompts must
            have the same length.
        model_checkpoint (EVO1_MODEL_CHECKPOINTS): Evo1 model checkpoint to use.
        top_k (int): Top-k sampling parameter.
        temperature (float): Sampling temperature.
        prepend_prompt (bool): Whether to prepend the prompt to the output.
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors.
        verbose (bool): Whether to print generation progress.
    """

    prompts: List[str] = ConfigField(
        title="Prompts",
        description="Prompt sequences for DNA generation (single prompt or multiple)",
    )
    model_checkpoint: EVO1_MODEL_CHECKPOINTS = ConfigField(
        default="evo-1-8k-base",
        title="Model Checkpoint",
        description="Evo1 model checkpoint to use",
    )

    # Advanced parameters
    top_k: int = ConfigField(
        default=4,
        ge=1,
        title="Top-k",
        description="Top-k sampling parameter",
        advanced=True,
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Sampling temperature",
        advanced=True,
    )
    prepend_prompt: bool = ConfigField(
        default=False,
        title="Prepend Prompt",
        description="Whether to prepend prompt to generation",
        hidden=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
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
        """Convert single string to list for consistent handling."""
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_prompts_length(self):
        """Validate that all prompts have the same length."""
        if len(set(len(seq) for seq in self.prompts)) != 1:
            raise ValueError(
                f"All prompts must have same length, got: {[len(seq) for seq in self.prompts]}"
            )
        return self


@generator(
    key="evo1",
    label="Evo1 DNA Language Model",
    config=Evo1GeneratorConfig,
    description="Evo1 genome language model for DNA sequence generation",
    uses_gpu=True,
    tools_called=["evo1"],
    category="autoregressive",
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
        >>> gen.assign(segment)  # num_tokens = 1003 - 3 = 1000
        >>> gen.sample()
    """

    def __init__(self, config: Evo1GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.prompts = config.prompts
        self.model_checkpoint = config.model_checkpoint
        self.top_k = config.top_k
        self.temperature = config.temperature
        self.prepend_prompt = config.prepend_prompt
        self.batch_size = config.batch_size
        self.verbose = config.verbose

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prepend_prompt: Optional[bool] = None,
    ) -> None:
        """Generate sequences using the Evo1 model.

        Args:
            prompts (list[str] | None): Optional prompts to use instead of self.prompts.
            prepend_prompt (bool | None): Optional override for prepend_prompt setting.
        """
        self._validate_generator()

        sampling_prompts = prompts if prompts is not None else self._replicate_prompts(self.prompts)
        prepend_prompt = prepend_prompt if prepend_prompt is not None else self.prepend_prompt
        num_tokens = self._compute_num_tokens(len(sampling_prompts[0]), prepend_prompt)

        inputs = Evo1SampleInput(prompts=sampling_prompts)
        sample_config = Evo1SampleConfig(
            prepend_prompt=prepend_prompt,
            model_name=self.model_checkpoint,
            top_k=self.top_k,
            temperature=self.temperature,
            num_tokens=num_tokens,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        evo1_output = run_evo1_sample(inputs=inputs, config=sample_config)
        generated_sequences = evo1_output.sequences

        for proposal, sequence in zip(
            self._assigned_segment.proposal_sequences, generated_sequences, strict=True
        ):
            proposal.sequence = sequence


        if evo1_output.scores:
            for proposal, score in zip(
                self._assigned_segment.proposal_sequences, evo1_output.scores, strict=True
            ):
                proposal._metadata["evo1_score"] = score

    def _replicate_prompts(self, prompts: List[str]) -> List[str]:
        """Match prompt count to proposal count, replicating single prompts."""
        num_proposals = len(self._assigned_segment.proposal_sequences)
        if len(prompts) == num_proposals:
            return prompts
        if len(prompts) == 1:
            return prompts * num_proposals
        raise ValueError(f"Expected 1 or {num_proposals} prompts, got {len(prompts)}")

    def _compute_num_tokens(
        self, prompt_length: int, prepend_prompt: bool
    ) -> int:
        """Compute tokens to generate based on segment length and prompt settings."""
        segment_length = self._assigned_segment.sequence_length
        num_tokens = ((segment_length - prompt_length) if prepend_prompt else segment_length)
        if num_tokens < 1:
            raise ValueError(f"Prompt length ({prompt_length}) exceeds segment length ({segment_length})")
        return num_tokens
