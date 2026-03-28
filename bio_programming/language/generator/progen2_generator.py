"""
proto_language/language/generator/progen2_generator.py

ProGen2 Generator for protein sequence generation.
"""

from __future__ import annotations

from typing import List, Optional, final

from proto_tools import (
    ProGen2SampleConfig,
    ProGen2SampleInput,
    run_progen2_sample,
)
from proto_tools.tools.causal_models.progen2.progen2_sample import (
    PROGEN2_MODEL_CHECKPOINTS,
)
from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import generator


class ProGen2GeneratorConfig(BaseConfig):
    """Configuration object for ProGen2Generator.

    This class defines configuration parameters for the ProGen2 generator, which uses
    the ProGen2 protein language model to autoregressively generate protein sequences
    from prompt sequences.

    Models are loaded from HuggingFace: https://huggingface.co/hugohrban/

    Attributes:
        prompts (list[str]): Prompt sequence(s) to start protein generation.
            Can be a single prompt string (automatically converted to list) or list of
            prompts for batch generation.

            ProGen2 uses special tokens: '1' (start) and '2' (end/stop).

            **Important**: If you provide only amino acid characters (e.g., "MKTL"),
            the underlying ProGen2 tool will automatically prepend the start token '1'
            during sampling. For explicit control, include the start token yourself:
            "1MKTL".

        model_checkpoint (PROGEN2_MODEL_CHECKPOINTS): ProGen2 model checkpoint to use. Options:
            - ``"progen2-small"``: 151M parameters (fastest)
            - ``"progen2-medium"``: 754M parameters
            - ``"progen2-oas"``: 754M parameters, trained on OAS antibody sequences
            - ``"progen2-large"``: 2B parameters (default)
            - ``"progen2-BFD90"``: 2B parameters, trained on BFD90
            - ``"progen2-xlarge"``: 6B parameters (highest quality, slowest)
            Default: ``"progen2-large"``.

        local_path (str | None): Path to local model weights directory for custom
            or fine-tuned models. If ``None``, downloads from HuggingFace (hugohrban/).
            Default: ``None``.

        temperature (float): Scales randomness of amino acid sampling by adjusting
            probability distribution sharpness:
            - ``< 1.0``: More deterministic, favors high-probability amino acids
            - ``1.0``: Standard sampling from model distribution
            - ``> 1.0``: More diverse, explores lower-probability amino acids
            Must be greater than 0. Default: 0.2 (following ProGen2 defaults).

        top_p (float): Nucleus sampling parameter. Chooses tokens whose cumulative
            probability mass is at least ``top_p``. Range: (0.0, 1.0].
            Default: 0.95 (following ProGen2 defaults).

        top_k (int): Limits sampling to the top-k most probable tokens at each
            generation step. Set to 0 to disable. Default: 0 (disabled, use top_p).

        max_length: Maximum total sequence length including prompt.
            Generation stops when this length is reached or a stop token is encountered.
            Must be at least 1. Default: 256.

        truncate_at_stop (bool): Whether to truncate generated sequences at the
            first stop token ('1' or '2'). If ``True``, returns clean protein
            sequences. Default: ``True``.

        strip_special_tokens (bool): Whether to remove the ProGen2 start and stop tokens
            ('1' or '2'). If ``True``, returns the stripped sequence.
            Default: ``True``.

        prepend_prompt (bool): Whether to include the prompt in the returned
            sequence. If ``False``, only newly generated tokens are returned.
            Default: ``True``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.

        verbose (bool): Whether to print detailed generation progress and timing.
            Default: ``False``.
    Note:
        For detailed information on ProGen2, see:
        - HuggingFace: https://huggingface.co/hugohrban/
        - GitHub: https://github.com/hugohrban/ProGen2-finetuning
        - Original GitHub: https://github.com/enijkamp/progen2
        - Original paper: https://www.cell.com/cell-systems/fulltext/S2405-4712(23)00272-7
    """

    # Required parameters.
    prompts: List[str] = ConfigField(
        title="Prompts",
        description="Prompt sequences for protein sequence generation",
    )
    model_checkpoint: PROGEN2_MODEL_CHECKPOINTS = ConfigField(
        default="progen2-large",
        title="Model Checkpoint",
        description="ProGen2 model checkpoint to use",
    )

    # Advanced parameters
    local_path: Optional[str] = ConfigField(
        default=None,
        title="Local Model Path",
        description="Path to local model weights",
        hidden=True,
    )
    temperature: float = ConfigField(
        default=0.2,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling.",
        advanced=True,
    )
    top_p: float = ConfigField(
        default=0.95,
        gt=0.0,
        le=1.0,
        title="Top-p",
        description="Nucleus sampling parameter.",
        advanced=True,
    )
    top_k: int = ConfigField(
        default=0,
        ge=0,
        title="Top-k",
        description="Limits sampling to the top-k most probable tokens.",
        advanced=True,
    )
    truncate_at_stop: bool = ConfigField(
        default=True,
        title="Truncate at Stop Token",
        description="Whether to truncate sequences at stop tokens",
        advanced=True,
    )
    strip_special_tokens: bool = ConfigField(
        default=True,
        title="Strip Special Tokens",
        description="Whether to strip start and stop tokens from final output",
        advanced=True,
    )
    prepend_prompt: bool = ConfigField(
        default=True,
        title="Prepend Prompt",
        description="Whether to prepend prompt to generation",
        hidden=True,
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
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
        """Convert single string to list for consistent internal handling."""
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_prompts_length(self):
        """Validate that all prompts have the same length."""
        if len(set(len(seq) for seq in self.prompts)) != 1:
            raise ValueError(f"All prompts must have same length, got: {[len(seq) for seq in self.prompts]}")
        return self


@generator(
    key="progen2",
    label="ProGen2 Protein Language Model",
    config=ProGen2GeneratorConfig,
    description="ProGen2 autoregressive protein language model for protein sequence generation",
    uses_gpu=True,
    tools_called=["progen2-sample"],
    category="autoregressive",
    supported_sequence_types=["protein"],
)
@final
class ProGen2Generator(Generator):
    """Protein sequence generator using ProGen2 autoregressive language model."""

    def __init__(self, config: ProGen2GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.prompts = config.prompts
        self.model_checkpoint = config.model_checkpoint
        self.local_path = config.local_path
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.truncate_at_stop = config.truncate_at_stop
        self.strip_special_tokens = config.strip_special_tokens
        self.prepend_prompt = config.prepend_prompt
        self.batch_size = config.batch_size
        self.verbose = config.verbose

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prepend_prompt: Optional[bool] = None,
    ) -> None:
        """Generate protein sequences using ProGen2 tool.

        Args:
            prompts (list[str] | None): Optional prompts to use instead of self.prompts.
            prepend_prompt (bool | None): Optional override for prepend_prompt setting.
        """
        self._validate_generator()
        sampling_prompts = prompts if prompts is not None else self._replicate_prompts(self.prompts)
        prepend_prompt = prepend_prompt if prepend_prompt is not None else self.prepend_prompt
        max_length = self._compute_max_length(len(sampling_prompts[0]), prepend_prompt)

        tool_input = ProGen2SampleInput(prompts=sampling_prompts)

        tool_config = ProGen2SampleConfig(
            model_checkpoint=self.model_checkpoint,
            local_path=self.local_path,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_length=max_length,
            truncate_at_stop=self.truncate_at_stop,
            strip_special_tokens=self.strip_special_tokens,
            prepend_prompt=prepend_prompt,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        output = run_progen2_sample(tool_input, tool_config)
        generated_sequences = output.sequences

        for proposal, sequence in zip(
            self._assigned_segment.proposal_sequences, generated_sequences, strict=True
        ):
            proposal.sequence = sequence

    def _replicate_prompts(self, prompts: List[str]) -> List[str]:
        """Match prompt count to proposal count, replicating single prompts."""
        num_proposals = len(self._assigned_segment.proposal_sequences)
        if len(prompts) == num_proposals:
            return prompts
        if len(prompts) == 1:
            return prompts * num_proposals
        raise ValueError(f"Expected 1 or {num_proposals} prompts, got {len(prompts)}")

    def _compute_max_length(self, prompt_length: int, prepend_prompt: bool) -> int:
        """Compute max_length for ProGen2 based on segment length and prompt settings."""
        segment_length = self._assigned_segment.sequence_length
        if prepend_prompt:
            if prompt_length >= segment_length:
                raise ValueError(f"Prompt length ({prompt_length}) must be less than segment length ({segment_length}) when prepend_prompt=True")
            return segment_length
        return segment_length + prompt_length
