"""
ProGen2 Generator for protein sequence generation.
"""

from __future__ import annotations

from typing import List, Optional, final

from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator, Segment
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.tools.language_models.progen2 import (
    ProGen2SampleConfig,
    ProGen2SampleInput,
    run_progen2_sample,
)
from proto_language.tools.language_models.progen2.inference import (  # noqa: F401
    PROGEN2_END_TOKEN,
    PROGEN2_MODEL_CHECKPOINTS,
    PROGEN2_START_TOKEN,
)


class ProGen2GeneratorConfig(BaseConfig):
    """Configuration object for ProGen2Generator.

    This class defines configuration parameters for the ProGen2 generator, which uses
    the ProGen2 protein language model to autoregressively generate protein sequences
    from prompt sequences.

    Models are loaded from HuggingFace: https://huggingface.co/hugohrban/

    Attributes:
        prompts (List[str]): Prompt sequence(s) to start protein generation.
            Can be a single prompt string (automatically converted to list) or list of
            prompts for batch generation.

            ProGen2 uses special tokens: '1' (start) and '2' (end/stop).

            **Important**: If you provide only amino acid characters (e.g., "MKTL"),
            the generator will automatically prepend the start token '1' and emit
            a warning. For explicit control, include the start token yourself: "1MKTL".

        model_checkpoint (str): ProGen2 model checkpoint to use. Options:
            - ``"progen2-small"``: 151M parameters (fastest)
            - ``"progen2-medium"``: 754M parameters
            - ``"progen2-oas"``: 754M parameters, trained on OAS antibody sequences
            - ``"progen2-large"``: 2B parameters (default)
            - ``"progen2-BFD90"``: 2B parameters, trained on BFD90
            - ``"progen2-xlarge"``: 6B parameters (highest quality, slowest)
            Default: ``"progen2-large"``.

        local_path (Optional[str]): Path to local model weights directory for custom
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

        max_length (int): Maximum total sequence length including prompt.
            Generation stops when this length is reached or a stop token is encountered.
            Must be at least 1. Default: 256.

        truncate_at_stop (bool): Whether to truncate generated sequences at the
            first stop token ('1' or '2'). If ``True``, returns clean protein
            sequences. Default: ``True``.

        strip_special_tokens (bool): Whether to remove the ProGen2 start and stop tokens
            ('1' or '2'). If ``True``, returns the stripped seequence.
            Default: ``True``.

        prepend_prompt (bool): Whether to include the prompt in the returned
            sequence. If ``False``, only newly generated tokens are returned.
            Default: ``True``.

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

    # Advanced parameters.
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
    )
    top_p: float = ConfigField(
        default=0.95,
        gt=0.0,
        le=1.0,
        title="Top-p",
        description="Nucleus sampling parameter.",
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


@GeneratorRegistry.register(
    key="progen2",
    label="ProGen2 Protein Language Model",
    config=ProGen2GeneratorConfig,
    description="ProGen2 autoregressive protein language model for protein sequence generation",
    requires_gpu=True,
    tools_called=["progen2"],
    category="autoregressive",
    supported_sequence_types=["protein"],
)
@final
class ProGen2Generator(Generator):
    """Protein sequence generator using ProGen2 autoregressive language model."""

    def __init__(self, config: ProGen2GeneratorConfig) -> None:
        super().__init__()
        self.prompts = config.prompts
        self.model_checkpoint = config.model_checkpoint
        self.local_path = config.local_path
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.truncate_at_stop = config.truncate_at_stop
        self.strip_special_tokens = config.strip_special_tokens
        self.prepend_prompt = config.prepend_prompt
        self.verbose = config.verbose

        # This should get assigned during `self.assign()`.
        self.max_length: Optional[int] = None

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a Segment to this generator and calculate lengths."""
        super().assign(assigned_segment)

        prompt_length = (
            len(self.prompts[0])
            if isinstance(self.prompts, list)
            else len(self.prompts)
        )

        self.max_length = assigned_segment.sequence_length
        if self.prepend_prompt:
            self.max_length = assigned_segment.sequence_length - prompt_length
        else:
            self.max_length = assigned_segment.sequence_length

    def sample(self, prompts: Optional[List[str]] = None) -> None:
        """Generate protein sequences using ProGen2 tool."""
        self._validate_generator()
        sampling_prompts = prompts if prompts is not None else self.prompts
        num_candidates = len(self._assigned_segment.candidate_sequences)

        # Handle prompt count matching
        if len(sampling_prompts) != num_candidates:
            if len(sampling_prompts) == 1:
                # Replicate single prompt for all candidates
                sampling_prompts = sampling_prompts * num_candidates
            else:
                raise ValueError(f"Number of prompts ({len(sampling_prompts)}) must either be 1 (will be replicated) or match the number of candidates ({num_candidates})")

        tool_input = ProGen2SampleInput(prompts=sampling_prompts)

        tool_config = ProGen2SampleConfig(
            model_checkpoint=self.model_checkpoint,
            local_path=self.local_path,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_length=self.max_length,
            truncate_at_stop=self.truncate_at_stop,
            strip_special_tokens=self.strip_special_tokens,
            prepend_prompt=self.prepend_prompt,
            verbose=self.verbose,
            keep_on_gpu=True,
        )

        output = run_progen2_sample(tool_input, tool_config)
        generated_sequences = output.sequences

        for idx, sequence in enumerate(generated_sequences):
            if idx < len(self._assigned_segment.candidate_sequences):
                self._assigned_segment.candidate_sequences[idx].sequence = sequence
