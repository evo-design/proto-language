# Generator Implementation Templates

Complete templates for config class and generator class by `input_type`. Load this file on demand when implementing a new generator.

## `_validate_generator()` Per-`input_type` Behavior

The class's `input_type` classvar determines what `_validate_generator()` does at the start of `_sample()`:

- **`STARTING_SEQUENCE`** (mutation): If proposals have no sequence, raises `RuntimeError`. No random-init fallback — the user must provide `segment.input_sequence` or rely on a prior optimizer stage's output.
- **`PROMPT`** (autoregressive): If proposals are already populated, logs a warning (they will be overwritten).
- **`STRUCTURE`** (inverse folding): If proposals have no sequence, seeds `"X" * length` and logs at INFO. The structure determines residues during design.
- **`LOGITS`** (gradient): No special init. Each proposal must carry `seq.logits` from a prior `GradientOptimizer` stage; reading code raises if missing.

A Program-build-time validator (`Program._validate_generator_inputs` in `core/program.py`) catches missing inputs at `Program.__init__` time — before any stage runs — with errors that name the offending stage and segment.

## Batching

Generators pass all proposals to the tool in one call; the tool owns the chunking loop and uses `batch_size` from the tool config. See `notes/batching.md` for the full architecture.

## Assign-Time Validation

When a generator carries segment-dependent state (alignment length, bias matrix, frozen-position list), override `assign()` and validate after `super().assign(segments)`:

```python
def assign(self, segments: Segment | Iterable[Segment]) -> None:
    super().assign(segments)
    if self.alignment_length != self.segment.sequence_length:
        raise ValueError(...)
    self._bias_matrix = build_bias(self._bias_config, self.segment)
```

## Preserving Logits / Structure Across Stages

Defaults: `LOGITS` generators preserve `seq.logits`; `STRUCTURE` generators preserve `seq.structure`; everything else clears both after `_sample()`. Override `_preserve_logits_after_sample()` or `_preserve_structure_after_sample()` to opt back in (e.g. an MCMC stage that wants to keep gradient-derived logits available).

## Config Class Template

File: `proto_language/generator/{name}_generator.py`

```python
import logging
from typing import final

from pydantic import field_validator, model_validator

from proto_language.utils.base import BaseConfig, ConfigField
from proto_language.core import Generator, GeneratorInputType, Segment
from proto_language.generator.generator_registry import generator

logger = logging.getLogger(__name__)


class MyGeneratorConfig(BaseConfig):
    """Configuration for MyGenerator.

    Detailed description of what this generator does and its parameters.
    """

    model_name: str = ConfigField(
        title="Model Name",
        description="Which model checkpoint to use",
    )

    temperature: float = ConfigField(
        default=1.0,
        title="Temperature",
        description="Sampling temperature (higher = more random)",
        gt=0.0,
        le=2.0,
    )

    # Include `batch_size` only if the generator calls a GPU tool that batches
    # internally. Tool-less generators (random sampling, gradient decoders) don't
    # need it.
    batch_size: int = ConfigField(
        default=1,
        title="Batch Size",
        description="Number of sequences to process per batch on the GPU",
        ge=1,
    )

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model(cls, v):
        valid = ["model_a", "model_b"]
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v
```

## Mutation Generator Template (`input_type = STARTING_SEQUENCE`)

```python
@generator(
    key="my-generator",
    label="My Generator",
    config=MyGeneratorConfig,
    description="Refines existing protein sequences using ...",
    uses_gpu=True,
    tools_called=["my-tool"],
    supported_sequence_types=["protein"],
)
@final
class MyGenerator(Generator):
    """Generate sequences using MyModel.

    Detailed description of the generation approach.
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.batch_size = config.batch_size

    def _sample(self) -> None:
        self._validate_generator()

        proposals = self.segment.proposal_sequences
        sequences = [seq.sequence for seq in proposals]

        result = run_my_tool(
            sequences=sequences,
            model=self.model_name,
            temperature=self.temperature,
            batch_size=self.batch_size,
            seed=self._next_seed(),
        )

        # Update sequences IN PLACE; store per-proposal diagnostics under
        # _generator_metadata[<registry_key>] so they don't collide with other
        # generators or the free-form user bag at proposal._metadata.
        key = self._spec.key
        for proposal, new_seq, score in zip(proposals, result.sequences, result.scores, strict=True):
            proposal.sequence = new_seq
            proposal._generator_metadata[key] = {"score": score, "model": self.model_name}
```

## Autoregressive Generator Template (`input_type = PROMPT`)

Prompts come from config or the `prompts` kwarg from `CyclingOptimizer`. Two helpers are usually needed: replicate a single prompt to match proposal count, and compute `max_new_tokens` from `segment.sequence_length` (minus prompt length when `prepend_prompt=True`).

```python
@generator(
    key="my-autoregressive",
    label="My Autoregressive Generator",
    config=MyGeneratorConfig,
    description="Generates DNA sequences left-to-right using ...",
    uses_gpu=True,
    tools_called=["my-tool"],
    supported_sequence_types=["dna"],
)
@final
class MyAutoregressiveGenerator(Generator):
    input_type = GeneratorInputType.PROMPT

    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.prompts = config.prompts
        self.prepend_prompt = config.prepend_prompt
        self.temperature = config.temperature

    def _sample(
        self,
        prompts: list[str] | None = None,
        prepend_prompt: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self._validate_generator()
        sampling_prompts = prompts or self._replicate_prompts(self.prompts)
        prepend = self.prepend_prompt if prepend_prompt is None else prepend_prompt
        if max_new_tokens is None:
            max_new_tokens = self._compute_max_new_tokens(len(sampling_prompts[0]), prepend)

        result = run_my_tool(
            MyToolInput(prompts=sampling_prompts),
            config=MyToolConfig(
                max_new_tokens=max_new_tokens, temperature=self.temperature,
                prepend_prompt=prepend, batch_size=self.batch_size, seed=self._next_seed(),
            ),
        )
        scores = result.scores or [None] * len(result.sequences)  # checkpoints may not return scores
        key = self._spec.key
        for proposal, seq, score in zip(self.segment.proposal_sequences, result.sequences, scores, strict=True):
            proposal.sequence = seq
            proposal._generator_metadata[key] = {"score": score}

    def _replicate_prompts(self, prompts: list[str]) -> list[str]:
        n = len(self.segment.proposal_sequences)
        if len(prompts) == n:
            return prompts
        if len(prompts) == 1:
            return prompts * n
        raise ValueError(f"Expected 1 or {n} prompts, got {len(prompts)}")

    def _compute_max_new_tokens(self, prompt_length: int, prepend_prompt: bool) -> int:
        seg_len = self.segment.sequence_length
        n = seg_len - prompt_length if prepend_prompt else seg_len
        if n < 1:
            raise ValueError(f"Prompt length ({prompt_length}) exceeds segment length ({seg_len})")
        return n
```

## Inverse Folding Generator Template (`input_type = STRUCTURE`)

Inverse folding generators receive structure inputs via config or via the `structure_inputs` kwarg from `CyclingOptimizer`:

```python
@generator(
    key="my-inverse-folding",
    label="My Inverse Folding Generator",
    config=MyGeneratorConfig,
    description="Designs sequences conditioned on structure using ...",
    uses_gpu=True,
    tools_called=["my-tool"],
    supported_sequence_types=["protein"],
)
@final
class MyInverseFoldingGenerator(Generator):
    input_type = GeneratorInputType.STRUCTURE

    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.structure_inputs = config.structure_inputs
        self.model_name = config.model_name

    def _sample(self, structure_inputs: list[...] | None = None) -> None:
        self._validate_generator()
        sampling_inputs = structure_inputs or self.structure_inputs
        if sampling_inputs is None:
            raise ValueError("No structure_inputs provided")

        # 1 structure → num_proposals seqs from it. N structures → 1 seq each (N must equal num_proposals).
        num_proposals = len(self.segment.proposal_sequences)
        if len(sampling_inputs) == 1:
            num_seqs, bs = num_proposals, self.batch_size
        elif len(sampling_inputs) == num_proposals:
            num_seqs, bs = 1, 1
        else:
            raise ValueError(f"structure_inputs ({len(sampling_inputs)}) must be 1 or match num_proposals ({num_proposals})")

        # ... call tool with (num_seqs, bs), then ...
        for proposal, struct_input in zip(self.segment.proposal_sequences, sampling_inputs, strict=True):
            proposal.sequence = ...
            proposal.structure = struct_input.structure
```

## Gradient Generator Template (`input_type = LOGITS`)

Decodes per-position logits from an upstream `GradientOptimizer` into discrete sequences. Typically exposes `sampling_mode` (`"argmax"` / `"categorical"`) and applies a temperature-scaled softmax before decoding:

```python
@generator(
    key="my-gradient",
    label="My Gradient Generator",
    config=MyGeneratorConfig,
    description="Decodes sequences from gradient-optimized logits using ...",
    uses_gpu=False,
    supported_sequence_types=["protein"],
)
@final
class MyGradientGenerator(Generator):
    input_type = GeneratorInputType.LOGITS

    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.sampling_mode = config.sampling_mode
        self.temperature = config.temperature

    def _sample(self) -> None:
        self._validate_generator()
        vocab = self.segment.ordered_vocab()
        rng = np.random.default_rng(self._next_seed()) if self.sampling_mode == "categorical" else None
        for proposal in self.segment.proposal_sequences:
            if proposal.logits is None:
                raise RuntimeError(f"Proposal on segment '{self.segment.label}' has no logits.")
            matrix = softmax(proposal.logits / self.temperature, axis=-1)
            proposal.sequence = _decode_argmax(matrix, vocab) if self.sampling_mode == "argmax" else _decode_categorical(matrix, vocab, rng)
```
