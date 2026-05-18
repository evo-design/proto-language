# Generator Implementation Templates

Complete templates for config class and generator class by `input_type`. Load this file on demand when implementing a new generator.

## `_validate_generator()` Per-`input_type` Behavior

The class's `input_type` classvar determines what `_validate_generator()` does at the start of `_sample()`:

- **`STARTING_SEQUENCE`** (mutation): If proposals have no sequence, raises `RuntimeError`. No random-init fallback — the user must provide `segment.input_sequence` or rely on a prior optimizer stage's output.
- **`PROMPT`** (autoregressive): If proposals are already populated, logs a warning (they will be overwritten).
- **`STRUCTURE`** (inverse folding): If proposals have no sequence, seeds `"X" * length` and logs at INFO. The structure determines residues during design.
- **`LOGITS`** (gradient): No special init. Each proposal must carry `seq.logits` from a prior `GradientOptimizer` stage; reading code raises if missing.

A Program-build-time validator (`Program._validate_generator_inputs` in `core/program.py`) catches missing inputs at `Program.__init__` time — before any stage runs — with errors that name the offending stage and segment.

## Batching Data Flow

```
Generator.sample()
    -> Collects ALL proposal sequences from segment
    -> Creates ToolInput with all sequences
    -> Creates ToolConfig with batch_size=self.batch_size
    -> Calls run_tool(inputs, config)
         -> Tool chunks sequences into batches of batch_size
         -> Processes each batch on GPU
         -> Returns concatenated results
    -> Updates proposal_sequences in-place from results
```

## Config Class Template

File: `proto_language/language/generator/{name}_generator.py`

```python
import logging
from typing import final

from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator, GeneratorInputType, Segment
from proto_language.language.generator.generator_registry import generator

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

Autoregressive generators receive prompts via config or via the `prompts` kwarg injected by `CyclingOptimizer`:

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
        self.model_name = config.model_name
        self.temperature = config.temperature

    def _sample(self, prompts: list[str] | None = None) -> None:
        self._validate_generator()
        sampling_prompts = prompts if prompts is not None else self.prompts
        ...
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
        # ... generate sequences ...
        for proposal, struct_input in zip(self.segment.proposal_sequences, sampling_inputs, strict=True):
            proposal.sequence = ...  # designed sequence
            proposal.structure = struct_input.structure
```

## Gradient Generator Template (`input_type = LOGITS`)

Gradient generators read per-position logits written by an upstream `GradientOptimizer`:

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

    def _sample(self) -> None:
        self._validate_generator()
        for proposal in self.segment.proposal_sequences:
            if proposal.logits is None:
                raise RuntimeError(f"Proposal on segment '{self.segment.label}' has no logits.")
            proposal.sequence = decode(proposal.logits, ...)
```

## Full Tool Integration Pattern

For generators that call external tools (via proto-tools):

```python
from proto_tools import (
    run_{tool},
    {Tool}Input,
    {Tool}Config,
)

def _sample(self) -> None:
    self._validate_generator()

    sequences = [seq.sequence for seq in self.segment.proposal_sequences]

    tool_input = ToolInput(sequences=sequences)
    tool_config = ToolConfig(
        model=self.model_name,
        temperature=self.temperature,
        batch_size=self.batch_size,
        seed=self._next_seed(),
    )

    result = run_tool(inputs=tool_input, config=tool_config)

    for proposal, sequence in zip(self.segment.proposal_sequences, result.sequences, strict=True):
        proposal.sequence = sequence
```
