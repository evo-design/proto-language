# Generator Implementation Templates

Complete templates for config class and generator class by category. Load this file on demand when implementing a new generator.

## Config Class Template

File: `proto_language/language/generator/{name}_generator.py`

```python
from __future__ import annotations

import logging
from typing import List, Optional, final

from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator, Segment
from proto_language.language.generator.generator_registry import generator

logger = logging.getLogger(__name__)


class MyGeneratorConfig(BaseConfig):
    """Configuration for MyGenerator.

    Detailed description of what this generator does and its parameters.
    """

    # Required parameter
    model_name: str = ConfigField(
        title="Model Name",
        description="Which model checkpoint to use",
    )

    # Optional with default
    temperature: float = ConfigField(
        default=1.0,
        title="Temperature",
        description="Sampling temperature (higher = more random)",
        gt=0.0,
        le=2.0,
        advanced=True,
    )

    batch_size: int = ConfigField(
        default=1,
        title="Batch Size",
        description="Number of sequences to process per batch on the GPU",
        ge=1,
        advanced=True,
    )

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model(cls, v):
        valid = ["model_a", "model_b"]
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v
```

## Mutation Generator Template

```python
@generator(
    key="my-generator",
    label="My Generator",
    config=MyGeneratorConfig,
    description="Generates sequences using ...",
    category="mutation",
    requires_gpu=True,
    tools_called=["my-tool"],
    supported_sequence_types=["protein"],
)
@final
class MyGenerator(Generator):
    """Generate sequences using MyModel.

    Detailed description of the generation approach.
    """

    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.batch_size = config.batch_size

    def assign(self, assigned_segment: Segment) -> None:
        super().assign(assigned_segment)
        # Custom validation after base class validation
        if assigned_segment.sequence_length < 10:
            raise ValueError("Sequence must be at least 10 residues")

    def sample(self) -> None:
        self._validate_generator()

        candidates = self._assigned_segment.candidate_sequences
        sequences = [seq.sequence for seq in candidates]

        # Call external tool or compute locally
        result = run_my_tool(
            sequences=sequences,
            model=self.model_name,
            temperature=self.temperature,
        )

        # Update sequences IN PLACE
        for candidate, new_seq in zip(candidates, result.sequences):
            candidate.sequence = new_seq

        # Optionally store metadata
        for candidate, score in zip(candidates, result.scores):
            candidate._metadata.update({
                "my_generator_score": score,
                "my_generator_model": self.model_name,
            })
```

## Autoregressive Generator Template

Autoregressive generators often support prompts and KV caching:

```python
@generator(
    key="my-autoregressive",
    label="My Autoregressive Generator",
    config=MyGeneratorConfig,
    description="Generates sequences left-to-right using ...",
    category="autoregressive",
    requires_gpu=True,
    tools_called=["my-tool"],
    supported_sequence_types=["dna"],
)
@final
class MyAutoregressiveGenerator(Generator):
    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.temperature = config.temperature

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prepend_prompt: Optional[bool] = None,
        old_kv_cache: Optional[Dict] = None,
    ) -> None:
        self._validate_generator()
        # Use provided prompts or defaults
        sampling_prompts = prompts if prompts is not None else self.prompts
        ...
```

## Inverse Folding Generator Template

Inverse folding generators take structure inputs:

```python
@generator(
    key="my-inverse-folding",
    label="My Inverse Folding Generator",
    config=MyGeneratorConfig,
    description="Designs sequences conditioned on structure using ...",
    category="inverse_folding",
    requires_gpu=True,
    tools_called=["my-tool"],
    supported_sequence_types=["protein"],
)
@final
class MyInverseFoldingGenerator(Generator):
    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.model_name = config.model_name

    def sample(self, structure_inputs: Optional[List[...]] = None) -> None:
        self._validate_generator()
        sampling_inputs = structure_inputs or self.structure_inputs
        if sampling_inputs is None:
            raise ValueError("No structure_inputs provided")
        ...
```

## Full Tool Integration Pattern

For generators that call external tools deployed on cloud:

```python
from proto_tools import (
    run_{tool},
    {Tool}Input,
    {Tool}Config,
)

def sample(self) -> None:
    self._validate_generator()

    sequences = [seq.sequence for seq in self._assigned_segment.candidate_sequences]

    # Build tool input/config
    tool_input = ToolInput(sequences=sequences)
    tool_config = ToolConfig(
        model=self.model_name,
        temperature=self.temperature,
        batch_size=self.batch_size,
    )

    # Call tool
    result = run_tool(inputs=tool_input, config=tool_config)
    generated = result.sequences

    # Update candidates in-place
    for i, sequence in enumerate(generated):
        self._assigned_segment.candidate_sequences[i].sequence = sequence
```
