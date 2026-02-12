---
name: implement-generator
description: >
  Use this skill when the user asks to create, modify, or debug a generator
  in the proto-language language core. This covers the full lifecycle:
  config class, Generator subclass with __init__/assign/sample, decorator
  registration, export chain, and tests.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# implement-generator skill

## Before You Start

1. **Read the registry** to see all existing generators and naming conventions:
   - `proto_language/language/generator/__init__.py`
2. **Find a similar implementation** by category:
   - Mutation (CPU): `proto_language/language/generator/uniform_mutation_generator.py`
   - Mutation (GPU tool): `proto_language/language/generator/esm2_generator.py`
   - Autoregressive: `proto_language/language/generator/evo2_generator.py`
   - Inverse folding: `proto_language/language/generator/proteinmpnn_generator.py`
3. **Read the base class**: `proto_language/language/core/generator.py`
4. **Read the decorator/registry**: `proto_language/language/generator/generator_registry.py`

## Generator ABC Contract

```python
class Generator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self._assigned_segment: Optional[Segment] = None

    def assign(self, assigned_segment: Segment) -> None:
        # Validates: not ligand, sequence type compatible
        # Category-specific init: mutation→random, autoregressive→none, inverse_folding→"X"

    @abstractmethod
    def sample(self) -> None:
        # Modifies self._assigned_segment.candidate_sequences IN PLACE

    def _validate_generator(self) -> None:
        # Called at start of sample(). Validates state, performs lazy initialization.
```

**Critical rules**:
- Always call `super().__init__()` in `__init__`
- Always call `super().assign(assigned_segment)` as first line in custom `assign()`
- Always call `self._validate_generator()` as first line in `sample()`
- Always use `@final` decorator on the class to prevent subclassing
- `sample()` modifies sequences **in-place** — it returns nothing

## Category Behavior

| Category | `assign()` auto-init | `sample()` behavior |
|----------|---------------------|---------------------|
| `"mutation"` | Random sequence from `valid_chars` if empty | Refines existing sequences |
| `"autoregressive"` | No initialization | Generates from scratch (left-to-right) |
| `"inverse_folding"` | `"X" * length` if empty | Structure-conditioned design |

The category determines what `_validate_generator()` does at the start of `sample()`:
- **mutation**: If candidate has no sequence, initializes random from `valid_chars`
- **autoregressive**: No random init (generates entirely new sequences)
- **inverse_folding**: If candidate has no sequence, initializes with `"X"` (unknown)

## Complete Implementation Template

### Step 1: Config Class

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

    batch_size: Optional[int] = ConfigField(
        default=None,
        title="Batch Size",
        description="Override automatic batch sizing",
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

### Step 2: Generator Class

```python
@generator(
    key="my-generator",                           # Unique, kebab-case
    label="My Generator",                         # Human-readable display name
    config=MyGeneratorConfig,                     # Config class from Step 1
    description="Generates sequences using ...",  # UI description
    category="mutation",                          # "mutation" | "autoregressive" | "inverse_folding"
    requires_gpu=True,                            # True if calls GPU tools
    tools_called=["my-tool"],                     # Tool names this generator invokes
    supported_sequence_types=["protein"],          # Empty list [] = all types supported
)
@final
class MyGenerator(Generator):
    """Generate sequences using MyModel.

    Detailed description of the generation approach.
    """

    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        # Store config values as instance attributes
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

        # Get current candidates
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

## Decorator Argument Reference

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `key` | `str` | Yes | Unique kebab-case identifier |
| `label` | `str` | Yes | Human-readable name for UI |
| `config` | `Type[BaseModel]` | Yes | Pydantic config class |
| `description` | `str` | Yes | What this generator does |
| `category` | `str` | Yes | `"mutation"`, `"autoregressive"`, or `"inverse_folding"` |
| `requires_gpu` | `bool` | No | Default `False` |
| `tools_called` | `List[str]` | No | Default `[]` |
| `supported_sequence_types` | `List[str]` | No | Default `[]` (= all types). Options: `"dna"`, `"rna"`, `"protein"` |

## Export Chain

Add to `proto_language/language/generator/__init__.py`:

```python
# Import both class and config
from .my_generator import MyGenerator, MyGeneratorConfig

# Add both to __all__
__all__ = [
    ...
    "MyGenerator",
    "MyGeneratorConfig",
]
```

## Tool Integration Pattern

For generators that call external tools deployed on cloud:

```python
from proto_tools.tools.{category}.{tool} import (
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

## Autoregressive Generator Special Patterns

Autoregressive generators often support:
- **Prompts**: Initial prefix sequences for generation
- **KV caching**: Store/reuse attention caches across beam search steps
- **`sample()` overrides**: Extra parameters like `prompts`, `old_kv_cache`

```python
def sample(self, prompts: Optional[List[str]] = None,
           old_kv_cache: Optional[Dict] = None) -> None:
    self._validate_generator()
    # Use provided prompts or defaults
    sampling_prompts = prompts if prompts is not None else self.prompts
    ...
```

## Inverse Folding Generator Special Patterns

Inverse folding generators take structure inputs:

```python
def sample(self, structure_inputs: Optional[List[...]] = None) -> None:
    self._validate_generator()
    sampling_inputs = structure_inputs or self.structure_inputs
    if sampling_inputs is None:
        raise ValueError("No structure_inputs provided")
    ...
```

## Test Requirements

File: `tests/language_tests/generator_tests/test_{name}_generator.py`

Every generator needs these tests:
1. **Initialization** — verify config values stored correctly
2. **Assign** — verify segment assignment, custom validation
3. **Sample** — verify sequences are modified in-place
4. **Batch** — verify multiple candidates are mutated independently
5. **Sequence type validation** — verify supported/unsupported types
6. **Config validation** — verify invalid configs raise errors
7. **Edge cases** — short sequences, large num_mutations, etc.

For GPU generators, mark tests with `@pytest.mark.uses_gpu`.
For CPU generators, no marker needed (auto-applied).

See the testing skill for complete test templates.
