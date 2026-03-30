---
name: implement-generator
description: >
  Implements, modifies, or debugs generators in the proto-language DSL.
  Covers the full lifecycle: config class, Generator subclass with __init__/assign/sample,
  category-specific patterns (mutation, autoregressive, inverse folding),
  decorator registration, export chain, and tests. Use when working with
  generators, sequence sampling, Evo2, ESM2, ESM3, ProteinMPNN, LigandMPNN,
  or mutation strategies.
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
        # Category-specific init: mutation->random, autoregressive->none, inverse_folding->"X"

    @abstractmethod
    def sample(self) -> None:
        # Modifies self._assigned_segment.proposal_sequences IN PLACE

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

## Implementation Steps

For complete config class and generator class templates, use the `Read` tool to load:
- **Templates**: `.claude/skills/implement-generator/TEMPLATES.md`

Summary of the workflow:
1. **Config class** — inherit `BaseConfig`, use `ConfigField` (supports `advanced`, `hidden`, and `depends_on` for conditional visibility), declare model params
2. **Generator class** — `@generator` decorator, `@final`, implement `__init__`, `assign`, `sample`
3. **Export chain** — add to `generator/__init__.py`
4. **Tests** — init, assign, sample, batch, type validation, config validation

## Decorator Argument Reference

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `key` | `str` | Yes | Unique kebab-case identifier |
| `label` | `str` | Yes | Human-readable name for UI |
| `config` | `Type[BaseModel]` | Yes | Pydantic config class |
| `description` | `str` | Yes | What this generator does |
| `category` | `str` | Yes | `"mutation"`, `"autoregressive"`, or `"inverse_folding"` |
| `uses_gpu` | `bool` | No | Whether generator requires GPU |
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
from proto_tools import run_{tool}, {Tool}Input, {Tool}Config

def sample(self) -> None:
    self._validate_generator()
    sequences = [seq.sequence for seq in self._assigned_segment.proposal_sequences]
    tool_input = ToolInput(sequences=sequences)
    tool_config = ToolConfig(model=self.model_name, temperature=self.temperature, batch_size=self.batch_size)
    result = run_tool(inputs=tool_input, config=tool_config)
    for i, sequence in enumerate(result.sequences):
        self._assigned_segment.proposal_sequences[i].sequence = sequence
```

## Batching Architecture

Generators do NOT implement batching loops. The tool layer owns all batching logic.

**Key rules:**
1. Default `batch_size = 1` — safe by default, prevents OOM
2. Generator config stores `batch_size` — passed through to tool config unchanged
3. Never write a batching loop in a generator — the tool handles chunking internally
4. Inverse folding special case (e.g., ProteinMPNN): When one structure generates N sequences, `batch_size` controls sequences per forward pass. When N structures each generate 1 sequence, `batch_size` is forced to 1 by the generator.

## Documentation

Documentation reference pages are auto-generated from Python docstrings and field descriptions. To update documentation, update the Python config docstrings/field descriptions in the source code.

## Test Requirements

File: `tests/language_tests/generator_tests/test_{name}_generator.py`

Every generator needs these tests:
1. **Initialization** — verify config values stored correctly
2. **Assign** — verify segment assignment, custom validation
3. **Sample** — verify sequences are modified in-place
4. **Batch** — verify multiple proposals are mutated independently
5. **Sequence type validation** — verify supported/unsupported types
6. **Config validation** — verify invalid configs raise errors
7. **Edge cases** — short sequences, large num_mutations, etc.

For GPU generators, mark tests with `@pytest.mark.uses_gpu`.
For CPU generators, no marker needed (auto-applied).

See the testing skill for complete test templates.

## Validation Checklist

Copy this and check off as you go:

- [ ] Config class inherits `BaseConfig` with `ConfigField` (use `depends_on` for conditionally visible fields)
- [ ] `@generator` decorator with unique kebab-case key and correct category
- [ ] `@final` decorator on class
- [ ] `__init__` calls `super().__init__()`
- [ ] `assign()` calls `super().assign(assigned_segment)` first
- [ ] `sample()` calls `self._validate_generator()` first
- [ ] `sample()` modifies `proposal_sequences` in-place (returns nothing)
- [ ] No batching loop in generator (tool handles batching)
- [ ] Export chain updated: `generator/__init__.py` (class + config)
- [ ] Tests cover: init, assign, sample, batch, type validation, config validation
- [ ] Tests pass: `pytest tests/language_tests/generator_tests/ --cpu -x`
- [ ] Lint passes: `ruff check proto_language/language/generator/`

If any check fails, fix before proceeding.
