---
name: implement-generator
description: >
  Implements, modifies, or debugs generators in the proto-language DSL.
  Covers the full lifecycle: config class, Generator subclass with __init__/assign/sample,
  input_type-specific patterns (mutation, autoregressive, inverse folding, gradient),
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
2. **Find a similar implementation** by `input_type`:
   - Mutation (CPU): `proto_language/language/generator/random_nucleotide_generator.py`
   - Mutation (GPU tool): `proto_language/language/generator/esm2_generator.py`
   - Autoregressive: `proto_language/language/generator/evo2_generator.py`
   - Inverse folding: `proto_language/language/generator/proteinmpnn_generator.py`
   - Gradient: `proto_language/language/generator/position_weight_generator.py`
3. **Read the base class**: `proto_language/language/core/generator.py`
4. **Read the decorator/registry**: `proto_language/language/generator/generator_registry.py`

## Generator ABC Contract

```python
class Generator(ABC):
    # Required ClassVar on every concrete subclass:
    input_type: ClassVar[GeneratorInputType]        # what kind of starting input

    @abstractmethod
    def __init__(self) -> None:
        # Always call super().__init__()

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        # Validates: not ligand, sequence type compatible, tied segments agree on type/length

    @abstractmethod
    def _sample(self, *args, **kwargs) -> None:
        # Modifies self.segment.proposal_sequences IN PLACE

    def _validate_generator(self) -> None:
        # Called at start of _sample(). Dispatches on self.input_type:
        # - STARTING_SEQUENCE: raises if proposals empty (subclasses may seed first; random mutation generators do)
        # - PROMPT: warns if proposals already populated (will be overwritten)
        # - STRUCTURE: seeds 'X' * length if empty, logs INFO
        # - LOGITS: no special init
```

**Critical rules**:
- Always call `super().__init__()` in `__init__`
- Always call `super().assign(segments)` as first line in custom `assign()`
- Always call `self._validate_generator()` as first line in `_sample()`
- Always use `@final` decorator on the class to prevent subclassing
- `_sample()` modifies sequences **in-place** — it returns nothing
- Declare the `input_type` classvar on the concrete class (no default on the base — every concrete generator must set it). For generators that take dynamic conditioning data via `CyclingOptimizer`, the conditioning data is passed as the **first non-self positional argument** to `_sample()` — make that the conditioning kwarg (`prompts` for autoregressive, `structure_inputs` for inverse folding).
- Per-proposal diagnostics go to ``proposal._generator_metadata[self._spec.key]`` (dict), namespaced by the registry key. Don't write to ``proposal._metadata``: that's a free-form user bag and prefixed keys collide across generators.

## Input Kinds

Each generator declares its `input_type` via a classvar. The client reads this from `GeneratorRegistry` to render the right input UI. The Program-build validator (`Program._validate_generator_inputs` in `core/program.py`) walks stages in order and verifies the input is satisfiable; failures raise at `Program.__init__` time.

| `input_type` | Category | Examples | Runtime input source |
|---|---|---|---|
| `STARTING_SEQUENCE` | `mutation` | ESM2, ESM3, MSA, Random*, SemigreedyMutation | `segment.proposal_sequences[].sequence` (from `segment.input_sequence` or prior stage); base validator raises if empty. Random mutation generators override `_sample()` to seed empty proposals with a fully random sequence on the first call. |
| `PROMPT` | `autoregressive` | Evo1, Evo2, ProGen2 | `config.prompts` or `_sample(prompts=...)` (first positional kwarg) from CyclingOptimizer |
| `STRUCTURE` | `inverse_folding` | ProteinMPNN, LigandMPNN | `config.structure_inputs` or `_sample(structure_inputs=...)` (first positional kwarg) from CyclingOptimizer; segment seeded with `'X' * length` if no prior sequence |
| `LOGITS` | `gradient` | PositionWeight | `segment.proposal_sequences[].logits` from a prior `GradientOptimizer` stage |

## Implementation Steps

For complete config class and generator class templates, use the `Read` tool to load:
- **Templates**: `.claude/skills/implement-generator/TEMPLATES.md`

Summary of the workflow:
1. **Config class** — inherit `BaseConfig`, use presentation-agnostic `ConfigField`, declare model params
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
| `uses_gpu` | `bool` | No | Whether generator requires GPU |
| `tools_called` | `List[str]` | No | Default `[]` |
| `supported_sequence_types` | `List[str]` | No | Default `[]` (= all types). Options: `"dna"`, `"rna"`, `"protein"`. Ligand segments are immutable fixed targets — `Generator.assign()` rejects them, so do not list `"ligand"` here. |

`category` is **not** a decorator argument — it's auto-derived from the class's `input_type` classvar via `INPUT_TYPE_TO_CATEGORY` in `generator_registry.py`.

Example registration:

```python
@generator(
    key="my-generator",
    label="My Generator",
    config=MyGeneratorConfig,
    description="...",
    uses_gpu=False,
)
@final
class MyGenerator(Generator):
    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self, config: MyGeneratorConfig) -> None:
        super().__init__()
        ...
```

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

For generators that call external tools (via proto-tools):

```python
from proto_tools import run_{tool}, {Tool}Input, {Tool}Config

def sample(self) -> None:
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
    for i, sequence in enumerate(result.sequences):
        self.segment.proposal_sequences[i].sequence = sequence
```

Pass `seed=self._next_seed()` when the underlying tool output should participate
in seeded program determinism. Do not pass `seed_per_item`; `proto-tools`
automatically derives per-item seeds for `seed_sensitive=True` iterable tools.

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

- [ ] Config class inherits `BaseConfig` with presentation-agnostic `ConfigField`
- [ ] `@generator` decorator with unique kebab-case key (category is auto-derived from input_type)
- [ ] `@final` decorator on class
- [ ] `input_type` classvar set to the right `GeneratorInputType` value
- [ ] If the generator takes dynamic conditioning data via `CyclingOptimizer`, the conditioning kwarg is the **first non-self positional argument** of `_sample()` (`prompts` for autoregressive, `structure_inputs` for inverse folding)
- [ ] `__init__` calls `super().__init__()`
- [ ] `assign()` calls `super().assign(segments)` first (only if overriding)
- [ ] `_sample()` calls `self._validate_generator()` first
- [ ] `_sample()` modifies `proposal_sequences` in-place (returns nothing)
- [ ] No batching loop in generator (tool handles batching)
- [ ] Export chain updated: `generator/__init__.py` (class + config)
- [ ] Tests cover: init, assign, sample, batch, type validation, config validation
- [ ] Tests pass: `pytest tests/language_tests/generator_tests/ --cpu -x`
- [ ] Lint passes: `ruff check proto_language/language/generator/`
- [ ] Type check passes: `mypy proto_language/language/generator/`

If any check fails, fix before proceeding.
