---
name: general-dev
description: >
  General development conventions for proto-language: architecture overview,
  coding conventions, Pydantic config patterns (BaseConfig, ConfigField),
  data models (Sequence, Segment, Construct), registry system, export chains,
  batching architecture, naming conventions. Use for general development,
  code review, refactoring, adding utilities, or understanding project structure.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# general-dev skill

## File Header (every new file)

```python
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
```

Never use `print()`. Always use `logger.info()` / `logger.debug()` / `logger.warning()` / `logger.error()`.

## Import Ordering (enforced by ruff)

1. `from __future__ import annotations`
2. Standard library (`os`, `logging`, `typing`, etc.)
3. Third-party (`pydantic`, `numpy`, etc.)
4. Local (`proto_language.*`, `api.*`, `agent.*`)

## Data Model (`proto_language/language/core/`)

```
Sequence          A string + type (dna/rna/protein/ligand). Has ._metadata dict.
    ↓
Segment           Groups proposal sequences for one design region.
                  Has: .sequence_type, .sequence_length, .valid_chars,
                       .proposal_sequences (List[Sequence]), .result_sequences (List[Sequence]),
                       .original_sequence (Sequence), .num_proposals, .num_results
    ↓
Construct         Joins multiple Segments into a complete design. List[Segment].
```

Key: `Segment(sequence="ATCG", sequence_type="dna")` or `Segment(length=100, sequence_type="protein")`.

## Result Export (`proto_language/language/core/`)

Both `Program` and `Optimizer` provide 3 export methods:

```python
# Export files (csv/tsv/json/xlsx). All 4 tables or a single table.
.export(path="./results/", format="csv")
.export(path="seqs.csv", table="sequences")

# Get a pandas DataFrame
df = .to_dataframe(table="sequences")

# FASTA output (string or file)
fasta = .to_fasta()
.to_fasta(path="output.fasta")
```

Tables: `sequences`, `constraints`, `constructs`, `optimization`.
`Program` also accepts `stage=` to filter by optimizer stage.
Underlying utilities live in `proto_language/utils/export.py`.

## Config Pattern (`proto_language/base_config.py`)

All configs inherit `BaseConfig` and use `ConfigField` (not Pydantic's `Field`):

```python
from proto_language.base_config import BaseConfig, ConfigField

class MyConfig(BaseConfig):
    required_param: float = ConfigField(
        title="Display Name",        # UI label
        description="What it does",  # UI tooltip
        ge=0.0, le=100.0,           # Pydantic validators
    )
    optional_param: str = ConfigField(
        default="value",
        title="Optional Param",
        description="...",
        advanced=True,   # Shows in "Advanced" UI section
        hidden=False,    # True = completely hidden from UI
    )
    conditional_param: float = ConfigField(
        default=0.5,
        title="Conditional Param",
        description="Only shown when optional_param is 'value'",
        depends_on={"field": "optional_param", "value": "value"},
    )
```

**`depends_on`** — conditionally show/hide a field based on another field's value:

```python
from typing import List, TypedDict, Union

class DependsOn(TypedDict, total=False):
    field: str                                     # Required: sibling field key to watch
    value: Union[str, int, float, bool, List]      # Show when field == value (or field in value if list)
    not_null: bool                                 # Show when field is not None
```

Evaluation rules (first matching rule wins):
- `{"field": "mode", "value": "percentile"}` — show when `mode == "percentile"`
- `{"field": "use_weights"}` — show when `use_weights` is truthy (omit `value` and `not_null`)
- `{"field": "reference_seq", "not_null": True}` — show when `reference_seq` is not None

Only one of `value` or `not_null` should be specified. Omitting both means "show when truthy."

**BaseConfig behavior** (from `ConfigDict`):
- `extra='ignore'` — unknown fields silently ignored
- `validate_assignment=True` — validates on field updates
- `use_enum_values=True` — enums serialize as values
- `validate_default=True` — default values are validated

**Validators**:
```python
from pydantic import field_validator, model_validator

class MyConfig(BaseConfig):
    @field_validator('param', mode='before')
    @classmethod
    def normalize(cls, v):
        if isinstance(v, str): return [v]  # str → list
        return v

    @model_validator(mode='after')
    def cross_validate(self):
        if self.a > self.b: raise ValueError("a must be <= b")
        return self
```

## Registry Pattern (universal across constraints, generators, optimizers)

All three components follow the same pattern:

```python
# Registration — decorator on function (constraint) or class (generator/optimizer)
@constraint(key="my-key", label="My Label", config=MyConfig, ...)
def my_constraint(input_sequences, config): ...

@generator(key="my-key", label="My Label", config=MyConfig, ...)
class MyGenerator(Generator): ...

@optimizer(key="my-key", label="My Label", config=MyConfig, ...)
class MyOptimizer(Optimizer): ...

# Discovery
Registry.list_all()          # → List[Spec] with all metadata
Registry.get(key)            # → Spec for one component
Registry.get_schema(key)     # → JSON schema for client

# Factory (constraints only — generators/optimizers instantiated directly)
ConstraintRegistry.create(key, segments, config_dict, label=None, threshold=None, weight=None)
```

## Batching Architecture

`batch_size` controls GPU memory usage across the generator → tool boundary:

```
GeneratorConfig.batch_size (default=1)
    ↓ stored as self.batch_size in __init__
Generator.sample()
    ↓ passes ALL sequences + batch_size to tool
ToolConfig.batch_size (default=1)
    ↓ tool chunks sequences into batches
Standalone inference.py
    ↓ processes batch_size sequences per GPU forward pass
    returns all results concatenated
```

**Key rules:**
- `batch_size` defaults to `1` everywhere — safe by default, users opt in to higher throughput
- Generators NEVER implement batching loops — they pass all sequences to the tool in one call
- The tool layer (in proto-tools) owns the actual batching loop
- Both GeneratorConfig and ToolConfig define `batch_size: int = ConfigField(default=1, ...)`

## Export Chain

Every new component must be exported through the `__init__.py` chain:

```
proto_language/language/constraint/{category}/{name}_constraint.py
    → proto_language/language/constraint/{category}/__init__.py
    → proto_language/language/constraint/__init__.py

proto_language/language/generator/{name}_generator.py
    → proto_language/language/generator/__init__.py

proto_language/language/optimizer/{name}_optimizer.py
    → proto_language/language/optimizer/__init__.py
```

For generators, export **both** the class and its config: `from .my_generator import MyGenerator, MyGeneratorConfig`.

## Naming Conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| Registry key | kebab-case | `"gc-content"`, `"random-protein"`, `"mcmc"` |
| Tool registry key | `{tool}-{action}` kebab-case | `"esm2-sample"`, `"boltz2-prediction"`, `"blast-create-db"` |
| Config class | `{Name}Config` | `GCContentConfig`, `MCMCOptimizerConfig` |
| Constraint file | `{name}_constraint.py` | `gc_content_constraint.py` |
| Generator file | `{name}_generator.py` | `uniform_mutation_generator.py` |
| Optimizer file | `{name}_optimizer.py` | `mcmc_optimizer.py` |
| Test file | `test_{name}.py` | `test_gc_content_constraint.py` |
| Constraint function | `{name}_constraint` | `gc_content_constraint` |
| Generator class | `{Name}Generator` | `RandomProteinGenerator` |
| Optimizer class | `{Name}Optimizer` | `MCMCOptimizer` |

## Tool Registry Key Pattern (`proto-tools`)

All tool keys follow **`{tool}-{action}`** in kebab-case:
- `{tool}` = the software/model name (e.g., `esm2`, `boltz2`, `blast`, `alphafold3`)
- `{action}` = what the tool does (e.g., `sample`, `score`, `prediction`, `embedding`, `search`)

## Linting & Formatting

```bash
ruff check proto_language api agent tests    # Lint: F401 (unused imports) + F841 (unused vars) + import sorting
pre-commit run --all-files                    # All checks
```
