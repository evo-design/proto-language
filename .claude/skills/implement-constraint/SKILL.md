---
name: implement-constraint
description: >
  Implements, modifies, or debugs constraints in the proto-language DSL.
  Covers the full lifecycle: BaseConfig class with ConfigField, scoring function
  returning List[float], @constraint decorator registration, 3-level export chain,
  and pytest test coverage. Use when working with constraints, scoring functions,
  GC content, structure prediction scores (pLDDT, pTM, pAE), protein quality,
  sequence motifs, RNA structure, or splicing predictions.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# implement-constraint skill

## Before You Start

1. **Read the registry** to see all existing constraints and naming conventions:
   - `proto_language/language/constraint/__init__.py`
2. **Find a similar implementation** to use as a template. Read both its source and tests:
   - Simple (no tools): `proto_language/language/constraint/sequence_composition/gc_content_constraint.py`
   - Tool-based: `proto_language/language/constraint/protein_quality/protein_complexity_constraint.py`
   - Complex config: `proto_language/language/constraint/sequence_annotation/seq_motif_constraint.py`
   - GPU + structure: `proto_language/language/constraint/protein_structure/structure_similarity_constraint.py`
3. **Read the decorator/registry**: `proto_language/language/constraint/constraint_registry.py`

## Complete Implementation Template

### Step 1: Config Class

File: `proto_language/language/constraint/{category}/{name}_constraint.py`

```python
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY

logger = logging.getLogger(__name__)


class MyConstraintConfig(BaseConfig):
    """Configuration for MyConstraint.

    Detailed description of what this constraint evaluates and how scoring works.
    """

    # Required parameters (no default)
    target_value: float = ConfigField(
        title="Target Value",
        description="The target value to optimize toward",
        ge=0.0,
        le=100.0,
    )

    # Optional parameters (with default)
    tolerance: float = ConfigField(
        default=5.0,
        title="Tolerance",
        description="Acceptable deviation from target",
        ge=0.0,
        advanced=True,
    )

    # Field validator (single field, runs before model creation)
    @field_validator("target_value", mode="before")
    @classmethod
    def validate_target(cls, v):
        if isinstance(v, str):
            return float(v)
        return v

    # Model validator (cross-field, runs after model creation)
    @model_validator(mode="after")
    def validate_config(self):
        if self.tolerance > self.target_value:
            raise ValueError("tolerance cannot exceed target_value")
        return self
```

### Step 2: Constraint Function

```python
@constraint(
    key="my-constraint",                          # Unique, kebab-case
    label="My Constraint",                        # Human-readable display name
    config=MyConstraintConfig,                    # Config class from Step 1
    description="Evaluates sequences for ...",    # UI description
    gpu_required=False,                           # True if calls GPU tools
    tools_called=[],                              # e.g. ["esmfold", "segmasker"]
    category="sequence_composition",              # Must match directory name
    supported_sequence_types=["dna", "rna"],      # MUST be non-empty
    num_input_sequences_per_tuple=1,              # 1 = single segment, None = any
)
def my_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: MyConstraintConfig,
) -> List[float]:
    """Evaluate sequences against target value.

    Args:
        input_sequences: List of sequence tuples. Each tuple has one Sequence
            when num_input_sequences_per_tuple=1.
        config: Validated configuration object.

    Returns:
        List of float scores in [0.0, 1.0]. 0.0 = perfect, 1.0 = worst.
    """
    scores = []

    for (seq,) in input_sequences:
        # Handle edge cases
        if len(seq.sequence) == 0:
            seq._metadata["my_metric"] = 0.0
            scores.append(MAX_ENERGY)
            continue

        # Calculate the metric
        metric = _compute_metric(seq.sequence, config)

        # Store metadata (visible in UI and downstream)
        seq._metadata["my_metric"] = metric

        # Calculate penalty: 0.0 = in range, up to 1.0 = worst
        deviation = abs(metric - config.target_value)
        if deviation <= config.tolerance:
            score = MIN_ENERGY
        else:
            excess = deviation - config.tolerance
            score = min(MAX_ENERGY, excess / 100.0)

        scores.append(score)

    return scores
```

## Decorator Argument Reference

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `key` | `str` | Yes | Unique kebab-case identifier (e.g., `"gc-content"`) |
| `label` | `str` | Yes | Human-readable name for UI |
| `config` | `Type[BaseModel]` | Yes | Pydantic config class |
| `description` | `str` | Yes | What this constraint evaluates |
| `gpu_required` | `bool` | No | Default `False`. Set `True` if calling GPU tools |
| `tools_called` | `List[str]` | No | Default `[]`. Tool names this constraint invokes |
| `category` | `str` | No | Must match the subdirectory name (e.g., `"sequence_composition"`) |
| `supported_sequence_types` | `List[str]` | Yes | Non-empty list from: `"dna"`, `"rna"`, `"protein"`, `"ligand"` |
| `num_input_sequences_per_tuple` | `int \| None` | No | `1` for single-segment, `2+` for multi-segment, `None` for any |

## Scoring Convention

- **0.0** (`MIN_ENERGY`) = perfect score, constraint fully satisfied
- **1.0** (`MAX_ENERGY`) = worst score, constraint maximally violated
- Always clamp: `min(MAX_ENERGY, computed_score)`
- Import from: `from proto_language.utils import MAX_ENERGY, MIN_ENERGY`

Common scoring utilities (in `proto_language/utils/`):
- `calculate_percentage_range_deviation(actual, min_val, max_val)` — returns 0.0 if in range, fractional deviation otherwise
- `sigmoid_score(metric, inflection, slope=3.0)` — smooth 0-1 scoring via sigmoid

## Metadata Pattern

Store computed values on the Sequence object for downstream visibility:

```python
seq._metadata["my_metric"] = metric_value
seq._metadata["my_detail"] = {"sub_key": sub_value}
```

After constraint evaluation, metadata is accessible via:
```python
segment.candidate_sequences[i]._constraints_metadata["my_constraint"]["data"]["my_metric"]
# Or via the computed .metadata property:
segment.candidate_sequences[i].metadata["constraints"]["my_constraint"]["data"]["my_metric"]
```

## Tool Integration Pattern

For constraints that call external bioinformatics tools:

```python
from proto_tools import run_{tool}, {Tool}Input, {Tool}Config

@constraint(tools_called=["{tool}"], ...)
def my_tool_constraint(input_sequences, config):
    # Build tool input from sequences
    tool_input = ToolInput(sequences=[seq.sequence for (seq,) in input_sequences])
    tool_config = ToolConfig(param=config.tool_param)

    # Run tool
    result = run_tool(inputs=tool_input, config=tool_config)

    # Handle failure
    if not result.success:
        error_msg = result.errors[0] if result.errors else "Unknown error"
        for (seq,) in input_sequences:
            seq._metadata["tool_error"] = True
            seq._metadata["tool_error_message"] = error_msg
        raise ValueError(f"Tool failed: {error_msg}")

    # Process results
    scores = []
    for (seq,), tool_result in zip(input_sequences, result.per_sequence_results):
        seq._metadata["tool_metric"] = tool_result.value
        seq._metadata["tool_error"] = False
        score = _compute_score(tool_result.value, config)
        scores.append(score)

    return scores
```

### Batching Note

Constraints that call GPU tools should include `batch_size` in their config and pass it
through to the tool config. The tool layer handles the actual batching loop — constraints
should NOT implement their own sequence chunking. Default `batch_size = 1` for safety.

## Export Chain

1. **Category `__init__.py`**: `proto_language/language/constraint/{category}/__init__.py`
   ```python
   from .my_constraint import my_constraint
   ```

2. **Constraint `__init__.py`**: `proto_language/language/constraint/__init__.py`
   ```python
   from .{category} import my_constraint
   # Add to __all__ list
   ```

## Multi-Segment Constraints

For constraints that evaluate relationships between segments (e.g., binding affinity):

```python
@constraint(
    num_input_sequences_per_tuple=2,  # or None for variable
    supported_sequence_types=["protein", "ligand"],
    ...
)
def binding_constraint(input_sequences, config):
    for seq_tuple in input_sequences:
        protein_seq = seq_tuple[0]
        ligand_seq = seq_tuple[1]
        # Evaluate relationship between the two
        ...
```

## Documentation

Documentation `.mdx` files in `docs/` are auto-generated by `generate_docs.py` (run by pre-commit hooks). Never manually edit `.mdx` files — update the Python config docstrings/field descriptions instead.

## Test Requirements

File: `tests/language_tests/constraint_tests/test_{category}/test_{name}_constraint.py`

Every constraint needs these tests:
1. **Parametrized scoring** — multiple sequence/config combos with expected scores
2. **Wrong sequence type** — `pytest.raises(TypeError, match="does not support sequence type")`
3. **Invalid config** — `pytest.raises` on bad config values
4. **Metadata propagation** — verify metadata is set on sequences after evaluation
5. **Edge cases** — empty sequences, boundary values

See the testing skill for complete test templates.

## Validation Checklist

Copy this and check off as you go:

- [ ] Config class inherits `BaseConfig` with `ConfigField`
- [ ] `@constraint` decorator with unique kebab-case key
- [ ] `supported_sequence_types` is non-empty
- [ ] Scoring function returns `List[float]` with scores in [0.0, 1.0]
- [ ] Metadata stored on `seq._metadata` for downstream visibility
- [ ] Edge cases handled (empty sequences, boundary values)
- [ ] Export chain updated at all 3 levels (category `__init__`, constraint `__init__`, `__all__`)
- [ ] Tests cover: parametrized scoring, wrong type, invalid config, metadata, edge cases
- [ ] Tests pass: `pytest tests/language_tests/constraint_tests/ --cpu -x`
- [ ] Lint passes: `flake8 proto_language/language/constraint/`

If any check fails, fix before proceeding.
