---
name: implement-constraint
description: >
  Implements, modifies, or debugs constraints in the proto-language DSL.
  Covers the full lifecycle: BaseConfig class with ConfigField, scoring function
  returning list[ConstraintOutput], @constraint decorator registration, 3-level export chain,
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
import logging

from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
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
    )

    aggregation_mode: str = ConfigField(
        default="mean",
        title="Aggregation Mode",
        description="How to aggregate per-residue scores",
    )
    percentile_value: float = ConfigField(
        default=75.0,
        title="Percentile Value",
        description="Which percentile to use when mode is 'percentile'",
        ge=0.0,
        le=100.0,
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
    uses_gpu=False,                               # True if calls GPU tools
    tools_called=[],                              # e.g. ["esmfold", "segmasker"]
    category="sequence_composition",              # Must match directory name
    supported_sequence_types=["dna", "rna"],      # MUST be non-empty
)
def my_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: MyConstraintConfig,
) -> list[ConstraintOutput]:
    """Evaluate sequences against target value.

    Args:
        input_sequences: List of sequence tuples. Each tuple has one Sequence
            per input label (default: single segment).
        config: Validated configuration object.

    Returns:
        One ``ConstraintOutput`` per proposal. ``score`` is in [0.0, 1.0]
        (0.0 = perfect, 1.0 = worst); ``metadata`` carries per-proposal values
        that land under ``_constraints_metadata[label]["data"]``.
    """
    results = []

    for (seq,) in input_sequences:
        if len(seq.sequence) == 0:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"my_metric": 0.0}))
            continue

        metric = _compute_metric(seq.sequence, config)
        deviation = abs(metric - config.target_value)
        if deviation <= config.tolerance:
            score = MIN_ENERGY
        else:
            excess = deviation - config.tolerance
            score = min(MAX_ENERGY, excess / 100.0)

        results.append(ConstraintOutput(score=score, metadata={"my_metric": metric}))

    return results
```

## Decorator Argument Reference

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `key` | `str` | Yes | Unique kebab-case identifier (e.g., `"gc-content"`) |
| `label` | `str` | Yes | Human-readable name for UI |
| `config` | `type[BaseModel]` | Yes | Pydantic config class |
| `description` | `str` | Yes | What this constraint evaluates |
| `uses_gpu` | `bool` | No | Default `False`. Set `True` if calling GPU tools |
| `tools_called` | `list[str]` | No | Default `[]`. Tool names this constraint invokes |
| `category` | `str` | No | Must match the subdirectory name (e.g., `"sequence_composition"`) |
| `supported_sequence_types` | `list[str]` | Yes | Non-empty list from: `"dna"`, `"rna"`, `"protein"`, `"ligand"` |
| `input_labels` | `list[str \| InputSlot] \| None` | No | Default `["Sequence"]`. Strings for plain labels (`["Query", "Reference"]`), or `InputSlot(label=..., requires_logits=True, requires_structure=True)` for per-slot swap-detection. Use `None` for any number of interchangeable inputs |
| `requires_generators` | `list[str] \| None` | No | Generator registry keys required in the same optimizer stage. Validated at construction time |
| `backward` | `Callable \| None` | No | Gradient callable: `(input_sequences, *, config, **kwargs) -> list[GradientConstraintOutput]` |
| `backward_config` | `Type[BaseModel] \| None` | No | Separate config class for backward callable. If `None`, uses `config` |

## Constraint Modes

A Constraint can expose three capability shapes:

| Mode | `function` | `backward` | Registered how | Used by |
|---|---|---|---|---|
| `"discrete"` | ✅ | — | Decorated function returns `list[ConstraintOutput]` | MCMC, BeamSearch, RejSamp |
| `"gradient"` | — | ✅ | Decorated function returns `list[GradientConstraintOutput]` | GradientOptimizer |
| `"dual"` | ✅ | ✅ | Decorated forward fn + `backward=` kwarg | Any optimizer — each picks the right path |

**Discovery:**

```python
spec = ConstraintRegistry.get("my-constraint")
spec.mode              # "discrete" | "gradient" | "dual"
c = ConstraintRegistry.create("my-constraint", segments, config_dict)
c.supports_discrete    # True if forward scoring function is set
c.supports_gradient    # True if backward callable is set
```

**Optimizer routing** is automatic: `GradientOptimizer` filters on `supports_gradient`; MCMC/Beam/RejSamp call `constraint.evaluate()` which guards on `_function is None`. Dual-mode constraints pass both filters and route to the correct callable per optimizer.

**Standard kwargs forwarded by GradientOptimizer:** `temperature` (softmax temperature), `soft` (soft blending 0-1), `hard` (straight-through estimator blending 0-1). Accept these via `**kwargs` or explicitly in the backward signature.

### Gradient-Only Constraints

For constraints that only compute gradients (no discrete scoring path). The `@constraint`
decorator auto-detects the role from the return type annotation: `-> list[GradientConstraintOutput]`
registers the function as the backward callable and sets `mode="gradient"`.

```python
from proto_language.language.core.constraint import GradientConstraintOutput

@constraint(
    key="my-gradient-only",
    label="My Gradient-Only Constraint",
    config=MyConfig,
    description="Gradient-only scorer; no meaningful forward mode",
    uses_gpu=True,
    supported_sequence_types=["protein"],
)
def my_backward(
    input_sequences: list[tuple[Sequence, ...]], *, config: MyConfig, **kwargs: Any,
) -> list[GradientConstraintOutput]:
    """Compute gradient; no discrete scoring path exists."""
    results: list[GradientConstraintOutput] = []
    for (seq,) in input_sequences:
        grad, loss = run_tool(seq.logits, config)
        results.append(GradientConstraintOutput(gradient=(grad,), loss=loss))
    return results
```

This constraint is invisible to MCMC / BeamSearch / RejSamp — `evaluate()` raises because
`spec.function is None`. Use `"gradient"`-only when the operation **cannot** be scored
discretely — e.g., the backward is the only meaningful output, and attempting a forward-only
mode makes no semantic sense. For constraints that CAN be scored both ways (most AF2/AbLang
setups), use the dual-mode pattern below instead.

### Dual-Mode Constraints (Canonical for Multi-Stage Pipelines)

**This is the canonical pattern** for any constraint whose underlying computation can
produce both a discrete score and a gradient — forward and backward of the same
evaluator. Register ONE `@constraint` on the forward function and pair it with
`backward=` for the gradient callable:

```python
from proto_language.language.core.constraint import GradientConstraintOutput

def my_backward(
    input_sequences: list[tuple[Sequence, ...]], *, config: MyConfig, temperature: float, **kwargs: Any,
) -> list[GradientConstraintOutput]:
    """Gradient path — returns list[GradientConstraintOutput]."""
    ...


@constraint(
    key="my-constraint",
    label="My Constraint",
    config=MyConfig,
    description="Scores discrete sequences (MCMC) or computes gradients (GradientOptimizer).",
    uses_gpu=True,
    supported_sequence_types=["protein"],
    backward=my_backward,          # <-- pair the backward callable
    backward_config=MyConfig,      # optional: separate config class for the backward
)
def my_forward(
    input_sequences: list[tuple[Sequence, ...]], *, config: MyConfig,
) -> list[ConstraintOutput]:
    """Forward path — returns one ConstraintOutput per proposal."""
    ...
```

**Why dual-mode over two separate registry entries.** The underlying computation is one
thing; forward vs backward are two *queries* against the same evaluator, not two
evaluators. Splitting by caller-side concern (GradientOptimizer vs MCMC) makes constraint
identity depend on how it's used — backwards. Dual-mode:

- **One registry key, one UI entry, one factory.** Multi-stage pipelines (e.g. Germinal:
  GradientOptimizer → MCMCOptimizer) use one `Constraint(function=..., backward=..., ...)`
  across all stages. The optimizer picks the right path.
- **Single `_constraints_metadata[label]` namespace** uniform across stages, so
  confidence gates that read metadata between stages work transparently.
- **No duplicated decorator metadata.** Config, label, `tools_called`,
  `supported_sequence_types`, `input_labels`, description — one source of truth.
- **Shared config guaranteed.** If forward and backward share a config class (as in AF2
  binder design), users can't accidentally configure one mode differently from the other.

**Canonical in-tree example.** `af2-binder`
(`proto_language/language/constraint/differentiable/af2_binder_constraint.py`)
registers `af2_binder_forward` as the forward callable, pairs with `af2_binder_backward`,
and both paths construct the same `AlphaFold2BinderConfig` from shared `AF2BinderConstraintConfig`
fields — only `soft`, `hard`, and `compute_gradient` differ between modes.

**Rule of thumb:** if your constraint's underlying computation supports both forward and
backward, register as dual-mode. Single-mode is only appropriate when the other mode
genuinely doesn't exist (pure discrete heuristics with no differentiable form; pure
gradient hooks with no scoring interpretation).

## Scoring Convention

- **0.0** (`MIN_ENERGY`) = perfect score, constraint fully satisfied
- **1.0** (`MAX_ENERGY`) = worst score, constraint maximally violated
- Always clamp: `min(MAX_ENERGY, computed_score)`
- Import from: `from proto_language.utils import MAX_ENERGY, MIN_ENERGY`

Common scoring utilities (in `proto_language/utils/`):
- `calculate_percentage_range_deviation(actual, min_val, max_val)` — returns 0.0 if in range, fractional deviation otherwise
- `sigmoid_score(metric, inflection, slope=3.0)` — smooth 0-1 scoring via sigmoid

## Metadata Pattern

Pass per-proposal values through the `metadata` field of `ConstraintOutput`. The framework
stores the dict under `_constraints_metadata[label]["data"]`:

```python
ConstraintOutput(score=s, metadata={"my_metric": metric_value, "my_detail": {"sub_key": sub_value}})
```

After evaluation, read it back via:
```python
segment.proposal_sequences[i]._constraints_metadata["<constraint_key>"]["data"]["my_metric"]
# Or the computed .metadata view:
segment.proposal_sequences[i].metadata["constraints"]["<constraint_key>"]["data"]["my_metric"]
```

Metadata keys may freely collide with infrastructure keys (`score`, `weight`, `weighted_score`) —
user metadata lands one level deeper under `"data"`.

### Attaching Structures and Logits

When a constraint predicts a structure or logits for its inputs, pass them through the result
so the framework assigns them to the proposal `Sequence` objects. Align the tuple length with
the input tuple:

```python
n = len(proposal_tuple)  # number of input segments
# Attach predicted structure to slot 0; leave other slots untouched.
results.append(ConstraintOutput(
    score=s,
    structures=(structure,) + (None,) * (n - 1),
))
```

Non-`None` entries in `structures` / `logits` are written to `inputs[i].structure` /
`inputs[i].logits`; `None` entries and empty tuples are no-ops.

### Metadata: keep it inline

Constraints put raw Python objects (PDB strings, hit-dict lists, ORF annotations) into
`metadata` directly. Use `None` — not `[]` — as the empty sentinel to keep value types
homogeneous when readers iterate.

```python
results.append(ConstraintOutput(
    score=s,
    metadata={
        "pdb_output": structure.structure_pdb,
        "mmseqs_results": hits or None,
        "avg_plddt": 0.85,
        "hit_count": len(hits),
    },
    structures=(structure,) + (None,) * (len(proposal_tuple) - 1),
))
```

## Tool Integration Pattern

For constraints that call external bioinformatics tools:

```python
from proto_tools import run_{tool}, {Tool}Input, {Tool}Config

@constraint(tools_called=["{tool}"], ...)
def my_tool_constraint(input_sequences, config):
    tool_input = ToolInput(sequences=[seq.sequence for (seq,) in input_sequences])
    tool_config = ToolConfig(param=config.tool_param)
    result = run_tool(inputs=tool_input, config=tool_config)

    # Handle failure: surface the error via each result's metadata, then propagate.
    if not result.success:
        error_msg = result.errors[0] if result.errors else "Unknown error"
        return [
            ConstraintOutput(
                score=MAX_ENERGY,
                metadata={"tool_error": True, "tool_error_message": error_msg},
            )
            for _ in input_sequences
        ]

    # Success path
    results = []
    for (seq,), tool_result in zip(input_sequences, result.per_sequence_results):
        score = _compute_score(tool_result.value, config)
        results.append(ConstraintOutput(
            score=score,
            metadata={"tool_metric": tool_result.value, "tool_error": False},
        ))
    return results
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
    input_labels=["Protein", "Ligand"],
    supported_sequence_types=["protein", "ligand"],
    ...
)
def binding_constraint(input_sequences, config):
    for protein_seq, ligand_seq in input_sequences:
        # Evaluate relationship between the two
        ...
```

## Documentation

Documentation reference pages are auto-generated from Python docstrings and field descriptions. To update documentation, update the Python config docstrings/field descriptions in the source code.

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
- [ ] Mode chosen correctly: discrete (score only), gradient (backward only, `-> list[GradientConstraintOutput]`), or dual (`backward=` paired with forward scoring). Prefer dual when the computation supports both.
- [ ] `supported_sequence_types` is non-empty
- [ ] Scoring function returns `list[ConstraintOutput]`; `score` in [0.0, 1.0]
- [ ] Per-proposal data passed via `ConstraintOutput.metadata`; predicted structures / logits passed via `ConstraintOutput.structures` / `.logits` (tuple aligned with inputs)
- [ ] Edge cases handled (empty sequences, boundary values)
- [ ] Export chain updated at all 3 levels (category `__init__`, constraint `__init__`, `__all__`)
- [ ] UI visibility rules for conditionally relevant fields live in client overlays, not in `ConfigField`
- [ ] Tests cover: parametrized scoring, wrong type, invalid config, metadata, edge cases
- [ ] Tests pass: `pytest tests/language_tests/constraint_tests/ --cpu -x`
- [ ] Lint passes: `ruff check proto_language/language/constraint/`
- [ ] Type check passes: `mypy proto_language/language/constraint/`

If any check fails, fix before proceeding.
