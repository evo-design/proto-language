---
name: implement-constraint
description: >
  Implements, modifies, or debugs constraints in the proto-language DSL.
  Covers the full lifecycle: BaseConfig class with ConfigField, scoring function
  returning list[float], @constraint decorator registration, 3-level export chain,
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

    # Conditional field — only visible when a specific mode is selected
    aggregation_mode: str = ConfigField(
        default="mean",
        title="Aggregation Mode",
        description="How to aggregate per-residue scores",
    )
    percentile_value: float = ConfigField(
        default=75.0,
        title="Percentile Value",
        description="Which percentile to use (only when mode is 'percentile')",
        ge=0.0,
        le=100.0,
        depends_on={"field": "aggregation_mode", "value": "percentile"},
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
) -> list[float]:
    """Evaluate sequences against target value.

    Args:
        input_sequences: List of sequence tuples. Each tuple has one Sequence
            per input label (default: single segment).
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
| `config` | `type[BaseModel]` | Yes | Pydantic config class |
| `description` | `str` | Yes | What this constraint evaluates |
| `uses_gpu` | `bool` | No | Default `False`. Set `True` if calling GPU tools |
| `tools_called` | `list[str]` | No | Default `[]`. Tool names this constraint invokes |
| `category` | `str` | No | Must match the subdirectory name (e.g., `"sequence_composition"`) |
| `supported_sequence_types` | `list[str]` | Yes | Non-empty list from: `"dna"`, `"rna"`, `"protein"`, `"ligand"` |
| `input_labels` | `list[str \| InputSlot] \| None` | No | Default `["Sequence"]`. Strings for plain labels (`["Query", "Reference"]`), or `InputSlot(label=..., requires_logits=True, requires_structure=True)` for per-slot swap-detection. Use `None` for any number of interchangeable inputs |
| `backward` | `Callable \| None` | No | Gradient callable: `(inputs, *, config, **kwargs) -> GradientResult` |
| `backward_config` | `Type[BaseModel] \| None` | No | Separate config class for backward callable. If `None`, uses `config` |

## Constraint Modes

A Constraint can expose three capability shapes:

| Mode | `function` | `backward` | Registered how | Used by |
|---|---|---|---|---|
| `"discrete"` | ✅ | — | Decorated function returns `list[float]` | MCMC, BeamSearch, RejSamp |
| `"gradient"` | — | ✅ | Decorated function returns `GradientResult` | GradientOptimizer |
| `"dual"` | ✅ | ✅ | Decorated forward fn + `backward=` kwarg | Any optimizer — each picks the right path |

**Discovery:**

```python
spec = ConstraintRegistry.get("my-constraint")
spec.mode              # "discrete" | "gradient" | "dual"
c = ConstraintRegistry.create("my-constraint", segments, config_dict)
c.supports_discrete    # True if forward scoring function is set
c.supports_gradient    # True if backward callable is set
```

**Optimizer routing** is automatic: `GradientOptimizer` filters on `supports_gradient` (`proto_language/language/optimizer/gradient_optimizer.py:357`); MCMC/Beam/RejSamp call `constraint.evaluate()` which guards on `_function is None` (`proto_language/language/core/constraint.py:292`). Dual-mode constraints pass both filters and route to the correct callable per optimizer.

### Gradient-Only Constraints

For constraints that only compute gradients (no discrete scoring path) — e.g., gradient-only
naturalness terms that wrap a model's backward pass. The `@constraint` decorator auto-detects
the role from the return type annotation: `-> GradientResult` registers the function as the
backward callable and sets `mode="gradient"`.

```python
from proto_language.language.core.constraint import GradientResult

@constraint(
    key="ablang-vhh-gradient",
    label="AbLang VHH Naturalness Gradient",
    config=AbLangGradientConstraintConfig,
    description="Antibody naturalness gradient for VHH redesign",
    uses_gpu=True,
    supported_sequence_types=["protein"],
)
def ablang_vhh_gradient_backward(
    inputs: tuple[Sequence, ...], *, config, temperature: float, **kwargs: Any,
) -> GradientResult:
    """Compute gradient of naturalness w.r.t. relaxed logits."""
    grad, loss = run_ablang_gradient(inputs[0].logits, temperature, config)
    return GradientResult(gradient=(grad,), loss=loss)
```

This constraint is invisible to MCMC / BeamSearch / RejSamp (they skip it with a warning
from GradientOptimizer-style filters; evaluate() would raise). Use `"gradient"`-only when
the operation **cannot** be scored discretely — e.g., the backward is the only meaningful
output, and attempting a forward-only mode makes no semantic sense.

### Dual-Mode Constraints (Canonical for Multi-Stage Pipelines)

**This is the canonical pattern** for any constraint whose underlying computation can
produce both a discrete score and a gradient — forward and backward of the same
evaluator. Register ONE `@constraint` on the forward function and pair it with
`backward=` for the gradient callable:

```python
from proto_language.language.core.constraint import GradientResult

def my_backward(
    inputs: tuple[Sequence, ...], *, config: MyConfig, temperature: float, **kwargs: Any,
) -> GradientResult:
    """Gradient path — returns GradientResult."""
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
) -> list[float]:
    """Forward path — returns [0, 1] scores."""
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
and both paths construct the same `AlphaFold2BinderConfig` from shared `AF2BinderConfig`
fields — only `soft` and `compute_gradient` differ between modes.

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

Store computed values on the Sequence object for downstream visibility:

```python
seq._metadata["my_metric"] = metric_value
seq._metadata["my_detail"] = {"sub_key": sub_value}
```

After constraint evaluation, metadata is accessible via:
```python
segment.proposal_sequences[i]._constraints_metadata["my_constraint"]["data"]["my_metric"]
# Or via the computed .metadata property:
segment.proposal_sequences[i].metadata["constraints"]["my_constraint"]["data"]["my_metric"]
```

### Externalizing Large Metadata

When a constraint produces large metadata (structure files, search hit lists, ORF annotations,
domain results — anything that could exceed ~1KB), externalize it to the content-addressed
file store instead of storing it inline. This prevents bloating `seq._metadata` and database rows.

```python
import json
from proto_language.storage import store_file, FileType

# Write Structure onto first sequence for in-memory data flow (optimizer/generator reads this):
proposal_tuple[0].structure = structure

# Large file content (PDB, CIF, etc.) — store for export pipeline:
seq._metadata["pdb_output"] = store_file(structure.structure_pdb, FileType.PDB)

# Large JSON data (hit lists, ORF annotations, etc.) — serialize then store:
# Use None (not []) as the empty sentinel to avoid mixed types (dict vs list).
seq._metadata["mmseqs_results"] = store_file(
    json.dumps(hits), FileType.JSON
) if hits else None

# Small scalar values — keep inline (no store_file needed):
seq._metadata["avg_plddt"] = 0.85
seq._metadata["hit_count"] = len(hits)
```

**When to use `store_file()`:**
- Structure files (PDB, CIF): always externalize
- Tool output lists/dicts (MMseqs hits, ORF predictions, domain results): externalize
- Scalar metrics, short strings, small dicts: keep inline

**Available FileTypes:** `PDB`, `CIF`, `HMM`, `FASTA`, `CSV`, `JSON`, `BINARY`

**Reading externalized data:** Use `get_file_content()` — it transparently handles both
inline strings and file references:

```python
from proto_language.storage import get_file_content
content = get_file_content(seq._metadata["pdb_output"])  # works with both formats
```

The export pipeline handles file references automatically — no special handling needed
in export code.

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
- [ ] Mode chosen correctly: discrete (score only), gradient (backward only, `-> GradientResult`), or dual (`backward=` paired with forward scoring). Prefer dual when the computation supports both.
- [ ] `supported_sequence_types` is non-empty
- [ ] Scoring function returns `list[float]` with scores in [0.0, 1.0]
- [ ] Metadata stored on `seq._metadata` for downstream visibility
- [ ] Edge cases handled (empty sequences, boundary values)
- [ ] Export chain updated at all 3 levels (category `__init__`, constraint `__init__`, `__all__`)
- [ ] Use `depends_on` for fields that are only relevant when another field has a specific value
- [ ] Tests cover: parametrized scoring, wrong type, invalid config, metadata, edge cases
- [ ] Tests pass: `pytest tests/language_tests/constraint_tests/ --cpu -x`
- [ ] Lint passes: `ruff check proto_language/language/constraint/`
- [ ] Type check passes: `mypy proto_language/language/constraint/`

If any check fails, fix before proceeding.
