---
name: implement-optimizer
description: >
  Use this skill when the user asks to create, modify, or debug an optimizer
  in the proto-language language core. This covers the full lifecycle:
  config class, Optimizer subclass with __init__/run, dual-pool architecture,
  constraint evaluation, decorator registration, export chain, and tests.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# implement-optimizer skill

## Before You Start

1. **Read the registry** to see all existing optimizers:
   - `proto_language/language/optimizer/__init__.py`
2. **Find a similar implementation** by type:
   - Iterative (MCMC): `proto_language/language/optimizer/mcmc_optimizer.py`
   - Batch (greedy): `proto_language/language/optimizer/topk_optimizer.py`
   - Autoregressive (beam): `proto_language/language/optimizer/beam_search_optimizer.py`
   - Cycling: `proto_language/language/optimizer/cycling_optimizer.py`
3. **Read the base class**: `proto_language/language/core/optimizer.py`
4. **Read the decorator/registry**: `proto_language/language/optimizer/optimizer_registry.py`

## Optimizer ABC Contract

```python
class Optimizer(ABC):
    @abstractmethod
    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: BaseModel,
    ) -> None:
        # Stores constructs, generators, constraints, config
        # Calls _validate_optimizer()

    @abstractmethod
    def run(self) -> None:
        # Executes the optimization loop
        # Modifies segments' selected_sequences and candidate_sequences
```

## Dual-Pool Architecture

Every optimizer manages two sequence pools per segment:

```
selected_sequences    Persistent top-K results across iterations
                      Size: num_selected (from config)

candidate_sequences   Temporary proposals generated each step
                      Size: num_candidates (computed from config)
```

**Flow per optimization step**:
1. Copy `selected_sequences` → `candidate_sequences` (expanded/contracted as needed)
2. Apply generators to mutate `candidate_sequences`
3. Evaluate constraints on `candidate_sequences`
4. Update `selected_sequences` based on scores

## Filter vs Scoring Constraints

Constraints split into two evaluation phases:

```
Filter constraints (threshold set)     Evaluated FIRST, binary pass/fail
    ↓ only passing candidates proceed
Scoring constraints (no threshold)     Evaluated on survivors only
    ↓
Aggregate score = weighted sum/product of scoring constraint results
```

- Rejected candidates receive `filter_penalty` (default: `inf`) and skip scoring entirely
- Use `constraint.evaluate(mask=...)` to selectively evaluate only certain candidates

## Key Base Class Methods

### `score_energy(operation="add", filter_penalty=float("inf"))`

Evaluates ALL constraints on current `candidate_sequences`:

```python
# In your run() method:
self.score_energy(operation="add")      # Additive scoring (default)
self.score_energy(operation="multiply") # Multiplicative scoring

# After calling, self.energy_scores is populated:
# List[float] of length num_candidates
```

### `_initialize_sequence_pools()`

Sets up `candidate_sequences` from `selected_sequences` with cycling:

```python
# If num_candidates > num_selected, cycles through selected to fill
# If num_candidates < num_selected, takes first N
# Preserves diversity by round-robin assignment
```

### `_save_progress_snapshot(time_step)`

Saves current state to `self.history`:

```python
self._save_progress_snapshot(step)
# Appends: {"time_step": step, "constructs": [...], "energy_scores": [...]}
```

### `_validate_optimizer()`

Comprehensive validation called in `__init__`:
- Non-empty constructs, generators, constraints lists
- All generators assigned to segments
- No duplicate constraint labels
- All constraint inputs reference segments in constructs
- Generator segments exist in constructs

### State Management

```python
self._prepare_run()              # Reset history, prepare for fresh run
self._capture_initial_state()    # Snapshot state before run (for multi-run)
self._restore_initial_state()    # Restore to captured state
```

## Complete Implementation Template

### Step 1: Config Class

File: `proto_language/language/optimizer/{name}_optimizer.py`

```python
from __future__ import annotations

import copy
import logging
from typing import List, Optional

from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Construct, Constraint, Generator, Optimizer
from proto_language.language.optimizer.optimizer_registry import optimizer

logger = logging.getLogger(__name__)


class MyOptimizerConfig(BaseConfig):
    """Configuration for MyOptimizer.

    Detailed description of the optimization algorithm and its parameters.
    """

    num_selected: int = ConfigField(
        default=10,
        title="Number Selected",
        description="Number of top sequences to maintain",
        ge=1,
    )

    num_steps: int = ConfigField(
        default=100,
        title="Number of Steps",
        description="Total optimization iterations",
        ge=1,
    )

    verbose: bool = ConfigField(
        default=True,
        title="Verbose",
        description="Log progress during optimization",
        advanced=True,
    )

    @model_validator(mode="after")
    def validate_config(self):
        return self
```

### Step 2: Optimizer Class

```python
@optimizer(
    key="my-optimizer",                         # Unique, kebab-case
    label="My Optimizer",                       # Human-readable name
    config=MyOptimizerConfig,                   # Config class
    description="Optimizes sequences using ...",# UI description
)
class MyOptimizer(Optimizer):
    """Optimize sequences using MyAlgorithm.

    Detailed description of the algorithm.
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: MyOptimizerConfig,
    ) -> None:
        # Store config BEFORE calling super().__init__
        # (super validates, which may need config values)
        self._num_selected = config.num_selected
        self._num_steps = config.num_steps
        self._verbose = config.verbose

        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            config=config,
        )

    @property
    def num_selected(self) -> int:
        return self._num_selected

    @property
    def num_candidates(self) -> int:
        """Number of candidate sequences per step."""
        return self._num_selected  # Override as needed

    def run(self) -> None:
        """Execute the optimization loop."""
        self._prepare_run()

        # Initialize sequence pools
        self._initialize_sequence_pools()

        # Score initial state
        self.score_energy()
        self._save_progress_snapshot(0)

        for step in range(1, self._num_steps + 1):
            # 1. Prepare candidates from selected
            self._initialize_sequence_pools()

            # 2. Apply generators (mutate candidates)
            for gen in self.generators:
                gen.sample()

            # 3. Score candidates
            self.score_energy()

            # 4. Select top candidates → update selected_sequences
            self._select_top_candidates()

            # 5. Track progress
            self._save_progress_snapshot(step)

            if self._verbose:
                best_score = min(self.energy_scores)
                logger.info(f"Step {step}/{self._num_steps}: best={best_score:.4f}")

    def _select_top_candidates(self) -> None:
        """Select top-scoring candidates into selected_sequences."""
        # Sort by energy score (lower = better)
        scored = list(zip(self.energy_scores, range(len(self.energy_scores))))
        scored.sort(key=lambda x: x[0])

        # Take top num_selected
        for seg in self.segments:
            new_selected = []
            for _, idx in scored[:self.num_selected]:
                new_selected.append(copy.deepcopy(seg.candidate_sequences[idx]))
            seg.selected_sequences = new_selected
```

## Decorator Argument Reference

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `key` | `str` | Yes | Unique kebab-case identifier |
| `label` | `str` | Yes | Human-readable name |
| `config` | `Type[BaseModel]` | Yes | Pydantic config class |
| `description` | `str` | Yes | What this optimizer does |

## Single-Segment Optimizers

If your optimizer only works with one segment (like BeamSearch):

```python
# Add to OPTIMIZERS_WITH_TARGET_SEGMENT in optimizer_registry.py
OPTIMIZERS_WITH_TARGET_SEGMENT = frozenset({"beam-search", "cycling", "my-optimizer"})
```

This enables the `target_segment` field in the API parser for single-segment selection.

## Export Chain

Add to `proto_language/language/optimizer/__init__.py`:

```python
from .my_optimizer import MyOptimizer, MyOptimizerConfig

__all__ = [
    ...
    "MyOptimizer",
    "MyOptimizerConfig",
]
```

## Documentation

Documentation `.mdx` files in `docs/` are auto-generated by `generate_docs.py` (run by pre-commit hooks). Never manually edit `.mdx` files — update the Python config docstrings/field descriptions instead.

## Test Requirements

File: `tests/language_tests/optimizer_tests/test_{name}_optimizer.py`

### Setup Helper Pattern

Every optimizer test file should have a `_setup_components()` helper:

```python
from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.core import Constraint, Construct, Segment
from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig

def _setup_components(
    seq_length=10,
    num_selected=5,
    num_steps=10,
):
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
    gen.assign(segment)
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=40, max_gc=60),
    )
    config = MyOptimizerConfig(num_selected=num_selected, num_steps=num_steps)
    opt = MyOptimizer(
        constructs=[construct],
        generators=[gen],
        constraints=[constraint],
        config=config,
    )
    return opt, gen, constraint, segment
```

### Required Tests

1. **Initialization** — verify config storage and validation
2. **Config validation** — invalid configs raise `ValidationError`
3. **Run completes** — verify `run()` completes without error
4. **Score improves** — verify scores improve over steps (for iterative optimizers)
5. **History tracking** — verify snapshots saved at correct steps
6. **Multi-segment** — verify works with multiple constructs/segments
7. **Filter constraints** — verify filter + scoring constraint interaction

See the testing skill for complete test templates.
