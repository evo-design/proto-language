---
name: implement-optimizer
description: >
  Implements, modifies, or debugs optimizers in the proto-language DSL.
  Covers the full lifecycle: BaseOptimizerConfig with ConfigField, Optimizer
  subclass with __init__/run, dual-pool architecture (result/proposal sequences),
  constraint evaluation (filter + scoring), decorator registration, export chain,
  and pytest test coverage. Use when working with optimizers, MCMC, beam search,
  TopK selection, cycling, or sequence optimization algorithms.
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

## Optimizer ABC Contract (Summary)

The `Optimizer` ABC requires two abstract methods: `__init__` and `run`.

- **`__init__`**: Takes `constructs`, `generators`, `constraints`, plus config. Calls `super().__init__()` which runs `_validate_optimizer()`.
- **`run`**: Executes the optimization loop. Modifies segments' `result_sequences` and `proposal_sequences`.

**Note**: Subclass `__init__` signatures take `config` as a single parameter and unpack it into the ABC's individual parameters via `super().__init__()`.

## Dual-Pool Architecture

Every optimizer manages two sequence pools per segment:

```
result_sequences    Persistent top-K results across iterations
                      Size: num_results (from config or program-level default)

proposal_sequences   Temporary proposals generated each step
                      Size: num_proposals (computed from config)
```

**Flow per optimization step**:
1. Copy `result_sequences` -> `proposal_sequences` (expanded/contracted as needed)
2. Apply generators to mutate `proposal_sequences`
3. Evaluate constraints on `proposal_sequences`
4. Update `result_sequences` based on scores

## Filter vs Scoring Constraints

```
Filter constraints (threshold set)     Evaluated FIRST, binary pass/fail
    | only passing proposals proceed
Scoring constraints (no threshold)     Evaluated on survivors only
    |
Aggregate score = weighted sum/product of scoring constraint results
```

- Rejected proposals receive `filter_penalty` (default: `inf`) and skip scoring entirely
- Use `constraint.evaluate(mask=...)` to selectively evaluate only certain proposals

## Key Base Class Methods

| Method | Purpose |
|--------|---------|
| `score_energy(operation="add")` | Evaluate ALL constraints; populates `self.energy_scores` |
| `_initialize_sequence_pools()` | Set up `proposal_sequences` from `result_sequences` with cycling |
| `_save_progress_snapshot(step)` | Save current state to `self.history` |
| `_validate_optimizer()` | Comprehensive validation (called in `__init__`) |
| `_prepare_run()` | Reset history, prepare for fresh run |
| `_capture_initial_state()` | Snapshot state before run (for multi-run) |
| `_restore_initial_state()` | Restore to captured state |

## Implementation Steps

For complete config class and optimizer class templates, use the `Read` tool to load:
- **Templates**: `.claude/skills/implement-optimizer/TEMPLATES.md`

Summary of the workflow:
1. **Config class** — inherit `BaseOptimizerConfig`, use `ConfigField` (supports `advanced`, `hidden`, and `depends_on` for conditional visibility), declare `num_steps`/`num_results`
2. **Optimizer class** — `@optimizer` decorator, `@final`, implement `__init__` and `run`
3. **`_update_results`** — implement your selection logic (greedy, MCMC, etc.)

## Decorator Argument Reference

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `key` | `str` | Yes | Unique kebab-case identifier |
| `label` | `str` | Yes | Human-readable name |
| `config` | `Type[BaseModel]` | Yes | Pydantic config class |
| `description` | `str` | Yes | What this optimizer does |

## Single-Segment Optimizers

If your optimizer only works with one segment (like BeamSearch or Cycling):

1. Set `targets_single_segment=True` in the `@optimizer(...)` decorator
2. Add `target_segment: Segment` as the first parameter in `__init__` (before constructs)

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

Documentation reference pages are auto-generated from Python docstrings and field descriptions. To update documentation, update the Python config docstrings/field descriptions in the source code.

## Test Requirements

File: `tests/language_tests/optimizer_tests/test_{name}_optimizer.py`

Required tests:
1. **Initialization** — verify config storage and validation
2. **Config validation** — invalid configs raise `ValidationError`
3. **Run completes** — verify `run()` completes without error
4. **Score improves** — verify scores improve over steps (for iterative optimizers)
5. **History tracking** — verify snapshots saved at correct steps
6. **Multi-segment** — verify works with multiple constructs/segments
7. **Filter constraints** — verify filter + scoring constraint interaction

See the testing skill for complete test templates (including `_setup_components` helper pattern).

## Validation Checklist

Copy this and check off as you go:

- [ ] Config class inherits `BaseOptimizerConfig` with `ConfigField` (use `depends_on` for conditionally visible fields)
- [ ] `@optimizer` decorator with unique kebab-case key
- [ ] `@final` decorator on class
- [ ] `__init__` calls `super().__init__()` with unpacked config
- [ ] `run()` calls `_prepare_run()`, `_initialize_sequence_pools()`, `score_energy()`, `_save_progress_snapshot()`
- [ ] `_update_results()` implements correct selection logic
- [ ] Export chain updated: `optimizer/__init__.py`
- [ ] Tests cover: init, config validation, run, score improvement, history, filter constraints
- [ ] Tests pass: `pytest tests/language_tests/optimizer_tests/ --cpu -x`
- [ ] Lint passes: `ruff check proto_language/language/optimizer/`

If any check fails, fix before proceeding.
