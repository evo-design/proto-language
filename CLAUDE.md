## Project Overview

`proto-language` is the Proto constraint-based optimization framework for
designing biological sequences. Core abstractions are `Sequence`, `Segment`,
`Construct`, `Generator`, `Constraint`, `Optimizer`, and `Program`. The
`proto-tools/` submodule supplies bioinformatics tool wrappers and has its own
repo instructions, notes, tests, and CI.

## References

- `README.md`: user-facing overview, setup, core model, and examples.
- `notes/README.md`: index of the developer/agent/advanced-user notes (start here).
  - `notes/dev.md`: dev workflow — setup, submodule sync, worktrees, export validation, CI.
  - `notes/batching.md`: batching across pools, generators, constraints, scorers, backends.
  - `notes/error-handling.md`: raise vs. worst-score contracts in evaluate/sample/run.
  - `notes/filesystem.md`: where files live and where runtime artifacts are written.
  - `notes/testing.md`: markers, fixtures, placement, mocks, and component coverage.
  - `notes/biological-design-loop.md`, `notes/planning-quick-reference.md`,
    `notes/component-planning-example.md`: design-planning guidance and a worked example.
- `examples/scripts/` and `examples/jsons/`: example starter programs.

## Development Setup

Use the `proto-language` conda environment setup from `README.md`. 
Project configurations live in `pyproject.toml`.

See `notes/dev.md` for setup, submodules, export validation, and CI. See
`notes/testing.md` for markers, fixtures, placement, and mocks.

## Repository Map

- `proto_language/core/`: data model, ABCs, program orchestration, export, and
  validation.
- `proto_language/constraint/`: constraint implementations grouped by domain and the constraint registry.
- `proto_language/generator/`: sequence proposal generators and registry.
- `proto_language/optimizer/`: optimization algorithms and constraint compiler
  providers.
- `proto_language/utils/`: shared config, serialization, scoring, IO, logging,
  gradients, and scheduling helpers.
- `tests/language_tests/`: core, constraint, generator, and optimizer behavior
  tests.
- `tests/utils_tests/` and `tests/tests_cpu/`: utility tests and CPU integration
  regressions.
- `notes/`: canonical long-form developer references.

## Contributor Conventions

- Registries use decorators: `@constraint`, `@generator`, and `@optimizer`.
- Use `logging.getLogger(__name__)`, never `print()`, in framework code.
- Config classes inherit the local `BaseConfig` and use `ConfigField`.
- Registry keys are kebab-case. Follow neighboring file, class, function,
  config, and test naming patterns.
- Google-style docstrings, enforced by `tests/test_docstring_consistency.py`:
  class/function `Args`/`Attributes`/`Returns` types match signatures; Pydantic
  classes carry an `Attributes:` section.
- `proto_language/core/` is held to a documentation standard (gated by the same
  test): every component module has a header (summary → blank line → role
  overview → `Examples:` with a `>>> expr  # result` snippet) and every public
  behavioral class has an `Examples:` section; Pydantic models and enums use
  `Attributes:`/values instead, and the package `__init__` aggregator stays a
  plain summary. Apply the same pattern to new modules and components elsewhere.
  See `notes/dev.md`.
- Mypy is strict. Every `# type: ignore` needs an error code; prefer runtime
  `assert` narrowing over `cast()`, ad-hoc `Protocol`, or `TYPE_CHECKING`.
- Framework helpers raise by default. Per-proposal failures inside a scoring
  batch may soft-fail only when the local contract allows it. See
  `notes/error-handling.md`.
- Program-level seeds own run determinism and derive downstream seeds.
- Multi-stage programs reuse the same construct objects by identity across
  optimizers.

## Documentation

Generated docs come from source code, docstrings, field descriptions,
registries, README inputs, and examples. Update those source inputs rather
than generated docs.

When behavior changes, update the relevant `notes/` file, source docstrings,
field descriptions, examples, tests, and skill guidance in the same commit.

## Skills

Skill files live in `.claude/skills/` (`.agents/skills/` symlink). Read the relevant `SKILL.md` before using a workflow.

- `write-program`: composing optimization programs.
- `implement-constraint`, `implement-generator`, `implement-optimizer`:
  implementing or modifying framework components.

The `proto-tools/` submodule has `implement-tool` and `fix-env`; read its repo
instructions before editing that submodule.
