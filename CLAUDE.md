# CLAUDE.md

Short entrypoint for coding agents contributing to `proto-language`. Keep
long-form guidance in `notes/`, source docstrings, tests, examples, and local
skills so instructions do not drift.

`SYSTEM_PROMPT.md` is for agents that use the existing framework to write
programs and scripts. Use this file when editing the repo itself.

## Project Overview

`proto-language` is the Proto Bio constraint-based optimization framework for
designing biological sequences. Core abstractions are `Sequence`, `Segment`,
`Construct`, `Generator`, `Constraint`, `Optimizer`, and `Program`. The
`proto-tools/` submodule supplies bioinformatics tool wrappers and has its own
repo instructions, notes, tests, and CI.

## Read Before Editing

- `README.md`: user-facing overview, setup, core model, and examples.
- `notes/dev.md`, `notes/testing.md`, `notes/batching.md`,
  `notes/error-handling.md`, `notes/filesystem.md`, and
  `notes/claude-code.md`: team-shared references for setup, CI, tests,
  runtime behavior, file layout, and agent workflows.
- `.claude/skills/`: implementation workflows for constraints, generators,
  optimizers, and program writing.
- `examples/scripts/`, `examples/jsons/`, and `examples/data/`: idiomatic
  programs and realistic inputs.
- Source and tests: the final authority for signatures, registry keys,
  schemas, exports, and behavior.

Use auto-memory only for personal, non-obvious debugging discoveries. Put
team-shared setup steps, architecture decisions, and gotchas in `notes/`.

## Development Setup

Use the `proto-language` conda environment. Assume it is already active; do
not create or activate a virtual environment. Python, ruff, mypy, and pytest
configuration live in `pyproject.toml`.

```bash
pytest
pytest --integration
pytest --all
pytest --cpu --skip-ci
pytest --gpu --all
ruff check proto_language tests
ruff format proto_language tests
mypy proto_language/
python .github/scripts/validate_exports.py --verbose
```

See `notes/dev.md` for setup, submodules, export validation, and CI. See
`notes/testing.md` for markers, fixtures, placement, and mocks.

## Repository Map

- `proto_language/core/`: data model, ABCs, program orchestration, export, and
  validation.
- `proto_language/constraint/`, `proto_language/generator/`,
  `proto_language/optimizer/`: registered components and configs.
- `proto_language/language/`: public compatibility/export layer.
- `proto_language/utils/`: shared config, serialization, IO, registry, and
  validation helpers.
- `tests/language_tests/`: behavior tests for constraints, generators,
  optimizers, and programs.
- `notes/`: canonical long-form developer references.

## Contributor Conventions

- Registries use decorators: `@constraint`, `@generator`, and `@optimizer`.
- Use `logging.getLogger(__name__)`, never `print()`, in framework code.
- Config classes inherit the local `BaseConfig` and use `ConfigField`.
- Registry keys are kebab-case. Follow neighboring file, class, function,
  config, and test naming patterns.
- Google-style docstrings are enforced by
  `tests/test_docstring_consistency.py`; Pydantic classes include an
  `Attributes:` section.
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
field descriptions, examples, tests, and `.claude/skills/` guidance in the
same commit.

## Skills

- `write-program`: composing optimization programs.
- `implement-constraint`, `implement-generator`, `implement-optimizer`:
  implementing or modifying framework components.

The `proto-tools/` submodule has `implement-tool` and `fix-env`; read its repo
instructions before editing that submodule.
