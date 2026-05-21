# CLAUDE.md

proto-language: constraint-based optimization framework for designing biological sequences (DNA, RNA, proteins).

## Architecture

- `proto_language/`: Constraints, Generators, Optimizers, Programs (DSL). All three components register via a `@constraint` / `@generator` / `@optimizer` decorator and are discovered/instantiated through `ConstraintRegistry` / `GeneratorRegistry` / `OptimizerRegistry`. See the `implement-{constraint,generator,optimizer}` skills for full implementation lifecycles.
- `proto-tools/`: bioinformatics tool wrappers. Git submodule (`evo-design/proto-tools`, branch `main`). Has its own `CLAUDE.md`, `notes/`, tests, and CI.

### Data Model (`proto_language/core/`)

`Sequence` (str + `sequence_type` + optional `logits` + optional `structure` + metadata bags) → `Segment` (groups `proposal_sequences` / `result_sequences` for one design region) → `Construct` (joins segments into a complete design, `list[Segment]`).

Metadata bags on `Sequence` are **namespaced** — don't conflate them:
- `._constraints_metadata[<constraint_label>]["data"]` — written by the constraint framework from `ConstraintOutput.metadata`.
- `._generator_metadata[<generator_key>]` — written by `Generator.sample()`.
- `._metadata` — free-form user bag for ad-hoc keys. Framework code never writes here; prefixed keys would collide across components.

### Result Export

Both `Program` and `Optimizer` expose three export methods: `.export(path=..., format="csv"|"xlsx")` writes a folder with 4 tables + `sequences.fasta` + `assets/` sidecars; `.to_dataframe(table=...)` returns a single pandas DataFrame; `.to_fasta(path=...)` returns FASTA (string or file). Tables: `sequences`, `constraints`, `constructs`, `optimization`. `Program.export()` also accepts `stage=N` to filter by optimizer stage. Underlying utilities: `proto_language/utils/io.py`.

## Environment

- **Conda env**: `proto-language`. Assumed active; do NOT create/activate venvs. Python version, ruff/mypy/pytest pins live in `pyproject.toml`.

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace gated models (ESM3, AlphaGenome) | *(unset)* |

## Commands

```bash
pytest                                # Fast unit tests (skips slow, integration)
pytest --integration                  # Include integration tests (require MAFFT etc.)
pytest --all                          # Everything: unit + slow + integration
pytest --cpu --skip-ci                # Mimic CI
pytest --gpu --all                    # GPU + slow + integration
pytest -k "name"                      # Filter by name
ruff check proto_language tests       # Lint
ruff format proto_language tests      # Format (enforced in CI)
mypy proto_language/                  # Type check (strict)
```

## Coding Conventions

Things that differ from Python/Pydantic defaults — Claude won't infer these from the code:

- **Logging**: `logging.getLogger(__name__)`, never `print()`.
- **Pydantic**: inherit `BaseConfig`, use `ConfigField` (not Pydantic's `Field`).
- **Names**: registry keys are kebab-case (`"gc-content"`, `"mcmc"`); config classes `{Name}Config`; constraint function `{name}_constraint`; generator class `{Name}Generator`; optimizer class `{Name}Optimizer`; files `{name}_{component}.py`; tests `test_{name}.py`.
- **Mypy strict**: every `# type: ignore` includes an error code. Prefer `assert` guards over `# type: ignore`. Do NOT use `cast()`, ad-hoc `Protocol`, or `TYPE_CHECKING` to dodge type errors.

## Error Handling

`Constraint.evaluate(...)`, `Generator.sample(...)`, `Optimizer.run()`, and helpers **raise by default**. The only soft-fail allowed is inside a `for proposal in batch:` loop, where a per-proposal failure becomes a `MAX_ENERGY` score + `metadata["<key>_error"]` and the batch continues. Canonical: `gap_gini_constraint.py`. Full design and the `format_pydantic_error()` reformat at `Registry.create()`: `notes/error-handling.md`.

## Docstring Conventions

Google style. Types required in `Args:`, `Attributes:`, `Returns:` and must match the signature. Pydantic classes always include an `Attributes:` section. Enforced by `tests/test_docstring_consistency.py` — that test is the canonical rule; consult it for edge cases.

## Test Conventions

| Tier | Command | Marker |
|------|---------|--------|
| **Unit** | `pytest` | *(none)* — auto-applied |
| **Integration** | `pytest --integration` | `@pytest.mark.integration` |
| **Slow** | `pytest --slow` | `@pytest.mark.slow` |
| **GPU** | `pytest --gpu` | `@pytest.mark.uses_gpu` |
| **CI-skip** | `pytest --skip-ci` | `@pytest.mark.skip_ci` |

Tests live under `tests/language_tests/{constraint,generator,optimizer}_tests/`. Mock generators in `tests/conftest.py` substitute for real models. Branch-coverage floor is set in `pyproject.toml` (`fail_under`); run `pytest --cov` to check.

Full reference (test placement, naming, mock fixtures, per-component templates, bug-fixing workflow): `notes/testing.md`.

## Knowledge Management

Team-shared knowledge: this `CLAUDE.md` for conventions; `notes/` for setup, architecture, and long-form rules; auto-memory for personal cross-session discoveries.

Index of `notes/`:
- `dev.md` — setup, submodule sync, CI checks, docs generation.
- `batching.md` — batching architecture across generator → tool → GPU boundary.
- `error-handling.md` — raise vs soft-fail rules, `format_pydantic_error()`.
- `filesystem.md` — where source, tests, examples, logs, and persistent caches live.
- `testing.md` — markers, placement, templates per component type, conftest fixtures, mock scoring functions.
- `claude-code.md` — skills, commands, CI integration, common workflows for the Claude Code layer.

## Keeping Docs in Sync

When code changes alter documented behavior, update the docs in the same commit:

| Code area | Update |
|---|---|
| `proto_language/constraint/` | `implement-constraint` SKILL.md |
| `proto_language/generator/` | `implement-generator` SKILL.md |
| `proto_language/optimizer/` | `implement-optimizer` SKILL.md |
| `proto_language/core/`, `proto_language/utils/base.py` | docstrings (code is the canonical surface for these data models and config patterns) |
| `tests/conftest.py`, pytest markers | this file + `notes/testing.md` |
| Error-handling rules | `notes/error-handling.md` |
| Docstring conventions | this file + `tests/test_docstring_consistency.py` |

The `proto-tools/` submodule has its own `CLAUDE.md` with its own mappings.

## Skills (`.claude/skills/`)

For users (writing programs):
- **write-program** — composing optimization programs (segments, constructs, generators, constraints, optimizers).

For developers (extending the framework):
- **implement-constraint** — full constraint implementation lifecycle.
- **implement-generator** — full generator implementation lifecycle (ABC contract, categories, templates).
- **implement-optimizer** — full optimizer implementation lifecycle (dual-pool architecture, templates).

General coding conventions live in this file; long-form testing reference (templates, fixtures, mocks) in `notes/testing.md`.

The `proto-tools/` submodule has `implement-tool` and `fix-env`. See its CLAUDE.md.
