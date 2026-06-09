# Development Guide

Dev workflow for `proto-language` contributors: commands, initial setup, submodule sync, worktrees, the export-chain validator, and the CI workflows that gate PRs. For testing specifics see `notes/testing.md`; for batching see `notes/batching.md`.

## Quick Reference

```bash
ruff check proto_language tests        # lint (mirrors checks.yml lint job)
ruff format --check                    # formatting (mirrors checks.yml lint job)
mypy proto_language/                   # types (mirrors checks.yml mypy job)
pytest --cpu-only                      # CPU unit tests (mirrors unit-tests.yml)
pytest --cpu-only --skip-ci            # additionally skip skip_ci tests, hide CUDA
pytest --integration --cpu-only -v     # external-tool tests (mirrors integration-tests.yml)
python .github/scripts/validate_exports.py --verbose  # export-chain consistency
```

## Initial Setup

Follow the [README](../README.md#installation) for the conda environment and the editable install of `proto-language` plus the `proto-tools` submodule. The README is the source of truth; don't duplicate setup steps here.

## Submodule Sync

`proto-tools` is a git submodule tracking `main` (`.gitmodules` sets `branch = main`, `ignore = dirty`).

Pull both repos together when the submodule reference changes upstream:

```bash
git pull --recurse-submodules
# or:
git pull
git submodule update --init --recursive
```

Bump `proto-tools` to the latest published `main` and commit the new pointer in the parent repo:

```bash
git submodule update --remote proto-tools
git add proto-tools
git commit -m "Bump proto-tools submodule"
```

Set once per clone to auto-recurse on every `git pull`:

```bash
git config submodule.recurse true
```

CI workflows that touch `proto-tools` need a `CI_SUBMODULE_ACCESS` PAT (fine- grained or `repo`-scoped classic) granting access to both `evo-design/proto- language` and `evo-design/proto-tools`, stored under `Settings → Secrets and variables → Actions`.

## Git Worktrees

Worktrees check out multiple branches into separate directories without stashing the current branch.

```bash
git worktree list
git worktree add /path/to/wt existing-branch
git worktree add -b new-branch /path/to/wt
git worktree remove /path/to/wt
```

Parent repo — submodules are not auto-initialized in a fresh worktree:

```bash
git worktree add ../proto-language-feature feature-branch
cd ../proto-language-feature
git submodule update --init --recursive
```

Submodule (independent of any parent worktree):

```bash
cd proto-tools
git worktree add ../proto-tools-feature feature-branch
git worktree add -b my-feature ../proto-tools-my-feature
```

Edits inside a submodule worktree do not auto-update the parent's submodule pointer; commit the pointer update in the parent separately.

## Logging and the tool spinner

`proto_language/utils/logging_config.py` centralizes logging: `setup_logging()`
installs a console handler (stdout, print-like via `SelectiveLevelFormatter`) and
an optional timestamped file handler under `logs/`. Use `get_logger(__name__)` in
framework code; never `print()`.

The console handler is **bar-aware**. proto-tools tools render an animated
spinner/progress bar while they run (structure predictors, pLMs, etc.); a plain
`StreamHandler` would print log lines straight onto that line and corrupt the
frame. `_BarAwareStreamHandler` routes through `tqdm.write` whenever a tool bar is
active (`proto_tools.utils.progress.has_active_progress_bar`) and otherwise behaves
exactly like a normal handler, so there is no change outside an active spinner.

The spinner **renderer is reused from proto-tools**, not reimplemented:
`proto_language/utils/spinner.py` re-exports `progress_bar`, `set_substatus`, and
`has_active_progress_bar` from `proto_tools.utils.progress` (the module is named
`spinner` to avoid colliding with the `progress` scheduling helper;
`status_indicator` is left out because proto-tools treats it as an internal
fallback). The language layer currently **does not drive the spinner** — the tools
own the status channel, so we don't add `set_substatus`/`update_status` calls in
optimizer or `Program` loops. The re-exports exist for future use; for now the only
integration is keeping our logging a good citizen around the tools' spinner.

## Export Chain Validator

`.github/scripts/validate_exports.py` runs AST-based checks on `__init__.py` files in both repos to catch missing or stale exports before they surface as runtime `ImportError`s. The script is stdlib-only and never executes the target modules.

```bash
python .github/scripts/validate_exports.py                # all domains
python .github/scripts/validate_exports.py --domain Tools # single domain
python .github/scripts/validate_exports.py --verbose      # show every check
```

Exit code is `0` on pass, `1` on errors. The script's module docstring is the canonical list of which checks run per domain. Intentional omissions live in the `exceptions` section of `.github/scripts/export_config.json`; add new exceptions there rather than mutating `__all__` to satisfy the check.

## CI Workflows

| Workflow | Trigger | What it runs |
|---|---|---|
| `unit-tests.yml` | non-draft PR + manual `workflow_dispatch` | `pytest --cpu-only -q --override-ini="log_cli=false" --cov --cov-report=term-missing` |
| `checks.yml` | non-draft PR | Three parallel jobs: `ruff check` + `ruff format --check`; `mypy proto_language/`; `python .github/scripts/validate_exports.py --verbose` |
| `submodule-check.yml` | non-draft PR | Fails when the pinned `proto-tools` SHA differs from `evo-design/proto-tools@main`; fix with `git submodule update --remote proto-tools` and commit |
| `integration-tests.yml` | scheduled (daily 06:00 UTC) + `workflow_dispatch` | Install MAFFT; `pytest --integration --cpu-only -v`. **Not** PR-triggered |
| `claude.yml` | `@claude` in issue/PR/review comment | Code review or scoped question response |

See `notes/testing.md` for the test-side details (markers, fixtures, mocks) that these workflows exercise.

## Documentation

Reference docs are generated externally from this repo's registries, docstrings, and field descriptions plus `proto-tools` tool READMEs. Update source inputs — not generated pages — and merge; downstream regeneration picks up the change. There is no `docs_autogen.yml` workflow in this repo.

### Docstring standard

Google convention (ruff `D`, `convention = "google"`). `tests/test_docstring_consistency.py` checks that class/function `Args`/`Attributes`/`Returns` types match signatures, and holds `proto_language/core/` to a stricter standard: every `core/*.py` component module has a detailed header with an `Examples:` section, and every public behavioral core class has an `Examples:` section. Pydantic models (`BaseModel`) and enums are exempt — they document shape via `Attributes:`/values — as is the package `__init__` aggregator.

Module-header template:

```python
"""<one-line summary ending with a period>.

<2-5 sentences: what the module provides and its role in the data model /
optimization loop.>

Examples:
    >>> from proto_language.core import Thing
    >>> thing = Thing(...)
    >>> thing.attr  # expected value
"""
```

Examples are illustrative, not executed (no `--doctest-modules`): use `>>> expr  # result` inline comments, never a separate expected-output line. ruff's `docstring-code-format` reformats the `>>>` blocks, so keep them valid, canonically-formatted Python. Module headers serve source readers; the generated reference docs are driven by class/function and `ConfigField` docstrings via `proto_language/utils/docs_api.py`, so user-facing classes need their own `Examples:`. The test gates `core/` only — apply the same pattern to new modules and components elsewhere.
