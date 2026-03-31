# Development Guide

This guide covers the development workflow, including pre-commit hooks and what is tested by CI checks.

## Quick Reference

```bash
# Important commands to know
ruff check proto_language api agent tests # Run by Checks CI to check code style
pytest --cpu --skip-ci                 # Run by Unit Test CI to run CPU-only unit tests (mimics exact CI conditions)
pytest --e2e -v                        # Run by E2E Test CI (starts real a cache + API server)

python deployment/deploy_cloud_functions.py        # Deploy all services to cloud
python deployment/deploy_cloud_functions.py --test # Deploy and run smoke tests (you should do this if you modify cloud service implementations)

python .github/scripts/validate_exports.py          # Validate export chain consistency across both repos
python .github/scripts/validate_exports.py --verbose # Same, with detailed output
python .github/scripts/generate_openapi.py           # Regenerate openapi.json from FastAPI app
python .github/scripts/generate_openapi.py --check   # Verify openapi.json is up to date (run by Checks CI)
```

## Table of Contents
- [Initial Setup](#initial-setup)
- [Keeping the Submodule in Sync](#keeping-the-submodule-in-sync)
- [Git Worktrees](#git-worktrees)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Export Chain Validator](#export-chain-validator)
- [Continuous Integration (CI) Checks](#continuous-integration-ci-checks)

---

## Initial Setup

Follow the setup instructions in the [README](../README.md#setup) to create your conda environment and install dependencies.

---

## Keeping the Submodule in Sync

`proto-tools` is a git submodule that tracks the `main` branch. Keep it in sync when pulling changes:

> [!NOTE] **CI / GitHub Actions**: The proto-tools submodule is a private repo. Workflows require a `CI_SUBMODULE_ACCESS` secret. Create a fine-grained PAT (or classic PAT with `repo` scope) that has access to both this repo and `evo-design/proto-tools`, then add it under **Settings → Secrets and variables → Actions** as `CI_SUBMODULE_ACCESS`.

**When someone else updates the submodule reference:**
```bash
git pull --recurse-submodules
```
Or: `git pull` then `git submodule update --init --recursive`

**To pull the latest from proto-tools and update the parent repo:**
```bash
git submodule update --remote proto-tools
git add proto-tools
git commit -m "Update proto-tools submodule"
```

**Optional – auto-update submodules on pull** (set once per repo):
```bash
git config submodule.recurse true
```

---

## Git Worktrees

Git worktrees allow you to check out multiple branches simultaneously in separate directories. This is useful for working on multiple features or reviewing PRs without stashing changes.

### Common Worktree Commands

```bash
# List all worktrees
git worktree list

# Add worktree with new branch
git worktree add -b new-branch /path/to/worktree

# Add worktree for existing branch
git worktree add /path/to/worktree existing-branch

# Remove a worktree
git worktree remove /path/to/worktree
```

### Worktrees for the Parent Repo

To work on a different branch of `proto-language` while keeping your current branch intact:

```bash
# From the repo root, create a worktree for another branch
git worktree add ../proto-language-feature feature-branch

# Initialize submodules in the new worktree
cd ../proto-language-feature
git submodule update --init --recursive
```

### Worktrees for the Submodule

To work on multiple branches of `proto-tools` simultaneously:

```bash
# Navigate into the submodule
cd proto-tools

# Create a worktree for a feature branch
git worktree add ../proto-tools-feature feature-branch

# Or create a new branch in a worktree
git worktree add -b my-new-feature ../proto-tools-my-feature
```

> [!NOTE] When using worktrees with submodules, changes in a submodule worktree won't automatically update the parent repo's submodule reference. You'll still need to commit the submodule pointer update in the parent repo.

---

## Pre-commit Hooks

Pre-commit hooks run automatically before every commit to ensure code quality.

### Manual Installation

If you need to set up pre-commit hooks manually:
```bash
uv pip install pre-commit
pre-commit install
```

### What the Hooks Do

1. **Linting + import sorting** - Runs `ruff` to lint and sort imports
2. **Basic checks** - Removes trailing whitespace, fixes end-of-file issues, validates YAML, checks for large files
3. **Export chain validation** - Validates `__init__.py` export chains when any `__init__.py` is staged (see [Export Chain Validator](#export-chain-validator))

### Running Hooks Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py

# Run a specific hook
pre-commit run ruff --all-files
```

### Bypassing Hooks (Not Recommended)

```bash
git commit --no-verify
```

**Note:** CI will still catch issues if you bypass hooks.

---

## Export Chain Validator

AST-based tool that validates `__init__.py` export chains across both `proto-language` and `proto-tools`. Catches the #1 silent failure mode: adding a new tool/constraint/generator but missing an `__init__.py` export level, causing `ImportError` at runtime.

### What It Checks

1. **Upward chain completeness** — Every symbol in `__all__` at level N is importable at level N+1
2. **`__all__` consistency** — Every item in `__all__` is actually imported or defined in that module (catches stale entries)
3. **Registry registration** — Every `@tool`/`@constraint`/`@generator`/`@optimizer` decorated function is exported by its parent `__init__.py`

### Running It

```bash
python .github/scripts/validate_exports.py                # All domains
python .github/scripts/validate_exports.py --domain Tools  # Single domain
python .github/scripts/validate_exports.py --verbose       # Show all checks
```

Exit code 0 = pass, 1 = errors found. Errors go to stderr with actionable messages.

### Exceptions

Known intentional omissions (internal base configs, private subpackages, etc.) are listed in the `exceptions` section of `.github/scripts/export_config.json`. Add new exceptions there if a symbol is intentionally not propagated.

### CI Integration

Runs automatically as a pre-commit hook when any `__init__.py` is staged. See [Pre-commit Hooks](#pre-commit-hooks).

---

## Continuous Integration (CI) Checks

### Conditional Automatic CIs
The following CIs run automatically on pull requests that are in `ready_for_review` state:

#### CPU Unit Tests
**File:** `.github/workflows/run-unit-tests.yml`
**Triggers:** On non-draft PRs
**What it does:** Runs fast CPU-only unit tests, skips tests marked with `@pytest.mark.skip_ci`

**Run locally:**
```bash
# Mimic exact CI behavior
pytest --cpu

# Additionally skip tests marked with skip_ci (stricter than CI)
pytest --cpu --skip-ci
```

**Note:** Tests marked with `@pytest.mark.skip_ci` are skipped when `--skip-ci` is passed. CI does NOT use `--skip-ci` — it runs `pytest --cpu` directly. Use `--skip-ci` locally if you want to skip tests that depend on remote APIs or rate-limited services.

**Chimera-only tests:** Tests marked with `@pytest.mark.only_chimera` only run on the Chimera cluster (where `SLURM_CLUSTER_NAME=arc-slurm`). These tests are automatically skipped on other machines.

#### Integration Tests
**File:** `.github/workflows/integration_tests.yml`
**Triggers:** On non-draft PRs
**What it does:** Runs tests requiring external tools (MAFFT, etc.)

**Run locally:**
```bash
pytest --integration --cpu -v
```

#### End-to-End Tests
**File:** `.github/workflows/e2e_tests.yml`
**Triggers:** On non-draft PRs
**What it does:** Starts real a cache + API server and runs end-to-end HTTP tests

**Run locally:**
```bash
pytest --e2e -v
```

### Constant Automatic CIs
This CI always runs automatically on pull requests regardless of state.

#### Checks (Lint, Exports, OpenAPI)
**File:** `.github/workflows/checks.yml`
**Triggers:** On all PR pushes and main branch
**What it does:** Three parallel jobs:
1. **Ruff Lint** — checks code style (F401 + F841 + import sorting)
2. **Validate Export Chains** — verifies `__init__.py` exports are consistent
3. **Verify OpenAPI Spec** — ensures `openapi.json` matches current Pydantic models

**Run locally:**
```bash
ruff check proto_language api agent tests
python .github/scripts/validate_exports.py --verbose
python .github/scripts/generate_openapi.py --check
```

### Manual CIs
The following CIs run manually when requested by the user:

#### Claude Code Review
**File:** `.github/workflows/claude.yml`
**Triggers:** Only when `@claude` is mentioned in a PR comment
**What it does:**
- If comment is just `@claude` or explicitly asks for review → Full code review
- If comment asks a specific question → Answers that question only

**Usage examples:**
```
@claude
# Triggers full code review

@claude please review this PR
# Triggers full code review

@claude why does this function use caching?
# Answers specific question only
```

#### Documentation Generation

Documentation reference pages are auto-generated from Python docstrings and field descriptions. The `docs_autogen.yml` workflow no longer exists in this repo; generation now happens externally from the source code in this repo.

#### Deploy to Staging
**File:** `.github/workflows/release-to-staging.yml`
**Triggers:** Manual dispatch
**What it does:** Deploys the application to the staging environment

#### Release and Deploy to Production
**File:** `.github/workflows/release-to-prod.yml`
**Triggers:** Manual dispatch
**What it does:** Creates a release and deploys the application to production

---

## Documentation Generation

Documentation reference pages are auto-generated from Python docstrings, field descriptions, and tool READMEs. The generation scripts introspect the Python registries in this repo and `proto-tools`.

### Adding Documentation

**For constraints/generators/optimizers:**
1. Add Google-style docstrings to your Python class/function
2. Open and merge your PR in this repo
3. Reference pages regenerate automatically from the source code

**For tools:**
1. Update tool README/source in `proto-tools`
2. Merge that change in the tools repo
3. Reference pages regenerate automatically from the source code

---

## Batching

`batch_size` defaults to `1` everywhere (tools and generators) — safe by default.
The tool layer owns the batching loop; generators/constraints pass all sequences
plus `batch_size` through. See `.claude/skills/general-dev/SKILL.md` "Batching Architecture"
for the full data flow diagram.
