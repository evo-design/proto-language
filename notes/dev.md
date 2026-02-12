# Development Guide

This guide covers the development workflow, including pre-commit hooks and what is tested by CI checks.

## Quick Reference

```bash
# Important commands to know
python docs/generate_docs.py          # Command used to auto-generate docs (this is done automatically by pre-commit hooks)
flake8 proto_language api agent tests # Run by Lint Check CI to check code style
pytest --cpu --skip-ci                 # Run by Unit Test CI to run CPU-only unit tests (mimics exact CI conditions)
python tests/run_integration_tests.py  # Run by Integration Test CI to run integration tests

python deployment/deploy_cloud_functions.py # Deploy all services to cloud and run simple execution tests (you should do this if you modify cloud service implementations)
```

## Table of Contents
- [Initial Setup](#initial-setup)
- [Keeping the Submodule in Sync](#keeping-the-submodule-in-sync)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Continuous Integration (CI) Checks](#continuous-integration-ci-checks)

---

## Initial Setup

If you haven't already, run the installation script which will set up your environment and install pre-commit hooks automatically:

> [!NOTE] As of 01/14/2026, the installation script is failing on Chimera due to dependency issues.

```bash
bash install.sh
conda activate proto-language
```

---

## Keeping the Submodule in Sync

`proto-tools` is a git submodule that tracks the `main` branch. Keep it in sync when pulling changes:

> [!NOTE] **CI / GitHub Actions**: The bio-tools submodule is a private repo. Workflows require a `CI_SUBMODULE_ACCESS` secret. Create a fine-grained PAT (or classic PAT with `repo` scope) that has access to both this repo and `evo-design/proto-tools`, then add it under **Settings → Secrets and variables → Actions** as `CI_SUBMODULE_ACCESS`.

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

## Pre-commit Hooks

Pre-commit hooks run automatically before every commit to ensure code quality and keep documentation in sync.

### Manual Installation

If you need to set up pre-commit hooks manually:
```bash
uv pip install pre-commit
pre-commit install
```

### What the Hooks Do

1. **Auto-generate documentation** - Extracts docstrings from Python files and converts tool READMEs to MDX format
2. **Code formatting** - Runs `black` and `isort` to format Python code
3. **Basic checks** - Removes trailing whitespace, fixes end-of-file issues, validates YAML, checks for large files

### Running Hooks Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py

# Run a specific hook
pre-commit run generate-docs --all-files
pre-commit run black --all-files
```

### Bypassing Hooks (Not Recommended)

```bash
git commit --no-verify
```

**Note:** CI will still catch issues if you bypass hooks.

---

## Continuous Integration (CI) Checks

### Conditional Automatic CIs
The following CIs run automatically on pull requests that are in `ready_for_review` state:

#### Auto-Generate Documentation
**File:** `.github/workflows/docs_check.yml`
**Triggers:** When doc-related files change (docstrings, READMEs, generate_docs.py)
**What it does:** Verifies that generated docs are in sync with source files

This should be covered automatically by the pre-commit hooks, but you can also manually run and commit the files via:

```bash
python docs/generate_docs.py
git add docs/
git commit -m "docs: Auto-generate documentation"
```

#### CPU Unit Tests
**File:** `.github/workflows/run-unit-tests.yml`
**Triggers:** On non-draft PRs
**What it does:** Runs fast CPU-only unit tests, skips tests marked with `@pytest.mark.skip_ci`

**Run locally:**
```bash
# Run all CPU tests (including skip_ci tests)
pytest --cpu

# Mimic exact CI behavior (skip tests marked with skip_ci)
pytest --cpu --skip-ci
```

**Note:** Tests marked with `@pytest.mark.skip_ci` are automatically skipped in CI (e.g., remote API tests that may hit rate limits). Use `--skip-ci` locally to test exactly what CI will run.

#### Integration Tests
**File:** `.github/workflows/integration_tests.yml`
**Triggers:** On non-draft PRs
**What it does:** Runs comprehensive integration tests with external dependencies

**Run locally:**
```bash
python tests/run_integration_tests.py --verbose
```

### Constant Automatic CIs
This CI always runs automatically on pull requests regardless of state.

#### Lint Check
**File:** `.github/workflows/flake8_check.yml`
**Triggers:** On all PR pushes and main branch
**What it does:** Checks code style with flake8

**Run locally:**
```bash
flake8 proto_language api agent tests
```

### Manual CIs
The following CIs run manually when requested by the user:

#### Claude Code Review
**File:** `.github/workflows/claude-code-review.yml`
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

#### the docs site Documentation Validation
**File:** `.github/workflows/docs.yml`
**Triggers:** On pushes to main branch that modify docs
**What it does:** Validates documentation and checks for broken links

**Run locally:**
```bash
cd docs


```

---

## Documentation Generation

Documentation is auto-generated from:

1. **Python docstrings** - Constraints, generators, optimizers
2. **Tool READMEs** - Individual tool documentation in `proto-tools/proto_tools/tools/*/README.md`

### Documentation Structure

```
docs/
├── language/
│   ├── constraints/    # Auto-generated from constraint docstrings
│   ├── generators/     # Auto-generated from generator docstrings
│   └── optimizers/     # Auto-generated from optimizer docstrings
├── tools/              # Auto-generated from tool READMEs
└── docs.json           # Navigation structure (auto-updated)
```

### Adding Documentation

**For constraints/generators/optimizers:**
1. Add Google-style docstrings to your Python class/function
2. Commit the code (docs will auto-generate via pre-commit)

**For tools:**
1. Create a `README.md` in your tool directory: `proto-tools/proto_tools/tools/category/tool_name/README.md`
2. Follow the existing README structure
3. Commit (docs will auto-generate via pre-commit)

### Manual Documentation Generation

```bash
python docs/generate_docs.py
```

This will:
- Scan all registered constraints, generators, and optimizers
- Parse their docstrings
- Convert tool READMEs to MDX format
- Update `docs.json` navigation
