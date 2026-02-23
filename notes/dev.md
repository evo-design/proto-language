# Development Guide

This guide covers the development workflow, including pre-commit hooks and what is tested by CI checks.

## Quick Reference

```bash
# Important commands to know
python docs/generate_docs.py          # Manually regenerate docs locally (CI auto-generates on main)
flake8 proto_language api agent tests # Run by Lint Check CI to check code style
pytest --cpu --skip-ci                 # Run by Unit Test CI to run CPU-only unit tests (mimics exact CI conditions)
python tests/run_integration_tests.py  # Run by Integration Test CI to run integration tests

python deployment/deploy_cloud_functions.py # Deploy all services to cloud and run simple execution tests (you should do this if you modify cloud service implementations)
```

## Table of Contents
- [Initial Setup](#initial-setup)
- [Keeping the Submodule in Sync](#keeping-the-submodule-in-sync)
- [Git Worktrees](#git-worktrees)
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

1. **Import sorting** - Runs `isort` to sort imports
2. **Basic checks** - Removes trailing whitespace, fixes end-of-file issues, validates YAML, checks for large files

### Running Hooks Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py

# Run a specific hook
pre-commit run isort --all-files
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

#### Auto-Generate Unified Documentation (Main only)
**File:** `.github/workflows/docs_autogen.yml`
**Triggers:** Pushes to `main` when language source, docs content/generator, or submodule pointer changes
**What it does:** Regenerates docs, validates links (``), and auto-commits `docs/` updates with the bot

**Run locally (optional):**
```bash
python docs/generate_docs.py

```

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

Documentation is auto-generated from:

1. **Python docstrings** - Constraints, generators, optimizers (parsed in this repo)
2. **Tool docs artifacts** - Synced from `proto-tools/docs/tools/*.mdx` at the pinned submodule commit

### Documentation Structure

```
docs/
├── language/
│   ├── constraints/    # Auto-generated from constraint docstrings
│   ├── generators/     # Auto-generated from generator docstrings
│   └── optimizers/     # Auto-generated from optimizer docstrings
├── tools/              # Synced from proto-tools generated docs
└── docs.json           # Navigation structure (auto-updated)
```

### Adding Documentation

**For constraints/generators/optimizers:**
1. Add Google-style docstrings to your Python class/function
2. Open and merge your PR without committing generated docs
3. After merge to `main`, docs are regenerated automatically by CI

**For tools:**
1. Update tool README/source in `proto-tools`
2. Merge that change in the tools repo (its own main-only autogen updates tool docs artifacts)
3. Bump the `proto-tools` submodule pointer in this repo
4. Merge pointer update; outer main autogen then syncs those artifacts into unified docs

### Manual Documentation Generation

```bash
python docs/generate_docs.py
```

This will:
- Scan all registered constraints, generators, and optimizers
- Parse their docstrings
- Sync tool MDX artifacts from the pinned submodule commit snapshot
- Update `docs.json` navigation

---

## Batching

`batch_size` defaults to `1` everywhere (tools and generators) — safe by default.
The tool layer owns the batching loop; generators/constraints pass all sequences
plus `batch_size` through. See `.claude/skills/general-dev/SKILL.md` "Batching Architecture"
for the full data flow diagram.
