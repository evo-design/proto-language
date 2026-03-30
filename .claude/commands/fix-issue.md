# Fix GitHub Issue $ARGUMENTS

## Step 0: Set Up Worktree

Create an isolated worktree so this fix doesn't block or conflict with other in-progress work:

```bash
git fetch origin main
USER=$(git config user.name | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
git worktree add .claude/worktrees/issue-$ARGUMENTS -B "$USER/fix-issue-$ARGUMENTS" origin/main
cd .claude/worktrees/issue-$ARGUMENTS
git submodule update --init --recursive
```

`-B` (not `-b`) ensures this works even if the branch exists from a previous attempt — it resets it to `origin/main`. The submodule init is required because worktrees don't auto-initialize submodules, and most tests import from `proto_tools`.

Work inside this worktree for all subsequent steps. If the branch name doesn't capture the intent (e.g., it's a feature, not a fix), rename it after reading the issue with `git branch -m $USER/better-name`.

If a worktree already exists for this issue, `cd` into it and `git pull origin main && git submodule update --init --recursive` to stay current.

## Step 1: Read the Issue

```bash
gh issue view $ARGUMENTS
gh issue view $ARGUMENTS --comments
```

Extract from the issue:
- **What's broken or requested** — the core problem or feature
- **Reproduction steps** — if it's a bug
- **Affected components** — which part of the system (language core, API, agent, deployment)
- **Labels/assignees** — for priority and area context

## Step 2: Explore the Codebase

Use sub-agents in parallel to investigate all relevant areas simultaneously:

- **Search for keywords** from the issue (error messages, function names, config keys) across the codebase
- **Read related source files** identified from the issue description or search results
- **Read existing tests** for the affected component to understand expected behavior
- **Check recent commits** touching the affected files: `git log --oneline -20 -- <file>`

Parallelize exploration aggressively — launch multiple sub-agents to search different areas at once rather than searching sequentially.

### Where to Look by Component

| Component | Source | Tests |
|-----------|--------|-------|
| Constraint | `proto_language/language/constraint/{category}/` | `tests/language_tests/constraint_tests/` |
| Generator | `proto_language/language/generator/` | `tests/language_tests/generator_tests/` |
| Optimizer | `proto_language/language/optimizer/` | `tests/language_tests/optimizer_tests/` |
| Program | `proto_language/language/program/` | `tests/language_tests/test_program.py` |
| Core (Segment, Construct, Sequence) | `proto_language/language/core/` | `tests/language_tests/` |
| API | `api/` | `tests/api_tests/` |
| Agent | `agent/` | `tests/agent_tests/` |
| Config system | `proto_language/base_config.py` | `tests/language_tests/` |
| Tool integrations | `proto_tools/` (submodule) | `tests/tool_tests/` |

## Step 3: Present Findings — STOP and wait for user

**Do not write code yet.** Present your interpretation of the problem and proposed approach:

- **My read on the issue**: one paragraph — what's actually broken/needed and why
- **Root cause hypothesis**: what you think is wrong, with evidence from the code you read
- **Proposed approach**: which files you'll touch and what changes you'll make
- **Scope check**: is this a clean fix, or does it touch the public API / export chain / multiple components?
- **Branch name**: confirm or suggest renaming the branch to something more descriptive

Wait for the user to confirm, redirect, or add context before proceeding.

## Step 4: Write a Failing Test

**Always write a test that reproduces the bug before attempting a fix.**

Place the test in the correct location per test conventions:
- `tests/language_tests/constraint_tests/test_{category}/test_{name}_constraint.py`
- `tests/language_tests/generator_tests/test_{name}_generator.py`
- `tests/language_tests/optimizer_tests/test_{name}_optimizer.py`
- `tests/api_tests/test_{area}.py`

```bash
# Verify the test fails as expected
pytest -xvs -k "test_name" tests/path/to/test_file.py
```

For feature requests (not bugs), skip the failing-test step — but still plan the tests you'll write alongside the implementation.

## Step 5: Implement the Fix

Follow the coding conventions:
- `from __future__ import annotations` at top of every file
- `logging.getLogger(__name__)` — never `print()`
- Ruff (line length 88, import sorting)
- Pydantic v2 with `BaseConfig` / `ConfigField` for configs
- Registry keys: kebab-case

Keep the fix minimal and focused. Don't refactor surrounding code unless the issue specifically asks for it.

## Step 6: Verify

Run these checks in parallel using sub-agents where possible:

```bash
# 1. Verify the new test passes
pytest -xvs -k "test_name" tests/path/to/test_file.py

# 2. Run the broader test suite for the affected component
pytest tests/language_tests/constraint_tests/ --cpu    # (or whichever area)

# 3. Run the full fast test suite to check for regressions
pytest --cpu --skip-ci

# 4. Lint
ruff check proto_language api agent tests
```

If any test fails, fix it before proceeding. Don't ask — just fix regressions.

## Step 7: Push & PR

After all checks pass, push the branch and create a PR:

```bash
git push -u origin HEAD
gh pr create --title "Fix #$ARGUMENTS: <concise description>" --body "Closes #$ARGUMENTS

## Summary
<1-3 bullets>

## Test plan
- [ ] New test reproduces the bug and passes with fix
- [ ] Existing test suite passes (`pytest --cpu --skip-ci`)
- [ ] Lint clean (`ruff check`)"
```

Then offer to clean up the worktree:
```bash
REPO_ROOT=$(git worktree list --porcelain | head -1 | sed 's/worktree //')
cd "$REPO_ROOT"
git worktree remove --force .claude/worktrees/issue-$ARGUMENTS
```

`--force` handles the case where uncommitted changes remain in the worktree (e.g., debug files, scratch notes).

## Step 8: Summary

Provide a concise summary:
- **Issue**: one-line restatement of the problem
- **Root cause**: what was wrong
- **Fix**: what changed (files + brief description)
- **Tests**: what tests were added/modified
- **PR**: link to the created PR

## Tips

- For issues that span multiple components, use the todo list to track each piece
- If the issue is ambiguous, read the full comment thread (`gh issue view $ARGUMENTS --comments`) before starting
- If reproduction requires GPU or external services, mark new tests with appropriate markers (`@pytest.mark.uses_gpu`, `@pytest.mark.slow`, `@pytest.mark.skip_ci`)
- When fixing constraint/generator/optimizer bugs, always check the registry export chain — missing exports are a common source of "not found" issues
