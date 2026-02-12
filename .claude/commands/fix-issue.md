# Fix GitHub Issue $ARGUMENTS

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

## Step 3: Write a Failing Test

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

## Step 4: Implement the Fix

Follow the coding conventions:
- `from __future__ import annotations` at top of every file
- `logging.getLogger(__name__)` — never `print()`
- Black (line length 88), isort (black-compatible profile)
- Pydantic v2 with `BaseConfig` / `ConfigField` for configs
- Registry keys: kebab-case

Keep the fix minimal and focused. Don't refactor surrounding code unless the issue specifically asks for it.

## Step 5: Verify

Run these checks in parallel using sub-agents where possible:

```bash
# 1. Verify the new test passes
pytest -xvs -k "test_name" tests/path/to/test_file.py

# 2. Run the broader test suite for the affected component
pytest tests/language_tests/constraint_tests/ --cpu    # (or whichever area)

# 3. Run the full fast test suite to check for regressions
pytest --cpu --skip-ci

# 4. Lint
flake8 proto_language api agent tests
```

If any test fails, fix it before proceeding. Don't ask — just fix regressions.

## Step 6: Summary

After all checks pass, provide a concise summary:
- **Issue**: one-line restatement of the problem
- **Root cause**: what was wrong
- **Fix**: what changed (files + brief description)
- **Tests**: what tests were added/modified
- **Verification**: confirmation that all tests and lint pass

## Tips

- For issues that span multiple components, use the todo list to track each piece
- If the issue is ambiguous, read the full comment thread (`gh issue view $ARGUMENTS --comments`) before starting
- If reproduction requires GPU or external services, mark new tests with appropriate markers (`@pytest.mark.uses_gpu`, `@pytest.mark.slow`, `@pytest.mark.skip_ci`)
- When fixing constraint/generator/optimizer bugs, always check the registry export chain — missing exports are a common source of "not found" issues
