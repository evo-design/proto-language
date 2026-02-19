# CLAUDE.md

proto-language: constraint-based optimization framework for designing biological sequences (DNA, RNA, proteins).

## Architecture

1. **Language core** (`proto_language/language/`) — Constraints, Generators, Optimizers, Programs (DSL)
2. **Tools** (`proto-tools/`) — 25+ bioinformatics tool wrappers.
   Git submodule tracking `evo-design/proto-tools` (branch: main).
   Has its own CLAUDE.md, notes/, tests, and CI.
3. **API** (`api/`) — FastAPI + the task queue/a cache for async optimization jobs
4. **Agent** (`agent/`) — AI agent (OpenAI Agents SDK + LiteLLM)
5. **Deployment** (`deployment/`) — cloud GPU cloud functions

All three language components (constraints, generators, optimizers) use a registry pattern:
```python
@constraint(key="gc-content", label="GC Content", config=GCContentConfig, ...)
def gc_content_constraint(input_sequences, config) -> List[float]: ...

# Discovery: ConstraintRegistry.list_all(), .get(key), .get_schema(key)
# Factory:   ConstraintRegistry.create(key, segments, config_dict)
# Same pattern for @generator / GeneratorRegistry and @optimizer / OptimizerRegistry
```

## Environment

- **Conda env**: `proto-language` (Python >=3.12). Assumed active — do NOT create/activate venvs.
- `.env` contains API keys — never expose or commit.

## Commands

```bash
pytest                                # Fast tests (skips slow)
pytest --cpu --skip-ci                # Mimic CI
pytest --gpu --all                    # GPU + slow tests
pytest -k "name"                      # Filter by name
flake8 proto_language api agent tests  # Lint (F401, F841 only)
black proto_language api agent tests   # Format
isort proto_language api agent tests   # Sort imports
pre-commit run --all-files              # All checks
python api/start_dev.py                 # API dev server
python deployment/deploy_cloud_functions.py  # Deploy cloud services
```

## Notes (`notes/`)

Dynamic development notes that evolve as the repo grows. **Read these at the start of relevant tasks. Actively update them** when you discover new gotchas, resolve issues, or learn something future sessions should know — don't ask, just update and mention what you added.

- `dev.md` — Setup, submodule sync, pre-commit hooks, CI checks, docs generation.

The `proto-tools/` submodule has its own `notes/` directory.

## Coding Conventions

- `from __future__ import annotations` at top of every file
- `logging.getLogger(__name__)` — never `print()`
- Black (line length 88), isort (black-compatible profile), flake8 (F401 + F841 only)
- Pydantic v2 for all configs — inherit `BaseConfig`, use `ConfigField` (not `Field`)
- Registry keys: kebab-case. Config classes: `{Name}Config`. Files: `{name}_constraint.py` / `{name}_generator.py`
- **When modifying existing code**: Thoroughly find and update ALL callsites, imports, docstrings, comments, tests, and documentation that reference the changed code. Use sub-agents to search the entire codebase in parallel. Leave no dangling references.

## Test Conventions

- Markers: `@pytest.mark.uses_gpu`, `@pytest.mark.slow`, `@pytest.mark.skip_ci`, `@pytest.mark.integration`
- CPU tests need no marker (auto-applied). External deps (the task queue, a cache, DB) auto-mocked in `tests/conftest.py`.
- Mock generators in conftest.py for testing optimizers/programs without real models.
- Structure: `tests/language_tests/`, `tests/api_tests/`, `tests/agent_tests/`, `tests/tool_tests/`
- **Before running GPU tests**, check if a GPU is available. If no GPU is detected, run CPU tests by default (`pytest --cpu`).

## Sub-Agent Parallelization

Use sub-agents aggressively to parallelize independent work:

- **Codebase exploration**: Launch multiple sub-agents simultaneously to search different areas (e.g., one searching constraints, another searching generators, another reading tests)
- **Test + lint**: Run `pytest` and `flake8` in parallel sub-agents after making changes
- **Multi-file investigation**: When an issue touches several components, fan out sub-agents to read all relevant files at once rather than reading them sequentially
- **Fix verification**: Run the specific failing test, the broader component suite, and lint all in parallel

The general rule: if two tasks don't depend on each other's output, run them in parallel sub-agents.

## Skills (`.claude/skills/`) & Commands (`.claude/commands/`)

### For users (writing programs)

- **write-program** — composing optimization programs in Python (segments, constructs, generators, constraints, optimizers)

### For developers (extending the framework)

Skills (auto-loaded when relevant):

- **general-dev** — coding conventions, config patterns, registry system, data model, export chains
- **implement-constraint** — full constraint implementation lifecycle with templates and examples
- **implement-generator** — full generator implementation lifecycle (ABC contract, categories, templates)
- **implement-optimizer** — full optimizer implementation lifecycle (dual-pool architecture, templates)
- **testing** — comprehensive test patterns, fixtures, markers, templates for each component type

Commands (invoked with `/command-name [args]`):

- **`/fix-issue <number>`** — full GitHub issue fix lifecycle (read issue, explore, reproduce, fix, test, verify)

### In `proto-tools/` submodule

See the submodule's own CLAUDE.md for its skills and commands.
