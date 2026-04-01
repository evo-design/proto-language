# CLAUDE.md

proto-language: constraint-based optimization framework for designing biological sequences (DNA, RNA, proteins).

## Architecture

1. **Language core** (`proto_language/language/`): Constraints, Generators, Optimizers, Programs (DSL)
2. **Tools** (`proto-tools/`): 25+ bioinformatics tool wrappers.
   Git submodule tracking `evo-design/proto-tools` (branch: main).
   Has its own CLAUDE.md, notes/, tests, and CI.

All three language components (constraints, generators, optimizers) use a registry pattern:
```python
@constraint(key="gc-content", label="GC Content", config=GCContentConfig, ...)
def gc_content_constraint(input_sequences, config) -> List[float]: ...

# Discovery: ConstraintRegistry.list_all(), .get(key), .get_schema(key)
# Factory:   ConstraintRegistry.create(key, segments, config_dict)
# Same pattern for @generator / GeneratorRegistry and @optimizer / OptimizerRegistry
```

## Environment

- **Conda env**: `proto-language` (Python >=3.10). Assumed active; do NOT create/activate venvs.
- Local dev needs no env vars; sensible defaults are built in.

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace gated models (ESM3, AlphaGenome) | *(unset)* |

## Commands

```bash
pytest                                # Fast unit tests (skips slow, integration)
pytest --integration                  # Include integration tests (require MAFFT etc.)
pytest --all                          # Everything including slow + integration
pytest --cpu --skip-ci                # Mimic CI
pytest --gpu --all                    # GPU + slow + integration tests
pytest -k "name"                      # Filter by name
ruff check proto_language tests       # Lint (F401, F841, import sorting)
pre-commit run --all-files            # All checks
```

## Knowledge Management

Three layers for persistent knowledge. Put information in the right one:

| Layer | Location | Shared? | Best For |
|-------|----------|---------|----------|
| **CLAUDE.md** | Repo root (git) | Team | Conventions, architecture, commands, standards |
| **notes/** | `notes/` (git) | Team | Setup guides, CI procedures, architecture decisions, platform reports |
| **Auto-memory** | `~/.claude/.../memory/` | Personal | Debugging patterns, tool/model quirks, non-obvious discoveries |

### notes/

Team-shared development docs. Read at the start of relevant tasks.

- `dev.md`: Setup, submodule sync, pre-commit hooks, CI checks, docs generation
- `batching.md`: Batching architecture across generator → tool → GPU boundary

Update notes/ when you discover something **every developer needs to know** (CI changes, new setup steps, architecture decisions).

The `proto-tools/` submodule has its own `notes/` directory.

### Auto-memory

Claude's personal memory across sessions. Save to auto-memory when you discover something **non-obvious during a session**:

- Debugging that took multiple attempts → save root cause + fix
- Undocumented tool/model behavior → save the quirk + workaround
- Non-obvious architectural coupling → save the discovery
- Platform-specific issues (GPU memory limits, cluster-specific quirks, etc.)

Do NOT save to auto-memory: anything already in CLAUDE.md or notes/ (avoid duplication), temporary task context, or information other developers need (use notes/ instead).

## Keeping Docs in Sync

When a code change alters behavior documented in this file or any `SKILL.md`, update the docs in the same change. Key mappings:

| Code area | Update in |
|---|---|
| `proto_language/language/constraint/` | CLAUDE.md Architecture, `implement-constraint` SKILL.md |
| `proto_language/language/generator/` | CLAUDE.md Architecture, `implement-generator` SKILL.md |
| `proto_language/language/optimizer/` | CLAUDE.md Architecture, `implement-optimizer` SKILL.md |
| `proto_language/language/core/` | `general-dev` SKILL.md (Data Model, Result Export) |
| `proto_language/base_config.py` | `general-dev` SKILL.md (Config Pattern) |
| `tests/conftest.py`, pytest markers | CLAUDE.md Test Conventions, `testing` SKILL.md |
| New skills or commands added | CLAUDE.md Skills & Commands section |
| Docstring conventions | CLAUDE.md (Docstring Conventions), `tests/test_docstring_consistency.py` |

The `proto-tools/` submodule has its own CLAUDE.md with its own mappings.

## Coding Conventions

- `from __future__ import annotations` at top of every file
- `logging.getLogger(__name__)`, never `print()`
- Ruff (line length 88, checks F401 + F841 + import sorting)
- Pydantic v2 for all configs: inherit `BaseConfig`, use `ConfigField` (not `Field`). Use `depends_on` for conditional field visibility (show/hide fields based on another field's value).
- Registry keys: kebab-case. Config classes: `{Name}Config`. Files: `{name}_constraint.py` / `{name}_generator.py`
- **When modifying existing code**: Thoroughly find and update ALL callsites, imports, docstrings, comments, tests, and documentation that reference the changed code. Use sub-agents to search the entire codebase in parallel. Leave no dangling references.

## Docstring Conventions

Google style everywhere. Enforced by `tests/test_docstring_consistency.py`.

- **Module docstrings**: First line is the relative path from repo root. Blank line, then short description. More content after that is optional. `__init__.py` files are exempt.
  ```python
  """
  proto_language/language/core/constraint.py

  Constraint evaluation and metadata propagation for sequences.
  """
  ```
- **One-liners**: Acceptable for simple functions. No structured sections needed.
- **Multi-line docstrings** (anything with a blank line): Google style. Summary line, blank line, then sections as needed: `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Example:`, `Note:`.
- **Types required in docstrings**: Every `Args:`, `Attributes:`, and `Returns:` entry must include the type annotation matching the function signature or class annotation. Use modern Python syntax (`list[str]`, `X | None`). Consistency tests enforce that docstring types match signatures.
  ```python
  Args:
      sequences (list[str]): Input protein sequences.
      config (GCContentConfig | None): Optional configuration.

  Attributes:
      min_gc (float): Minimum acceptable GC content percentage.

  Returns:
      list[float]: Constraint scores for each sequence.
  ```
- **Pydantic classes**: Always include `Attributes:` section with full descriptions. These intentionally duplicate the short `ConfigField(description=...)` strings; field descriptions are short tooltips for the client UI, while docstring descriptions are longer developer-facing explanations.

## Test Conventions

Three test tiers:

| Tier | Command | What runs | Marker |
|------|---------|-----------|--------|
| **Unit** | `pytest` | Fast, fully mocked, no I/O | *(none needed)* |
| **Integration** | `pytest --integration` | Requires external tools (MAFFT, etc.) | `@pytest.mark.integration` |
Other markers: `@pytest.mark.uses_gpu`, `@pytest.mark.slow`, `@pytest.mark.skip_ci`

- CPU tests need no marker (auto-applied).
- Mock generators in conftest.py for testing optimizers/programs without real models.
- Structure: `tests/language_tests/`
- **Before running GPU tests**, check if a GPU is available. If no GPU is detected, run CPU tests by default (`pytest --cpu`).

## Sub-Agent Parallelization

Use sub-agents aggressively to parallelize independent work:

- **Codebase exploration**: Launch multiple sub-agents simultaneously to search different areas (e.g., one searching constraints, another searching generators, another reading tests)
- **Test + lint**: Run `pytest` and `ruff check` in parallel sub-agents after making changes
- **Multi-file investigation**: When an issue touches several components, fan out sub-agents to read all relevant files at once rather than reading them sequentially
- **Fix verification**: Run the specific failing test, the broader component suite, and lint all in parallel

The general rule: if two tasks don't depend on each other's output, run them in parallel sub-agents.

## Skills (`.claude/skills/`) & Commands (`.claude/commands/`)

### For users (writing programs)

- **write-program**: composing optimization programs in Python (segments, constructs, generators, constraints, optimizers)

### For developers (extending the framework)

Skills (auto-loaded when relevant):

- **general-dev**: coding conventions, config patterns, registry system, data model, export chains
- **implement-constraint**: full constraint implementation lifecycle with templates and examples
- **implement-generator**: full generator implementation lifecycle (ABC contract, categories, templates)
- **implement-optimizer**: full optimizer implementation lifecycle (dual-pool architecture, templates)
- **testing**: comprehensive test patterns, fixtures, markers, templates for each component type

Commands (invoked with `/command-name [args]`):

- **`/fix-issue <number>`**: full GitHub issue fix lifecycle (read issue, explore, reproduce, fix, test, verify)

### In `proto-tools/` submodule

See the submodule's own CLAUDE.md for its skills and commands.
