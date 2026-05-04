# CLAUDE.md

proto-language: constraint-based optimization framework for designing biological sequences (DNA, RNA, proteins).

## Architecture

1. **Language core** (`proto_language/language/`): Constraints, Generators, Optimizers, Programs (DSL)
2. **Tools** (`proto-tools/`): Bioinformatics tool wrappers.
   Git submodule tracking `evo-design/proto-tools` (branch: main).
   Has its own CLAUDE.md, notes/, tests, and CI.

All three language components (constraints, generators, optimizers) use a registry pattern:
```python
@constraint(key="gc-content", label="GC Content", config=GCContentConfig, ...)
def gc_content_constraint(input_sequences, config) -> list[ConstraintOutput]: ...

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
ruff check proto_language tests       # Lint
ruff format proto_language tests      # Format (enforced in CI)
mypy proto_language/                  # Type check (strict)
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

- `dev.md`: Setup, submodule sync, CI checks, docs generation
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
| New skills added | CLAUDE.md Skills section |
| Docstring conventions | CLAUDE.md (Docstring Conventions), `tests/test_docstring_consistency.py` |

The `proto-tools/` submodule has its own CLAUDE.md with its own mappings.

## Coding Conventions

- `logging.getLogger(__name__)`, never `print()`
- Ruff (line length 120, Google-convention pydocstyle — see `pyproject.toml [tool.ruff.lint]` for full config)
- Mypy strict mode with Pydantic plugin — all code must pass `mypy proto_language/` with zero errors. Every `# type: ignore` must include the error code (e.g. `# type: ignore[arg-type]`). Use only for genuinely unfixable external-lib issues. Prefer `assert` guards for type narrowing over `# type: ignore`. Do NOT use `cast()`, arbitrary `Protocol` definitions, or `TYPE_CHECKING` blocks to work around type issues.
- Pydantic v2 for all configs: inherit `BaseConfig`, use `ConfigField` (not `Field`). Use `depends_on` for conditional field visibility (show/hide fields based on another field's value).
- Registry keys: kebab-case. Config classes: `{Name}Config`. Files: `{name}_constraint.py` / `{name}_generator.py`
- **When modifying existing code**: Thoroughly find and update ALL callsites, imports, docstrings, comments, tests, and documentation that reference the changed code. Use sub-agents to search the entire codebase in parallel. Leave no dangling references.

## Error Handling Policy

**Default: raise.** Inside `constraint.evaluate(...)`, `generator.sample(...)`, `Optimizer.run()`, and any helper they call, raise on failure. The earlier "soft-fail to preserve compute" approach was wrong for the user: when a tool call crashes (CUDA OOM, missing binary, target prep returned None, reference folding returned empty), the failure is almost always **deterministic** for the current config — soft-failing produces 100 iterations of all-`MAX_ENERGY` garbage that looks like a real result. Raising surfaces the actual error immediately so the user fixes the config and reruns; lost iterations of progress are cheap to redo, ambiguous results are not.

**The one exception — per-proposal failure inside a `for proposal in batch:` loop.** If MAFFT can align 31 of 32 sequence pairs but fails on the 32nd, the other 31 are useful. The bad item should soft-fail without killing the batch:

```python
for proposal_pair in input_sequences:
    try:
        score = run_mafft_align(...)
    except Exception as e:
        logger.warning("gap-gini: alignment failed for pair (...): %s", e)
        results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"gap_gini_error": str(e)}))
        continue
    results.append(ConstraintOutput(score=score, ...))
```

This is the only place soft-fail belongs. Canonical: `gap_gini_constraint.py`. Other examples: `structure_ensemble_similarity` per-sequence, `structure_confidence` per-proposal missing-metric, `specific_kmer` sequence-too-short, `gyration_radius` no-metric, per-DNA-proposal sites in `protein_globularity` and `protein_symmetry_ring` (where ORFipy may find no canonical ATG-to-stop ORF, or ESMFold may fail for the selected longest CDS).

**Config-construction-time errors raise too**, with reformatted messages:

- **Pydantic `ValidationError` at `Registry.create()`**: caught and reformatted via `format_pydantic_error()` (in `proto_language/utils/helpers.py`) → `ValueError("<type> '<key>' config invalid — <field>: <msg> [got=<value>]")`. Optimizer config validation lives in `Optimizer.__init__` and in the `test_constraint` / `test_generator` / `test_optimizer` helpers in `proto_language/utils/component_validation.py` — all using the same helper.
- **Function-entry hard config checks** (file existence, mutually-exclusive options, list-of-required-fields-empty): raise `ValueError`/`RuntimeError` naming the bad value.

**Programming-bug invariants** (e.g. `"Inconsistent state: N energy_scores for M proposals"`, `"Proposal has no logits"`, `"NaN where impossible"`): raise. Soft-fail would mask a real bug.

Style: one line per error, name the operation / tool / failing value, and (when natural) a one-clause fix hint. Use `logging.getLogger(__name__)` not `print()` or `warnings.warn()`.

## Docstring Conventions

Google style everywhere. Enforced by `tests/test_docstring_consistency.py`.

- **Module docstrings**: A one-line Google-style summary ending with a period, or a summary line + blank line + details for longer descriptions. `__init__.py` files are exempt (D104 ignored). No path-header prefix.
  ```python
  """Constraint evaluation and metadata propagation for sequences."""
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
      list[ConstraintOutput]: One result per input sequence.
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
- **Coverage**: Branch coverage floor is configured in `pyproject.toml` (`fail_under`). Run `pytest --cov --cov-report=term-missing` to check.

## Sub-Agent Parallelization

Use sub-agents aggressively to parallelize independent work:

- **Codebase exploration**: Launch multiple sub-agents simultaneously to search different areas (e.g., one searching constraints, another searching generators, another reading tests)
- **Test + lint**: Run `pytest` and `ruff check` in parallel sub-agents after making changes
- **Multi-file investigation**: When an issue touches several components, fan out sub-agents to read all relevant files at once rather than reading them sequentially
- **Fix verification**: Run the specific failing test, the broader component suite, and lint all in parallel

The general rule: if two tasks don't depend on each other's output, run them in parallel sub-agents.

## Skills (`.claude/skills/`)

### For users (writing programs)

- **write-program**: composing optimization programs in Python (segments, constructs, generators, constraints, optimizers)

### For developers (extending the framework)

Skills (auto-loaded when relevant):

- **general-dev**: coding conventions, config patterns, registry system, data model, export chains
- **implement-constraint**: full constraint implementation lifecycle with templates and examples
- **implement-generator**: full generator implementation lifecycle (ABC contract, categories, templates)
- **implement-optimizer**: full optimizer implementation lifecycle (dual-pool architecture, templates)
- **testing**: comprehensive test patterns, fixtures, markers, templates for each component type

### In `proto-tools/` submodule

See the submodule's own CLAUDE.md for its skills and commands.
