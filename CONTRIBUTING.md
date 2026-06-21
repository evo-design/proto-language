# Contributing to proto-language

Thank you for your interest in contributing! proto-language is the constraint-based optimization framework for designing biological sequences, built on the [proto-tools](https://github.com/evo-design/proto-tools) execution layer. The codebase has well-established patterns but is very much in active development, and contributions of all kinds are welcome: new constraints, generators, and optimizers, core framework improvements, documentation, and more. Please follow the existing patterns and conventions as closely as possible. (Coding agents are very helpful for this!)

This guide covers the conventions and workflows used in the project.

## Development Setup

Contributors install editable checkouts of **both** layers, using the proto-tools submodule:

```bash
git clone https://github.com/evo-design/proto-language.git
cd proto-language
git submodule update --init --recursive

pip install -e ".[dev]"               # language layer and dev tools (proto-tools installed from git)
pip install -e "./proto-tools[dev]"   # override with the editable submodule
```

The proto-tools editable installation must be run **last**: it replaces the git-installed proto-tools with the local submodule, so that edits within `proto-tools/` take effect immediately. System build tools are still automatically provisioned through the foundation environment.

Persistent data (model weights, tool environments) lives under `PROTO_HOME` (defaults to `~/.proto/`), inherited from proto-tools. See the [README](README.md) for storage and gated-model setup.

### Troubleshooting: `proto_tools` import fails after install

If `import proto_tools` fails (e.g. `cannot import name 'ProdigalConfig' from 'proto_tools' (unknown location)`, or `proto_tools.__file__` is `None`), the environment likely has a **stale, partial `proto_tools` directory** in `site-packages` left over from an install attempt made before the submodule was checked out. Because it lacks a top-level `__init__.py`, Python treats it as a namespace package that silently shadows the editable install — and plain `pip uninstall proto_tools` does **not** remove it (those files are untracked). Reinstalling repeatedly will not fix it.

Recover by deleting the stale directory, then reinstalling:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
pip uninstall -y proto_tools
rm -rf "$SITE"/proto_tools "$SITE"/proto_tools-*.dist-info "$SITE"/__editable__*proto_tools*
pip install -e "./proto-tools[dev]"
python -c "import proto_tools; print(proto_tools.__file__)"   # should point at proto-tools/proto_tools/__init__.py
```

To avoid this in the first place, always run `git submodule update --init --recursive` **before** any `pip install`.

## Using with coding agents

Conventions are documented in [`CLAUDE.md`](CLAUDE.md) (symlinked as [`AGENTS.md`](AGENTS.md) and [`GEMINI.md`](GEMINI.md)); long-form references are in [`notes/`](notes/), including [`notes/biological-design-loop.md`](notes/biological-design-loop.md) for guidance on the biological design loop.

Per-task skills are located under [`.claude/skills/`](.claude/skills/):

- **`write-program`** — composing optimization programs (segments, constructs, generators, constraints, and optimizers).
- **`implement-constraint`** — the full constraint implementation lifecycle (categories, registry, gradient variants, and tests).
- **`implement-generator`** — the full generator lifecycle (ABC contract, category-specific templates, batching, and seeding).
- **`implement-optimizer`** — the full optimizer lifecycle (dual-pool architecture, templates, and gradient-compiler integration).

The `proto-tools/` submodule provides its own agent-conventions documentation and skills (`implement-tool`, `fix-env`).

## Code Style

### Formatting

- **ruff**: enforced for both linting and formatting (line length 120). CI runs `ruff check` and `ruff format --check`; run them before committing.

### Conventions

- **mypy** strict mode with the Pydantic plugin. Every `# type: ignore` must include the error code. Prefer runtime `assert` narrowing over `cast()`, ad-hoc `Protocol`, or `TYPE_CHECKING` blocks.
- Use `logging.getLogger(__name__)`, never `print()`, in framework code.
- Config classes inherit the local `BaseConfig` and declare parameters with `ConfigField`.
- Registries use decorators (`@constraint`, `@generator`, `@optimizer`) with kebab-case keys. Follow neighboring file, class, function, and test naming.
- Google-style docstrings, enforced by `tests/test_docstring_consistency.py`. `proto_language/core/` is held to a higher documentation standard (module headers and `Examples:` snippets); apply the same pattern to new components. See [`notes/dev.md`](notes/dev.md).
- Program-level seeds own run determinism and derive downstream seeds; multi-stage programs reuse the same construct objects by identity across optimizers.

## Testing

Run `pytest`. Use `--cpu-only` for normal local and CI-equivalent runs; plain `pytest` skips slow and integration tests but still runs `uses_gpu` tests when a GPU is visible.

```bash
pytest --cpu-only -x                  # fast CPU feedback
pytest --integration --cpu-only -v    # external-tool tests, CPU only
pytest --gpu-only -k "esm2" -x        # GPU-marked tests only
pytest --all --cpu-only               # add slow + integration, skip GPU
```

Common markers: `uses_gpu`, `uses_cpu`, `slow`, `integration`, `extensive`, `skip_ci`. See [`tests/README.md`](tests/README.md) and [`notes/testing.md`](notes/testing.md) for the full set of flags, markers, and per-component conventions.

CI runs the unit tests and `checks.yml` (`ruff check`, `ruff format --check`, `mypy proto_language/`, and export validation) on non-draft PRs. Add the `skip-tests` label to skip the unit-test run on docs-only or submodule-pointer-bump PRs. Integration tests run on a schedule or via manual dispatch.

## Implementing components

Constraints, generators, and optimizers each follow a registered lifecycle (decorator registration, a `BaseConfig` class, the export chain, and tests). The `write-program`, `implement-constraint`, `implement-generator`, and `implement-optimizer` skills provide step-by-step guides with templates. See [`CLAUDE.md`](CLAUDE.md) for the full architecture reference.

## Branch Naming

Use descriptive branch names with a category prefix:

- `feat/description`: new features or components
- `fix/description`: bug fixes
- `refactor/description`: code restructuring without behavior change
- `docs/description`: documentation-only changes
- `test/description`: test additions or fixes

## Questions?

Open an issue or start a discussion if you're unsure about anything. We're happy to help!
