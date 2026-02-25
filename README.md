# High-Level Programming Language for Generative Biology

[Unit Tests](https://github.com/evo-design/proto-language/actions/workflows/run-unit-tests.yml)
[Integration Tests](https://github.com/evo-design/proto-language/actions/workflows/integration_tests.yml)
[Lint Check](https://github.com/evo-design/proto-language/actions/workflows/flake8_check.yml)

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

## Related Repositories

### Backend

- `[proto-tools](https://github.com/evo-design/proto-tools/tree/main)` – Standalone tool layer for biological models.

### Client

- `[proto-tools-ui](https://github.com/evo-design/proto-tools-ui)` – Mock UI for demonstrating tool usage.
- `[proto-language-lang](https://github.com/evo-design/proto-language-lang)` – Primary client interface for the biological programming language.

## Installation

```bash
# 1. Clone and initialize submodules
git clone https://github.com/evo-design/proto-language.git
cd proto-language
git submodule update --init --recursive

# 2. Create and activate conda environment
conda create --name proto-language python=3.12 -y
conda activate proto-language

# 3. Install dependencies
pip install uv
uv pip install -e .                      # Users: core language
uv pip install -e ./proto-tools
```

**For developers** (API server, AI agent, testing, linting):

```bash
pip install uv
uv pip install -e ".[all]"               # Developers: + API, agent, testing, linting
uv pip install -e ./proto-tools

# 4. (Optional) Install pre-commit hooks
pre-commit install
```

> [!NOTE]
> Beam search with Evo2 does not work in the main environment. To fix this issue, clone from Brian's env
> on Chimera (instructions below) and use Evo2 `_in_process_mode`:

```bash
conda create --name proto-language --clone /home/brianhie/miniconda/envs/gpro/
pip uninstall -y numcodecs zarr
pip install --no-cache-dir --force-reinstall numcodecs zarr
pip install -e /home/{USERNAME}/proto-language
```

## Running the API

### Local Development

```bash
python api/start_dev.py
```

### Docker

```bash
docker-compose up
```

API will be available at [http://localhost:8000](http://localhost:8000)

## Tests

Tests can be run with various filtering options based on hardware utilization and execution time.
See [tests/README.md](tests/README.md) for more details. A few commonly used commands are listed below.

- By default, slow tests *are skipped*. You must specify `--all` to run them.
- By default, we don't filter tests by hardware utilization. Specify `--cpu` to run only CPU-based tests and `--gpu` to run only GPU-based tests.

```bash
# Run all tests that are not marked as slow (both CPU and GPU based)
pytest

# Run all tests, including slow ones (both CPU and GPU based)
pytest --all

# Run fast CPU-based tests
pytest --cpu

# Run all GPU-based tests
pytest --gpu --all
```

## HuggingFace Authentication

Some tools (e.g. ESM3, AlphaGenome) use gated HuggingFace models that require both authentication and accepting the model's license/terms on the HuggingFace model page. See the [proto-tools README](./proto-tools/README.md#huggingface-authentication) for the full list of gated models and setup instructions.

## Using with Claude Code

This repo includes [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills for both users writing programs and developers extending the framework. Launch `claude` from the repo root:

```bash
claude
```

### For users (writing programs)

- **write-program** — composing optimization programs in Python (segments, constructs, generators, constraints, optimizers)

### For developers (extending the framework)

Skills (auto-loaded when relevant):

- **general-dev** — coding conventions, config patterns, registry system, data model, export chains
- **implement-constraint** — full constraint implementation lifecycle with templates and examples
- **implement-generator** — full generator implementation lifecycle (ABC contract, categories, templates)
- **implement-optimizer** — full optimizer implementation lifecycle (dual-pool architecture, templates)
- **testing** — comprehensive test patterns, fixtures, markers, templates for each component type

Commands (invoked with `/command-name`):

- `**/fix-issue <number>`** — full GitHub issue fix lifecycle (read issue, explore, reproduce, fix, test, verify)

The `proto-tools/` submodule has its own skills and commands — see its [README](./proto-tools/README.md#using-with-claude-code).
