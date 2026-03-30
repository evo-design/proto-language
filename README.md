# High-Level Programming Language for Generative Biology

[![Unit Tests](https://github.com/evo-design/proto-language/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/run-unit-tests.yml)
[![Integration Tests](https://github.com/evo-design/proto-language/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/integration_tests.yml)
[![E2E Tests](https://github.com/evo-design/proto-language/actions/workflows/e2e_tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/e2e_tests.yml)
[![Lint Check](https://github.com/evo-design/proto-language/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/flake8_check.yml)

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

## Related Repositories

### Backend

- [`proto-tools`](https://github.com/evo-design/proto-tools/tree/main) – Standalone tool layer for biological models.


## Installation

### With conda (recommended)

Includes compilers and system libraries needed by tool environments.

```bash
# 1. Clone and initialize submodules
git clone https://github.com/evo-design/proto-language.git
cd proto-language
git submodule update --init --recursive

# 2. Create conda environment (compilers, system libs, core Python deps, and tools)
conda env create -f environment.yml
conda activate proto-language
```

### With pip only

If you already have compilers (`gcc`, `g++`, `cmake`) installed system-wide:

```bash
git clone https://github.com/evo-design/proto-language.git
cd proto-language
git submodule update --init --recursive
pip install -e .
pip install -e ./proto-tools
```

### Developers

```bash
# 3. (Optional) Install full dev dependencies (API, agent, testing, linting)
pip install -e ".[all]"
pip install -e "./proto-tools[dev]"

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

## Environment Setup

Local dev needs no env vars — sensible defaults are built in. Optionally copy the template to customize:

```bash
cp .env.example .env
```

a cache must be running (`brew services start cache` or `docker run -d -p 6379:6379 cache:alpine`). The API checks a cache on startup and warns if unavailable.

See `.env.example` for all variables with descriptions, or `CLAUDE.md` for a categorized reference table.

## Running the API

### Local Development

```bash
python api/main.py
```

Runs on [http://localhost:8000](http://localhost:8000) with local execution mode (no cloud), SQLite database, and a cache on localhost.

### Docker

```bash
docker-compose up
```

Starts a cache + API together. Available at [http://localhost:8000](http://localhost:8000).

### Production

Deployed via GitHub Actions (`release-to-prod.yml`). Requires `DATABASE_URL`, `REDIS_URL`, `LOCAL_EXECUTION=false`, and cloud tokens set in the deploy platform environment variables. Set `CLOUD_ENVIRONMENT=staging` on the staging the deploy platform service.

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
