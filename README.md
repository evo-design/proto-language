# High-Level Design Language for Generative Biology

[![Checks](https://github.com/evo-design/proto-language/actions/workflows/checks.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/checks.yml)
[![Unit Tests](https://github.com/evo-design/proto-language/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/evo-design/proto-language/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/integration-tests.yml)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/evs3Unkegv)

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

## Related Repositories

- [`proto-tools`](https://github.com/evo-design/proto-tools) – Standalone tool layer for biological models
- [`proto-client`](https://github.com/evo-design/proto-client) – Python SDK for Proto Bio APIs


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
# 3. (Optional) Install dev dependencies (testing, linting)
pip install -e ".[dev]"
pip install -e "./proto-tools[dev]"

```

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

- **write-program**: composing optimization programs in Python (segments, constructs, generators, constraints, optimizers)

### For developers (extending the framework)

Skills (auto-loaded when relevant):

- **general-dev**: coding conventions, config patterns, registry system, data model, export chains
- **implement-constraint**: full constraint implementation lifecycle with templates and examples
- **implement-generator**: full generator implementation lifecycle (ABC contract, categories, templates)
- **implement-optimizer**: full optimizer implementation lifecycle (dual-pool architecture, templates)
- **testing**: comprehensive test patterns, fixtures, markers, templates for each component type

The `proto-tools/` submodule has its own skills and commands; see its [README](./proto-tools/README.md#using-with-claude-code).
