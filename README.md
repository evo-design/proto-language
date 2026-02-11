# High-Level Programming Language for Generative Biology

[![Unit Tests](https://github.com/evo-design/proto-language/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/run-unit-tests.yml)
[![Integration Tests](https://github.com/evo-design/proto-language/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/integration_tests.yml)
[![Lint Check](https://github.com/evo-design/proto-language/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/flake8_check.yml)

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

Related repos:
- [`bio-tools` (standalone tool layer)](https://github.com/evo-design/bio-tools/tree/main)
- [`proto-language-lang` (front end)](https://github.com/evo-design/proto-language-lang)

> [!NOTE]
> CI's are currently failing on main. We are working on fixing them now


## Installation

### Quick Start

Run the install script (it initializes git submodules and creates the conda environment):

```bash
bash install.sh
conda activate proto-language
```

### Manual Setup

If you prefer to set up manually, or if the install script fails:

```bash
# 1. Initialize submodules
git submodule update --init --recursive

# 2. Create and activate conda environment
conda create --name proto-language python=3.12 -y
conda activate proto-language

# 3. Install dependencies (choose one)
pip install uv
uv pip install -e .[tools,api,dev]          # CPU-only
# uv pip install -e .[tools,api,gpu,dev]    # GPU

# 4. Install proto-tools submodule
uv pip install -e ./proto-tools

# 5. Install pre-commit hooks
pre-commit install
```
>[!NOTE] The installation script is currently failing on Chimera due to dependency issues.
> For now, use the following to clone from Brian's env:

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

API will be available at http://localhost:8000

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


## Running the Toy Example

```bash
jupyter notebook
```

Open and run `notebooks/toy_example.ipynb`
