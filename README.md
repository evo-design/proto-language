# High-Level Programming Language for Generative Biology

[![Unit Tests](https://github.com/evo-design/proto-language/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/run-unit-tests.yml)
[![Integration Tests](https://github.com/evo-design/proto-language/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/integration_tests.yml)

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

## Installation


1. Create the conda environment: 
```bash
conda create --name proto-language python=3.12 -y
conda activate proto-language
conda install -c conda-forge -c bioconda -c nvidia mmseqs2 blast cuda-nvcc cuda-cudart-dev transformer-engine-torch=2.3.0 -y
git submodule update --init --recursive
```
2. Install as an editable package:

    A) Only with CPU tool + API  dependencies:
    ```bash
    pip install uv
    uv pip install -e .[tools,api]
    ```

    B) Or with GPU dependencies:
    ```bash
    pip install uv
    uv pip install -e .[tools,api,gpu]
    uv pip install flash-attn==2.8.0.post2 --no-build-isolation
    pip install evo2
    ```

    C) Or with full dependencies (including dev):
    ```bash
    pip install uv
    uv pip install -e .[all]
    uv pip install flash-attn==2.8.0.post2 --no-build-isolation
    pip install evo2
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
