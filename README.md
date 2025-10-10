# Proto Language

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

## Installation

The package is now structured with `pyproject.toml`. Install with:

1. Create the conda environment:
```bash
conda create --name proto-language python=3.12 -y
conda activate proto-language
conda install -c conda-forge -c bioconda mmseqs2 blast -y
```

2. Install as an editable package:

    A) Only with CPU dependencies:
    ```bash
    pip install uv
    uv pip install -e .
    ```

    B) Or with GPU dependencies:
    ```bash
    pip install uv
    uv pip install -e .[gpu]
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

```bash
pytest -sv
```

For CPU-specific tests:

```bash
pytest -sv --cpu
```
NOTE: CPU tests require the installation of the `dev` dependencies.

For GPU-specific tests:

```bash
pytest -sv --gpu
```
NOTE: GPU tests require the installation of the `gpu` dependencies.

## Running the Toy Example

```bash
jupyter notebook
```

Open and run `notebooks/toy_example.ipynb`
