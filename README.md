# Proto Language

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

## Local environment setup 

1. Create a virtual environment:
   ```bash
   conda create --name proto-language-local python=3.10 -y
   conda activate proto-language-local
   ```

2. Install dependencies:
   ```bash
   pip install uv
   uv pip install -r requirements_local.txt
   ```

   [MMseqs2](https://github.com/soedinglab/MMseqs2?tab=readme-ov-file#installation) is also needed to run unit tests

   On Mac:
      ```bash
      brew install mmseqs2
      ```
   

## Tests

Unit tests can be run with the command:
```bash
pytest -sv
```

For CPU-specific tests:
```bash
pytest -sv tests/tests_cpu/
```

## Running the Toy Example

Run a basic DNA sequence optimization with GC content constraints:

```bash
jupyter notebook
```
Open and run `notebooks/toy_example.ipynb`

