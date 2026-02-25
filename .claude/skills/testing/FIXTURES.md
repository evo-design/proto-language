# Fixtures & Mocks Reference

Detailed reference for conftest.py fixtures and mock scoring functions. Load this file on demand when setting up test infrastructure.

## conftest.py Fixtures Reference (`tests/conftest.py`)

All fixtures below are `autouse=True` — they apply to every test automatically.

### `mock_generator_registry` (autouse)
Patches `GeneratorRegistry.get_key()` and `.get()` to handle mock generators:
- `MockAutoregressiveGenerator` -> category `"autoregressive"`, types `["dna"]`
- `MockMutationGenerator` -> category `"mutation"`, types `["dna"]`
- `MockInverseFoldingGenerator` -> category `"inverse_folding"`, types `["protein"]`
- `MockAutoregressiveGeneratorNoKVCache` -> category `"autoregressive"`
- `ControlledMockGenerator` -> category `"autoregressive"`
- `SegmentAwareMockGenerator` -> category `"autoregressive"`
- `AccumulativeTrackingGenerator` -> category `"autoregressive"`

### `mock_celery` (autouse)
Mocks the task queue app and tasks. No real the task queue/a cache needed for tests.

### `mock_redis` (autouse)
Mocks both sync and async a cache clients. Patches `cache.a cache`, `cache.StrictRedis`, and `SSEManager` methods.

### `mock_database` (autouse)
Mocks SQLAlchemy sessions and database operations. Patches `DatabaseManager` CRUD methods.

### `setup_test_logging` (session-scoped, autouse)
Configures logging to `logs/pytest_{timestamp}.log`. Suppresses noisy third-party loggers.

### `setup_cloud_environment` (session-scoped, autouse)
Loads cloud credentials from `~/.cloud.toml` for tests that call deployed services.

### `gpu_available` (session-scoped, NOT autouse)
Returns `True` if GPU is available. Use in tests: `@pytest.mark.skipif(not gpu_available, reason="No GPU")`.

## Mock Scoring Functions (`tests/language_tests/constraint_tests/utils.py`)

For testing the `Constraint` class without real constraint functions:

```python
from tests.language_tests.constraint_tests.utils import (
    mock_single_input_scoring_function,       # Scores by T-fraction in sequence
    mock_multi_input_scoring_function,        # Same as single (batched)
    mock_multi_input_scoring_function_disjoint,  # Two-sequence tuples (T% + C%)
    mock_dna_only_scoring_function,           # Only supports DNA
    mock_protein_only_scoring_function,       # Only supports protein
)
```

## Creating Custom Mock Scoring Functions

```python
def my_mock_scoring(input_sequences, config=None):
    return [0.5 for _ in input_sequences]

# REQUIRED: Set these attributes (normally set by @constraint decorator)
my_mock_scoring._constraint_config_class = None
my_mock_scoring._constraint_supported_sequence_types = ["dna", "rna", "protein"]
```
