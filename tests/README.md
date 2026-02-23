# Running Tests

The test suite includes options for filtering tests by hardware requirements and execution time.

## Test Markers

When writing tests, use these markers:
```python
import pytest

# Mark a test as requiring GPU
@pytest.mark.uses_gpu
def test_gpu_function():
    ...

# Mark a test as slow (will be skipped unless --all or --slow specified)
@pytest.mark.slow
def test_long_running_operation():
    ...

# CPU tests don't need explicit marking (auto-marked)
def test_cpu_function():
    ...
```

- By default, all tests are marked as CPU-only unless explicitly marked with `@pytest.mark.uses_gpu`.
- By default, all tests are marked as fast unless explicitly marked with `@pytest.mark.slow`.

## External Dependencies

The test suite automatically mocks external dependencies (a cache, databases) so you can run tests without setting up these services. cloud credentials are loaded from `~/.cloud.toml` if available.
