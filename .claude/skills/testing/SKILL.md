---
name: testing
description: >
  Comprehensive testing patterns for the proto-language project. Covers running
  tests (pytest markers, CPU/GPU/slow), writing new tests for constraints, generators,
  and optimizers, debugging test failures, conftest fixtures, and mock scoring functions.
  Use when writing tests, debugging failures, or setting up test infrastructure.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# testing skill

## Running Tests

```bash
# Standard
pytest                                    # All fast tests (skips slow)
pytest --cpu                              # CPU-only (skip GPU tests)
pytest --gpu                              # GPU-only (skip CPU tests)
pytest --cpu --skip-ci                    # Exact CI behavior
pytest --gpu --all                        # GPU + slow tests
pytest --slow                             # ONLY slow tests (skip others)
pytest --all                              # Everything including slow
pytest -k "mcmc"                          # Filter by name
pytest --no-log-console                   # Suppress console logging

# By area
pytest tests/language_tests/              # Language core
pytest tests/language_tests/constraint_tests/  # Constraints only
pytest tests/language_tests/generator_tests/   # Generators only
pytest tests/language_tests/optimizer_tests/   # Optimizers only
pytest tests/api_tests/                   # API
pytest tests/agent_tests/                 # Agent
pytest tests/tool_tests/                  # Tool integrations

# Linting (CI checks F401 unused imports, F841 unused vars only)
flake8 proto_language api agent tests
black --check proto_language api agent tests
isort --check-only proto_language api agent tests
```

## Test File Placement

```
tests/
├── conftest.py                                     # Shared fixtures (auto-use)
├── language_tests/
│   ├── constraint_tests/
│   │   ├── utils.py                                # Mock scoring functions
│   │   ├── test_base_constraint.py                 # Constraint class tests
│   │   ├── test_constraint_registry.py             # Registry tests
│   │   ├── test_sequence_composition/
│   │   │   ├── test_gc_content_constraint.py
│   │   │   └── test_{name}_constraint.py
│   │   ├── test_protein_structure/
│   │   ├── test_protein_quality/
│   │   ├── test_rna_secondary_structure/
│   │   ├── test_rna_splicing/
│   │   └── test_sequence_annotation/
│   ├── generator_tests/
│   │   ├── test_uniform_mutation_generator.py
│   │   └── test_{name}_generator.py
│   ├── optimizer_tests/
│   │   ├── test_base_optimizer.py
│   │   ├── test_mcmc_optimizer.py
│   │   └── test_{name}_optimizer.py
│   └── test_program.py
├── api_tests/
├── agent_tests/
└── tool_tests/
```

**Naming rules**:
- File: `test_{component_name}.py` (e.g., `test_gc_content_constraint.py`)
- Class: `Test{ComponentName}` (e.g., `TestGCContentConstraint`)
- Method: `test_{behavior}` (e.g., `test_dna_sequences`, `test_wrong_sequence_type`)

## Markers

| Marker | When to use | Effect |
|--------|------------|--------|
| `@pytest.mark.uses_gpu` | Test requires GPU (CUDA) | Skipped with `--cpu` |
| `@pytest.mark.slow` | Test takes >30s | Skipped by default; needs `--all` or `--slow` |
| `@pytest.mark.skip_ci` | Test can't run in GitHub Actions | Skipped with `--skip-ci` or in CI |
| `@pytest.mark.integration` | End-to-end integration test | Informational |
| `@pytest.mark.asyncio` | Async test function | Required for `async def test_*` |
| *(no marker)* | CPU test (fast) | Auto-marked `uses_cpu` by conftest |

**Rule**: CPU tests need NO marker. Only add markers for GPU, slow, or special tests.

## Writing New Tests

For complete test templates by component type, use the `Read` tool to load:
- **Constraint tests**: `.claude/skills/testing/TEMPLATES.md` (section: Constraint Test Template)
- **Generator tests**: `.claude/skills/testing/TEMPLATES.md` (section: Generator Test Template)
- **Optimizer tests**: `.claude/skills/testing/TEMPLATES.md` (section: Optimizer Test Template)

For conftest fixtures and mock scoring functions, use the `Read` tool to load:
- **Fixtures reference**: `.claude/skills/testing/FIXTURES.md`

## Bug-Fixing Workflow

When a user reports a bug:

1. **Write a failing test FIRST** that reproduces the bug
2. **Verify the test fails** as expected (`pytest -k "test_name" -x`)
3. **Fix the bug** in the source code
4. **Verify the test passes** (`pytest -k "test_name" -x`)
5. **Run broader suite** to check regressions: `pytest tests/ --cpu`

## Common Assertion Patterns

```python
# Exact float comparison with tolerance
assert abs(actual - expected) < 1e-9

# Approximate comparison
assert actual == pytest.approx(expected, abs=1e-6)

# Exception with message match
with pytest.raises(ValueError, match="must be positive"):
    do_something()

# Exception type only
with pytest.raises(TypeError):
    do_something()

# Parametrized tests
@pytest.mark.parametrize("input,expected", [("A", 1), ("B", 2)])
def test_mapping(self, input, expected):
    assert func(input) == expected

# Skip conditionally (GPU tests)
@pytest.mark.gpu
def test_gpu_feature(self):
    ...
```

## Validation Checklist

Copy this and check off as you go:

- [ ] Test file placed in correct directory (`tests/language_tests/{component}_tests/`)
- [ ] Test class named `Test{ComponentName}`
- [ ] Correct markers applied (GPU, slow, skip_ci)
- [ ] Parametrized scoring tests cover edge cases (empty, boundary, perfect, worst)
- [ ] Wrong sequence type test included
- [ ] Invalid config test included
- [ ] Metadata propagation test included (for constraints)
- [ ] Tests pass: `pytest tests/language_tests/{area}_tests/ --cpu -x`
- [ ] Lint passes: `flake8 tests/`

If any check fails, fix before proceeding.
