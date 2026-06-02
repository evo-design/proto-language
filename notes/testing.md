# Testing

Long-form testing reference for `proto-language`: commands, markers, placement, fixtures, mocks, and component coverage. `tests/conftest.py`, `pyproject.toml`, and `.github/workflows/` are the source of truth when this guide and code disagree.

## Commands

Use `--cpu` for normal local and CI-equivalent runs. Plain `pytest` skips slow and integration tests, but it does not skip tests marked `uses_gpu`.

```bash
pytest --cpu -x                                   # fast CPU-focused feedback
pytest tests/language_tests/generator_tests --cpu -x
pytest -k "test_name" --cpu -x
pytest --cpu --skip-ci                           # additionally skip skip_ci and hide CUDA
pytest --integration --cpu -v                    # external-tool tests, CPU only
pytest --gpu -k "esm2" -x                         # GPU-marked tests only, still respects slow/integration gates
pytest --gpu --slow -k "beam_search" -x           # slow GPU subset

ruff check proto_language tests
ruff format --check
mypy proto_language/
python .github/scripts/validate_exports.py --verbose
```

Current GitHub workflows:

- `unit-tests.yml`: non-draft PRs and `main`, runs `pytest --cpu -q --override-ini="log_cli=false" --cov --cov-report=term-missing`.
- `integration-tests.yml`: scheduled/manual, installs MAFFT and runs `pytest --integration --cpu -v`.
- `checks.yml`: non-draft PRs, runs `ruff check`, `ruff format --check`, `mypy proto_language/`, and export validation.
- `submodule-check.yml`: non-draft PRs, verifies `proto-tools` points at the latest `main`.

## Markers and Flags

| Marker | When to use | Selection behavior |
|---|---|---|
| `@pytest.mark.uses_gpu` | Test requires CUDA/model GPU execution | Included by default unless also slow/integration; skipped by `--cpu`; selected by `--gpu` |
| `@pytest.mark.slow` | Test takes long enough to disrupt default feedback | Skipped by default; included by `--all`; selected alone by `--slow` |
| `@pytest.mark.integration` | Test requires external tools/services such as MAFFT | Skipped by default; included by `--integration` or `--all` |
| `@pytest.mark.skip_ci` | Test is not valid in GitHub Actions or local CI simulation | Skipped when `GITHUB_ACTIONS=true` or `--skip-ci` |
| `@pytest.mark.only_chimera` | Test only runs on Chimera (`SLURM_CLUSTER_NAME=arc-slurm`) | Skipped everywhere else |
| `@pytest.mark.asyncio` | Optional explicit marker for async tests | `asyncio_mode=auto` is enabled, so async tests do not normally need it |
| *(no marker)* | Fast CPU test | Auto-marked `uses_cpu` by `conftest.py` |

Flag interactions:

- `--cpu` skips `uses_gpu`.
- `--gpu` skips auto-marked CPU tests, but does not by itself include slow or integration tests.
- `--all` includes slow and integration tests.
- `--slow` runs only tests marked `slow`; slow tests that are also `integration` are still skipped unless you also pass `--integration` (with both flags, all slow tests run — slow-only and slow+integration alike, while non-slow tests stay skipped). Use `--all` to run the full default + slow + integration set instead.
- `--skip-ci` skips `skip_ci` tests and sets `CUDA_VISIBLE_DEVICES=""`.

CPU tests need no marker. Add only the markers that change selection or environment assumptions.

## Placement

```
tests/
├── conftest.py
├── language_tests/
│   ├── constraint_tests/
│   │   ├── utils.py
│   │   ├── test_base_constraint.py
│   │   ├── test_constraint_registry.py
│   │   ├── test_sequence_composition/
│   │   ├── test_sequence_annotation/
│   │   ├── test_sequence_alignment/
│   │   ├── test_sequence_scoring/
│   │   ├── test_protein_quality/
│   │   ├── test_protein_structure/
│   │   ├── test_rna_secondary_structure/
│   │   └── test_rna_splicing/
│   ├── generator_tests/
│   ├── optimizer_tests/
│   └── test_*.py
├── utils_tests/
├── tests_cpu/
└── test_codebase_consistency.py
```

Naming:

- File: `test_{component_name}.py`.
- Class: `Test{ComponentName}` for component suites. This repo uses test classes; the proto-tools submodule has different conventions.
- Method/function: `test_{behavior}`.

Put broad framework behavior in the nearest existing core test file. Put component-specific behavior beside neighboring component tests.

## Fixtures

Autouse fixtures in `tests/conftest.py`:

- `mock_generator_registry`: maps optimizer-test mock generator classes to real registry keys so compatibility checks see the expected generator category and input type.
- `setup_test_logging`: writes `logs/pytest_{timestamp}.log` or `logs/pytest_{k_expression}.log`, suppresses noisy third-party loggers, and keeps console logging configurable with `--no-log-console`.

Other common fixtures:

- `sample_pdb_content`: minimal PDB content as a string.
- `temp_pdb_file`: temporary PDB file path with cleanup.
- `toy_json`: contents of `examples/jsons/toy.json`.

There is no global `gpu_available` fixture in this repo. If a GPU test needs a hardware check beyond marker selection, add a local helper or skip near the test body, following existing Malinois tests.

## Mock Generator Registry Map

`mock_generator_registry` maps these class names:

| Mock class | Registry key |
|---|---|
| `MockAutoregressiveGenerator` | `evo1` |
| `MockAutoregressiveGeneratorNoKVCache` | `evo1` |
| `ControlledMockGenerator` | `evo1` |
| `SegmentAwareMockGenerator` | `evo1` |
| `AccumulativeTrackingGenerator` | `evo1` |
| `TrackingKVCacheGenerator` | `evo1` |
| `MockMutationGenerator` | `random-protein` |
| `MockCyclingGenerator` | `random-protein` |
| `MockInverseFoldingGenerator` | `proteinmpnn` |

Mock classes should still declare matching `input_type` classvars in the test file. The fixture only patches registry key lookup.

## Constraint Mocks

Use `tests/language_tests/constraint_tests/utils.py` when testing the `Constraint` wrapper without a real registered constraint:

```python
from tests.language_tests.constraint_tests.utils import (
    mock_dna_only_scoring_function,
    mock_multi_input_scoring_function,
    mock_multi_input_scoring_function_disjoint,
    mock_protein_only_scoring_function,
    mock_single_input_scoring_function,
)
```

Custom mock scoring functions must mimic decorator metadata:

```python
from proto_language.core import ConstraintOutput


def my_mock_scoring(input_sequences, config=None):
    return [ConstraintOutput(score=0.5, metadata={}) for _ in input_sequences]


my_mock_scoring._constraint_config_class = None
my_mock_scoring._constraint_supported_sequence_types = ["dna", "rna", "protein"]
```

If the test covers registry behavior, use a real `@constraint` function instead of patching metadata manually.

## Assertion Patterns

```python
assert actual == pytest.approx(expected, abs=1e-6)

with pytest.raises(ValueError, match="must be positive"):
    do_something()

@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("GCGC", 0.0),
        ("AAAA", 1.0),
    ],
)
def test_scoring(sequence: str, expected: float) -> None:
    ...
```

Prefer message matches for user-facing validation errors. For internal invariants, matching the exception type is often enough unless the message is part of the contract.

## Component Coverage

Constraints should usually cover:

- scoring formula and edge cases: perfect, worst, boundary, empty/short input when relevant;
- supported and unsupported sequence types;
- invalid config validation;
- `ConstraintOutput` metadata propagation under `_constraints_metadata[<label>]["data"]`;
- threshold/filter behavior when the constraint is commonly used as a filter;
- tool-backed failure behavior: raise for hard failures, `MAX_ENERGY` only for proposal-local soft failures.

Generators should usually cover:

- config storage and validation;
- `assign()` validation, including unsupported sequence type and tied segments when relevant;
- `sample()` mutates each proposal in place and preserves expected length/type;
- multiple proposal sequences in `segment.proposal_sequences`;
- seed determinism when the generator owns stochastic behavior;
- GPU tests marked `uses_gpu` and slow tests marked `slow`.

Optimizers should usually cover:

- config validation and `num_results` resolution;
- constructor validation for unassigned generators/constraints and duplicate objects;
- `run()` completion on a small deterministic setup;
- filter constraints and all-filter behavior;
- history/tracking fields that downstream exports consume;
- multi-stage handoff or seed propagation when changing program-level behavior.

## Bug-Fixing Workflow

1. Write or identify a failing test that reproduces the bug.
2. Verify the failure with a narrow command, for example `pytest -k "test_name" --cpu -x`.
3. Fix the source.
4. Rerun the narrow test.
5. Run the relevant directory with `--cpu -x`.
6. Run `ruff check` on touched source/tests, and `mypy proto_language/` if the change touches typed framework code.

## Checklist

- [ ] Test file is in the closest matching directory.
- [ ] CPU tests are unmarked; GPU/slow/integration/skip_ci markers are applied only when needed.
- [ ] Edge cases cover boundary, invalid, and representative success behavior.
- [ ] Config validation has explicit tests for custom validators.
- [ ] Constraint tests assert score, metadata, and sequence-type behavior.
- [ ] Generator tests assert assignment, proposal mutation, and batch/proposal behavior.
- [ ] Optimizer tests assert validation, run loop, filtering, and history where relevant.
- [ ] Narrow tests pass with `pytest ... --cpu -x`.
- [ ] `ruff check proto_language tests` and `ruff format --check` pass for relevant files.
- [ ] `mypy proto_language/` passes when typed source changed.
