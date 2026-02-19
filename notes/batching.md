# Batching Architecture

How GPU memory is managed across generators, constraints, and tools.

## Generator Batching (Framework-Level)

Generators have a **framework-level** `batch_size` attribute on the `Generator` base class (`core/generator.py`):

```python
class Generator(ABC):
    batch_size: int = 1
```

The framework splits candidates into chunks of `batch_size` and processes each chunk on GPU. All generators default to `batch_size=1` — users increase it via config to enable batching:

- **ESM2 / ESM3 / Evo1 / Evo2**: `batch_size` config field, default `1`. Set higher (e.g., 8-16) for throughput.
- **ProteinMPNN / LigandMPNN**: No `batch_size` config field. Batch size is computed dynamically in `sample()` based on the number of input structures.
- **CPU generators** (UniformMutation, MSA): `batch_size = 1` (no batching needed).

## Constraint Evaluation (No Framework-Level Batching)

Constraints do **not** have a framework-level `batch_size`. `Constraint.evaluate()` passes all passing candidates to the constraint function in a single call (`core/constraint.py:253`):

```python
raw_scores = self._function(input_sequences_to_evaluate, config=self._function_config)
```

The full `List[Tuple[Sequence, ...]]` is passed at once. GPU memory management is handled internally by each tool.

### Why No Framework-Level Batching for Constraints?

1. **GPU memory depends on sequence characteristics, not candidate count.** Structure prediction memory scales with residue count and complex size, not with the number of candidates. A single 1000-residue protein can OOM, while 100 small peptides fit easily.

2. **Tools know their own memory characteristics.** ESMFold can batch by total residue count. Boltz2/AF3/Chai1 are inherently sequential (one complex at a time). A framework-level `batch_size` based on candidate count would be the wrong abstraction.

3. **The two-pass filter→score strategy already reduces workload.** Cheap filters reject bad candidates before expensive GPU constraints run, which is a more effective optimization than chunking.

## Tool-Level Batching Matrix

| Tool | Batching Strategy | User Config |
|------|------------------|-------------|
| **ESMFold** | Batches by total residue count | `max_batch_residues` |
| **BioEmu** | Fixed batch size | `batch_size` |
| **Boltz2** | Sequential (one complex at a time) | — |
| **AlphaFold3** | Sequential | — |
| **Chai-1** | Sequential | — |
| **MMseqs2** | All at once (CPU) | — |

## Key Code Paths

- **Generator `batch_size`**: `proto_language/language/core/generator.py` — `Generator` base class attribute
- **`Constraint.evaluate()`**: `proto_language/language/core/constraint.py` — builds `input_sequences_to_evaluate` and calls function once
- **`score_energy()` two-pass strategy**: `proto_language/language/core/optimizer.py` — filters first, then scorers on surviving candidates
- **`BoltzBindingStrengthConfig`**: `proto_language/language/constraint/protein_structure/boltz_binding_strength_constraint.py` — no `batch_size` field (removed as dead code)
