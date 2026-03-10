# Stress Test Programs

GPU-based stress tests for the ToolPool multi-GPU dispatch system. Each program uses ESMFold structure prediction constraints and/or ESM2 generators to exercise parallel GPU scheduling.

## Programs

| # | Program | Run Command |
|---|---------|-------------|
| 01 | Protein pLDDT (ESMFold) | `python examples/scripts/run_program.py examples/stress-test-programs/01-protein-plddt-esmfold.json` |
| 02 | ESM2 + Structure Scoring | `python examples/scripts/run_program.py examples/stress-test-programs/02-esm2-structure-scoring.json` |
| 03 | Globularity + Symmetry Ring | `python examples/scripts/run_program.py examples/stress-test-programs/03-globularity-symmetry.json` |
| 04 | Multi-Construct Protein | `python examples/scripts/run_program.py examples/stress-test-programs/04-multi-construct-protein.json` |
| 05 | Heavy Multi-Stage Pipeline | `python examples/scripts/run_program.py examples/stress-test-programs/05-heavy-multi-stage-pipeline.json` |
| 06 | High-Volume pLDDT | `python examples/scripts/run_program.py examples/stress-test-programs/06-high-volume-plddt.json` |

The ToolPool auto-detects available GPUs and splits work across them.
