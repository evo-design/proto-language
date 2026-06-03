# High-Level Programming Language for Generative Biology

[![Checks](https://github.com/evo-design/proto-language/actions/workflows/checks.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/checks.yml)
[![Unit Tests](https://github.com/evo-design/proto-language/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/evo-design/proto-language/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/integration-tests.yml)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/evs3Unkegv)

proto-language is a constraint-based optimization framework for designing biological sequences (DNA, RNA, proteins).

## Related Repositories

- [`proto-tools`](https://github.com/evo-design/proto-tools) — tool execution layer (submodule).

## Installation

### With conda (recommended)

Includes compilers and system libraries needed by tool environments.

```bash
git clone https://github.com/evo-design/proto-language.git
cd proto-language
git submodule update --init --recursive

conda env create -f environment.yml
conda activate proto-language
```

### With pip only

If you already have compilers (`gcc`, `g++`, `cmake`) installed system-wide:

```bash
git clone https://github.com/evo-design/proto-language.git
cd proto-language
git submodule update --init --recursive
pip install -e .
pip install -e ./proto-tools
```

### Developers

```bash
pip install -e ".[dev]"
pip install -e "./proto-tools[dev]"
```

## Quickstart

Working programs ship under [`examples/`](examples/):

- **[`examples/scripts/`](examples/scripts/)** — runnable Python programs, from a fast end-to-end smoke test ([`toy.py`](examples/scripts/toy.py)) to broader workloads. The fastest way to run something end-to-end.
- **[`examples/jsons/`](examples/jsons/)** — declarative JSON program definitions emitted by the client (the `optimization_stages` schema). These illustrate program shape; they are not loaded by a Python consumer.

## Architecture

The framework is built around seven primitives in [`proto_language/core/`](proto_language/core/) — three data containers, three pluggable interfaces, and one orchestrator:

- **`Sequence`** — typed string (DNA / RNA / protein) plus optional logits, folded structure, and namespaced metadata bags. The atomic unit of design.
- **`Segment`** — one design region. Holds the proposal `Sequence`s for that region and the surviving result `Sequence`s after scoring.
- **`Construct`** — an ordered list of `Segment`s that concatenate into a full biological construct (e.g. promoter + coding region; multi-chain protein; a designed gene).
- **`Constraint`** *(registered via `@constraint`)* — scores a `Sequence` against a target property; returns a score plus namespaced metadata; can optionally provide gradients.
- **`Generator`** *(registered via `@generator`)* — proposes new `Sequence`s for a `Segment`.
- **`Optimizer`** *(registered via `@optimizer`)* — search strategy that drives the propose-score-refine loop.
- **`Program`** — top-level orchestrator. Owns the `Construct` and composes one or more `Optimizer` stages.

All three pluggable interfaces share a `BaseConfig` Pydantic config pattern and declare parameters with `ConfigField`.

### The optimization loop

`Program.run()` walks its optimizer stages. Each stage iterates:

1. The `Optimizer` asks its `Generator` for proposal `Sequence`s on one or more `Segment`s.
2. Each `Constraint` evaluates the proposals and writes its score + metadata onto the proposal `Sequence`s.
3. The `Optimizer` aggregates the constraint scores and selects survivors. Those become the `Segment`'s result `Sequence`s and feed into the next iteration (or the next stage).

When the program finishes, `Program.export(path=...)` writes a folder containing tables for sequences / constraints / constructs / optimization steps, a FASTA, and an `assets/` sidecar directory.

## Repo Layout

For the canonical guide to where source, tests, examples, logs, and persistent caches live — including how `PROTO_HOME` / `PROTO_MODEL_CACHE` are inherited from proto-tools — see [`notes/filesystem.md`](notes/filesystem.md).

## Tests

Run `pytest`. See [`tests/README.md`](tests/README.md) for tiers, markers, and the full flag set; [`notes/testing.md`](notes/testing.md) for per-component conventions.

## HuggingFace Authentication

Some generators load gated HuggingFace models. Set `HF_TOKEN` in your environment after accepting the model's license. See [`proto-tools/README.md`](./proto-tools/README.md#step-3-gated-model-access-optional-) for the full setup flow and the list of gated models.

## Using with coding agents

Conventions live in [`CLAUDE.md`](CLAUDE.md) (symlinked as [`AGENTS.md`](AGENTS.md) and [`GEMINI.md`](GEMINI.md)); long-form references in [`notes/`](notes/), including [`notes/biological-design-loop.md`](notes/biological-design-loop.md) for biological design loop guidance.

Per-task skills live under [`.claude/skills/`](.claude/skills/):

- **`write-program`** — composing optimization programs (segments, constructs, generators, constraints, optimizers).
- **`implement-constraint`** — full constraint implementation lifecycle (categories, registry, gradient variants, tests).
- **`implement-generator`** — full generator lifecycle (ABC contract, category-specific templates, batching, seeding).
- **`implement-optimizer`** — full optimizer lifecycle (dual-pool architecture, templates, gradient compiler integration).

The `proto-tools/` submodule carries its own agent-conventions doc and skills (`implement-tool`, `fix-env`).
