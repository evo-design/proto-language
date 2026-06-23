# Proto Language

![Proto Tools](https://proto-bio.github.io/proto-assets/covers/open-wings-code/carousel.png)

[![Checks](https://github.com/evo-design/proto-language/actions/workflows/checks.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/checks.yml)
[![Unit Tests](https://github.com/evo-design/proto-language/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/evo-design/proto-language/actions/workflows/unit-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/evo-design/proto-language/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-proto.evodesign.org-blue)](https://proto.evodesign.org/docs/language/introduction)

Welcome! This repository contains the open-source implementation of `proto-language`, a Python package for designing biological sequences (DNA, RNA, and proteins) through constraint-based optimization. A design is specified as a set of constraints, and the framework runs a propose–score–refine loop to search for sequences that satisfy them, drawing on a large suite of computational biology and biological AI tools to score candidates.

`proto-language` is built on top of the [proto-tools](https://github.com/evo-design/proto-tools) execution layer, so each computationally intensive tool (structure predictors, protein language models, inverse folding, sequence and structure aligners, gene annotation, and more) runs in its own automatically managed, isolated environment. Programs can run locally or as hosted optimization runs through the [proto-client](https://github.com/evo-design/proto-client) Python SDK. 

Proto-language is open source under an MIT license. Contributions are welcome!

## Setup

### Step 1: Install the package

The package requires Python 3.10 or later and pip:

```bash
pip install git+https://github.com/evo-design/proto-language.git
```

System tools that standalone tool environments require in order to build (git, curl, gcc, make, cmake) are automatically provisioned on first use through proto-tools' shared **foundation environment**, so no manual setup is necessary.

> [!NOTE]
> A direct PyPI install (`pip install proto-language`) is planned.

> [!NOTE]
> Contributors should instead use the editable installation described in [CONTRIBUTING.md](CONTRIBUTING.md#development-setup).

### Step 2: Configure storage (optional)

All persistent data (model weights, tool environments, micromamba) is stored under `PROTO_HOME`, which defaults to `~/.proto/` and is inherited from proto-tools.

To customize the storage location (recommended for laboratory and HPC environments):

```bash
# Add to your shell profile:
export PROTO_HOME=/path/to/your/proto_home
```

To override only the model-weights location, set `export PROTO_MODEL_CACHE=/path/to/shared/weights`. See [`notes/filesystem.md`](notes/filesystem.md) for all options.

### Step 3: Gated model access (optional)

Some generators and constraints load gated models (for example ESM3, AlphaGenome, and AlphaFold3) that require accepting a license and authenticating with HuggingFace. Set `HF_TOKEN` in the environment after accepting each model's terms. See [`proto-tools/README.md`](https://github.com/evo-design/proto-tools#step-3-gated-model-access-optional-) for the full procedure and the list of gated models.

> [!TIP]
> Setup is complete. See the [Quickstart](#quickstart) to run a program from end to end.

## Quickstart

Working programs are provided under [`examples/`](examples/):

- **[`examples/scripts/`](examples/scripts/)** — runnable Python programs, ranging from a minimal end-to-end example ([`toy.py`](examples/scripts/toy.py)) to broader workloads.
- **[`examples/jsons/`](examples/jsons/)** — declarative JSON program definitions (the `optimization_stages` schema). These illustrate program structure and are not loaded by a Python consumer.

## Architecture

The framework is built around seven primitives in [`proto_language/core/`](proto_language/core/) — three data containers, three pluggable interfaces, and one orchestrator:

- **`Sequence`** — a typed string (DNA, RNA, or protein) together with optional logits, a folded structure, and namespaced metadata. The atomic unit of design.
- **`Segment`** — a single design region. It holds the proposal `Sequence`s for that region and the surviving result `Sequence`s after scoring.
- **`Construct`** — an ordered list of `Segment`s that concatenate into a full biological construct (for example, a promoter plus a coding region; a multi-chain protein; or a designed gene).
- **`Constraint`** *(registered via `@constraint`)* — scores a `Sequence` against a target property, returning a score and namespaced metadata, and may optionally provide gradients.
- **`Generator`** *(registered via `@generator`)* — proposes new `Sequence`s for a `Segment`.
- **`Optimizer`** *(registered via `@optimizer`)* — a search strategy that drives the propose–score–refine loop.
- **`Program`** — the top-level orchestrator. It owns the `Construct` and composes one or more `Optimizer` stages.

All three pluggable interfaces share a `BaseConfig` Pydantic configuration pattern and declare parameters with `ConfigField`.

### The optimization loop

`Program.run()` iterates through its optimizer stages. Each stage performs the following steps:

1. The `Optimizer` requests proposal `Sequence`s from its `Generator` for one or more `Segment`s.
2. Each `Constraint` evaluates the proposals and records its score and metadata on the proposal `Sequence`s.
3. The `Optimizer` aggregates the constraint scores and selects survivors. These become the `Segment`'s result `Sequence`s and feed into the next iteration, or the next stage.

When the program finishes, `Program.export(path=...)` writes a directory containing tables for sequences, constraints, constructs, and optimization steps, a FASTA file, and an `assets/` sidecar directory.

## Development & Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer setup, code style, testing, and agent conventions.
