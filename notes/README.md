# proto-language notes

This directory holds the reference notes for `proto-language`, the
constraint-based optimization framework for designing biological sequences. They
are written for **developers, agents, and advanced users**. Each note is the
canonical source for one area: start here to find where a topic is covered, then
read the source, docstrings, and tests for the final word on signatures and
behavior.

## Setup and workflow

- `dev.md`: contributor dev workflow — initial setup, submodule sync, git
  worktrees, the export-chain validator, and the CI workflows that gate PRs.

## Runtime behavior and layout

- `batching.md`: how batching is split across proposal pools, generators,
  constraint calls, compiled scorers, and proto-tools backends.
- `error-handling.md`: when to raise vs. return a worst-score output inside
  `Constraint.evaluate`, `Generator.sample`, `Optimizer.run`, and their helpers.
- `filesystem.md`: where files live and where runtime artifacts are written.

## Contributing

- `testing.md`: test commands, markers, placement, fixtures, mocks, and the
  component coverage each constraint/generator/optimizer should have.

## Design planning

Guidance for *planning* a biological design task — for agents and advanced users
writing programs, and companions to the `write-program` and `implement-*` skills.

- `biological-design-loop.md`: general guidance for writing automated biological
  design scripts (sequence, structure, regulatory, and multi-part designs).
- `planning-quick-reference.md`: a one-page cheat sheet of the most-used rules for
  the design planning loop.
- `component-planning-example.md`: a worked Phase 2 component plan for a concrete
  task (de novo PD-L1 mini-binder), as a shape reference.

## Tools layer

The `proto-tools/` submodule carries its own notes under `proto-tools/notes/`
for tool-layer topics: storage and model weights, tool environments, device
management, seeding, logging, error handling, and tool testing. The language
notes above point into them where the boundary matters. Read the submodule's
notes and repo instructions before changing behavior inside `proto-tools/`.
