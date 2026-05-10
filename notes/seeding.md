# Seeding

`proto-language` owns run-level seed hierarchy. `proto-tools` executes the
explicit seeds it receives and does not know about programs, optimizers, or
search steps.

## Runtime Hierarchy

The deterministic hierarchy is:

```text
Program(seed)
  -> Optimizer.config.seed
     -> generator call seeds
     -> constraint config seeds
        -> proto-tools config.seed / config.seeds
```

`Optimizer.seed` is a property backed by `optimizer.config.seed`. Program-level
seeds overwrite optimizer config seeds with optimizer-specific child seeds, so
there is no separate optimizer seed value to keep in sync.

## Generators

Generators receive an optimizer-derived seed via `_set_program_seed()`. Each
subsequent `_next_seed()` call draws from the reset generator RNG. Unseeded
optimizer runs clear generator runtime seed streams so reused generators pass
`seed=None` to tools again.

## Constraints

Seeded optimizers apply constraint runtime seeds before each run. The seed
setter walks the constraint's Pydantic config tree and updates seed-bearing
fields:

- `seed` fields receive an integer child seed.
- `seeds` fields receive a one-element list, replacing any multi-seed value.
- Private per-evaluation cursors such as AF2 multimer's
  `_evaluation_seed_offset` are reset.

When an optimizer itself is unseeded, constraint seed values are not overwritten,
but private seed cursors are still reset so manually seeded constraint configs
replay on repeated runs.

## Tools Boundary

Language code should pass explicit seeds into tool configs when reproducibility
is intended.

On the tools side, `seed=None` remains cacheable by default. Only cacheable
tools marked `generative=True` skip cache and iterable dedup while unseeded so
repeated sampler calls can diversify; do not use `generative` for prediction,
scoring, relax, or analysis tools merely because they accept a seed.
