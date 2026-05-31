# Planning Quick Reference

A one-page cheat sheet of the most frequently used rules for the biological design planning loop. For broader generator, validator, candidate-pool, and failure-handling guidance, see `notes/biological-design-loop.md`. Refer to the full system prompt for definitions, edge cases, and reasoning. For a worked example of the Phase 2 output these rules produce, see `notes/component-planning-example.md`.

## The Five Hard Constraints

1. **No invented biology.** Sequences and panels come from the task, staged assets, or sourced databases. Never memory.
2. **No static candidate tables.** Final candidates come from an optimizer-owned pool, not from memory or a fetched sequence.
3. **Validate against the named comparison set.** When the task names one (UniRef50, a wildtype, a panel), it is non-substitutable.
4. **No silent failures.** No broad `try/except` around decisive calls.
5. **Model priors are not validity checks.** Likelihood does not satisfy a structural or functional requirement.

## The Planning Loop, at a Glance

```
Phase 1: biological_planning.md       → review pass
   │
   ▼
Phase 2: component_planning.md        → review pass
   │
   ▼ (only on a named blocker)
Phase 3: external_research.md         → review pass
   │
   ▼ (if upstream changed, loop back)
   ▼
Converged? (3 conditions all hold)    → cap at 3 cycles
   │
   ▼
final_biological_design_plan.md       → final review → implement
```

## Optimizer Decision Tree, Compressed


| Situation                                                      | Branch                                                                |
| -------------------------------------------------------------- | --------------------------------------------------------------------- |
| Capable generator + cheap filters                              | Best-of-K / rejection sampling                                        |
| Refinement task with starting candidate                        | MCMC with intelligent mutations                                       |
| De novo + property predictor + conditioned generator           | Cycling                                                               |
| De novo + differentiable generator + differentiable constraint | Gradient                                                              |
| De novo + capable scoring, no cycling/gradient path            | Black-box MCMC / simulated annealing                                  |
| Multiple branches fit different phases                         | Compose (broad-then-narrow, generate-then-verify, coarse-then-fine)   |
| No branch fits                                                 | Upstream choices are wrong — return to generator/constraint selection |


## Constraint Roles

Every constraint plays exactly one.


| Role             | When applied                   | Purpose                                                  |
| ---------------- | ------------------------------ | -------------------------------------------------------- |
| Cheap filter     | Every proposal                 | Reject malformed / obviously-bad before expensive checks |
| Heavy validation | Survivors of cheap filters     | Go/no-go on a must-pass requirement                      |
| Ranker           | Pool of validated candidates   | Order surviving pool by task's ranking criterion         |
| Steering signal  | During gradient/cycling search | Bias generation toward objective                         |


## Deep-Dive Is Mandatory When

- The requirement is high-stakes: structure folding, binding, off-target search, novelty against a named database.
- The initial survey returned more than two or three plausible tool candidates.

A deep-dive produces: recommended primary tool, independent secondary tools for agreement, and ruled-out lookalikes.

## Oracle Agreement Guidance

- Must-pass high-stakes check: minimum two direct checks + one independent proxy.
- Cheap, unambiguous check (length, alphabet): one is fine.

