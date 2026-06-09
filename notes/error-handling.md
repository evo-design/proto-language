# Error Handling

Rules for raising vs returning worst-score outputs inside `Constraint.evaluate(...)`, `Generator.sample(...)`, `Optimizer.run()`, and helpers they call.

## Default: Raise

Raise on framework, configuration, and backend failures. Examples:

- A required file, model, binary, weight directory, target structure, or config value is missing or invalid.
- A tool call crashes with CUDA OOM, import failure, timeout, malformed output, failed target preparation, or an empty reference prediction.
- A generator cannot produce valid proposals for the configured segment.
- An optimizer reaches an inconsistent internal state.

These failures are usually deterministic for the current config. Returning `MAX_ENERGY` for every proposal makes a broken run look like a real result; raising gives the user the actionable failure immediately.

## Allowed Soft Failures

Soft-fail only when the failure is local to one proposal and the remaining proposals in the same batch are still meaningful.

Canonical shape:

```python
for proposal_pair in input_sequences:
    try:
        score = run_mafft_align(...)
    except Exception as exc:
        logger.warning("gap-gini: alignment failed for pair (...): %s", exc)
        results.append(
            ConstraintOutput(
                score=MAX_ENERGY,
                metadata={"gap_gini_error": str(exc)},
            )
        )
        continue
    results.append(ConstraintOutput(score=score, metadata={...}))
```

Use soft failure for proposal-local failures such as:

- one item in a batched external alignment or structure-comparison call fails, while the other items succeeded;
- a proposal is missing the metric needed for one score, but the backend call itself completed;
- a proposal-local biological precondition is not met, such as a sequence being too short for the requested k-mer or having no resolvable ORF for a DNA-to-protein score.

Raise instead when the shared setup, target preparation, model invocation, or batch-level tool call fails. Those failures invalidate the whole evaluation call, not just one proposal.

Soft failures must include metadata that names the reason. Use a stable key such as `<constraint>_error` and keep successful metadata shape unchanged.

## Config Errors

Config construction errors raise.

- Constraint and generator registry `.create()` helpers catch Pydantic `ValidationError` and reformat via `format_pydantic_error()` in `proto_language/utils/serialization.py`: `"<component> '<key>' config invalid — <field>: <msg> [got=<value>]; ..."`. Multi-field errors are joined with `; `. The constraint registry applies the same reformatting to the optional `backward_config_dict`, with prefix `"constraint '<key>' backward config invalid"`.
- Function-entry hard checks, such as mutually exclusive fields, missing target assets, empty required lists, or invalid paths, should raise `ValueError` or `RuntimeError` naming the bad value and operation.

## Programming Invariants

Raise for impossible states and programmer bugs. Examples:

- Returned constraint result count does not match evaluated proposals.
- A scoring constraint returns booleans or a filter returns invalid result shapes.
- A `ConstraintOutput` has an out-of-range score, wrong type, or wrong structures/logits arity.
- A proposal that passed all filters has `NaN` energy.
- A logit-consuming generator receives a proposal with no logits.
- A generator or constraint is unassigned, duplicated where forbidden, or points at incompatible segments.

Soft-failing these cases hides real framework bugs.

## Tool-Side Policy

proto-tools also raises by default. `PROTO_CAPTURE_ERRORS=1` switches tool wrapper failures into `success=False` outputs, but language code should not depend on capture mode unless it explicitly opts into that contract.

`MissingAssetError` always raises in proto-tools, regardless of `PROTO_CAPTURE_ERRORS`, so unprovisioned gated weights or large databases can be reported or skipped by the caller.

See `proto-tools/notes/error-handling.md` for the full tool-side contract.

## Style

- Use `logging.getLogger(__name__)`; do not use `print()` or `warnings.warn()` in framework code.
- Keep error messages one line.
- Name the operation, component/tool, and failing value when available.
- Add a short fix hint when it is obvious.
- When soft-failing, log at warning level and include the failure in `ConstraintOutput.metadata`.

## Key Files

| File | Role |
|---|---|
| `proto_language/core/constraint.py` | Result validation, mask handling, metadata writes |
| `proto_language/core/optimizer.py` | Filter-then-score, energy invariants, filter penalties |
| `proto_language/core/generator.py` | Generator assignment and proposal validation |
| `proto_language/utils/serialization.py` | `format_pydantic_error()` |
| `proto_language/utils/scoring.py` | `MAX_ENERGY`, `MIN_ENERGY` constants (re-exported from `proto_language.utils`) |
