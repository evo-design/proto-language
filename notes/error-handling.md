# Error Handling

Long-form rules for raising vs soft-failing inside `Constraint.evaluate(...)`, `Generator.sample(...)`, `Optimizer.run()`, and their helpers. CLAUDE.md carries the one-line summary; this file is the canonical reference.

## Default: raise

Inside `constraint.evaluate(...)`, `generator.sample(...)`, `Optimizer.run()`, and any helper they call, raise on failure. The earlier "soft-fail to preserve compute" approach was wrong for the user: when a tool call crashes (CUDA OOM, missing binary, target prep returned None, reference folding returned empty), the failure is almost always **deterministic** for the current config — soft-failing produces 100 iterations of all-`MAX_ENERGY` garbage that looks like a real result. Raising surfaces the actual error immediately so the user fixes the config and reruns; lost iterations of progress are cheap to redo, ambiguous results are not.

## The one exception — per-proposal failure inside a `for proposal in batch:` loop

If MAFFT can align 31 of 32 sequence pairs but fails on the 32nd, the other 31 are useful. The bad item should soft-fail without killing the batch:

```python
for proposal_pair in input_sequences:
    try:
        score = run_mafft_align(...)
    except Exception as e:
        logger.warning("gap-gini: alignment failed for pair (...): %s", e)
        results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"gap_gini_error": str(e)}))
        continue
    results.append(ConstraintOutput(score=score, ...))
```

This is the only place soft-fail belongs. Canonical: `gap_gini_constraint.py`. Other examples: `structure_ensemble_similarity` per-sequence, `structure_confidence` per-proposal missing-metric, `specific_kmer` sequence-too-short, `gyration_radius` no-metric, per-DNA-proposal sites in `protein_globularity` and `protein_symmetry_ring` (where ORFipy may find no canonical ATG-to-stop ORF, or ESMFold may fail for the selected longest CDS).

## Config-construction-time errors raise too

With reformatted messages:

- **Pydantic `ValidationError` at `Registry.create()`**: caught and reformatted via `format_pydantic_error()` (in `proto_language/utils/helpers.py`) → `ValueError("<type> '<key>' config invalid — <field>: <msg> [got=<value>]")`. Optimizer config validation lives in `Optimizer.__init__`, using the same helper.
- **Function-entry hard config checks** (file existence, mutually-exclusive options, list-of-required-fields-empty): raise `ValueError`/`RuntimeError` naming the bad value.

## Programming-bug invariants raise

E.g. `"Inconsistent state: N energy_scores for M proposals"`, `"Proposal has no logits"`, `"NaN where impossible"`. Soft-fail would mask a real bug.

## Style

One line per error, name the operation / tool / failing value, and (when natural) a one-clause fix hint. Use `logging.getLogger(__name__)` not `print()` or `warnings.warn()`.

## Files

| File | Role |
|---|---|
| `proto_language/utils/helpers.py` | `format_pydantic_error()` — registry-create error formatter |
| `proto_language/utils/__init__.py` | `MAX_ENERGY`, `MIN_ENERGY` constants used in soft-fail scores |

For tool-side raise-vs-capture policy (`PROTO_CAPTURE_ERRORS`, `MissingAssetError` carve-out), see `proto-tools/notes/error-handling.md`.
