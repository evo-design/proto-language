# Batching Architecture

Batching is split across proposal pools, generators, constraint calls, compiled scorers, and proto-tools backends. Do not assume a single `batch_size` knob controls every expensive operation.

## Ownership Layers

1. **Optimizers choose proposal-pool size.** This is how many proposal rows exist in `Segment.proposal_sequences` for a step or internal batch.
2. **Generators may expose `batch_size`.** GPU-backed generators pass it to their proto-tools sampler; CPU/logit generators usually inherit the base default.
3. **Constraints receive masked proposal tuples.** `Constraint.evaluate()` calls each constraint function once with all proposals that passed the current mask.
4. **Compiled scorers can group compatible constraints.** Additive scoring routes through the constraint compiler, which may collapse several public constraints into one backend call.
5. **Tools own backend memory strategy.** Sequence count is only one dimension; structures, chains, ligands, recycles, MSA depth, and sample count often matter more.

## Optimizer Proposal Batching

`Generator` defines the shared fallback:

```python
class Generator(ABC):
    batch_size: int = 1
```

Current framework-level consumers:

- **`BeamSearchOptimizer`** copies `generator.batch_size` to `self.batch_size` and chunks proposal generation for each beam.
- **`RejectionSamplingOptimizer`** uses `proposal_batch_size` for internal proposal batches. If unset, it infers the largest positive `batch_size` found on generators and on top-level `constraint.function_config` / `constraint.backward_config`, capped at `num_samples`.

This is outer proposal batching. It does not guarantee that every downstream GPU call is chunked the same way.

## Generator Batching

These generators expose a user-facing `batch_size` config field, defaulting to `1`, and pass it to proto-tools:

- `ESM2Generator`, `ESM3Generator`, `Evo1Generator`, `Evo2Generator`, and `ProGen2Generator` pass `batch_size` to the language-model sampling tool.
- `ProteinMPNNGenerator` and `LigandMPNNGenerator` pass `batch_size` to inverse folding configs. With one input structure, they request `num_proposals` sequences and let the tool chunk by `batch_size`; with one structure per proposal, they force `num_sequences_per_structure=1` and `batch_size=1`.

CPU and logit-materialization generators such as `RandomNucleotideGenerator`, `RandomProteinGenerator`, `MSAGenerator`, `SemigreedyMutationGenerator`, and `PositionWeightGenerator` do not expose a config-level batching knob. They inherit `Generator.batch_size = 1`; their `_sample()` implementations loop over the current proposal pool.

## Constraint Evaluation

Individual `Constraint` objects do not have a framework-level `batch_size` attribute. `Constraint.evaluate()` resolves the mask, builds one tuple per proposal, and calls the scoring function once:

```python
input_sequences = [tuple(seg.proposal_sequences[idx] for seg in self._inputs) for idx in indices_to_evaluate]
results = self._function(input_sequences, config=self._function_config)
```

The constraint function receives every mask-passing proposal for that call. It may forward them to one tool call, split internally, or loop per proposal.

Some constraint configs include tool-level batching fields. Current examples include `ESM2PerplexityConfig.batch_size`, `MalinoisActivityConfig.batch_size`, `EnformerChromatinAccessibilityMorseConfig.batch_size`, and `BorzoiChromatinAccessibilityMorseConfig.batch_size`. Those fields tune the tool call and may also influence `RejectionSamplingOptimizer`'s inferred outer `proposal_batch_size`. Nested config fields, such as `BioEmuConfig.batch_size` inside `StructureEnsembleSimilarityConfig`, are tool parameters only and are not read by the rejection-sampling inference helper.

## Filter Then Score

`Optimizer.score_energy()` evaluates threshold constraints before scoring constraints:

1. Each filter receives the current `passed` mask.
2. Failed proposals are marked with the rejecting constraint label.
3. Later filters and scoring constraints only evaluate proposals that still pass.
4. Rejected proposals receive the optimizer's filter penalty.

For additive scoring, scorers go through the `evaluate_scoring_constraints()` helper in `proto_language.optimizer.constraint_compiler`. Providers currently include ESMFold, Protenix, Malinois, and AlphaFold2 binder, allowing related constraints to share a backend prediction or gradient call when their configs are compatible.

## Tool-Level Patterns

Treat these as patterns, not a registry. The source of truth is the selected generator, constraint config, and proto-tools runner.

- **Sequence model batches:** language-model samplers and some sequence scorers expose direct `batch_size` knobs.
- **Per-structure inverse folding:** MPNN-style tools loop over input structures and chunk requested designs per structure.
- **Residue-budget batching:** ESMFold-style tools batch by linked residue count rather than proposal count.
- **Length-scaled sampling:** tools such as BioEmu expose a `batch_size`, but effective memory use depends heavily on sequence length and sample count.
- **Sequential complex prediction:** diffusion and structure-prediction tools often process one complex at a time and tune memory with recycles, MSA depth, diffusion samples, or low-memory flags.
- **Bulk search:** tools such as MMseqs2 submit all queries together and expose resource knobs like threads, split, or GPU mode rather than proposal `batch_size`.

## Key Code Paths

- `proto_language/core/generator.py`
- `proto_language/core/constraint.py`
- `proto_language/core/optimizer.py`
- `proto_language/optimizer/beam_search_optimizer.py`
- `proto_language/optimizer/rejection_sampling_optimizer.py`
- `proto_language/optimizer/constraint_compiler/`
- The selected generator or constraint module plus its corresponding `proto_tools` input/config/runner.
