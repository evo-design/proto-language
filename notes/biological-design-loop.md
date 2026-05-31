# Biological Design Loop

Use this note when writing automated biological design scripts. It is general guidance for sequence, structure, regulatory, and multi-part designs; task prompts remain the source of truth for exact thresholds and forbidden methods.

## Start With A Capable Prior

The first generator should already concentrate probability mass near the biology being requested. Random sequence, motif sprinkling, or composition matching can create diverse candidates, but they rarely create function by themselves. If a small raw sample has no plausible candidates after cheap checks, change the generator or add conditioning before scaling the run.

Prefer direct conditioning when it exists:

- Use scaffold-, backbone-, motif-, target-, or property-conditioned generation before free generation plus rejection.
- For target-conditioned protein-protein interfaces, prefer workflows designed for binder hallucination, antibody design, or complex design when they match the target and output contract. Consider Protein Hunter alongside BindCraft, Germinal, and RFdiffusion-family methods when the requested output is a new amino-acid binder or interacting protein chain, especially for all-X, partially specified, or contact-conditioned searches. These workflows optimize interface contacts, interface confidence, and complex geometry directly; use generic monomer hallucination or unconditional backbone generation only when a binder-specific workflow is unavailable or incompatible.
- For local refinement, use mutation proposals informed by homologs, language-model scores, structural context, or the property being optimized.
- For de novo work, prefer cycling, gradient, or generate-then-verify pipelines over naive best-of-K from a flat prior.
- Before composing low-level tools by hand, check whether a higher-level domain workflow in proto-tools already targets the requested design class. Prefer the workflow when its conditioning inputs and output scores cover the task contract; compose lower-level generators and validators when it cannot.

## Match Tools To The Biological Quantity

Choose validators and rankers by objective fit, not by familiarity from the nearest example script. For every must-pass learned property, ask what the tool was trained or designed to predict, what conditioning it accepts, and which submetrics it returns. Prefer tools whose outputs directly match the task's pass/fail quantities.

- For interaction design, distinguish protein-protein binder generation from small-molecule or ligand design. Binder hallucination tools are appropriate when the output is an amino-acid binder or complex; ligand-like "compound" objectives need chemistry-aware generation, docking, and ligand-specific scoring instead.
- For high-stakes learned properties, use at least two independent predictors or proxy families for final validation when feasible. Agreement is stronger evidence than a single convenient predictor with a tight cutoff. For structure work, consider the current local panel before defaulting to old examples: ESMFold2 is available for fast, high-accuracy all-atom structure and interaction prediction, including antibody complexes; Boltz-2 can provide structure and protein-ligand affinity, and AF2/AF3-family, Chai, Protenix, TM-align/US-align, Foldseek, pDockQ2, ipSAE, and DSSP each answer different validation questions when available.
- If several candidate tools measure the same quantity, run a small representative comparison on the same candidate set and keep the tool or combination that best separates plausible successes from failures.
- If a secondary predictor or workflow is skipped, record the concrete reason: incompatible input/output, failed smoke test, unavailable model, or measured runtime budget. Do not skip it only because it is absent from the closest example.
- Keep predictor submetrics separate through filtering and logging. A combined score may rank survivors, but it should not hide which biological quantity is starving the pool.

## Build A Large Pool, Then Narrow It

Most tasks need more internal candidates than the requested final count. The script should keep a scored table or equivalent metadata for the internal pool: candidate id, source generator, cheap-filter status, proxy scores, heavy validator scores, final-selection reason, and failure reason when rejected.

Order work by cost:

1. Deterministic checks: length, alphabet, fixed regions, stop codons, duplicate sequences, forbidden residues, simple repeat/homopolymer checks.
2. Fast biological proxies: sequence models, motif/scaffold checks, local alignments against small panels, simple structural or regulatory proxies.
3. Heavy validators: structure, binding, genome-scale search, expression, splicing, or other model-backed checks. Use task-matched current tools where available: SpliceAI or Pangolin for splice-site and variant effects, Boltz-2 affinity for small-molecule ligand binding, and ESMFold2/Boltz-2 for fast biomolecular complex structure and interaction prediction before costlier or complementary final oracles.
4. Final selection: require hard gates, then rank by task score or the closest available proxy. Preserve diversity among survivors.

## Treat Proxies As Imperfect

When the target scoring tool or most direct validator is unavailable, choose available tools that measure the same biological quantity. Do not claim that a proxy is identical to the target validator. Add safety margin and independent checks:

- Keep each hard-filter quantity separate. A composite or weighted proxy can rank survivors, but it must not replace per-requirement gates and failure logs.
- Use stricter thresholds than the target cutoff when calibration is unknown.
- Prefer agreement across independent predictors or scoring families.
- Avoid selecting candidates that barely pass a single proxy.
- If a model-prior, likelihood, naturalness, or other rank-only plausibility signal is strongly relevant to a hard filter, use it as a guard band or secondary agreement check instead of relying on one uncalibrated validator.
- Treat continuous validator outputs as optimization targets. For learned or context-dependent properties, deterministic syntax or motif checks are only cheap filters; final selection should push the relevant model submetrics in the required directions with margin. If a batch shares one failing submetric, revise the generator, constraints, or ranker rather than resubmitting the same construction pattern.
- When the exact target validator cannot be used in the inner loop, calibrate the closest available substitute before trusting it. Run a small representative sweep across construction choices, inspect each relevant submetric separately, and choose settings that improve the hardest metric with margin. If no proxy measures a decisive submetric, keep that gap visible and require stronger independent mechanistic evidence instead of finalizing from motif heuristics alone.
- For context-dependent mechanisms such as splicing, expression, localization, binding, or folding, require redundant biological evidence rather than a single motif or "designed by construction" check. For splicing tasks, separate donor/acceptor probability, variant delta scores, Pangolin splice-site usage/P(splice), junction or retention proxies, and expression effects instead of replacing them with a GT-AG motif check.
- Record any requirement that cannot be directly or equivalently checked.

Model priors, likelihoods, and "designed by construction" arguments are useful rankers, not validity checks for explicit requirements.

## Match Optimizer To Feedback

Rejection sampling works only when the generator already produces a measurable fraction of candidates near the target property. If a first batch has little or no yield, do not scale the same generator/filter stack indefinitely. Change the search strategy:

- Use cycling or generate-then-verify loops when a predictor can create conditioning for the next generation round.
- Use MCMC or simulated annealing for local refinement when mutations can be scored by a decomposed objective.
- Use gradient or logit optimization when the generator and at least one relevant constraint are differentiable.
- Use beam or prefix search when partial sequences can be scored before full completion.
- Use a higher-level domain workflow when it directly optimizes the requested design class.

When using MCMC or another feedback optimizer, include the failing submetrics explicitly in the accept/reject or ranking objective. Do not optimize only a generic combined score if a required submetric is repeatedly failing. When pilot runs or target-validator feedback show concentrated failure in one or two submetrics, revise the proxy objective, generator conditioning, or proposal distribution around those submetrics before scaling the same search; more samples from an objective that ignores the failing quantity are unlikely to recover the batch.

## Verify Plan-to-Code Conformance

Before declaring a script ready, trace every final-plan heavy validator, named comparison-set query, and ranker into executable code that runs before the requested output is written. A plan that promises a decisive learned or model-backed validator—splicing, regulatory, homology/genome-scale search, structure prediction, docking, or similar—is not satisfied by deterministic motif filters alone. If a planned validator cannot be implemented, update the plan and fail loudly rather than silently shipping candidates that never passed the promised validation panel.

## Smoke-Test Before Expensive Runs

Before launching a full search, run tiny checks that exercise real code paths. First run a cheap compile/import check on the final script in the same environment that will execute it, so syntax errors, stale imports, renamed local APIs, and missing requirements are fixed before GPU time is requested. If a delegated review or helper process is available, use it to inspect the generated script for runnable-code defects and plan-to-code mismatches; if delegation fails, do a dedicated self-review and keep moving toward a runnable script rather than treating the delegation failure as a biological or implementation failure:

- `python -m py_compile` or equivalent on the final script, plus a tiny import smoke test for local proto-language/proto-tools objects used by the script.
- One generator call with the intended conditioning and output length.
- One optimizer step or a tiny proposal batch.
- One call to each decisive external tool with representative input.
- One parser pass for every format that will be filtered on: FASTA, TSV, BED, PDB, mmCIF, JSON, BLAST/MMseqs output, or model metrics.
- One output write/read validation against the requested final format.

For command-line tools, test the exact argument order and a small real-format input. A data conversion step that fails on a real BED or structure file should be fixed before any full candidate search starts.

Do not drop the strongest available validator only because its optional packages are absent from the authoring venv. Proto-tools can create tool-specific runtime environments, and submissions can declare requirements. Rule out a decisive available validator only after a concrete end-to-end smoke test or installation failure, and record the failed command, error, and whether the failure occurred in tool launch, output extraction, or downstream parsing. If the tool launches but returns no consumable artifact, fix the configuration or switch to an equivalent available validator that measures the same biological quantity before abandoning the run.

## Manage Time Without Lowering Standards

Write the requested output only after unique candidates have passed deterministic checks and the strongest validation panel currently available, then overwrite it only as better final-equivalent candidates pass the validation panel. Never pad a draft or final file with duplicates, relabeled records, or placeholder sequences to meet the requested count. A draft protects against timeouts, but it should not be populated by unvalidated placeholders.s

If the candidate count undershoots:

- Sample more if budget remains, and implement this as an explicit loop in the script rather than a single fixed batch.
- Keep valid partial survivors and continue from them.
- Log which stage is starving the pool.
- For high-rejection validators, minimally adjust the thresholds or replenish the upstream candidate pool in bounded rounds until the target count is reached or an explicit proposal, wall-clock, or compute cap is hit. Size the cap from the available execution budget so the script uses the budget intentionally instead of exiting after an arbitrary first undershoot.
- Revisit the generator or validation plan if no valid candidates appear.

Never lower a must-pass threshold to fill the requested count. A smaller validated set plus a loud failure is better than a full file of candidates that are low-quality.