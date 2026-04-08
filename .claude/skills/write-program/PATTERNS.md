# Common Program Patterns

Detailed examples of common program patterns. Load this file on demand when composing programs.

## Construct Identity (CORRECT vs WRONG)

All optimizers in a multi-stage program MUST share the **same construct objects by identity** (not copies). This is how state flows between stages:

```python
# CORRECT: Same construct object
construct = Construct([segment])
opt1 = RejectionSamplingOptimizer(constructs=[construct], ...)
opt2 = MCMCOptimizer(constructs=[construct], ...)  # Same object

# WRONG: Different construct objects — state won't flow between stages!
opt1 = RejectionSamplingOptimizer(constructs=[Construct([segment])], ...)
opt2 = MCMCOptimizer(constructs=[Construct([segment])], ...)  # Different!
```

## Multi-Stage: Rejection Sampling -> MCMC

Broad exploration then fine-tuning. See `examples/scripts/toy-multiple-optimizers.py`.

```python
segment = Segment(length=50, sequence_type="dna")
construct = Construct([segment])

# Stage 1: Explore broadly
gen1 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=10)))
gen1.assign(segment)
c1 = Constraint(inputs=[segment], function=gc_content_constraint,
                function_config={"min_gc": 50, "max_gc": 100})
opt1 = RejectionSamplingOptimizer(constructs=[construct], generators=[gen1], constraints=[c1],
                     config=RejectionSamplingOptimizerConfig(num_samples=100, num_results=5))

# Stage 2: Fine-tune
gen2 = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
gen2.assign(segment)
c2 = Constraint(inputs=[segment], function=gc_content_constraint,
                function_config={"min_gc": 70, "max_gc": 80})
opt2 = MCMCOptimizer(constructs=[construct], generators=[gen2], constraints=[c2],
                     config=MCMCOptimizerConfig(num_results=1, num_steps=200))

program = Program(optimizers=[opt1, opt2], num_results=5)
program.run()
```

## Program-Level `num_results`

Set `num_results` on Program to provide a default for all optimizers. Each optimizer resolves its result count as: **config num_results > program num_results > error**.

```python
# All optimizers default to 5 results unless overridden
program = Program(optimizers=[opt1, opt2], num_results=5)

# opt1 uses config.num_results=20 (overrides program default, logs warning)
# opt2 uses num_results=5 (from program default, since config.num_results is None)
```

## Multi-Segment: Fixed Flanks + Variable Region

See `examples/scripts/program_intron_design.py`.

```python
left = Segment(sequence="ATCGATCG", sequence_type="dna", label="left_flank")
variable = Segment(length=100, sequence_type="dna", label="intron")
right = Segment(sequence="GCTAGCTA", sequence_type="dna", label="right_flank")
construct = Construct([left, variable, right])

gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
gen.assign(variable)  # Only variable region gets a generator

# Constraint can span multiple segments
constraint = Constraint(inputs=[left, variable, right], function=splice_fn, ...)
```

## Multi-Constraint Protein Design

See `examples/scripts/program_symmetric_proteins.py`.

```python
monomer = Segment(length=100, sequence_type="protein")
construct = Construct([monomer])

gen = RandomProteinGenerator(RandomProteinGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
gen.assign(monomer)

plddt = Constraint(inputs=[monomer], function=structure_plddt_constraint,
                   function_config={"structure_tool": "esmfold", ...})
ptm = Constraint(inputs=[monomer], function=structure_ptm_constraint,
                 function_config={"structure_tool": "esmfold", ...})
symmetry = Constraint(inputs=[monomer], function=protein_symmetry_ring_constraint,
                      function_config={"n_symmetric_units": 3, ...})

optimizer = MCMCOptimizer(
    constructs=[construct],
    generators=[gen],
    constraints=[plddt, ptm, symmetry],
    config=MCMCOptimizerConfig(num_results=1, num_steps=5000),
)
program = Program(optimizers=[optimizer], num_results=1)
program.run()
```

## Incremental Stage Execution

```python
program = Program(optimizers=[opt1, opt2], num_results=5)

# Run stage 0 and inspect
program.run_stage(0)
results = program.get_stage_results(0)
print(f"Stage 0 best energy: {results['results'][results['best_result_idx']]['energy_score']}")

# Continue to stage 1
program.run_stage(1)
results = program.get_stage_results(1)
```

## Custom Logging

```python
def custom_logging(step: int, outputs: tuple[Segment]) -> None:
    seq = outputs[0].proposal_sequences[0]
    gc = seq._metadata["constraints"]["gc_content_constraint"]["data"].get("gc_content", "N/A")
    print(f"Step {step} | seq: {seq.sequence[:30]}... | gc: {gc}")

optimizer = MCMCOptimizer(..., custom_logging=custom_logging)
```

## Export Results

```python
program.run()

# Export all 4 tables at once (sequences, constraints, constructs, optimization)
program.export(path="./results/", format="csv")

# Export a single table
program.export(path="sequences.json", table="sequences", format="json")

# Get a DataFrame for analysis
df = program.to_dataframe(table="sequences")

# FASTA output for bioinformatics pipelines
fasta_str = program.to_fasta()
program.to_fasta(path="results.fasta")

# Stage-specific export (multi-optimizer programs)
program.export(path="stage0.csv", table="sequences", stage=0)
```

Optimizer instances also support the same export methods (without `stage`):
```python
optimizer.export(path="./results/", format="csv")
df = optimizer.to_dataframe(table="sequences")
fasta = optimizer.to_fasta()
```

## Accessing Results

```python
program.run()

# Final sequences
for construct in program.constructs:
    for seq in construct.joined_sequences:
        print(seq.sequence)

# Energy scores from final optimizer
print(program.energy_scores)

# Structured results per stage
results = program.get_stage_results(stage_index=0)
for result in results["results"]:
    print(f"Result {result['result_idx']}: energy={result['energy_score']:.4f}")

# Per-constraint metadata
seg = construct.segments[0]
for seq in seg.result_sequences:
    constraints = seq._metadata.get("constraints", {})
    for name, data in constraints.items():
        print(f"  {name}: score={data['score']:.4f}")

# Optimization history
for snapshot in program.optimizers[0].history:
    print(f"Step {snapshot['time_step']}: scores={snapshot['energy_scores']}")
```

## Running Programs

```bash
# Run directly
python3 examples/scripts/toy.py

# Run with arguments (if script supports them)
python3 examples/scripts/program_symmetric_proteins.py \
    --monomer-length 100 --n-symmetric-units 3 --n-steps 10000

# Run from JSON via parser
python3 examples/scripts/run_program.py
```
