---
name: write-program
description: >
  Use this skill when the user asks to write, create, or modify a proto-language
  optimization program in Python. This covers composing Segments, Constructs,
  Generators, Constraints, Optimizers, and Programs to design biological sequences.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# write-program skill

## Before You Start

1. **Read example programs** to match the user's design goal:
   - Simple DNA: `examples/scripts/toy.py`
   - Multi-stage: `examples/scripts/toy-multiple-optimizers.py`
   - TopK exploration: `examples/scripts/topk_example.py`
   - Autoregressive DNA: `examples/scripts/evo2_example.py`
   - Multi-segment intron design: `examples/scripts/program_intron_design.py`
   - Protein structure design: `examples/scripts/program_symmetric_proteins.py`
   - Protein hunting (cycling): `examples/scripts/protein_hunter.py`
   - Beam search with KV caching: `examples/scripts/beam_search_kv_caching.py`
   - 40+ protein system designs: `examples/scripts/human_systems_claude/programs/`
2. **Check available components** via registries:
   - Constraints: `proto_language/language/constraint/__init__.py`
   - Generators: `proto_language/language/generator/__init__.py`
   - Optimizers: `proto_language/language/optimizer/__init__.py`
3. **Read JSON examples** for schema reference: `examples/jsons/`

## Program Structure (6 Steps)

Every program follows this exact flow:

```
1. Segments     →  Define design regions (fixed or variable)
2. Constructs   →  Combine segments into complete designs
3. Generators   →  Create + assign to segments
4. Constraints  →  Define scoring functions on segments
5. Optimizers   →  Wire constructs + generators + constraints
6. Program      →  Run sequential optimizer stages
```

## Complete Template

```python
from __future__ import annotations

from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.constraint import gc_content_constraint
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig


# === Step 1: Segments ===
# Variable region (generator will fill)
variable = Segment(length=100, sequence_type="dna")

# Fixed region (no generator needed)
# flank = Segment(sequence="ATCGATCG", sequence_type="dna")

# === Step 2: Constructs ===
construct = Construct([variable])

# === Step 3: Generators ===
gen = UniformMutationGenerator(
    UniformMutationGeneratorConfig(num_mutations=1)
)
gen.assign(variable)  # MUST assign to target segment

# === Step 4: Constraints ===
gc = Constraint(
    inputs=[variable],
    function=gc_content_constraint,
    function_config={"min_gc": 40, "max_gc": 60},  # Dict or config object
)

# === Step 5: Optimizer ===
optimizer = MCMCOptimizer(
    constructs=[construct],
    generators=[gen],
    constraints=[gc],
    config=MCMCOptimizerConfig(
        num_selected=1,
        num_steps=100,
        track_step_size=10,
    ),
)

# === Step 6: Program ===
program = Program(optimizers=[optimizer])
program.run()

# === Results ===
for seq in construct.joined_sequences:
    print(seq.sequence)
```

## Key Rules

### Construct Identity
All optimizers in a multi-stage program MUST share the **same construct objects by identity** (not copies). This is how state flows between stages:

```python
# CORRECT: Same construct object
construct = Construct([segment])
opt1 = TopKOptimizer(constructs=[construct], ...)
opt2 = MCMCOptimizer(constructs=[construct], ...)  # Same object

# WRONG: Different construct objects
opt1 = TopKOptimizer(constructs=[Construct([segment])], ...)
opt2 = MCMCOptimizer(constructs=[Construct([segment])], ...)  # Different!
```

### Generator-Segment Assignment
Every variable segment needs exactly one generator per optimizer stage. Call `gen.assign(segment)` before creating the optimizer.

### Fresh Generators/Constraints Per Stage
Generators and constraints **cannot be reused** across optimizer stages. Create new instances for each stage:

```python
# Stage 1
gen1 = UniformMutationGenerator(config1)
gen1.assign(segment)
constraint1 = Constraint(inputs=[segment], function=gc_content_constraint, ...)
opt1 = TopKOptimizer(constructs=[construct], generators=[gen1], constraints=[constraint1], ...)

# Stage 2 — new generator and constraint instances
gen2 = UniformMutationGenerator(config2)
gen2.assign(segment)  # Same segment, new generator
constraint2 = Constraint(inputs=[segment], function=gc_content_constraint, ...)
opt2 = MCMCOptimizer(constructs=[construct], generators=[gen2], constraints=[constraint2], ...)
```

### Constraint Config
`function_config` accepts either a dict or a Pydantic config object:

```python
# Dict (auto-parsed by Constraint)
Constraint(function_config={"min_gc": 40, "max_gc": 60}, ...)

# Config object (explicit)
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
Constraint(function_config=GCContentConfig(min_gc=40, max_gc=60), ...)
```

### Filter vs Scoring Constraints
```python
# Scoring constraint (default) — contributes to energy score
Constraint(inputs=[seg], function=fn, function_config=cfg)

# Weighted scoring — multiplies score by weight
Constraint(inputs=[seg], function=fn, function_config=cfg, weight=2.0)

# Filter constraint — binary pass/fail gate
Constraint(inputs=[seg], function=fn, function_config=cfg, threshold=0.1)
```

## Discovering Available Components

Do NOT rely on hardcoded lists — always discover dynamically before writing a program.

### Find constraints, generators, and optimizers via `__init__.py` files:

```
proto_language/language/constraint/__init__.py   # All registered constraints
proto_language/language/generator/__init__.py     # All registered generators
proto_language/language/optimizer/__init__.py     # All registered optimizers
```

Read these files to see what's available, their import paths, and naming.

### Find config options for a specific component:

Read the source file to see the config class and its `ConfigField` parameters:
- Constraints: `proto_language/language/constraint/{category}/{name}_constraint.py`
- Generators: `proto_language/language/generator/{name}_generator.py`
- Optimizers: `proto_language/language/optimizer/{name}_optimizer.py`

### Find the right component for a design goal:

Constraints are organized by category subdirectory:
- `sequence_composition/` — GC content, homopolymers, k-mers, length
- `protein_structure/` — pLDDT, pTM, RMSD, TM-score, symmetry, globularity, binding
- `protein_quality/` — complexity, repetitiveness, diversity, balanced amino acids
- `rna_secondary_structure/` — property/motif/feature/basepair similarity
- `rna_splicing/` — intron boundary, tissue specificity
- `sequence_annotation/` — sequence similarity, promoter strength, motifs

Generators are organized by category:
- **mutation** — refine existing sequences (UniformMutation, ESM2, MSA)
- **autoregressive** — generate left-to-right from prompts (Evo2, ProGen2)
- **inverse_folding** — design sequences conditioned on structure (ProteinMPNN, LigandMPNN)

## Common Patterns

### Multi-Stage: TopK → MCMC
Broad exploration then fine-tuning. See `examples/scripts/toy-multiple-optimizers.py`.

```python
segment = Segment(length=50, sequence_type="dna")
construct = Construct([segment])

# Stage 1: Explore broadly
gen1 = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=10))
gen1.assign(segment)
c1 = Constraint(inputs=[segment], function=gc_content_constraint,
                function_config={"min_gc": 50, "max_gc": 100})
opt1 = TopKOptimizer(constructs=[construct], generators=[gen1], constraints=[c1],
                     config=TopKOptimizerConfig(num_samples=100, k=5))

# Stage 2: Fine-tune
gen2 = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
gen2.assign(segment)
c2 = Constraint(inputs=[segment], function=gc_content_constraint,
                function_config={"min_gc": 70, "max_gc": 80})
opt2 = MCMCOptimizer(constructs=[construct], generators=[gen2], constraints=[c2],
                     config=MCMCOptimizerConfig(num_selected=1, num_steps=200))

program = Program(optimizers=[opt1, opt2])
program.run()
```

### Multi-Segment: Fixed Flanks + Variable Region
See `examples/scripts/program_intron_design.py`.

```python
left = Segment(sequence="ATCGATCG", sequence_type="dna", label="left_flank")
variable = Segment(length=100, sequence_type="dna", label="intron")
right = Segment(sequence="GCTAGCTA", sequence_type="dna", label="right_flank")
construct = Construct([left, variable, right])

gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
gen.assign(variable)  # Only variable region gets a generator

# Constraint can span multiple segments
constraint = Constraint(inputs=[left, variable, right], function=splice_fn, ...)
```

### Multi-Constraint Protein Design
See `examples/scripts/program_symmetric_proteins.py`.

```python
monomer = Segment(length=100, sequence_type="protein")
construct = Construct([monomer])

gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
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
    config=MCMCOptimizerConfig(num_selected=1, num_steps=5000),
)
program = Program(optimizers=[optimizer])
program.run()
```

### Incremental Stage Execution
```python
program = Program(optimizers=[opt1, opt2])

# Run stage 0 and inspect
program.run_stage(0)
results = program.get_stage_results(0)
print(f"Stage 0 best energy: {results['batch_results'][results['best_batch_idx']]['energy_score']}")

# Continue to stage 1
program.run_stage(1)
results = program.get_stage_results(1)
```

### Custom Logging
```python
from typing import Tuple

def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
    seq = outputs[0].candidate_sequences[0]
    gc = seq._metadata["constraints"]["gc_content_constraint"]["data"].get("gc_content", "N/A")
    print(f"Step {step} | seq: {seq.sequence[:30]}... | gc: {gc}")

optimizer = MCMCOptimizer(..., custom_logging=custom_logging)
```

### Export Results
```python
program.run()

# Export all results as JSON
program.export_program(format="json", path="results.json")

# Export specific segment as CSV
program.export_segment("construct_0", "segment_0", batch_idx=0, format="csv")

# Export optimization history
program.export_batch_history(batch_idx=0, format="csv", path="history.csv")
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
for batch in results["batch_results"]:
    print(f"Batch {batch['batch_idx']}: energy={batch['energy_score']:.4f}")

# Per-constraint metadata
seg = construct.segments[0]
for seq in seg.selected_sequences:
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
