---
name: write-program
description: >
  Composes proto-language optimization programs in Python. Covers Segments,
  Constructs, Generators, Constraints, Optimizers, and Programs for designing
  biological sequences. Use when writing programs, composing optimization
  pipelines, designing DNA/protein/RNA sequences, or setting up multi-stage
  optimization with constraints like GC content, structure prediction, or
  protein quality.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# write-program skill

## Before You Start

1. **Browse example programs** to match the user's design goal:
   - Python scripts: `examples/scripts/` (DNA, RNA, protein, multi-chain designs)
   - JSON programs: `examples/jsons/` (declarative format, run via `examples/scripts/run_program.py`)
   - Read the example closest to the user's design goal before writing a new program.
2. **Discover available components** using the registry API (see Component Discovery below) to find constraints, generators, and optimizers and inspect their config schemas.

## Program Structure (6 Steps)

Every program follows this exact flow:

```
1. Segments     ->  Define design regions (fixed or variable)
2. Constructs   ->  Combine segments into complete designs
3. Generators   ->  Create + assign to segments
4. Constraints  ->  Define scoring functions on segments
5. Optimizers   ->  Wire constructs + generators + constraints
6. Program      ->  Run sequential optimizer stages
```

## Complete Template

```python
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.constraint import gc_content_constraint
from proto_language.language.generator import (
    MaskingStrategy,
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
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
gen = RandomNucleotideGenerator(
    RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
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
        num_results=1,
        num_steps=100,
    ),
)

# === Step 6: Program ===
# num_results sets the default number of output sequences for all optimizers.
# Individual optimizers can override via their config's num_results field.
program = Program(optimizers=[optimizer], num_results=1)
program.run()

# === Results ===
for seq in construct.joined_sequences:
    print(seq.sequence)
```

## Key Rules

### Construct Identity
All optimizers in a multi-stage program MUST share the **same construct objects by identity** (not copies). This is how state flows between stages.

### Generator-Segment Assignment
Every variable segment needs exactly one generator per optimizer stage. Call `gen.assign(segment)` before creating the optimizer.

### Fresh Generators/Constraints Per Stage
Generators and constraints **cannot be reused** across optimizer stages. Create new instances for each stage.

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

### Find config options for a specific component:

Read the source file to see the config class and its `ConfigField` parameters:
- Constraints: `proto_language/language/constraint/{category}/{name}_constraint.py`
- Generators: `proto_language/language/generator/{name}_generator.py`
- Optimizers: `proto_language/language/optimizer/{name}_optimizer.py`

### Component categories:

Constraints: `sequence_composition/` (GC content, homopolymers, k-mers, length), `protein_structure/` (pLDDT, pTM, RMSD, TM-score, symmetry, globularity, binding), `protein_quality/` (complexity, repetitiveness, diversity, balanced amino acids), `rna_secondary_structure/` (property/motif/feature/basepair similarity), `rna_splicing/` (intron boundary, tissue specificity), `sequence_annotation/` (sequence similarity, promoter strength, motifs), `sequence_alignment/` (sequence alignment similarity scoring)

Generators: **mutation** (RandomNucleotide, RandomProtein, ESM2, MSA), **autoregressive** (Evo2, ProGen2), **inverse_folding** (ProteinMPNN, LigandMPNN)

## Common Patterns

For detailed examples of common patterns, use the `Read` tool to load:
- **Patterns reference**: `.claude/skills/write-program/PATTERNS.md`

Covers: multi-stage (Rejection Sampling -> MCMC), program-level num_results, multi-segment (fixed flanks + variable region), multi-constraint protein design, incremental execution, custom logging, export results, and accessing results.

## Validation Checklist

Copy this and check off as you go:

- [ ] All variable segments have a generator assigned (`gen.assign(segment)`)
- [ ] Multi-stage programs share the same construct objects by identity
- [ ] Fresh generators and constraints created for each optimizer stage
- [ ] Constraint configs use valid parameters (check source for `ConfigField` options)
- [ ] Program `num_results` set or each optimizer config has `num_results`
- [ ] Program runs without error: `python3 my_program.py`
- [ ] Results accessible via `construct.joined_sequences`

If any check fails, fix before proceeding.
