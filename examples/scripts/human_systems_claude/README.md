# Human Complex Diversification Pipeline

This pipeline generates diversified protein sequences for human complexes and scores them with AlphaFold3.

## Directory Structure

```
human_systems_claude/
├── configs/                    # Generated JSON configurations (one per row)
├── programs/                   # Generated Python scripts (one per row)
├── outputs/                    # Results from each program run
│   └── {program_name}/
│       └── run_{timestamp}/    # Each run gets its own timestamped directory
│           ├── af3_outputs/    # AlphaFold3 output files
│           ├── results.json    # Scoring results
│           └── final_sequences.fasta
├── lib/                        # Shared library code
│   ├── __init__.py
│   ├── base_program.py         # Core utilities
│   └── stoichiometry.py        # Stoichiometry lookup
├── scripts/
│   └── generate_all.py         # Metascript to generate configs + programs
├── slurm/
│   ├── job_template.sbatch     # SLURM job template
│   └── submit_all.sh           # Batch submission script
└── manifest.json               # Generated manifest of all programs
```

## Quick Start

### 1. Generate configs and programs

```bash
cd ~/proto-language

python examples/scripts/human_systems_claude/scripts/generate_all.py \
    --excel /path/to/Major_pathways_and_complexes_in_human_biology.xlsx \
    --output-dir examples/scripts/human_systems_claude
```

This creates:
- `configs/*.json`: Configuration for each row in the spreadsheet
- `programs/*.py`: Executable Python script for each row
- `manifest.json`: Index of all generated programs

**Note**: If you've added custom constraints to a program, it will NOT be overwritten. Use `--force` to override this protection.

### 2. Run locally (single program)

```bash
cd ~/proto-language

python examples/scripts/human_systems_claude/programs/i_genetic_info_processing__dna_replication__origin_recognition.py
```

### 3. Submit to SLURM cluster

All commands should be run from the `proto-language` root directory:

```bash
cd ~/proto-language

# Submit all jobs
./examples/scripts/human_systems_claude/slurm/submit_all.sh

# Dry run (see what would be submitted)
./examples/scripts/human_systems_claude/slurm/submit_all.sh --dry-run

# Submit a single job by index
./examples/scripts/human_systems_claude/slurm/submit_all.sh --single 0

# Submit a range of jobs
./examples/scripts/human_systems_claude/slurm/submit_all.sh --range 0-10
```

## Multiple Design Runs

Each program run creates a timestamped subdirectory in the outputs folder:
```
outputs/i_genetic_info_processing__dna_replication__origin_recognition/
├── run_20240115_143022/
│   ├── af3_outputs/
│   ├── program.log
│   ├── results.json
│   └── final_sequences.fasta
├── run_20240116_091530/
│   ├── af3_outputs/
│   ├── program.log
│   ├── results.json
│   └── final_sequences.fasta
└── ...
```

This allows you to run the same program multiple times without overwriting previous results.

## Regenerating Programs (Preserving Custom Modifications)

When you call `generate_all.py` again:
- **Config files** are always regenerated (safe to update)
- **Program files** with custom constraints are **NOT overwritten**

To see which programs have modifications:
```bash
python scripts/generate_all.py --excel ... --output-dir ... --dry-run
```

To force regeneration of all programs:
```bash
python scripts/generate_all.py --excel ... --output-dir ... --force
```

## Configuration Format

Each JSON config contains:

```json
{
  "row_index": 0,
  "category": "I. Genetic Info Processing",
  "pathway": "DNA Replication",
  "component": "Origin Recognition",
  "filename_base": "i_genetic_info_processing__dna_replication__origin_recognition",
  "complexes": [
    {
      "complex_id": "COMPLEX::ORC_core",
      "complex_type": "COMPLEX",
      "complex_name": "ORC_core",
      "gene_ids": ["ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6", "CDC6"],
      "stoichiometry": {"ORC1": 1, "ORC2": 1, ...},
      "stoichiometry_inferred": false,
      "pdb_ids": ["5UJM", "7JPS"]
    }
  ],
  "all_gene_ids": ["ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6", "CDC6"],
  "n_steps_per_generator": 15
}
```

## Customizing Programs

Each generated program has a `add_custom_constraints()` function where you can add row-specific constraints:

```python
def add_custom_constraints(gene_id_to_segment: Dict[str, Segment]) -> List[Constraint]:
    constraints = []

    # Example: conservation constraint
    from proto_language.language.constraint import residue_constraint
    constraints.append(Constraint(
        inputs=[gene_id_to_segment['ORC1']],
        function=residue_constraint,
        function_config={'position': 100, 'allowed_residues': ['K', 'R']},
    ))

    # Example: cross-protein interaction
    constraints.append(Constraint(
        inputs=[gene_id_to_segment['ORC1'], gene_id_to_segment['ORC2']],
        function=interaction_constraint,
        function_config={'min_contacts': 10},
    ))

    return constraints
```

Once you add custom constraints, the program will not be overwritten by `generate_all.py` (unless you use `--force`).

## Stoichiometry

Stoichiometry is inferred from:
1. **Known structures**: Hardcoded for well-characterized complexes (e.g., ATP synthase F1: α3β3γδε)
2. **Naming conventions**: `_trimer` → 3 copies, `_tetramer` → 4 copies, etc.
3. **Defaults**: 1 copy per gene for heteromeric complexes

Check `lib/stoichiometry.py` to see or modify the inference logic.

## Output Format

Each run produces a `results.json` with:

```json
{
  "run_timestamp": "20240115_143022",
  "run_dir": "outputs/.../run_20240115_143022",
  "gene_ids": ["ORC1", "ORC2", ...],
  "final_sequences": {
    "ORC1": "MAEPRQ...",
    "ORC2": "MSTVKL..."
  },
  "complex_scores": [
    {
      "complex_id": "COMPLEX::ORC_core",
      "af3_confidence": {
        "plddt": 0.85,
        "ptm": 0.72,
        "iptm": 0.68,
        "pae": 8.5
      },
      "pdb_comparisons": {
        "7JPS": {
          "tmscore": 0.75,
          "rmsd": 3.2
        }
      },
      "errors": []
    }
  ],
  "summary": {
    "total_complexes": 1,
    "successful": 1,
    "failed": 0
  }
}
```

## SLURM Resource Defaults

- CPUs: 64
- GPUs: 1
- Memory: 256G
- Time: 48 hours

Modify `slurm/job_template.sbatch` to adjust resources.

## Dependencies

- proto_language
- BioPython
- pandas
- PyMOL (for PDB retrieval)
- AlphaFold3 (via proto_language)
- ESM3 / ESMFold (via proto_language)
