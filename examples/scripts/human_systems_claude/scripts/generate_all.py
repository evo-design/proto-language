#!/usr/bin/env python3
"""
Metascript to generate configuration files and Python programs
for all complexes in the human biology spreadsheet.

Usage:
    python generate_all.py --excel path/to/spreadsheet.xlsx --output-dir ./

This will create:
    - configs/*.json: Configuration files for each row
    - programs/*.py: Python scripts for each row
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import stoichiometry directly to avoid loading heavy dependencies
from lib.stoichiometry import get_stoichiometry

# Marker used to detect manual modifications
# This is the exact content of the default constraint function body (without leading indent on first line)
DEFAULT_CONSTRAINT_BODY = """constraints = []

    # TODO: Add custom constraints here

    return constraints"""


def script_has_custom_modifications(script_path: str) -> bool:
    """
    Check if a generated script has been manually modified.

    Detects modifications by checking if the add_custom_constraints function
    has been changed from its default template.

    Args:
        script_path: Path to the Python script

    Returns:
        True if the script appears to have custom modifications
    """
    if not os.path.exists(script_path):
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # Check if the default constraint body is still present
    # If it's not, the user has modified the function
    if DEFAULT_CONSTRAINT_BODY not in content:
        return True

    return False


def sanitize_filename(s: str) -> str:
    """Convert a string to a valid filename component."""
    # Replace spaces and special chars with underscores
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s-]+', '_', s)
    s = s.strip('_').lower()
    return s


def parse_complex_string(complex_str: str) -> List[Dict[str, Any]]:
    """
    Parse a complex definition string into structured data.

    Example input:
        "COMPLEX::ORC_core::ORC1,ORC2,ORC3; HOMO_FAMILY::PCNA_trimer::PCNA"

    Returns:
        List of dicts, each with:
        - complex_id: str (e.g., "COMPLEX::ORC_core")
        - complex_type: str (e.g., "COMPLEX")
        - complex_name: str (e.g., "ORC_core")
        - gene_ids: List[str]
    """
    if pd.isna(complex_str) or not complex_str.strip():
        return []

    complexes = []

    # Split by semicolon (multiple complexes per cell)
    for part in complex_str.split(';'):
        part = part.strip()
        if not part:
            continue

        # Parse TYPE::NAME::GENES format
        tokens = part.split('::')
        if len(tokens) < 3:
            print(f"WARNING: Could not parse complex string: {part}")
            continue

        complex_type = tokens[0].strip()
        complex_name = tokens[1].strip()
        genes_str = tokens[2].strip()

        # Parse gene list
        gene_ids = [g.strip() for g in genes_str.split(',') if g.strip()]

        complex_id = f"{complex_type}::{complex_name}"

        complexes.append({
            'complex_id': complex_id,
            'complex_type': complex_type,
            'complex_name': complex_name,
            'gene_ids': gene_ids,
        })

    return complexes


def parse_pdb_string(pdb_str: str) -> List[str]:
    """
    Parse PDB ID string into a list of IDs.

    Handles formats like:
        "5UJM, 7JPS"
        "6VVO (RCSB PDB)"
        "9B8S (RCSB PDB), 5VBN"
    """
    if pd.isna(pdb_str) or not pdb_str.strip():
        return []

    pdb_ids = []

    # Split by comma
    for part in pdb_str.split(','):
        part = part.strip()
        if not part:
            continue

        # Remove annotations like "(RCSB PDB)"
        part = re.sub(r'\s*\([^)]*\)\s*', '', part)
        part = part.strip()

        # Validate PDB ID format (4 alphanumeric characters)
        if re.match(r'^[A-Za-z0-9]{4}$', part):
            pdb_ids.append(part.upper())
        else:
            print(f"WARNING: Invalid PDB ID format: {part}")

    return pdb_ids


def build_pdb_mapping(df_pdb: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a mapping from complex_id to list of PDB IDs.
    """
    mapping = {}

    for _, row in df_pdb.iterrows():
        complex_label = row['Complex label (yours)']
        pdb_str = row['Representative multimer PDB(s)']

        if pd.isna(complex_label):
            continue

        # Clean up complex label
        complex_label = complex_label.strip()

        # Skip annotation rows (those starting with parentheses)
        if complex_label.startswith('('):
            continue

        pdb_ids = parse_pdb_string(pdb_str)

        if pdb_ids:
            mapping[complex_label] = pdb_ids

    return mapping


def generate_row_config(
    row_idx: int,
    row: pd.Series,
    pdb_mapping: Dict[str, List[str]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate configuration for a single row.

    Returns:
        Tuple of (filename_base, config_dict)
    """
    # Extract row metadata
    category = row['Category'] if pd.notna(row['Category']) else ''
    pathway = row['Pathway / Complex'] if pd.notna(row['Pathway / Complex']) else ''
    component = row['Component / Role'] if pd.notna(row['Component / Role']) else ''
    complexes_str = row['Complexes'] if pd.notna(row['Complexes']) else ''

    # Build filename
    parts = []
    if category:
        parts.append(sanitize_filename(category))
    if pathway:
        parts.append(sanitize_filename(pathway))
    if component:
        parts.append(sanitize_filename(component))

    filename_base = '__'.join(parts) if parts else f'row_{row_idx:02d}'

    # Parse complexes
    complexes = parse_complex_string(complexes_str)

    # Gather all gene IDs
    all_gene_ids = []
    seen_genes = set()
    for comp in complexes:
        for gene in comp['gene_ids']:
            if gene not in seen_genes:
                all_gene_ids.append(gene)
                seen_genes.add(gene)

    # Enrich each complex with stoichiometry and PDB IDs
    enriched_complexes = []
    for comp in complexes:
        stoichiometry, inferred = get_stoichiometry(
            comp['complex_id'], comp['gene_ids']
        )

        pdb_ids = pdb_mapping.get(comp['complex_id'], [])

        enriched_complexes.append({
            'complex_id': comp['complex_id'],
            'complex_type': comp['complex_type'],
            'complex_name': comp['complex_name'],
            'gene_ids': comp['gene_ids'],
            'stoichiometry': stoichiometry,
            'stoichiometry_inferred': inferred,
            'pdb_ids': pdb_ids,
        })

    config = {
        'row_index': row_idx,
        'category': category,
        'pathway': pathway,
        'component': component,
        'filename_base': filename_base,
        'complexes': enriched_complexes,
        'all_gene_ids': all_gene_ids,
        'n_steps_per_generator': 3,
    }

    return filename_base, config


def generate_program_script(config: Dict[str, Any]) -> str:
    """
    Generate Python script content for a row configuration.
    """
    filename_base = config['filename_base']
    category = config['category']
    pathway = config['pathway']
    component = config['component']

    script = f'''#!/usr/bin/env python3
"""
Diversification program for: {pathway} - {component}
Category: {category}

This script:
1. Loads wildtype sequences for all genes in this row
2. Runs MCMC-based diversification with structure constraints
3. Scores each complex with AlphaFold3
4. Saves results to the output directory

To customize:
- Add constraints in the `add_custom_constraints` function
- Modify hyperparameters in the config or below
"""

import os
import sys
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import (
    load_config,
    gene_ids_to_program,
    score_complexes_in_program_with_af3,
)
from proto_language.language.core import Constraint, Segment


# =============================================================================
# CUSTOMIZE HERE: Add row-specific constraints
# =============================================================================

def add_custom_constraints(
    gene_id_to_segment: Dict[str, Segment],
) -> List[Constraint]:
    """
    Add custom constraints for this specific system.

    This function is called during program construction. You can add
    constraints that:
    - Enforce specific residue conservation
    - Add cross-protein interaction constraints
    - Include custom scoring functions

    Args:
        gene_id_to_segment: Mapping from gene ID to its Segment object

    Returns:
        List of Constraint objects to add to the program

    Examples:
        # Conservation constraint for a specific residue
        from proto_language.language.constraint import residue_constraint
        constraints.append(Constraint(
            inputs=[gene_id_to_segment['ORC1']],
            function=residue_constraint,
            function_config={{'position': 100, 'allowed_residues': ['K', 'R']}},
        ))

        # Cross-protein interaction constraint
        constraints.append(Constraint(
            inputs=[gene_id_to_segment['ORC1'], gene_id_to_segment['ORC2']],
            function=interaction_constraint,
            function_config={{'min_contacts': 10}},
        ))
    """
    constraints = []

    # TODO: Add custom constraints here

    return constraints


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_path = os.path.join(project_dir, 'configs', '{filename_base}.json')
    output_dir = os.path.join(project_dir, 'outputs', '{filename_base}')

    # Load configuration
    config = load_config(config_path)

    print("=" * 70)
    print(f"Row: {{config['pathway']}} - {{config['component']}}")
    print(f"Category: {{config['category']}}")
    print(f"Genes: {{config['all_gene_ids']}}")
    print(f"Complexes: {{[c['complex_id'] for c in config['complexes']]}}")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build and run program
    gene_ids = config['all_gene_ids']
    n_steps = config.get('n_steps_per_generator', 3)

    print(f"\\nBuilding diversification program for {{len(gene_ids)}} genes...")
    program, gene_id_to_segment = gene_ids_to_program(
        gene_ids=gene_ids,
        n_steps_per_generator=n_steps,
        custom_constraints_fn=add_custom_constraints,
    )

    print(f"\\nRunning program ({{n_steps * len(gene_ids)}} total MCMC steps)...")
    program.run()

    # Score complexes with AF3
    print(f"\\nScoring {{len(config['complexes'])}} complexes with AlphaFold3...")
    results = score_complexes_in_program_with_af3(
        program=program,
        gene_ids=gene_ids,
        complexes=config['complexes'],
        output_dir=output_dir,
    )

    print(f"\\nDone! Results saved to: {{output_dir}}")

    # Return exit code based on success
    if results['summary']['failed'] > 0:
        print(f"WARNING: {{results['summary']['failed']}} complex(es) had errors")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''

    return script


def main():
    parser = argparse.ArgumentParser(
        description='Generate configs and programs for human complex diversification'
    )
    parser.add_argument(
        '--excel', '-e',
        required=True,
        help='Path to the Excel spreadsheet'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Output directory for configs and programs (default: current directory)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Print what would be generated without writing files'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite program scripts even if they have custom modifications'
    )

    args = parser.parse_args()

    # Read Excel file
    print(f"Reading Excel file: {args.excel}")
    xlsx = pd.ExcelFile(args.excel)

    df_complexes = pd.read_excel(xlsx, sheet_name='Complexes')
    df_pdb = pd.read_excel(xlsx, sheet_name='PDB IDs')

    print(f"Found {len(df_complexes)} rows in Complexes sheet")
    print(f"Found {len(df_pdb)} rows in PDB IDs sheet")

    # Build PDB mapping
    pdb_mapping = build_pdb_mapping(df_pdb)
    print(f"Built PDB mapping for {len(pdb_mapping)} complexes")

    # Create output directories
    configs_dir = os.path.join(args.output_dir, 'configs')
    programs_dir = os.path.join(args.output_dir, 'programs')

    if not args.dry_run:
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(programs_dir, exist_ok=True)

    # Forward-fill the Category column (it's merged in Excel)
    df_complexes['Category'] = df_complexes['Category'].ffill()

    # Generate configs and programs for each row
    generated = []
    skipped_modified = []

    for row_idx, row in df_complexes.iterrows():
        # Skip rows without complexes
        if pd.isna(row['Complexes']) or not row['Complexes'].strip():
            print(f"Skipping row {row_idx}: no complexes defined")
            continue

        filename_base, config = generate_row_config(row_idx, row, pdb_mapping)

        # Skip if no genes
        if not config['all_gene_ids']:
            print(f"Skipping row {row_idx} ({filename_base}): no genes parsed")
            continue

        config_path = os.path.join(configs_dir, f'{filename_base}.json')
        program_path = os.path.join(programs_dir, f'{filename_base}.py')

        # Check if program has been manually modified
        has_modifications = script_has_custom_modifications(program_path)

        print(f"\nRow {row_idx}: {filename_base}")
        print(f"  Complexes: {len(config['complexes'])}")
        print(f"  Genes: {len(config['all_gene_ids'])}")
        print(f"  Config: {config_path}")
        print(f"  Program: {program_path}")

        if has_modifications and not args.force:
            print(f"  ⚠️  SKIPPING: Program has custom modifications (use --force to overwrite)")
            skipped_modified.append(filename_base)
            # Still add to generated list for manifest, but mark as skipped
            generated.append({
                'filename_base': filename_base,
                'config_path': config_path,
                'program_path': program_path,
                'n_genes': len(config['all_gene_ids']),
                'n_complexes': len(config['complexes']),
                'has_custom_modifications': True,
                'script_regenerated': False,
            })
            # Still update the config (it's safe to overwrite)
            if not args.dry_run:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            continue

        if not args.dry_run:
            # Write config (always safe to update)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Write program
            script_content = generate_program_script(config)
            with open(program_path, 'w') as f:
                f.write(script_content)
            os.chmod(program_path, 0o755)

        generated.append({
            'filename_base': filename_base,
            'config_path': config_path,
            'program_path': program_path,
            'n_genes': len(config['all_gene_ids']),
            'n_complexes': len(config['complexes']),
            'has_custom_modifications': False,
            'script_regenerated': True,
        })

    # Write manifest
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    if not args.dry_run:
        with open(manifest_path, 'w') as f:
            json.dump({
                'generated_at': pd.Timestamp.now().isoformat(),
                'source_excel': os.path.abspath(args.excel),
                'total_programs': len(generated),
                'programs': generated,
            }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print('='*60)
    print(f"Total programs: {len(generated)}")
    print(f"  - Regenerated: {sum(1 for g in generated if g.get('script_regenerated', True))}")
    print(f"  - Skipped (custom modifications): {len(skipped_modified)}")
    print(f"Total genes across all programs: {sum(g['n_genes'] for g in generated)}")
    print(f"Total complexes across all programs: {sum(g['n_complexes'] for g in generated)}")

    if skipped_modified:
        print(f"\nPrograms with custom modifications (not overwritten):")
        for name in skipped_modified:
            print(f"  - {name}")
        print(f"\nUse --force to overwrite these files.")

    if not args.dry_run:
        print(f"\nManifest written to: {manifest_path}")
    else:
        print("\n(dry run - no files written)")


if __name__ == '__main__':
    main()
