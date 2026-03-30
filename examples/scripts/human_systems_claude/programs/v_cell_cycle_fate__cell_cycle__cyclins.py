#!/usr/bin/env python3
"""
Diversification program for: Cell Cycle - Cyclins
Category: V. Cell Cycle & Fate

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
    gene_ids_to_program,
    load_config,
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
            function_config={'position': 100, 'allowed_residues': ['K', 'R']},
        ))

        # Cross-protein interaction constraint
        constraints.append(Constraint(
            inputs=[gene_id_to_segment['ORC1'], gene_id_to_segment['ORC2']],
            function=interaction_constraint,
            function_config={'min_contacts': 10},
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
    config_path = os.path.join(project_dir, 'configs', 'v_cell_cycle_fate__cell_cycle__cyclins.json')
    output_dir = os.path.join(project_dir, 'outputs', 'v_cell_cycle_fate__cell_cycle__cyclins')

    # Load configuration
    config = load_config(config_path)

    print("=" * 70)
    print(f"Row: {config['pathway']} - {config['component']}")
    print(f"Category: {config['category']}")
    print(f"Genes: {config['all_gene_ids']}")
    print(f"Complexes: {[c['complex_id'] for c in config['complexes']]}")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build and run program
    gene_ids = config['all_gene_ids']
    n_steps = config.get('n_steps_per_generator', 3)

    print(f"\nBuilding diversification program for {len(gene_ids)} genes...")
    program, gene_id_to_segment = gene_ids_to_program(
        gene_ids=gene_ids,
        n_steps_per_generator=n_steps,
        custom_constraints_fn=add_custom_constraints,
    )

    print(f"\nRunning program ({n_steps * len(gene_ids)} total MCMC steps)...")
    program.run()

    # Score complexes with AF3
    print(f"\nScoring {len(config['complexes'])} complexes with AlphaFold3...")
    results = score_complexes_in_program_with_af3(
        program=program,
        gene_ids=gene_ids,
        complexes=config['complexes'],
        output_dir=output_dir,
    )

    print(f"\nDone! Results saved to: {output_dir}")

    # Return exit code based on success
    if results['summary']['failed'] > 0:
        print(f"WARNING: {results['summary']['failed']} complex(es) had errors")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
