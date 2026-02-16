#!/usr/bin/env python3
"""
Diversification program for: b2AR to TF Pathway - b2AR to TF Pathway
Category: VI. Signaling Pathways

This script:
1. Loads wildtype sequences for all genes in this row
2. Runs MCMC-based diversification with structure constraints
3. Scores each complex with AlphaFold3
4. Saves results to the output directory

To customize:
- Add constraints in the `add_custom_constraints` function
- Modify hyperparameters in the config or below
"""

import gc
import os
import sys
from datetime import datetime
from typing import Dict, List

import torch
from Bio.Seq import Seq

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import gene_ids_to_program, load_config, score_complexes_in_program_with_af3
from vi_signaling_pathways__b2ar_to_tf_pathway__creb_dna import (
    DESIGN_SEQ_LENGTH as CREB_DNA_LENGTH,
)
from vi_signaling_pathways__b2ar_to_tf_pathway__creb_dna import (
    generate_creb_dna_sequence,
)

from proto_language.language.constraint import structure_ensemble_rmsd_constraint
from proto_language.language.core import Constraint, Construct, Segment
from proto_language.utils import inverse_sigmoid_score

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

    # See program additions below.

    return constraints


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # =============================================================================
    # Design the CREB DNA sequence.
    # =============================================================================

    # Note: Design this first, as it is separate from the other parts of the file,
    # and Flashzoi has problems with GPU CUDA context corruption.

    creb_dna = generate_creb_dna_sequence()
    assert len(creb_dna) == CREB_DNA_LENGTH

    # Clean up the CUDA context just in case.
    gc.collect()
    torch.cuda.empty_cache()

    start = (CREB_DNA_LENGTH // 2) - 25
    end = (CREB_DNA_LENGTH // 2) + 25
    creb_dna_small = creb_dna[start:end]
    creb_dna_small_revcomp = str(Seq(creb_dna_small).reverse_complement())

    # =============================================================================
    # Proceed with the regular protein design program.
    # =============================================================================

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_path = os.path.join(project_dir, 'configs', 'vi_signaling_pathways__b2ar_to_tf_pathway__b2ar_to_tf_pathway.json')
    output_dir = os.path.join(project_dir, 'outputs', 'vi_signaling_pathways__b2ar_to_tf_pathway__b2ar_to_tf_pathway')

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
    n_steps = 5

    print(f"\nBuilding diversification program for {len(gene_ids)} genes...")
    program, gene_id_to_segment = gene_ids_to_program(
        gene_ids=gene_ids,
        n_steps_per_generator=n_steps,
        custom_constraints_fn=add_custom_constraints,
    )

    print(f"\nRunning program ({n_steps * len(gene_ids)} total MCMC steps)...")
    program.run()

    # =============================================================================
    # Score monomers based on BioEmu-predicted ensemble.
    # =============================================================================

    run_dir = os.environ.get('RUN_OUTPUT_DIR')
    if run_dir is None:
        # Fallback for local runs (not via SLURM)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(output_dir, f'run_{run_timestamp}')
        os.makedirs(run_dir, exist_ok=True)

    bioemu_dir_prefix = str(os.path.join(run_dir, 'bioemu_outputs'))

    rmsd_score_gnas_inactive = Constraint(
        inputs=[program.constructs[1].segments[0]],
        function=structure_ensemble_rmsd_constraint,
        function_config={
            'target_structure': 'examples/data/pdb_cache/6au6.pdb',
            'target_chain_id': 'A',
            'target_residue_range': (85, 394),
            'candidate_residue_range': (85, 394),
            'bioemu_config': {
                'num_samples': 3000,
                'output_dir': bioemu_dir_prefix + '_gnas',
                'batch_size': 100,
            },
            'rmsd_aggregation': 'min',
            'inflection_point_angstroms': 3.,
            'sigmoid_slope': 3.,
            'verbose': True,
        },
        label='ensemble_rmsd_gnas_inactive',
    ).evaluate()[0]
    rmsd_gnas_inactive = inverse_sigmoid_score(rmsd_score_gnas_inactive, 3., 3.)

    print(f"Min RMSD = {rmsd_gnas_inactive} against inactive Galpha-s (PDB 6AU6)")

    rmsd_score_gnas_exchange = Constraint(
        inputs=[program.constructs[1].segments[0]],
        function=structure_ensemble_rmsd_constraint,
        function_config={
            'target_structure': 'examples/data/pdb_cache/3sn6.pdb',
            'target_chain_id': 'A',
            'target_residue_range': (85, 394),
            'candidate_residue_range': (85, 394),
            'bioemu_config': {
                'num_samples': 3000,
                'output_dir': bioemu_dir_prefix + '_gnas',
                'batch_size': 100,
            },
            'rmsd_aggregation': 'min',
            'inflection_point_angstroms': 3.,
            'sigmoid_slope': 3.,
            'verbose': True,
        },
        label='ensemble_rmsd_gnas_exchange',
    ).evaluate()[0]
    rmsd_gnas_exchange = inverse_sigmoid_score(rmsd_score_gnas_exchange, 3., 3.)

    print(f"Min RMSD = {rmsd_gnas_exchange} against Galpha-s in exchange state (PDB 3SN6)")

    rmsd_score_prkar1a_homodimer = Constraint(
        inputs=[program.constructs[6].segments[0]],
        function=structure_ensemble_rmsd_constraint,
        function_config={
            'target_structure': 'examples/data/pdb_cache/1rl3.pdb',
            'target_chain_id': 'A',
            'target_residue_range': (119, 379),
            'candidate_residue_range': (119, 379),
            'bioemu_config': {
                'num_samples': 1000,
                'output_dir': bioemu_dir_prefix + '_prkar1a',
                'batch_size': 100,
            },
            'rmsd_aggregation': 'min',
            'inflection_point_angstroms': 3.,
            'sigmoid_slope': 3.,
            'verbose': True,
        },
        label='ensemble_rmsd_prkar1a_homodimer',
    ).evaluate()[0]
    rmsd_prkar1a_homodimer = inverse_sigmoid_score(rmsd_score_prkar1a_homodimer, 3., 3.)

    print(f"Min RMSD = {rmsd_prkar1a_homodimer} against homodimer PKA-R (PDB 1RL3)")

    rmsd_score_prkar1a_tetramer = Constraint(
        inputs=[program.constructs[6].segments[0]],
        function=structure_ensemble_rmsd_constraint,
        function_config={
            'target_structure': 'examples/data/pdb_cache/2qcs.pdb',
            'target_chain_id': 'B',
            'target_residue_range': (119, 379),
            'candidate_residue_range': (119, 379),
            'bioemu_config': {
                'num_samples': 1000,
                'output_dir': bioemu_dir_prefix + '_prkar1a',
                'batch_size': 100,
            },
            'rmsd_aggregation': 'min',
            'inflection_point_angstroms': 3.,
            'sigmoid_slope': 3.,
            'verbose': True,
        },
        label='ensemble_rmsd_prkar1a_tetramer',
    ).evaluate()[0]
    rmsd_prkar1a_tetramer = inverse_sigmoid_score(rmsd_score_prkar1a_tetramer, 3., 3.)

    print(f"Min RMSD = {rmsd_prkar1a_tetramer} against tetrameric PKA-R (PDB 2QCS)")

    # =============================================================================
    # Add DNA and ligands, then score with AF3.
    # =============================================================================

    # Add epinephrine to b2AR structure (index 0).
    gene_ids.append('L_epinephrine')
    config['complexes'][0]['gene_ids'].append('L_epinephrine')
    config['complexes'][0]['stoichiometry']['L_epinephrine'] = 1
    program.constructs.append(
        Construct([
            Segment(
                sequence='CNC[C@@H](c1ccc(c(c1)O)O)O',
                sequence_type='ligand',
            )
        ])
    )

    # Add ATP to adenylyl cyclase structure (index 3).
    gene_ids.append('ATP')
    config['complexes'][3]['gene_ids'].append('ATP')
    config['complexes'][3]['stoichiometry']['ATP'] = 1
    program.constructs.append(
        Construct([
            Segment(
                sequence='c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N',
                sequence_type='ligand',
            )
        ])
    )

    # Add KIX domain of CREB binding protein (CBP) (index 4).
    gene_ids.append('CREBBP_KIX')
    config['complexes'][5]['gene_ids'].append('CREBBP_KIX')
    config['complexes'][5]['stoichiometry']['CREBBP_KIX'] = 2  # There are two CREBs, so 2 CBP KIXs.
    program.constructs.append(
        Construct([
            Segment(
                sequence='GVRKGWHEHVTQDLRSHLVHKLVQAIFPTPDPAALKDRRMENLVAYAKKVEGDMYESANSRDEYYHLLAEKIYKIQKELE',
                sequence_type='protein',
            )
        ])
    )

    # Add (generated) DNA binding site of CREB (index 4).
    gene_ids += [ 'CREB_TF_motif1', 'CREB_TF_motif2' ]
    config['complexes'][5]['gene_ids'] += [ 'CREB_TF_motif1', 'CREB_TF_motif2' ]
    config['complexes'][5]['stoichiometry']['CREB_TF_motif1'] = 1
    config['complexes'][5]['stoichiometry']['CREB_TF_motif2'] = 1
    program.constructs.append(
        Construct([
            Segment(
                sequence=creb_dna_small,
                sequence_type='dna',
            )
        ])
    )
    program.constructs.append(
        Construct([
            Segment(
                sequence=creb_dna_small_revcomp,
                sequence_type='dna',
            )
        ])
    )

    # Score complexes with AF3.
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
