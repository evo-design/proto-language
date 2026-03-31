"""
Base program utilities for human complex diversification.

This module provides shared functionality for:
- Loading wildtype sequences from gene IDs
- Building diversification programs
- Scoring complexes with AlphaFold3
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pymol

pymol.finish_launching(['pymol', '-qc'])
from Bio import SeqIO
from proto_tools.tools.masked_models.masking import MaskingStrategy
from pymol import cmd

from proto_language.language.constraint import (
    overall_protein_quality_constraint,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
    structure_rmsd_constraint,
    structure_tmscore_constraint,
)
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
)
from proto_language.language.generator import ESM3Generator, ESM3GeneratorConfig
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.utils import inverse_sigmoid_score

from .stoichiometry import expand_gene_ids_by_stoichiometry

# Constants for RMSD scoring
INFLECTION_POINT_ANGSTROMS = 5.0  # Corresponds to 0.5 after sigmoid
SIGMOID_SLOPE = 3.0

# Default paths (can be overridden)
DEFAULT_HUMAN_GENES_TSV = 'examples/data/human_genes.tsv'
DEFAULT_HUMAN_GENES_FASTA = 'examples/data/human_genes.fasta'
DEFAULT_PDB_CACHE_DIR = 'examples/data/pdb_cache'


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_results(output_dir: str, results: Dict[str, Any]) -> None:
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def load_wildtype_seqs(
    gene_ids: List[str],
    genes_tsv: str = DEFAULT_HUMAN_GENES_TSV,
    genes_fasta: str = DEFAULT_HUMAN_GENES_FASTA,
) -> List[str]:
    """
    Map gene IDs (HGNC symbols) to protein sequences.

    Args:
        gene_ids: List of HGNC gene symbols (e.g., ["ORC1", "ORC2"])
        genes_tsv: Path to TSV mapping gene IDs to UniProt IDs
        genes_fasta: Path to FASTA file with UniProt sequences

    Returns:
        List of protein sequences in same order as gene_ids
    """
    # Gene ID to UniProt ID mapping
    df = pd.read_csv(genes_tsv, sep='\t')
    gene_id_to_uniprot_id = {
        row['From']: row['Entry'] for _, row in df.iterrows()
    }

    missing_genes = [g for g in gene_ids if g not in gene_id_to_uniprot_id]
    if missing_genes:
        raise ValueError(f"Gene IDs not found in mapping: {missing_genes}")

    uniprot_ids = [gene_id_to_uniprot_id[gene_id] for gene_id in gene_ids]

    # UniProt ID to sequence mapping
    uniprot_id_to_seq: Dict[str, str] = {}
    for record in SeqIO.parse(genes_fasta, 'fasta'):
        uniprot_id = record.id.split('|')[1]
        seq = str(record.seq)
        uniprot_id_to_seq[uniprot_id] = seq

    missing_uniprot = [u for u in uniprot_ids if u not in uniprot_id_to_seq]
    if missing_uniprot:
        raise ValueError(f"UniProt IDs not found in FASTA: {missing_uniprot}")

    return [uniprot_id_to_seq[uniprot_id] for uniprot_id in uniprot_ids]


def get_remote_pdb_contents(
    pdb_id: str,
    cache_dir: str = DEFAULT_PDB_CACHE_DIR,
    verbose: bool = False,
) -> str:
    """
    Retrieve PDB file contents. Checks local cache first; if missing,
    fetches from RCSB via PyMOL and caches it.

    Args:
        pdb_id: The 4-character PDB ID (e.g., '7JPS')
        cache_dir: Directory to store cached PDB files

    Returns:
        The full text contents of the PDB file
    """
    pdb_id = pdb_id.lower()

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    filename = f'{pdb_id}.pdb'
    file_path = os.path.join(cache_dir, filename)

    # Check cache
    if os.path.exists(file_path):
        if verbose:
            print(f'Found {pdb_id} in cache: {file_path}')
        with open(file_path, 'r') as f:
            return f.read()

    # Fetch from RCSB
    if verbose:
        print(f'Fetching {pdb_id} from RCSB...')

    cmd.reinitialize()
    result = cmd.fetch(pdb_id, path=cache_dir, type='pdb')

    if result == -1:
        raise ValueError(
            f'Failed to fetch PDB ID: {pdb_id}. Check ID or internet connection.'
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'PyMOL fetched {pdb_id}, but file was not found at {file_path}'
        )

    with open(file_path, 'r') as f:
        return f.read()


def gene_ids_to_program(
    gene_ids: List[str],
    n_steps_per_generator: int = 20,
    genes_tsv: str = DEFAULT_HUMAN_GENES_TSV,
    genes_fasta: str = DEFAULT_HUMAN_GENES_FASTA,
    custom_constraints_fn: Optional[callable] = None,
) -> Tuple[Program, Dict[str, Segment]]:
    """
    Create a diversification program for a list of gene IDs.

    Maps gene IDs to native human sequences, then uses sequence-based
    diversification and structure-based guidance.

    Args:
        gene_ids: List of HGNC gene symbols
        n_steps_per_generator: MCMC steps per generator
        genes_tsv: Path to gene ID mapping file
        genes_fasta: Path to sequence FASTA file
        custom_constraints_fn: Optional function(segments_dict) -> List[Constraint]
            to add custom constraints

    Returns:
        Tuple of:
        - Program: The assembled program (not yet run)
        - Dict[str, Segment]: Mapping from gene_id to its Segment
    """
    sequences = load_wildtype_seqs(gene_ids, genes_tsv, genes_fasta)

    constructs, generators, constraints = [], [], []
    gene_id_to_segment: Dict[str, Segment] = {}

    for gene_id, seq in zip(gene_ids, sequences):
        print(f'Adding {gene_id} to the program (length={len(seq)})...')

        # Create segment and generator
        protein_segment = Segment(
            sequence=seq,
            sequence_type='protein',
        )
        gene_id_to_segment[gene_id] = protein_segment

        esm3_config = ESM3GeneratorConfig(
            model_checkpoint='esm3_sm_open_v1',
            temperature=0.3,
            masking_strategy=MaskingStrategy(num_mutations=int(0.25 * len(seq))),
        )
        protein_generator = ESM3Generator(esm3_config)
        protein_generator.assign(protein_segment)
        generators.append(protein_generator)

        protein_construct = Construct([protein_segment])
        constructs.append(protein_construct)

        # Per-protein constraints.

        structure_plddt = Constraint(
            inputs=[protein_segment],
            function=structure_plddt_constraint,
            function_config={'structure_tool': 'esmfold'},
            label='esmfold_plddt',
        )
        constraints.append(structure_plddt)

        structure_ptm = Constraint(
            inputs=[protein_segment],
            function=structure_ptm_constraint,
            function_config={'structure_tool': 'esmfold'},
            label='esmfold_ptm',
        )
        constraints.append(structure_ptm)

        target_plddt = 1. - Constraint(
            inputs=[Segment(sequence=seq, sequence_type='protein')],
            function=structure_plddt_constraint,
            function_config={'structure_tool': 'esmfold'},
            label='esmfold_target_plddt',
        ).evaluate()[0]
        tmscore_threshold = 0.3  # Corresponds to TMscore of 0.7.
        min_target_plddt = 0.6
        structure_tmscore = Constraint(
            inputs=[protein_segment],
            function=structure_tmscore_constraint,
            function_config={
                'target_chains': (seq,),
                'structure_tool': 'esmfold',
                'min_target_plddt': min_target_plddt,
            },
            threshold=(
                # Only set threshold if this constraint is evaluated.
                tmscore_threshold if target_plddt >= min_target_plddt else None
            ),
            label='esmfold_tmscore_similarity',
        )
        constraints.append(structure_tmscore)

        structure_rmsd = Constraint(
            inputs=[protein_segment],
            function=structure_rmsd_constraint,
            function_config={
                'target_chains': (seq,),
                'structure_tool': 'esmfold',
                'min_target_plddt': 0.6,
                'inflection_point_angstroms': INFLECTION_POINT_ANGSTROMS,
                'sigmoid_slope': SIGMOID_SLOPE,
            },
            label='esmfold_rmsd_similarity',
        )
        constraints.append(structure_rmsd)

        protein_quality_config = {
            'quality_threshold': 0.,
            'enable_length': False,
            'enable_complexity': True,
            'complexity_max_low_complexity': 0.2,
            'enable_repetitiveness': True,
            'repetitiveness_max_repetitiveness': 0.1,
            'repetitiveness_min_repeat_length': 1,
            'enable_diversity': False,
            'enable_balanced_aas': False,
        }
        native_protein_quality_score = Constraint(
            inputs=[Segment(sequence=seq, sequence_type='protein')],
            function=overall_protein_quality_constraint,
            function_config={'protein_quality_config': protein_quality_config},
        ).evaluate()[0]
        protein_quality_threshold = max(0.2, native_protein_quality_score + 0.1)
        protein_quality_threshold = min(protein_quality_threshold, 1.)
        protein_quality = Constraint(
            inputs=[protein_segment],
            function=overall_protein_quality_constraint,
            function_config={'protein_quality_config': protein_quality_config},
            threshold=protein_quality_threshold,
            label='base_protein_quality',
        )
        constraints.append(protein_quality)

    # Add custom constraints if provided
    if custom_constraints_fn is not None:
        custom_constraints = custom_constraints_fn(gene_id_to_segment)
        constraints.extend(custom_constraints)
        print(f'Added {len(custom_constraints)} custom constraints')

    # Configure optimizer
    mcmc_optimizer_config = MCMCOptimizerConfig(
        num_results=1,
        proposals_per_result=1,
        num_steps=len(generators) * n_steps_per_generator,
        max_temperature=0.1,
        min_temperature=0.01,
        track_step_size=1,
        verbose=True,
    )

    def _custom_logging(step: int, outputs: Tuple[Segment]) -> None:
        """Print intermediate sequences."""
        assert len(outputs) == len(gene_ids)
        for gene_id, output in zip(gene_ids, outputs):
            print(f'\t{gene_id}: {str(output.result_sequences[0])}')

    optimizer = MCMCOptimizer(
        constructs=constructs,
        generators=generators,
        constraints=constraints,
        config=mcmc_optimizer_config,
        custom_logging=_custom_logging,
        clear_tool_cache=4 * 1024 * 1024 * 1024,  # 4 GB
    )

    program = Program(optimizers=[optimizer], num_results=1)

    return program, gene_id_to_segment


def score_single_complex_with_af3(
    gene_id_to_segment: Dict[str, Segment],
    complex_info: Dict[str, Any],
    af3_dir: str,
    pdb_cache_dir: str = DEFAULT_PDB_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Score a single complex with AlphaFold3.

    Args:
        gene_id_to_segment: Mapping from gene_id to final Segment
        complex_info: Dict with keys:
            - complex_id: str
            - gene_ids: List[str]
            - stoichiometry: Dict[str, int]
            - pdb_ids: List[str] or None
        af3_dir: Directory for AF3 outputs
        pdb_cache_dir: Directory for cached PDB files

    Returns:
        Dict with scoring results
    """
    complex_id = complex_info['complex_id']
    gene_ids = complex_info['gene_ids']
    stoichiometry = complex_info['stoichiometry']
    pdb_ids = complex_info.get('pdb_ids') or []

    print(f"\n{'='*60}")
    print(f"Scoring complex: {complex_id}")
    print(f"Genes: {gene_ids}")
    print(f"Stoichiometry: {stoichiometry}")
    print(f"PDB IDs: {pdb_ids}")
    print('='*60)

    # Expand gene IDs by stoichiometry
    expanded_gene_ids = expand_gene_ids_by_stoichiometry(gene_ids, stoichiometry)

    # Get segments for expanded gene list
    final_segments = [gene_id_to_segment[gene_id] for gene_id in expanded_gene_ids]

    results = {
        'complex_id': complex_id,
        'gene_ids': gene_ids,
        'expanded_gene_ids': expanded_gene_ids,
        'stoichiometry': stoichiometry,
        'pdb_ids': pdb_ids,
        'af3_confidence': {},
        'pdb_comparisons': {},
        'errors': [],
    }

    af3_tool_config = {
        'use_msa': True,
        'colabfold_search_config': {'search_mode': 'local'},
        'output_dir': af3_dir,
        'verbose': True,
    }

    # Compute AF3 confidence metrics
    try:
        final_plddt = 1. - Constraint(
            inputs=final_segments,
            function=structure_plddt_constraint,
            function_config={
                'structure_tool': 'alphafold3',
                'tool_config': af3_tool_config,
            },
            label='af3_plddt',
        ).evaluate()[0]

        final_ptm = 1. - Constraint(
            inputs=final_segments,
            function=structure_ptm_constraint,
            function_config={
                'structure_tool': 'alphafold3',
                'tool_config': af3_tool_config,
            },
            label='af3_ptm',
        ).evaluate()[0]

        final_iptm = 1. - Constraint(
            inputs=final_segments,
            function=structure_iptm_constraint,
            function_config={
                'structure_tool': 'alphafold3',
                'tool_config': af3_tool_config,
            },
            label='af3_iptm',
        ).evaluate()[0]

        final_pae = Constraint(
            inputs=final_segments,
            function=structure_pae_constraint,
            function_config={
                'structure_tool': 'alphafold3',
                'tool_config': af3_tool_config,
            },
            label='af3_pae',
        ).evaluate()[0] * 31.75  # Max PAE is 31.75 Angstroms

        results['af3_confidence'] = {
            'plddt': float(final_plddt),
            'ptm': float(final_ptm),
            'iptm': float(final_iptm),
            'pae': float(final_pae),
        }

        print(f'AlphaFold3 confidence for {complex_id}:')
        print(f'  pLDDT: {final_plddt:.3f}')
        print(f'  pTM:   {final_ptm:.3f}')
        print(f'  ipTM:  {final_iptm:.3f}')
        print(f'  pAE:   {final_pae:.2f} Å')

    except Exception as e:
        error_msg = f"AF3 confidence scoring failed for {complex_id}: {str(e)}"
        print(f"ERROR: {error_msg}")
        results['errors'].append(error_msg)

    # Compare to experimental PDB structures
    for pdb_id in pdb_ids:
        try:
            experimental_pdb_content = get_remote_pdb_contents(
                pdb_id, cache_dir=pdb_cache_dir
            )
        except Exception as e:
            error_msg = f"Could not load PDB {pdb_id}: {str(e)}"
            print(f"WARNING: {error_msg}")
            results['errors'].append(error_msg)
            continue

        try:
            final_tmscore = 1. - Constraint(
                inputs=final_segments,
                function=structure_tmscore_constraint,
                function_config={
                    'target_pdb_content': experimental_pdb_content,
                    'structure_tool': 'alphafold3',
                    'tool_config': af3_tool_config,
                    'plddt_threshold': 50.,  # Filter low pLDDT regions.
                    'tm_score_normalization': 'max',
                },
                label='af3_tmscore_similarity',
            ).evaluate()[0]

            final_rmsd_score = Constraint(
                inputs=final_segments,
                function=structure_rmsd_constraint,
                function_config={
                    'target_pdb_content': experimental_pdb_content,
                    'structure_tool': 'alphafold3',
                    'tool_config': af3_tool_config,
                    'inflection_point_angstroms': INFLECTION_POINT_ANGSTROMS,
                    'sigmoid_slope': SIGMOID_SLOPE,
                },
                label='af3_rmsd_similarity',
            ).evaluate()[0]

            final_rmsd = inverse_sigmoid_score(
                final_rmsd_score,
                INFLECTION_POINT_ANGSTROMS,
                SIGMOID_SLOPE,
            )

            results['pdb_comparisons'][pdb_id] = {
                'tmscore': float(final_tmscore),
                'rmsd': float(final_rmsd),
            }

            print(f'Comparison to PDB {pdb_id}:')
            print(f'  TMscore: {final_tmscore:.3f}')
            print(f'  RMSD:    {final_rmsd:.2f} Å')

        except Exception as e:
            error_msg = f"PDB comparison failed for {pdb_id}: {str(e)}"
            print(f"ERROR: {error_msg}")
            results['errors'].append(error_msg)

    return results


def score_complexes_in_program_with_af3(
    program: Program,
    gene_ids: List[str],
    complexes: List[Dict[str, Any]],
    output_dir: str,
    run_timestamp: Optional[str] = None,
    pdb_cache_dir: str = DEFAULT_PDB_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Score all complexes in a program with AlphaFold3.

    Continues scoring remaining complexes even if some fail.

    Args:
        program: The run program containing diversified sequences
        gene_ids: List of all gene IDs in the program
        complexes: List of complex info dicts, each with:
            - complex_id: str
            - gene_ids: List[str]
            - stoichiometry: Dict[str, int]
            - pdb_ids: List[str] or None
        output_dir: Base directory for outputs (a timestamped subdir will be created
            or read from an environment variable)
        run_timestamp: Optional timestamp for this run. If None, generates one.
        pdb_cache_dir: Directory for cached PDB files

    Returns:
        Dict with all results and any errors
    """
    # Build mapping from gene_id to final segment
    gene_id_to_final_segment: Dict[str, Segment] = {}

    print('\nFinal sequences:')
    for gene_id, construct in zip(gene_ids, program.constructs):
        print(f'\t{gene_id}: {construct.joined_sequences[0]}')
        final_segment = construct.segments[0]
        gene_id_to_final_segment[gene_id] = final_segment

    # Create timestamped run directory
    run_dir = os.environ.get('RUN_OUTPUT_DIR')
    if run_dir is None:
        # Fallback for local runs (not via SLURM)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(output_dir, f'run_{run_timestamp}')
        os.makedirs(run_dir, exist_ok=True)

    # Create AF3 output directory within run directory
    af3_dir = os.path.join(run_dir, 'af3_outputs')

    print(f'\nRun directory: {run_dir}')

    # Score each complex
    all_results = {
        'run_timestamp': run_timestamp,
        'run_dir': run_dir,
        'gene_ids': gene_ids,
        'final_sequences': {
            gene_id: str(construct.joined_sequences[0])
            for gene_id, construct in zip(gene_ids, program.constructs)
        },
        'complex_scores': [],
        'summary': {
            'total_complexes': len(complexes),
            'successful': 0,
            'failed': 0,
        },
    }

    for complex_info in complexes:
        result = score_single_complex_with_af3(
            gene_id_to_final_segment,
            complex_info,
            af3_dir,
            pdb_cache_dir,
        )
        all_results['complex_scores'].append(result)

        if result['errors']:
            all_results['summary']['failed'] += 1
        else:
            all_results['summary']['successful'] += 1

    # Save results to run directory
    results_path = os.path.join(run_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also save sequences as FASTA
    fasta_path = os.path.join(run_dir, 'final_sequences.fasta')
    with open(fasta_path, 'w') as f:
        for gene_id, seq in all_results['final_sequences'].items():
            f.write(f'>{gene_id}\n{seq}\n')
    print(f"Sequences saved to {fasta_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SCORING SUMMARY")
    print('='*60)
    print(f"Total complexes: {all_results['summary']['total_complexes']}")
    print(f"Successful: {all_results['summary']['successful']}")
    print(f"Failed: {all_results['summary']['failed']}")

    if all_results['summary']['failed'] > 0:
        print("\nErrors encountered:")
        for result in all_results['complex_scores']:
            for error in result['errors']:
                print(f"  - {error}")

    return all_results
