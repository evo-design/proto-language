"""
Protein symmetry ring constraint for symmetric multimeric structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional, List

import numpy as np
from biotite.structure import get_chains
from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.models.structure_prediction.esmfold import (
    ESMFoldInput,
    ESMFoldConfig,
    run_esmfold,
)
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from proto_language.utils import (
    adjacent_distances,
    get_backbone_atoms,
    get_centroid,
    pairwise_distances,
    pdb_file_to_atomarray,
    MAX_ENERGY,
)



class ProteinSymmetryRingConfig(BaseConfig):
    """Configuration for protein symmetry ring constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of protomers in the ring structure (2-12 typical). Defines the oligomeric state (dimer=2, trimer=3, hexamer=6, etc.)."
    )
    max_symmetry_std: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum std of inter-protomer distances (Angstroms) for normalization. Values above this get score 1.0. Typical: 5-10 Å for tight rings."
    )
    all_to_all_protomer_symmetry: bool = Field(
        default=False,
        description="If True, compute pairwise distances between all protomers. If False, only compute distances between adjacent protomers in the ring. False is faster and sufficient for most rings."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Advanced ESMFold configuration parameters. Leave as None to use defaults. Sequences are handled separately via ESMFoldInput."
    )


@ConstraintRegistry.register(
    key="protein-symmetry-ring",
    label="Protein Symmetry Ring Structure",
    config=ProteinSymmetryRingConfig,
    description="Constrain protein to form symmetric ring-like multimeric structure",
    vectorized=True,
    concatenate=True,
    gpu_required=True
)
def protein_symmetry_ring_constraint(sequences: List[Sequence], config: ProteinSymmetryRingConfig) -> List[float]:
    """
    Constrain proteins to form symmetric ring-like multimeric structures.
    
    Supports both protein and DNA sequences:
    - Protein: Direct structure prediction
    - DNA: Uses Prodigal to predict proteins first, then evaluates their structures

    Args:
        sequences: List of protein or DNA sequences to evaluate.
        config: Configuration containing n_replications, symmetry parameters, and esmfold_config.

    Returns:
        List of constraint scores based on standard deviation of inter-protomer distances.
        Lower values indicate more symmetric ring-like arrangements.
    """
    by_type = {SequenceType.DNA: [], SequenceType.PROTEIN: []}
    for seq in sequences:
        by_type[seq.sequence_type].append(seq)
    
    scores = [None] * len(sequences)
    
    if by_type[SequenceType.PROTEIN]:
        protein_scores = _evaluate_protein_symmetry(by_type[SequenceType.PROTEIN], config)
        _map_scores_to_original(sequences, by_type[SequenceType.PROTEIN], protein_scores, scores)
    
    if by_type[SequenceType.DNA]:
        dna_scores = _evaluate_dna_symmetry(by_type[SequenceType.DNA], config)
        _map_scores_to_original(sequences, by_type[SequenceType.DNA], dna_scores, scores)
    
    return scores

def _evaluate_protein_symmetry(
    protein_sequences: List[Sequence],
    config: ProteinSymmetryRingConfig
) -> List[float]:
    """Evaluate protein ring symmetry directly."""
    batch_sequences = [[seq.sequence] * config.n_replications for seq in protein_sequences]
    
    esmfold_input = ESMFoldInput(sequences=batch_sequences)
    esmfold_config = config.esmfold_config or ESMFoldConfig()
    output = run_esmfold(inputs=esmfold_input, config=esmfold_config)

    distance_func = pairwise_distances if config.all_to_all_protomer_symmetry else adjacent_distances
    
    scores = []
    for seq, structure in zip(protein_sequences, output.structures):
        seq._metadata.update({
            "avg_plddt": structure.avg_plddt,
            "ptm": structure.ptm,
            "pdb_output": structure.structure_pdb_output,
            "esmfolded_sequence": ":".join([seq.sequence] * config.n_replications),
        })
        
        # Calculate ring symmetry
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb_output))
        
        centroids = []
        for chain_id in get_chains(atom_array):
            chain_backbone = get_backbone_atoms(atom_array[atom_array.chain_id == chain_id]).coord
            centroids.append(get_centroid(chain_backbone))
        
        assert len(centroids) == config.n_replications
        centroids = np.vstack(centroids)
        
        symmetry_std = float(np.std(distance_func(centroids)))
        normalized_score = min(1.0, symmetry_std / config.max_symmetry_std)
        
        seq._metadata["symmetry_std_raw"] = symmetry_std
        seq._metadata["symmetry_score_normalized"] = normalized_score
        scores.append(normalized_score)
    
    return scores

def _evaluate_dna_symmetry(
    dna_sequences: List[Sequence],
    config: ProteinSymmetryRingConfig
) -> List[float]:
    """Evaluate DNA sequences via Prodigal then symmetry."""
    prodigal_result = run_prodigal_prediction(
        ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences]),
        ProdigalConfig()
    )
    
    distance_func = pairwise_distances if config.all_to_all_protomer_symmetry else adjacent_distances
    scores = []
    
    for dna_seq, proteins_df, num_genes in zip(
        dna_sequences,
        prodigal_result.results_per_sequence,
        prodigal_result.total_num_genes_per_sequence
    ):
        dna_seq._metadata.update({
            "prodigal_proteins": proteins_df,
            "prodigal_protein_count": num_genes
        })
        
        if num_genes == 0 or len(proteins_df) == 0:
            scores.append(MAX_ENERGY)
            continue
        
        protein_seqs = proteins_df['protein_sequence'].tolist()
        batch = [[seq] * config.n_replications for seq in protein_seqs]
        
        esmfold_output = run_esmfold(
            ESMFoldInput(sequences=batch),
            config.esmfold_config or ESMFoldConfig()
        )
        
        # Calculate symmetry for all proteins, use best (lowest std)
        symmetry_stds = []
        for structure in esmfold_output.structures:
            atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb_output))
            centroids = []
            for chain_id in get_chains(atom_array):
                chain_backbone = get_backbone_atoms(atom_array[atom_array.chain_id == chain_id]).coord
                centroids.append(get_centroid(chain_backbone))
            
            centroids = np.vstack(centroids)
            symmetry_stds.append(float(np.std(distance_func(centroids))))
        
        best_symmetry_std = min(symmetry_stds)
        normalized_score = min(1.0, best_symmetry_std / config.max_symmetry_std)
        
        dna_seq._metadata["esmfold_protein_symmetry_stds"] = symmetry_stds
        dna_seq._metadata["esmfold_best_symmetry"] = best_symmetry_std
        scores.append(normalized_score)
    
    return scores


def _map_scores_to_original(
    all_sequences: List[Sequence],
    subset_sequences: List[Sequence],
    subset_scores: List[float],
    scores: List[Optional[float]]
) -> None:
    """Map subset scores back to original sequence order."""
    subset_idx = 0
    for i, seq in enumerate(all_sequences):
        if seq in subset_sequences:
            scores[i] = subset_scores[subset_idx]
            subset_idx += 1