"""
Protein globularity constraint for compact protein structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional,List

import numpy as np
from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.models.structure_prediction.esmfold import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
)
from proto_language.utils import (
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
    MAX_ENERGY
)
from proto_language.tools.models.structure_prediction.esmfold import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
)
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)


MAX_GLOBULARITY = 20.0

class ProteinGlobularityConfig(BaseConfig):
    """Configuration for protein globularity constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Advanced ESMFold configuration parameters. Leave as None to use defaults.",
    )


@ConstraintRegistry.register(
    key="protein-globularity",
    label="Protein Globularity",
    config=ProteinGlobularityConfig,
    description="Encourage compact, globular protein structures",
    vectorized=True,
    concatenate=True,
    gpu_required=True
)
def protein_globularity_constraint(sequences: List[Sequence], config: ProteinGlobularityConfig) -> List[float]:
    """
    Encourage compact, globular protein structures.
    
    Supports both protein and DNA sequences:
    - Protein: Direct structure prediction
    - DNA: Uses Prodigal to predict proteins first, then evaluates their structures

    Args:
        sequences: List of protein or DNA sequences to evaluate.
        config: Configuration containing n_replications and esmfold_config parameters.

    Returns:
        List of constraint scores based on standard deviation of distances from backbone atoms to centroid.
        Lower values indicate more compact, globular structures.
    """
    # Separate by type
    by_type = {SequenceType.DNA: [], SequenceType.PROTEIN: []}
    for seq in sequences:
        by_type[seq.sequence_type].append(seq)
    
    scores = [None] * len(sequences)
    
    # Process proteins
    if by_type[SequenceType.PROTEIN]:
        protein_scores = _evaluate_protein_globularity(by_type[SequenceType.PROTEIN], config)
        _map_scores_to_original(sequences, by_type[SequenceType.PROTEIN], protein_scores, scores)
    
    # Process DNA
    if by_type[SequenceType.DNA]:
        dna_scores = _evaluate_dna_globularity(by_type[SequenceType.DNA], config)
        _map_scores_to_original(sequences, by_type[SequenceType.DNA], dna_scores, scores)
    
    return scores


def _evaluate_protein_globularity(
    protein_sequences: List[Sequence],
    config: ProteinGlobularityConfig
) -> List[float]:
    """Evaluate protein globularity directly."""
    batch_sequences = [[seq.sequence] * config.n_replications for seq in protein_sequences]
    
    esmfold_input = ESMFoldInput(sequences=batch_sequences)
    esmfold_config = config.esmfold_config or ESMFoldConfig()
    output = run_esmfold(inputs=esmfold_input, config=esmfold_config)

    scores = []
    for seq, structure in zip(protein_sequences, output.structures):
        seq._metadata.update({
            "avg_plddt": structure.avg_plddt,
            "ptm": structure.ptm,
            "pdb_output": structure.structure_pdb_output,
            "esmfolded_sequence": ":".join([seq.sequence] * config.n_replications),
        })
        
        # Calculate globularity from structure
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb_output))
        backbone = get_backbone_atoms(atom_array).coord
        globularity_score = float(np.std(distances_to_centroid(backbone)))
        
        seq._metadata["globularity_score"] = globularity_score
        scores.append(globularity_score)
    
    return scores


def _evaluate_dna_globularity(
    dna_sequences: List[Sequence],
    config: ProteinGlobularityConfig
) -> List[float]:
    """Evaluate DNA sequences via Prodigal then globularity."""
    prodigal_result = run_prodigal_prediction(
        ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences]),
        ProdigalConfig()
    )
    
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
        
        # Calculate globularity for all proteins, use best (lowest std)
        globularities = []
        for structure in esmfold_output.structures:
            atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb_output))
            backbone = get_backbone_atoms(atom_array).coord
            globularities.append(float(np.std(distances_to_centroid(backbone))))
        
        best_globularity = min(globularities)
        globularity_score = min(1.0, best_globularity / MAX_GLOBULARITY)
        dna_seq._metadata["esmfold_protein_globularities"] = globularities
        dna_seq._metadata["esmfold_best_globularity"] = best_globularity
        dna_seq._metadata["esmfold_normalized_globularity"] = globularity_score
        scores.append(globularity_score)
    
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
