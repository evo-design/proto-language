"""
ESMFold pTM constraint for protein structure quality evaluation.
"""

from __future__ import annotations

from typing import Optional,List

from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
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
from proto_language.utils import MAX_ENERGY


class ESMFoldPTMConfig(BaseConfig):
    """Configuration for ESMFold pTM constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers, 2+ for oligomers (dimers, trimers, etc.). Higher values increase computational cost."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Optional ESMFold configuration (residue_idx_offset, chain_linker, verbose). If None, uses defaults.",
    )


@ConstraintRegistry.register(
    key="esmfold-ptm",
    label="ESMFold pTM Score",
    config=ESMFoldPTMConfig,
    description="Evaluate protein structure quality using ESMFold predicted TM score",
    vectorized=True,
    concatenate=False,
    gpu_required=True
)
def esmfold_ptm_constraint(
    sequences: List[Sequence],
    config: ESMFoldPTMConfig
) -> List[float]:
    """
    Evaluate protein structure quality using ESMFold's predicted TM (pTM) score.

    Args:
        sequences: The protein sequences to evaluate.
        config: Configuration containing n_replications and esmfold_config parameters.

    Returns:
        Constraint score where 0.0 indicates perfect structure quality (pTM = 1.0)
        and higher values indicate lower structure quality.

    Examples:
        Evaluating protein structure confidence:

        >>> seqs = [Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)]
        >>> # Using defaults:
        >>> cfg = ESMFoldPTMConfig()
        >>> score = esmfold_ptm_constraint(seq, config=cfg)
        >>> # With custom ESMFold parameters:
        >>> esmfold_cfg = ESMFoldConfig(residue_idx_offset=256, verbose=True)
        >>> cfg = ESMFoldPTMConfig(n_replications=2, esmfold_config=esmfold_cfg)
        >>> score = esmfold_ptm_constraint(seq, config=cfg)
    """

    by_type = {SequenceType.DNA: [], SequenceType.PROTEIN: []}
    for seq in sequences:
        by_type[seq.sequence_type].append(seq)
    
    scores = [None] * len(sequences)
    
    # Process proteins
    if by_type[SequenceType.PROTEIN]:
        protein_scores = _predict_structures(by_type[SequenceType.PROTEIN], config)
        _map_scores_to_original(sequences, by_type[SequenceType.PROTEIN], protein_scores, scores)
    
    # Process DNA
    if by_type[SequenceType.DNA]:
        dna_scores = _evaluate_dna_via_prodigal(by_type[SequenceType.DNA], config)
        _map_scores_to_original(sequences, by_type[SequenceType.DNA], dna_scores, scores)
    
    return scores


def _predict_structures(
    sequences: List[Sequence],
    config: ESMFoldPTMConfig
) -> List[float]:
    """Predict structures and return pTM-based scores."""
    batch_sequences = [
        [seq.sequence] * config.n_replications  # Each complex is n_replications of the same sequence
        for seq in sequences
    ]
    
    esmfold_input = ESMFoldInput(sequences=batch_sequences)
    esmfold_config = (
        config.esmfold_config if config.esmfold_config is not None else ESMFoldConfig()
    )
    output = run_esmfold(inputs=esmfold_input, config=esmfold_config)
    scores = []
    for i, (seq, structure) in enumerate(zip(sequences, output.structures)):
        seq._metadata.update({
            "avg_plddt": structure.avg_plddt,
            "ptm": structure.ptm,
            "pdb_output": structure.structure_pdb_output,
            "esmfolded_sequence": ":".join([seq.sequence] * config.n_replications),
        })

        # Calculate constraint score (lower pTM = higher penalty)
        scores.append(1.0 - structure.ptm)

    return scores

def _evaluate_dna_via_prodigal(
    dna_sequences: List[Sequence],
    config: ESMFoldPTMConfig
) -> List[float]:
    """Evaluate DNA sequences by predicting proteins with Prodigal first."""
    # Batch predict all DNA sequences at once
    prodigal_result = run_prodigal_prediction(
        ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences]),
        ProdigalConfig()
    )
    
    scores = []
    
    # Process each DNA sequence's results
    for i, (dna_seq, proteins_df, num_genes) in enumerate(zip(
        dna_sequences,
        prodigal_result.results_per_sequence,
        prodigal_result.total_num_genes_per_sequence
    )):
        # Store Prodigal results
        dna_seq._metadata.update({
            "prodigal_proteins": proteins_df,
            "prodigal_protein_count": num_genes
        })
        
        if num_genes == 0:
            scores.append(MAX_ENERGY)
            continue
        
        if len(proteins_df) == 0:
            scores.append(MAX_ENERGY)
            continue
        
        # Predict structures for this DNA's proteins
        protein_seqs = proteins_df['protein_sequence'].tolist()
        batch = [[seq] * config.n_replications for seq in protein_seqs]
        
        esmfold_output = run_esmfold(
            ESMFoldInput(sequences=batch),
            config.esmfold_config or ESMFoldConfig()
        )
        
        ptms = [s.ptm for s in esmfold_output.structures]
        dna_seq._metadata["esmfold_protein_ptms"] = ptms
        best_ptm = max(ptms)
        dna_seq._metadata["esmfold_best_ptm"] = best_ptm
        score = 1.0 - best_ptm
        
        scores.append(score)
    
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