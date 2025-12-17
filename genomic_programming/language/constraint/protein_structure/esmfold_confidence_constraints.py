"""
esmfold_confidence_constraints.py

Contains implementation of ESMFold structure quality constraints.
- ESMFold pLDDT constraint
- ESMFold pTM constraint
"""

from __future__ import annotations

from typing import Optional, List

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
)
from proto_language.tools.structure_prediction import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
    StructurePredictionComplex,
)
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from proto_language.utils import MAX_ENERGY


class ESMFoldConfidenceConfig(BaseConfig):
    """Configuration for ESMFold structure confidence constraints.
    
    This class defines configuration parameters for evaluating protein structure
    quality using Meta's ESMFold structure prediction model. ESMFold predicts
    3D protein structures from amino acid sequences and provides confidence
    metrics including pLDDT (per-residue confidence) and pTM (overall structure
    accuracy). These constraints are useful for filtering proteins predicted to
    fold into stable, well-defined structures. ESMFold requires significant GPU
    memory, especially for long sequences or high n_replications values.
    
    Attributes:
        n_replications (int): Number of times to replicate the sequence for
            multimeric structure prediction. Must be a positive integer. Use 1
            for monomeric proteins (single chain), 2 for homodimers, 3 for
            homotrimers, etc. Higher values predict oligomeric structures but
            increase computational cost and GPU memory usage. Default: 1.

        esmfold_config (Optional[ESMFoldConfig]): Optional advanced ESMFold
            configuration parameters including residue indexing offset, chain
            linker settings, and verbosity. If None, uses default ESMFold settings
            (residue_idx_offset=512, chain_linker=25, verbose=False). The
            ``complexes`` field is set programmatically and should not be
            specified here. Default: None.
    """
    # Required parameter
    n_replications: int = ConfigField(
        title="Number of Replications",
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction.",
    )

    # Optional parameter
    esmfold_config: Optional[ESMFoldConfig] = ConfigField(
        title="ESMFold Config",
        default=None,
        description="Optional ESMFold configuration. If None, uses default configuration.",
        advanced=True,
    )


@ConstraintRegistry.register(
    key="esmfold-plddt",
    label="ESMFold pLDDT Score",
    config=ESMFoldConfidenceConfig,
    description="Evaluate protein structure quality using ESMFold predicted LDDT score",
    batched=True,
    concatenate=True,
    gpu_required=True,
    tools_called=["esmfold", "prodigal"],
    category="protein_structure",
)
def esmfold_plddt_constraint(
    sequences: List[Sequence], config: ESMFoldConfidenceConfig
) -> List[float]:
    """Evaluate protein structure quality using ESMFold pLDDT score.
    
    This constraint uses ESMFold to predict protein structures and
    evaluates their quality using the predicted Local Distance Difference Test
    (pLDDT) score. pLDDT measures per-residue confidence in the predicted structure,
    with values ranging from 0 (low confidence) to 100 (high confidence). Higher
    pLDDT indicates more reliable structure predictions.
    
    For DNA sequences, the function first runs Prodigal to predict protein-coding
    regions (ORFs), then evaluates the structure quality of each predicted protein
    using the best (highest) pLDDT score among all predictions.
    
    Structure prediction is GPU-intensive and may take several minutes per protein
    depending on length and hardware.

    Args:
        sequences (List[Sequence]): List of protein or DNA sequences to evaluate.
            All sequences in the list must be the same type (all DNA or all PROTEIN).
            For DNA sequences, ORF prediction is performed automatically.
            
        config (ESMFoldConfidenceConfig): Configuration object containing
            ``n_replications`` (oligomeric state, default: 1) and optional
            ``esmfold_config`` for advanced ESMFold settings.

    Returns:
        List[float]: Constraint scores for each sequence, where 0.0 indicates
            perfect structure confidence (pLDDT = 100) and higher values indicate
            lower confidence. The score is calculated as 1.0 - pLDDT, so a pLDDT 
            of 0.9 (90% confidence) gives a score of 0.1. For DNA sequences, the
            score is based on the best pLDDT among all predicted proteins.

    Raises:
        AssertionError: If any sequence in the input list is not a protein or DNA
            sequence.
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Metadata varies by
        sequence type:
        
        **For protein sequences:**
        - ``avg_plddt``: Float average pLDDT score across all residues (0.0-1.0)
        - ``ptm``: Float predicted TM-score for overall structure accuracy (0.0-1.0)
        - ``pdb_output``: String PDB format structure file content
        - ``esmfolded_sequence``: String sequence used for structure prediction
          (with chain separators for multimers)
        
        **For DNA sequences:**
        - ``prodigal_proteins``: DataFrame of predicted proteins from Prodigal
        - ``prodigal_protein_count``: Integer count of predicted ORFs
        - ``esmfold_protein_plddts``: List of float pLDDT scores for each
          predicted protein
        - ``esmfold_protein_ptms``: List of float pTM scores for each predicted
          protein
        - ``esmfold_best_plddt``: Float best (highest) pLDDT score among all
          predicted proteins
        - ``esmfold_best_ptm``: Float best (highest) pTM score among all
          predicted proteins
    
    Examples:
        Evaluating monomer structure confidence:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ESMFoldConfidenceConfig(n_replications=1)
        >>> scores = esmfold_plddt_constraint([seq], config)
        >>> print(scores[0])  # e.g., 0.15 (pLDDT of 0.85)
        >>> print(seq._metadata["avg_plddt"])  # e.g., 0.85
        >>> print(seq._metadata["ptm"])  # e.g., 0.78
        
        Using custom ESMFold configuration:
        
        >>> from proto_language.tools.structure_prediction import ESMFoldConfig
        >>> esmfold_cfg = ESMFoldConfig(
        ...     residue_idx_offset=256,  # Offset for residue numbering
        ...     chain_linker=25,         # Linker length between chains
        ...     verbose=True             # Print progress
        ... )
        >>> config = ESMFoldConfidenceConfig(
        ...     n_replications=3,
        ...     esmfold_config=esmfold_cfg
        ... )
        >>> scores = esmfold_plddt_constraint([seq], config)
        
        Evaluating DNA sequence (with automatic ORF prediction):
        
        >>> dna_seq = Sequence("ATGGTACTGAGCCCAGCG...", "dna")
        >>> config = ESMFoldConfidenceConfig(n_replications=1)
        >>> scores = esmfold_plddt_constraint([dna_seq], config)
        >>> print(dna_seq._metadata["prodigal_protein_count"])  # e.g., 2
        >>> print(dna_seq._metadata["esmfold_best_plddt"])  # e.g., 0.82 (best among predicted proteins)
        >>> print(dna_seq._metadata["esmfold_protein_plddts"])  # e.g., [0.75, 0.82]

    """
    return esmfold_confidence(sequences, config, target_metric="avg_plddt")


@ConstraintRegistry.register(
    key="esmfold-ptm",
    label="ESMFold pTM Score",
    config=ESMFoldConfidenceConfig,
    description="Evaluate protein structure quality using ESMFold predicted TM score",
    batched=True,
    concatenate=True,
    gpu_required=True,
    tools_called=["esmfold", "prodigal"],
    category="protein_structure",
)
def esmfold_ptm_constraint(
    sequences: List[Sequence], config: ESMFoldConfidenceConfig
) -> List[float]:
    """Evaluate protein structure quality using ESMFold's pTM score.
    
    This constraint function uses ESMFold to predict protein structures and
    evaluates their quality using the predicted Template Modeling (pTM) score. pTM
    measures overall structure accuracy and alignment quality, with values ranging
    from 0 (poor) to 1 (excellent). pTM is particularly useful for assessing
    global structure quality and is more sensitive to domain arrangements and
    overall fold accuracy than pLDDT.
    
    For DNA sequences, the function first runs Prodigal to predict protein-coding
    regions (ORFs), then evaluates the structure quality of each predicted protein
    using the best (highest) pTM score among all predictions. Structure prediction
    is GPU-intensive and may take several minutes per protein depending on length and
    hardware.

    Args:
        sequences (List[Sequence]): List of protein or DNA sequences to evaluate.
            All sequences in the list must be the same type (all DNA or all PROTEIN).
            For DNA sequences, ORF prediction is performed automatically.
        config (ESMFoldConfidenceConfig): Configuration object containing
            ``n_replications`` (oligomeric state, default: 1) and optional
            ``esmfold_config`` for advanced ESMFold settings.

    Returns:
        List[float]: Constraint scores for each sequence, where 0.0 indicates
            perfect structure quality (pTM = 1.0) and higher values indicate lower
            quality. The score is calculated as 1.0 - pTM, so a pTM of 0.85 gives
            a score of 0.15. For DNA sequences, the score is based on the best pTM
            among all predicted proteins.

    Raises:
        AssertionError: If any sequence in the input list is not a protein or DNA
            sequence.
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. The metadata structure is
        identical to ``esmfold_plddt_constraint``:
        
        **For protein sequences:**
        - ``avg_plddt``: Float average pLDDT score (0.0-1.0)
        - ``ptm``: Float predicted TM-score (0.0-1.0)
        - ``pdb_output``: String PDB format structure file
        - ``esmfolded_sequence``: String sequence used for prediction
        
        **For DNA sequences:**
        - ``prodigal_proteins``: DataFrame of predicted proteins
        - ``prodigal_protein_count``: Integer count of ORFs
        - ``esmfold_protein_plddts``: List of pLDDT scores
        - ``esmfold_protein_ptms``: List of pTM scores
        - ``esmfold_best_plddt``: Float best pLDDT score
        - ``esmfold_best_ptm``: Float best pTM score
    
    Examples:
        Evaluating global structure quality:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ESMFoldConfidenceConfig(n_replications=1)
        >>> scores = esmfold_ptm_constraint([seq], config)
        >>> print(scores[0])  # e.g., 0.22 (pTM of 0.78)
        >>> print(seq._metadata["ptm"])  # e.g., 0.78
        >>> print(seq._metadata["avg_plddt"])  # e.g., 0.85 (also available)
    """
    return esmfold_confidence(sequences, config, target_metric="ptm")


def esmfold_confidence(
    sequences: List[Sequence],
    config: ESMFoldConfidenceConfig,
    target_metric: str = "avg_plddt",
) -> List[float]:
    """
    Helper function to evaluate protein structure confidence using ESMFold.

    Args:
        sequences: The protein sequences to evaluate.
        config: Configuration containing n_replications and esmfold_config parameters.
        target_metric: The metric to evaluate. Either "plddt" or "ptm".

    Returns:
        Constraint score where 0.0 indicates perfect structure confidence (pLDDT = 1.0)
        and higher values indicate lower structure confidence.

    Examples:
        Evaluating protein structure confidence:

        >>> seqs = [Sequence("MVLSPADKTNVK", "protein")]
        >>> # Using defaults:
        >>> cfg = ESMFoldPLDDTConfig()
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
        >>> # With custom ESMFold parameters:
        >>> esmfold_cfg = ESMFoldConfig(residue_idx_offset=256, verbose=True)
        >>> cfg = ESMFoldPLDDTConfig(n_replications=2, esmfold_config=esmfold_cfg)
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
    """
    if target_metric not in ["avg_plddt", "ptm"]:
        raise ValueError(
            f"Invalid target metric: {target_metric}. Must be 'avg_plddt' or 'ptm'."
        )

    scores = []
    if sequences[0].sequence_type == "protein":
        scores = _predict_structures(sequences, config, target_metric)
    else:
        scores = _evaluate_dna_via_prodigal(sequences, config, target_metric)

    return scores


def _predict_structures(
    sequences: List[Sequence], config: ESMFoldConfidenceConfig, target_metric: str
) -> List[float]:
    """Predict structures and return specified confidence scores."""

    # Create complexes with n_replications of each protein sequence
    complexes = [
        StructurePredictionComplex(
            chains=[seq.sequence] * config.n_replications,
            entity_types=["protein"] * config.n_replications,
        )
        for seq in sequences
    ]

    esmfold_input = ESMFoldInput(complexes=complexes)
    output = run_esmfold(
        inputs=esmfold_input, config=config.esmfold_config or ESMFoldConfig()
    )
    scores = []
    for seq, comp, structure in zip(sequences, complexes, output):
        seq._metadata.update(
            {
                "avg_plddt": structure.avg_plddt,
                "ptm": structure.ptm,
                "pdb_output": structure.structure_pdb,
                "esmfolded_sequence": ":".join(comp.chains),
            }
        )

        # Calculate constraint score (lower pLDDT = higher penalty)
        scores.append(1.0 - getattr(structure, target_metric))

    return scores


def _evaluate_dna_via_prodigal(
    dna_sequences: List[Sequence], config: ESMFoldConfidenceConfig, target_metric: str
) -> List[float]:
    """Evaluate DNA sequences by predicting proteins with Prodigal first."""
    # Batch predict all DNA sequences at once
    prodigal_result = run_prodigal_prediction(
        ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences]),
        ProdigalConfig(),
    )

    scores = []

    # Process each DNA sequence's results
    for i, (dna_seq, proteins_df, num_genes) in enumerate(
        zip(
            dna_sequences,
            prodigal_result.results_per_sequence,
            prodigal_result.total_num_genes_per_sequence,
        )
    ):
        # Store Prodigal results
        dna_seq._metadata.update(
            {"prodigal_proteins": proteins_df, "prodigal_protein_count": num_genes}
        )

        if num_genes == 0:
            scores.append(MAX_ENERGY)
            continue

        if len(proteins_df) == 0:
            scores.append(MAX_ENERGY)
            continue

        # Predict structures for this DNA's proteins
        protein_seqs = proteins_df["protein_sequence"].tolist()
        complexes = [
            StructurePredictionComplex(
                chains=[seq] * config.n_replications,
                entity_types=["protein"] * config.n_replications,
            )
            for seq in protein_seqs
        ]

        esmfold_output = run_esmfold(
            inputs=ESMFoldInput(complexes=complexes),
            config=config.esmfold_config or ESMFoldConfig(),
        )

        plddts = [structure.avg_plddt for structure in esmfold_output]
        ptms = [structure.ptm for structure in esmfold_output]

        dna_seq._metadata["esmfold_protein_plddts"] = plddts
        dna_seq._metadata["esmfold_protein_ptms"] = ptms

        best_plddt = max(plddts)
        best_ptm = max(ptms)

        dna_seq._metadata["esmfold_best_plddt"] = best_plddt
        dna_seq._metadata["esmfold_best_ptm"] = best_ptm

        if target_metric == "avg_plddt":
            score = 1.0 - best_plddt
        else:
            score = 1.0 - best_ptm

        scores.append(score)

    return scores
