"""
proto_language/language/constraint/protein_structure/protein_globularity_constraint.py

Protein globularity constraint for compact protein structures.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import List, Tuple

import numpy as np
from proto_tools import (
    ESMFoldConfig,
    ESMFoldInput,
    ProdigalConfig,
    ProdigalInput,
    StructurePredictionComplex,
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
    run_esmfold,
    run_prodigal_prediction,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.storage import FileType, store_file
from proto_language.utils import MAX_ENERGY


class ProteinGlobularityConfig(BaseConfig):
    """Configuration for protein globularity constraint.

    This class defines configuration parameters for evaluating protein structural
    compactness using ESMFold structure prediction. Globularity measures how
    compact and spherical a protein structure is, based on the spatial distribution
    of backbone atoms around the structure's center of mass. More globular proteins
    have backbone atoms clustered tightly around the centroid, while extended
    structures show higher dispersion. Globularity is measured as the standard
    deviation of distances from backbone atoms (N, CA, C, O) to the structure's
    centroid. Lower values indicate more compact, spherical structures.
    The score is normalized by dividing by max_globularity (default 20.0 Ångströms) and
    capped at 1.0.

    Attributes:
        n_replications (int): Number of times to replicate the sequence for
            multimeric structure prediction. Must be a positive integer. Use 1
            for monomeric proteins (single chain). Higher values predict oligomeric
            structures (dimers, trimers, etc.) but increase computational cost.
            Default: 1.

        max_globularity (float): Maximum standard deviation from the backbone atoms
            to the structure's centroid to be considered highly extended or unfolded.
            Structures with globularity measurments greater than this value receive the
            maximum penalty score of 1.0, while more compact structures receive proportionally
            lower scores (e.g., 10 Å globularity = 0.5 score for max_globularity of 20.0 Å).
            Default: 20.0.

        esmfold_config (ESMFoldConfig): Advanced ESMFold configuration parameters
            including residue indexing offset, chain linker settings, and verbosity.
            The ``complexes`` field is set programmatically and should not be
            specified here. Default: ESMFoldConfig().
    """
    # Required parameter
    n_replications: int = ConfigField(
        title="Number of Replications",
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers.",
    )

    # Optional parameter
    max_globularity: float = ConfigField(
        title="Max Globularity Deviation",
        default=20.0,
        description="Max std from backbone atoms to the structure's centroid to be considered highly extended/ unfolded.",
        advanced=True,
    )
    esmfold_config: ESMFoldConfig = ConfigField(
        title="ESMFold Config",
        default_factory=ESMFoldConfig,
        description="ESMFold configuration for structure prediction.",
        advanced=True,
    )


@constraint(
    key="protein-globularity",
    label="Protein Globularity",
    config=ProteinGlobularityConfig,
    description="Encourage compact, globular protein structures",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "prodigal-prediction"],
    category="protein_structure",
    supported_sequence_types=["dna", "protein"],
    num_input_sequences_per_tuple=1,
)
def protein_globularity_constraint(input_sequences: List[Tuple[Sequence, ...]], config: ProteinGlobularityConfig) -> List[float]:
    """Encourage compact, globular protein structures using ESMFold.

    This constraint function uses ESMFold to predict protein 3D structures
    and evaluates their compactness by analyzing the spatial distribution of
    backbone atoms. Globularity is measured as the standard deviation of distances
    from backbone atoms (N, CA, C, O) to the structure's geometric centroid.
    Lower values indicate more compact, spherical structures characteristic of
    well-folded globular proteins, while higher values indicate extended,
    elongated, or poorly folded structures.

    For DNA sequences, the function first runs Prodigal to predict protein-coding
    regions (ORFs), then evaluates the globularity of each predicted protein
    structure, using the best (most compact) globularity score among all predictions.

    Structure prediction is GPU-intensive and may take several minutes per protein
    depending on length and hardware.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples to
            evaluate. Each tuple contains one protein or DNA sequence. All sequences
            must be the same type. For DNA sequences, ORF prediction is performed
            automatically.

        config (ProteinGlobularityConfig): Configuration object containing
            ``n_replications`` (oligomeric state, default: 1) and optional
            ``esmfold_config`` for advanced ESMFold settings.

    Returns:
        List[float]: Constraint scores for each sequence based on structural
            compactness. Returns normalized scores (0.0-1.0) for both protein
            and DNA sequences, where lower values indicate more compact structures.
            Scores are normalized by dividing by max_globularity (default 20.0 Å)
            and capped at 1.0.

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Metadata varies by
        sequence type:

        **For protein sequences:**
        - ``avg_plddt``: Float average pLDDT score for structure confidence (0.0-1.0)
        - ``ptm``: Float predicted TM-score for structure accuracy (0.0-1.0)
        - ``pdb_output``: String PDB format structure file content
        - ``esmfolded_sequence``: List of sequences used for structure prediction
        - ``raw_globularity``: Float standard deviation of backbone-to-centroid
          distances in Ångströms (lower = more compact)
        - ``normalized_globularity``: Float normalized globularity score (0.0-1.0,
          capped by max_globularity)

        **For DNA sequences:**
        - ``prodigal_proteins``: DataFrame of predicted proteins from Prodigal
        - ``prodigal_protein_count``: Integer count of predicted ORFs
        - ``esmfold_protein_globularities``: List of float globularity scores
          for each predicted protein (in Ångströms)
        - ``esmfold_best_globularity``: Float best (lowest) globularity score
          among all predicted proteins (in Ångströms)
        - ``esmfold_normalized_globularity``: Float normalized best globularity
          (0.0-1.0, capped by max_globularity)

    Examples:
        Evaluating protein structural compactness:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ProteinGlobularityConfig(n_replications=1)
        >>> scores = protein_globularity_constraint([(seq,)], config)
        >>> print(scores[0])  # e.g., 0.425 (normalized score, lower = more compact)
        >>> print(seq._metadata["raw_globularity"])  # e.g., 8.5 (raw Ångströms)
        >>> print(seq._metadata["normalized_globularity"])  # e.g., 0.425
        >>> print(seq._metadata["avg_plddt"])  # e.g., 0.85 (also available)

        Evaluating DNA sequence (with automatic ORF prediction):

        >>> dna_seq = Sequence("ATGGTACTGAGCCCAGCG...", "dna")
        >>> config = ProteinGlobularityConfig(n_replications=1)
        >>> scores = protein_globularity_constraint([(dna_seq,)], config)
        >>> print(scores[0])  # Normalized score (0.0-1.0)
        >>> print(dna_seq._metadata["prodigal_protein_count"])  # e.g., 2
        >>> print(dna_seq._metadata["esmfold_best_globularity"])  # e.g., 7.8 Å (best among predicted proteins)
        >>> print(dna_seq._metadata["esmfold_protein_globularities"])  # e.g., [9.2, 7.8]
    """

    # Extract sequences from tuples and delegate to type-specific handlers
    sequences = [seq for (seq,) in input_sequences]
    if sequences[0].sequence_type == "protein":
        return _evaluate_protein_globularity(sequences, config)
    else:
        return _evaluate_dna_globularity(sequences, config)


def _evaluate_protein_globularity(
    protein_sequences: List[Sequence], config: ProteinGlobularityConfig
) -> List[float]:
    """Evaluate protein globularity directly."""

    # Create complexes with n_replications of each protein sequence
    complexes = [
        StructurePredictionComplex(
            chains=[{"sequence": seq.sequence, "entity_type": "protein"}] * config.n_replications
        )
        for seq in protein_sequences
    ]

    # Create the ESMFold input containing all the complexes
    esmfold_input = ESMFoldInput(complexes=complexes)

    # Run ESMFold
    output = run_esmfold(
        inputs=esmfold_input, config=config.esmfold_config
    )

    scores = []
    for protein_seq, comp, structure in zip(
        protein_sequences, complexes, output.structures
    ):
        protein_seq._metadata.update(
            {
                "avg_plddt": structure.avg_plddt,
                "ptm": structure.ptm,
                "pdb_output": store_file(structure.structure_pdb, FileType.PDB),
                "esmfolded_sequence": comp.chains,
            }
        )

        # Calculate globularity from structure
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))
        backbone = get_backbone_atoms(atom_array).coord
        raw_globularity = float(np.std(distances_to_centroid(backbone)))
        normalized_globularity = min(1.0, raw_globularity / config.max_globularity)

        # Update the globularity score in the metadata (keep raw value for users who need physical measurement)
        protein_seq._metadata["raw_globularity"] = raw_globularity
        protein_seq._metadata["normalized_globularity"] = normalized_globularity
        scores.append(normalized_globularity)

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

    for dna_seq, proteins_list, num_genes in zip(
        dna_sequences,
        prodigal_result.predicted_orfs,
        prodigal_result.num_orfs_per_sequence
    ):
        orf_dicts = [orf.model_dump() for orf in proteins_list]
        dna_seq._metadata.update({
            "prodigal_proteins": store_file(
                json.dumps(orf_dicts), FileType.JSON
            ) if orf_dicts else None,
            "prodigal_protein_count": num_genes
        })

        if num_genes == 0 or len(proteins_list) == 0:
            scores.append(MAX_ENERGY)
            continue

        protein_seqs = [orf.amino_acid_sequence for orf in proteins_list]
        complexes = [
            StructurePredictionComplex(
                chains=[{"sequence": seq, "entity_type": "protein"}] * config.n_replications
            )
            for seq in protein_seqs
        ]

        # Run ESMFold
        esmfold_output = run_esmfold(
            inputs=ESMFoldInput(complexes=complexes),
            config=config.esmfold_config,
        )

        # Calculate globularity for all proteins, use best (lowest std)
        globularities = []
        for structure in esmfold_output.structures:
            atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))
            backbone = get_backbone_atoms(atom_array).coord
            globularities.append(float(np.std(distances_to_centroid(backbone))))

        best_globularity = min(globularities)
        globularity_score = min(1.0, best_globularity / config.max_globularity)
        dna_seq._metadata["esmfold_protein_globularities"] = globularities
        dna_seq._metadata["esmfold_best_globularity"] = best_globularity
        dna_seq._metadata["esmfold_normalized_globularity"] = globularity_score
        scores.append(globularity_score)

    return scores
