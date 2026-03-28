"""
proto_language/language/constraint/protein_structure/protein_symmetry_ring_constraint.py

Protein symmetry ring constraint for symmetric multimeric structures.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import List, Optional, Tuple

import numpy as np
from proto_tools import (
    ESMFoldConfig,
    ESMFoldInput,
    ProdigalConfig,
    ProdigalInput,
    StructurePredictionComplex,
    adjacent_distances,
    get_backbone_atoms,
    get_centroid,
    pairwise_distances,
    pdb_file_to_atomarray,
    run_esmfold,
    run_prodigal_prediction,
)
from biotite.structure import get_chains

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.storage import FileType, store_file
from proto_language.utils import MAX_ENERGY


class ProteinSymmetryRingConfig(BaseConfig):
    """Configuration for protein symmetry ring constraint.

    This class defines configuration parameters for evaluating whether proteins
    form symmetric ring-like multimeric structures using ESMFold structure
    prediction. Ring symmetry is measured by analyzing the spatial arrangement
    of protomer centroids in predicted oligomeric structures. Symmetric rings
    have protomers evenly distributed in a circular arrangement with consistent
    inter-protomer distances, characteristic of many functional protein complexes
    like chaperonins, proteasomes, and ring-shaped enzymes. Symmetry is calculated by
    taking the centroids of each protomer (using backbone atom coordinates) and
    measuring the standard deviation of distances between protomers. Lower standard
    deviation indicates more symmetric arrangements where all protomers are equally spaced.
    The score is normalized by dividing by ``max_symmetry_std`` and capped at 1.0.

    Attributes:
        n_replications (int): Number of protomers in the ring structure. Must
            be a positive integer (typically 3-12). Defines the oligomeric state:
            3 for trimers, 4 for tetramers, 5 for pentamers, 6 for hexamers, etc.
             Higher values increase computational cost. Default: 3.

        max_symmetry_std (float): Maximum standard deviation of inter-protomer
            distances (in Ångströms) used for score normalization. Must be a
            positive float. Structures with symmetry standard deviation at or
            below this value receive proportionally lower scores, while those
            exceeding it receive the maximum penalty (1.0). Typical values range
            from 5.0 Å (very tight, highly symmetric rings) to 10.0 Å (moderate
            symmetry tolerance). Well-formed symmetric rings typically have std
            < 3 Å. Default: 10.0.

        all_to_all_protomer_symmetry (bool): If True, computes pairwise distances
            between all protomers (N*(N-1)/2 distances for N protomers), providing
            a more comprehensive symmetry measure. If False, only computes distances
            between adjacent protomers in the ring (N distances), which is faster
            and sufficient for most symmetric rings. Use True for stringent
            symmetry requirements or asymmetric arrangements. Default: False.

        esmfold_config (ESMFoldConfig): Advanced ESMFold configuration parameters
            including residue indexing offset, chain linker settings, and verbosity.
            The ``complexes`` field is set programmatically and should not be
            specified here. Default: ESMFoldConfig().
    """
    # Required parameters
    n_replications: int = ConfigField(
        default=3,
        ge=1,
        title="Number of Replications",
        description="Number of protomers in the ring structure. Defines the oligomeric state (dimer=2, trimer=3, etc.).",
        examples=[3, 12],
    )

    # Advanced parameters
    max_symmetry_std: float = ConfigField(
        default=10.0,
        ge=0.0,
        title="Max Symmetry Standard Deviation",
        description="Maximum std of inter-protomer distances in Å for normalization. Values above this get score 1.0.",
        examples=[5, 10],  # Typical: 5-10 Å for tight rings.
    )
    all_to_all_protomer_symmetry: bool = ConfigField(
        default=False,
        title="All-to-All Protomer Symmetry",
        description="True uses pairwise distances between all protomers. Else, use distances between adjacent protomers",
        advanced=True,
    )
    esmfold_config: ESMFoldConfig = ConfigField(
        default_factory=ESMFoldConfig,
        title="ESMFold Config",
        description="ESMFold configuration for structure prediction.",
        advanced=True,
    )


@constraint(
    key="protein-symmetry-ring",
    label="Protein Symmetry Ring Structure",
    config=ProteinSymmetryRingConfig,
    description="Constrain protein to form symmetric ring-like multimeric structure",
    uses_gpu=True,
    tools_called=["esmfold-prediction", "prodigal-prediction"],
    category="protein_structure",
    supported_sequence_types=["dna", "protein"],
    num_input_sequences_per_tuple=1,
)
def protein_symmetry_ring_constraint(input_sequences: List[Tuple[Sequence, ...]], config: ProteinSymmetryRingConfig) -> List[float]:
    """Constrain proteins to form symmetric ring-like multimeric structures using ESMFold.

    This constraint function uses ESMFold to predict multimeric protein
    structures and evaluates whether they form symmetric ring-like arrangements.
    Ring symmetry is quantified by calculating the centroid (center of mass) of
    each protomer's backbone and measuring how uniformly the protomers are
    distributed around the ring. Perfect symmetric rings have all inter-protomer
    distances equal, resulting in zero standard deviation.

    Many functional protein complexes naturally form symmetric rings, including
    chaperonins, proteasomes (heptameric rings), hexameric helicases, and various
    ring-shaped enzymes. This constraint is useful for designing or selecting
    proteins that form such symmetric assemblies.

    For DNA sequences, the function first runs Prodigal to predict protein-coding
    regions (ORFs), then evaluates the ring symmetry of each predicted protein
    structure, using the best (most symmetric) score among all predictions.

    Structure prediction is GPU-intensive and may take several minutes per protein
    depending on length and hardware.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples to
            evaluate. Each tuple contains one protein or DNA sequence. All sequences
            must be the same type. For DNA sequences, ORF prediction is performed
            automatically.

        config (ProteinSymmetryRingConfig): Configuration object containing
            ``n_replications`` (number of protomers in ring, default: 2),
            ``max_symmetry_std`` (normalization threshold in Å, default: 10.0),
            ``all_to_all_protomer_symmetry`` (distance calculation mode, default: False),
            and optional ``esmfold_config`` for advanced ESMFold settings.

    Returns:
        List[float]: Constraint scores for each sequence based on ring symmetry.
            Scores range from 0.0 (perfect symmetry, all inter-protomer distances
            equal) to 1.0 (highly asymmetric or max penalty). The score equals
            (std of inter-protomer distances) / max_symmetry_std, capped at 1.0.
            For DNA sequences, the score is based on the best symmetry among all
            predicted proteins.

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Metadata varies by
        sequence type:

        **For protein sequences:**
        - ``avg_plddt``: Float average pLDDT score for structure confidence (0.0-1.0)
        - ``ptm``: Float predicted TM-score for structure accuracy (0.0-1.0)
        - ``pdb_output``: String PDB format structure file content
        - ``esmfolded_sequence``: String colon-separated sequence representation
        - ``symmetry_std_raw``: Float raw standard deviation of inter-protomer
          distances in Ångströms (lower = more symmetric)
        - ``symmetry_score_normalized``: Float normalized symmetry score (0.0-1.0)

        **For DNA sequences:**
        - ``prodigal_proteins``: DataFrame of predicted proteins from Prodigal
        - ``prodigal_protein_count``: Integer count of predicted ORFs
        - ``esmfold_protein_symmetry_stds``: List of float symmetry standard
          deviations for each predicted protein (in Ångströms)
        - ``esmfold_best_symmetry``: Float best (lowest) symmetry std among all
          predicted proteins (in Ångströms)

    Examples:
        Designing a symmetric hexameric ring:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ProteinSymmetryRingConfig(
        ...     n_replications=6,  # Hexamer
        ...     max_symmetry_std=10.0
        ... )
        >>> scores = protein_symmetry_ring_constraint([(seq,)], config)
        >>> print(scores[0])  # e.g., 0.35 (3.5 Å std / 10.0 Å max)
        >>> print(seq._metadata["symmetry_std_raw"])  # e.g., 3.5 Å
        >>> print(seq._metadata["symmetry_score_normalized"])  # 0.35
    """
    # Extract sequences from tuples and separate by type
    sequences = [seq for (seq,) in input_sequences]
    by_type = {"dna": [], "protein": []}
    for seq in sequences:
        by_type[seq.sequence_type].append(seq)

    scores = [None] * len(input_sequences)

    if by_type["protein"]:
        protein_scores = _evaluate_protein_symmetry(by_type["protein"], config)
        _map_scores_to_original(sequences, by_type["protein"], protein_scores, scores)

    if by_type["dna"]:
        dna_scores = _evaluate_dna_symmetry(by_type["dna"], config)
        _map_scores_to_original(sequences, by_type["dna"], dna_scores, scores)

    return scores

def _evaluate_protein_symmetry(
    protein_sequences: List[Sequence],
    config: ProteinSymmetryRingConfig
) -> List[float]:
    """Evaluate protein ring symmetry directly."""

    # Create complexes with n_replications of each protein sequence
    complexes = [
        StructurePredictionComplex(
            chains=[{"sequence": seq.sequence, "entity_type": "protein"}]
            * config.n_replications
        )
        for seq in protein_sequences
    ]

    # Run ESMFold
    output = run_esmfold(
        inputs=ESMFoldInput(complexes=complexes),
        config=config.esmfold_config,
    )

    # Determine distance function
    distance_func = pairwise_distances if config.all_to_all_protomer_symmetry else adjacent_distances

    # Update sequence metadata with ESMFold output and calculate scores for each sequence
    scores = []
    for seq, structure in zip(protein_sequences, output.structures):
        seq._metadata.update(
            {
                "avg_plddt": structure.avg_plddt,
                "ptm": structure.ptm,
                "pdb_output": store_file(structure.structure_pdb, FileType.PDB),
                "esmfolded_sequence": ":".join([seq.sequence] * config.n_replications),
            }
        )

        # Calculate ring symmetry
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))

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

        # If there are no genes predicted, score is MAX_ENERGY
        if num_genes == 0 or len(proteins_list) == 0:
            scores.append(MAX_ENERGY)
            continue

        # Create complexes with n_replications of each protein sequence
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

        # Calculate symmetry for all proteins, use best (lowest std)
        symmetry_stds = []
        for structure in esmfold_output.structures:
            atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))
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
