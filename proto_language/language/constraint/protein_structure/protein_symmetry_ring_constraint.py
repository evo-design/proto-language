"""Protein symmetry ring constraint for symmetric multimeric structures."""

import logging
from io import StringIO

import numpy as np
from biotite.structure import get_chains
from proto_tools import (
    ESMFoldConfig,
    ESMFoldInput,
    StructurePredictionComplex,
    adjacent_distances,
    get_backbone_atoms,
    get_centroid,
    pairwise_distances,
    pdb_file_to_atomarray,
    run_esmfold,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.storage import FileType, store_file
from proto_language.utils import MAX_ENERGY
from proto_language.utils.orf_selection import predict_longest_canonical_cds

logger = logging.getLogger(__name__)


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
    tools_called=["esmfold-prediction", "orfipy-prediction"],
    category="protein_structure",
    supported_sequence_types=["dna", "protein"],
)
def protein_symmetry_ring_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinSymmetryRingConfig
) -> list[ConstraintOutput]:
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

    For DNA sequences, the function first uses ORFipy to scan both strands for
    canonical ATG-to-stop ORFs, selects the longest ORF as the single CDS for
    that proposal, and evaluates the ring symmetry of the translated protein.

    Structure prediction is GPU-intensive and may take several minutes per protein
    depending on length and hardware.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples to
            evaluate. Each tuple contains one protein or DNA sequence. All sequences
            must be the same type. For DNA sequences, canonical ORF prediction is
            performed automatically and the longest ORF is scored.

        config (ProteinSymmetryRingConfig): Configuration object containing
            ``n_replications`` (number of protomers in ring, default: 2),
            ``max_symmetry_std`` (normalization threshold in Å, default: 10.0),
            ``all_to_all_protomer_symmetry`` (distance calculation mode, default: False),
            and optional ``esmfold_config`` for advanced ESMFold settings.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0.0, 1.0]`` where 0.0 is
            perfect ring symmetry. Protein results attach the predicted ``Structure``
            to slot 0. ``metadata`` carries:

            **For protein sequences:**

            - ``avg_plddt``: Float average pLDDT score for structure confidence (0.0-1.0)
            - ``ptm``: Float predicted TM-score for structure accuracy (0.0-1.0)
            - ``pdb_output``: String PDB format structure file content
            - ``esmfolded_sequence``: String colon-separated sequence representation
            - ``symmetry_std_raw``: Float raw standard deviation of inter-protomer
              distances in Ångströms (lower = more symmetric)
            - ``symmetry_score_normalized``: Float normalized symmetry score (0.0-1.0)

            **For DNA sequences:**

            - ``orfipy_orfs``: Stored JSON of canonical ORFs detected by ORFipy
            - ``orfipy_orf_count``: Integer count of candidate ORFs
            - ``selected_cds``: Coordinates and length for the longest ORF used as
              the single CDS for scoring
            - ``esmfold_cds_symmetry_std``: Float symmetry standard deviation for
              the selected CDS protein (in Ångströms)
            - ``esmfold_normalized_symmetry``: Float normalized selected-CDS
              symmetry score (0.0-1.0, capped by max_symmetry_std)

    Examples:
        Designing a symmetric hexameric ring:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ProteinSymmetryRingConfig(
        ...     n_replications=6,  # Hexamer
        ...     max_symmetry_std=10.0,
        ... )
        >>> results = protein_symmetry_ring_constraint([(seq,)], config)
        >>> print(results[0].score)  # e.g., 0.35 (3.5 Å std / 10.0 Å max)
        >>> print(results[0].metadata["symmetry_std_raw"])  # e.g., 3.5 Å
        >>> print(results[0].metadata["symmetry_score_normalized"])  # 0.35
    """
    sequences = [seq for (seq,) in input_sequences]
    by_type: dict[str, list[Sequence]] = {"dna": [], "protein": []}
    for seq in sequences:
        by_type[seq.sequence_type].append(seq)

    per_proposal: list[ConstraintOutput | None] = [None] * len(input_sequences)

    if by_type["protein"]:
        protein_results = _evaluate_protein_symmetry(by_type["protein"], config)
        _map_results_to_original(sequences, by_type["protein"], protein_results, per_proposal)

    if by_type["dna"]:
        dna_results = _evaluate_dna_symmetry(by_type["dna"], config)
        _map_results_to_original(sequences, by_type["dna"], dna_results, per_proposal)

    return [r for r in per_proposal if r is not None]


def _evaluate_protein_symmetry(
    protein_sequences: list[Sequence], config: ProteinSymmetryRingConfig
) -> list[ConstraintOutput]:
    """Evaluate protein ring symmetry directly."""
    complexes = [
        StructurePredictionComplex(
            chains=[{"sequence": seq.sequence, "entity_type": "protein"}] * config.n_replications
        )
        for seq in protein_sequences
    ]

    output = run_esmfold(
        inputs=ESMFoldInput(complexes=complexes),
        config=config.esmfold_config,
    )

    distance_func = pairwise_distances if config.all_to_all_protomer_symmetry else adjacent_distances

    results: list[ConstraintOutput] = []
    for seq, structure in zip(protein_sequences, output.structures, strict=False):
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))

        centroids = []
        for chain_id in get_chains(atom_array):
            chain_backbone = get_backbone_atoms(atom_array[atom_array.chain_id == chain_id]).coord
            centroids.append(get_centroid(chain_backbone))

        if len(centroids) != config.n_replications:
            raise ValueError(f"Expected {config.n_replications} centroids, got {len(centroids)}")
        centroids_arr = np.vstack(centroids)

        symmetry_std = float(np.std(distance_func(centroids_arr)))
        normalized_score = min(1.0, symmetry_std / config.max_symmetry_std)

        results.append(
            ConstraintOutput(
                score=normalized_score,
                metadata={
                    "avg_plddt": structure.metrics["avg_plddt"],
                    "ptm": structure.metrics["ptm"],
                    "pdb_output": store_file(structure.structure_pdb, FileType.PDB),
                    "esmfolded_sequence": ":".join([seq.sequence] * config.n_replications),
                    "symmetry_std_raw": symmetry_std,
                    "symmetry_score_normalized": normalized_score,
                },
                structures=(structure,),
            )
        )

    return results


def _evaluate_dna_symmetry(dna_sequences: list[Sequence], config: ProteinSymmetryRingConfig) -> list[ConstraintOutput]:
    """Evaluate DNA sequences via the longest canonical ORF on either strand."""
    distance_func = pairwise_distances if config.all_to_all_protomer_symmetry else adjacent_distances
    results: list[ConstraintOutput] = []

    for selected_orf, metadata in predict_longest_canonical_cds(dna_sequences):
        if selected_orf is None:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata=metadata))
            continue

        complexes = [
            StructurePredictionComplex(
                chains=[{"sequence": selected_orf.amino_acid_sequence, "entity_type": "protein"}]
                * config.n_replications
            )
        ]

        try:
            esmfold_output = run_esmfold(
                inputs=ESMFoldInput(complexes=complexes),
                config=config.esmfold_config,
            )
        except Exception as e:
            logger.warning(
                "protein-symmetry-ring: ESMFold failed for selected DNA CDS %s: %s; using worst score",
                selected_orf.id,
                e,
            )
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={**metadata, "symmetry_ring_error": f"ESMFold failed: {e}"},
                )
            )
            continue

        structure = esmfold_output.structures[0]
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))
        centroids = []
        for chain_id in get_chains(atom_array):
            chain_backbone = get_backbone_atoms(atom_array[atom_array.chain_id == chain_id]).coord
            centroids.append(get_centroid(chain_backbone))

        centroids_arr = np.vstack(centroids)
        symmetry_std = float(np.std(distance_func(centroids_arr)))
        normalized_score = min(1.0, symmetry_std / config.max_symmetry_std)

        metadata["esmfold_cds_symmetry_std"] = symmetry_std
        metadata["esmfold_normalized_symmetry"] = normalized_score
        results.append(ConstraintOutput(score=normalized_score, metadata=metadata))

    return results


def _map_results_to_original(
    all_sequences: list[Sequence],
    subset_sequences: list[Sequence],
    subset_results: list[ConstraintOutput],
    per_proposal: list[ConstraintOutput | None],
) -> None:
    """Map subset results back to original sequence order."""
    subset_idx = 0
    for i, seq in enumerate(all_sequences):
        if seq in subset_sequences:
            per_proposal[i] = subset_results[subset_idx]
            subset_idx += 1
