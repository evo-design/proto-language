"""Protein symmetry ring constraint for symmetric multimeric structures."""

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
from proto_language.utils import MAX_ENERGY
from proto_language.utils.orf_selection import resolve_protein_complex_chains


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
    )
    esmfold_config: ESMFoldConfig = ConfigField(
        default_factory=ESMFoldConfig,
        title="ESMFold Config",
        description="ESMFold configuration for structure prediction.",
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
    input_labels=None,
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

    Each input tuple is folded as one complex with an arbitrary number of protein
    chains. DNA chains are first resolved with ORFipy by scanning both strands
    for canonical ATG-to-stop ORFs and selecting the longest ORF as that chain's
    translated CDS.

    Structure prediction is GPU-intensive and may take several minutes per protein
    depending on length and hardware.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of proposal tuples to
            evaluate. Each tuple may contain any number of protein and/or DNA
            chains. DNA chains are translated through the longest canonical ORF.

        config (ProteinSymmetryRingConfig): Configuration object containing
            ``max_symmetry_std`` (normalization threshold in Å, default: 10.0),
            ``all_to_all_protomer_symmetry`` (distance calculation mode, default: False),
            and optional ``esmfold_config`` for ESMFold settings.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0.0, 1.0]`` where 0.0 is
            perfect ring symmetry. Protein results attach the predicted ``Structure``
            to slot 0. ``metadata`` carries:

            - ``avg_plddt``: Float average pLDDT score for structure confidence (0.0-1.0)
            - ``ptm``: Float predicted TM-score for structure accuracy (0.0-1.0)
            - ``pdb_output``: String PDB format structure file content
            - ``esmfolded_sequence``: String colon-separated protein-chain representation
            - ``symmetry_std_raw``: Float raw standard deviation of inter-protomer
              distances in Ångströms (lower = more symmetric)
            - ``symmetry_score_normalized``: Float normalized symmetry score (0.0-1.0)
            - ``dna_chain_orfs``: Per-DNA-chain ORFipy metadata when DNA chains are present

    Examples:
        Designing a symmetric hexameric ring:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ProteinSymmetryRingConfig(
        ...     max_symmetry_std=10.0,
        ... )
        >>> results = protein_symmetry_ring_constraint([(seq, seq, seq, seq, seq, seq)], config)
        >>> print(results[0].score)  # e.g., 0.35 (3.5 Å std / 10.0 Å max)
        >>> print(results[0].metadata["symmetry_std_raw"])  # e.g., 3.5 Å
        >>> print(results[0].metadata["symmetry_score_normalized"])  # 0.35
    """
    resolved_complexes = resolve_protein_complex_chains(input_sequences)
    distance_func = pairwise_distances if config.all_to_all_protomer_symmetry else adjacent_distances
    results: list[ConstraintOutput | None] = [None] * len(input_sequences)
    complexes: list[StructurePredictionComplex] = []
    valid_indices: list[int] = []
    valid_metadata: list[dict[str, object]] = []
    valid_chain_sequences: list[list[str]] = []

    for idx, (chain_sequences, metadata) in enumerate(resolved_complexes):
        if chain_sequences is None:
            results[idx] = ConstraintOutput(score=MAX_ENERGY, metadata=metadata)
            continue
        complexes.append(
            StructurePredictionComplex(
                chains=[{"sequence": sequence, "entity_type": "protein"} for sequence in chain_sequences]
            )
        )
        metadata["esmfolded_sequence"] = ":".join(chain_sequences)
        valid_indices.append(idx)
        valid_metadata.append(metadata)
        valid_chain_sequences.append(chain_sequences)

    if complexes:
        output = run_esmfold(
            inputs=ESMFoldInput(complexes=complexes),
            config=config.esmfold_config,
        )

        for idx, chain_sequences, metadata, structure in zip(
            valid_indices, valid_chain_sequences, valid_metadata, output.structures, strict=True
        ):
            atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))

            centroids = []
            for chain_id in get_chains(atom_array):
                chain_backbone = get_backbone_atoms(atom_array[atom_array.chain_id == chain_id]).coord
                centroids.append(get_centroid(chain_backbone))

            if len(centroids) != len(chain_sequences):
                raise ValueError(f"Expected {len(chain_sequences)} centroids, got {len(centroids)}")
            centroids_arr = np.vstack(centroids)

            symmetry_std = float(np.std(distance_func(centroids_arr)))
            normalized_score = min(1.0, symmetry_std / config.max_symmetry_std)

            metadata.update(
                {
                    "avg_plddt": structure.metrics["avg_plddt"],
                    "ptm": structure.metrics["ptm"],
                    "pdb_output": structure.structure_pdb,
                    "symmetry_std_raw": symmetry_std,
                    "symmetry_score_normalized": normalized_score,
                }
            )
            results[idx] = ConstraintOutput(
                score=normalized_score,
                metadata=metadata,
                # ESMFold returns one full-complex structure, not per-chain structures.
                # Following the protein-structure constraint convention, attach it
                # to slot 0 as the canonical carrier; metadata is broadcast to all inputs.
                structures=(structure,) + (None,) * (len(chain_sequences) - 1),
            )

    return [result for result in results if result is not None]
