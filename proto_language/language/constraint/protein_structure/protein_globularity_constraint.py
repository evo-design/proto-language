"""Protein globularity constraint for compact protein structures."""

from io import StringIO

import numpy as np
from proto_tools import (
    ESMFoldConfig,
    ESMFoldInput,
    StructurePredictionComplex,
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
    run_esmfold,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.orf_selection import resolve_protein_complex_chains


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
    tools_called=["esmfold-prediction", "orfipy-prediction"],
    category="protein_structure",
    supported_sequence_types=["dna", "protein"],
    input_labels=None,
)
def protein_globularity_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinGlobularityConfig
) -> list[ConstraintOutput]:
    """Encourage compact, globular protein structures using ESMFold.

    This constraint function uses ESMFold to predict protein 3D structures
    and evaluates their compactness by analyzing the spatial distribution of
    backbone atoms. Globularity is measured as the standard deviation of distances
    from backbone atoms (N, CA, C, O) to the structure's geometric centroid.
    Lower values indicate more compact, spherical structures characteristic of
    well-folded globular proteins, while higher values indicate extended,
    elongated, or poorly folded structures.

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

        config (ProteinGlobularityConfig): Configuration object containing
            ``max_globularity`` and optional ``esmfold_config`` for advanced
            ESMFold settings.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0.0, 1.0]`` (lower = more
            compact). The predicted complex ``Structure`` attaches to slot 0.
            ``metadata`` carries:

            - ``avg_plddt``: Float average pLDDT score for structure confidence (0.0-1.0)
            - ``ptm``: Float predicted TM-score for structure accuracy (0.0-1.0)
            - ``pdb_output``: String PDB format structure file content
            - ``esmfolded_sequence``: String colon-separated protein-chain representation
            - ``raw_globularity``: Float standard deviation of backbone-to-centroid
              distances in Ångströms (lower = more compact)
            - ``normalized_globularity``: Float normalized globularity score (0.0-1.0,
              capped by max_globularity)
            - ``dna_chain_orfs``: Per-DNA-chain ORFipy metadata when DNA chains are present

    Examples:
        Evaluating protein structural compactness:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ProteinGlobularityConfig()
        >>> results = protein_globularity_constraint([(seq,)], config)
        >>> print(results[0].score)  # e.g., 0.425 (normalized score, lower = more compact)
        >>> print(results[0].metadata["raw_globularity"])  # e.g., 8.5 (raw Ångströms)
        >>> print(results[0].metadata["normalized_globularity"])  # e.g., 0.425
        >>> print(results[0].metadata["avg_plddt"])  # e.g., 0.85 (also available)

        Evaluating DNA sequence (with automatic ORF prediction):

        >>> dna_seq = Sequence("ATGGTACTGAGCCCAGCG...", "dna")
        >>> config = ProteinGlobularityConfig()
        >>> results = protein_globularity_constraint([(dna_seq,)], config)
        >>> print(results[0].score)  # Normalized score (0.0-1.0)
        >>> # Single-DNA-chain proposals also flatten selected-CDS metadata.
        >>> print(results[0].metadata["orfipy_orf_count"])  # e.g., 2
        >>> print(results[0].metadata["selected_cds"]["amino_acid_length"])  # longest ORF length
        >>> # Multi-chain proposals carry per-chain CDS metadata.
        >>> print(results[0].metadata["translated_cds_by_chain"][0]["amino_acid_length"])
        >>> print(results[0].metadata["raw_globularity"])  # e.g., 7.8 Å
    """
    resolved_complexes = resolve_protein_complex_chains(input_sequences)
    results: list[ConstraintOutput | None] = [None] * len(input_sequences)
    complexes: list[StructurePredictionComplex] = []
    valid_indices: list[int] = []
    valid_metadata: list[dict[str, object]] = []

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

    if complexes:
        output = run_esmfold(inputs=ESMFoldInput(complexes=complexes), config=config.esmfold_config)
        for idx, comp, metadata, structure in zip(
            valid_indices, complexes, valid_metadata, output.structures, strict=True
        ):
            atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))
            backbone = get_backbone_atoms(atom_array).coord
            raw_globularity = float(np.std(distances_to_centroid(backbone)))
            normalized_globularity = min(1.0, raw_globularity / config.max_globularity)

            metadata.update(
                {
                    "avg_plddt": structure.metrics["avg_plddt"],
                    "ptm": structure.metrics["ptm"],
                    "pdb_output": structure.structure_pdb,
                    "raw_globularity": raw_globularity,
                    "normalized_globularity": normalized_globularity,
                }
            )
            results[idx] = ConstraintOutput(
                score=normalized_globularity,
                metadata=metadata,
                structures=(structure,) + (None,) * (len(comp.chains) - 1),
            )

    return [result for result in results if result is not None]
