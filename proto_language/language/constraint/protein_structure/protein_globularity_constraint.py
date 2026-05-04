"""Protein globularity constraint for compact protein structures."""

import logging
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
from proto_language.storage import FileType, store_file
from proto_language.utils import MAX_ENERGY
from proto_language.utils.orf_selection import predict_longest_canonical_cds

logger = logging.getLogger(__name__)


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
    tools_called=["esmfold-prediction", "orfipy-prediction"],
    category="protein_structure",
    supported_sequence_types=["dna", "protein"],
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

    For DNA sequences, the function first uses ORFipy to scan both strands for
    canonical ATG-to-stop ORFs, selects the longest ORF as the single CDS for
    that proposal, and evaluates the globularity of the translated protein.

    Structure prediction is GPU-intensive and may take several minutes per protein
    depending on length and hardware.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples to
            evaluate. Each tuple contains one protein or DNA sequence. All sequences
            must be the same type. For DNA sequences, canonical ORF prediction is
            performed automatically and the longest ORF is scored.

        config (ProteinGlobularityConfig): Configuration object containing
            ``n_replications`` (oligomeric state, default: 1) and optional
            ``esmfold_config`` for advanced ESMFold settings.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0.0, 1.0]`` (lower = more
            compact). For protein inputs the result also attaches the predicted
            ``Structure`` to slot 0. ``metadata`` carries:

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

            - ``orfipy_orfs``: Stored JSON of canonical ORFs detected by ORFipy
            - ``orfipy_orf_count``: Integer count of candidate ORFs
            - ``selected_cds``: Coordinates and length for the longest ORF used as
              the single CDS for scoring
            - ``esmfold_cds_globularity``: Float globularity score for the selected
              CDS protein (in Ångströms)
            - ``esmfold_normalized_globularity``: Float normalized selected-CDS
              globularity (0.0-1.0, capped by max_globularity)

    Examples:
        Evaluating protein structural compactness:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = ProteinGlobularityConfig(n_replications=1)
        >>> results = protein_globularity_constraint([(seq,)], config)
        >>> print(results[0].score)  # e.g., 0.425 (normalized score, lower = more compact)
        >>> print(results[0].metadata["raw_globularity"])  # e.g., 8.5 (raw Ångströms)
        >>> print(results[0].metadata["normalized_globularity"])  # e.g., 0.425
        >>> print(results[0].metadata["avg_plddt"])  # e.g., 0.85 (also available)

        Evaluating DNA sequence (with automatic ORF prediction):

        >>> dna_seq = Sequence("ATGGTACTGAGCCCAGCG...", "dna")
        >>> config = ProteinGlobularityConfig(n_replications=1)
        >>> results = protein_globularity_constraint([(dna_seq,)], config)
        >>> print(results[0].score)  # Normalized score (0.0-1.0)
        >>> print(results[0].metadata["orfipy_orf_count"])  # e.g., 2
        >>> print(results[0].metadata["selected_cds"]["amino_acid_length"])  # longest ORF length
        >>> print(results[0].metadata["esmfold_cds_globularity"])  # e.g., 7.8 Å
    """
    sequences = [seq for (seq,) in input_sequences]
    if sequences[0].sequence_type == "protein":
        return _evaluate_protein_globularity(sequences, config)
    return _evaluate_dna_globularity(sequences, config)


def _evaluate_protein_globularity(
    protein_sequences: list[Sequence], config: ProteinGlobularityConfig
) -> list[ConstraintOutput]:
    """Evaluate protein globularity directly."""
    complexes = [
        StructurePredictionComplex(
            chains=[{"sequence": seq.sequence, "entity_type": "protein"}] * config.n_replications
        )
        for seq in protein_sequences
    ]

    esmfold_input = ESMFoldInput(complexes=complexes)
    output = run_esmfold(inputs=esmfold_input, config=config.esmfold_config)

    results: list[ConstraintOutput] = []
    for comp, structure in zip(complexes, output.structures, strict=False):
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))
        backbone = get_backbone_atoms(atom_array).coord
        raw_globularity = float(np.std(distances_to_centroid(backbone)))
        normalized_globularity = min(1.0, raw_globularity / config.max_globularity)

        results.append(
            ConstraintOutput(
                score=normalized_globularity,
                metadata={
                    "avg_plddt": structure.metrics["avg_plddt"],
                    "ptm": structure.metrics["ptm"],
                    "pdb_output": store_file(structure.structure_pdb, FileType.PDB),
                    "esmfolded_sequence": comp.chains,
                    "raw_globularity": raw_globularity,
                    "normalized_globularity": normalized_globularity,
                },
                structures=(structure,),
            )
        )

    return results


def _evaluate_dna_globularity(
    dna_sequences: list[Sequence], config: ProteinGlobularityConfig
) -> list[ConstraintOutput]:
    """Evaluate DNA sequences via the longest canonical ORF on either strand."""
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
                "protein-globularity: ESMFold failed for selected DNA CDS %s: %s; using worst score",
                selected_orf.id,
                e,
            )
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={**metadata, "globularity_error": f"ESMFold failed: {e}"},
                )
            )
            continue

        structure = esmfold_output.structures[0]
        atom_array = pdb_file_to_atomarray(StringIO(structure.structure_pdb))
        backbone = get_backbone_atoms(atom_array).coord
        raw_globularity = float(np.std(distances_to_centroid(backbone)))
        globularity_score = min(1.0, raw_globularity / config.max_globularity)
        metadata["esmfold_cds_globularity"] = raw_globularity
        metadata["esmfold_normalized_globularity"] = globularity_score
        results.append(ConstraintOutput(score=globularity_score, metadata=metadata))

    return results
