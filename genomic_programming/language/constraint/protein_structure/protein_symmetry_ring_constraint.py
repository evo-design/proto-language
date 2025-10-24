"""
Protein symmetry ring constraint for symmetric multimeric structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import numpy as np
from pydantic import Field

from ...core import Sequence
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.models.structure_prediction.esmfold import ESMFoldInput, ESMFoldConfig
from ....utils import (
    adjacent_distances,
    get_backbone_atoms,
    get_centroid,
    pairwise_distances,
    pdb_file_to_atomarray,
)
from ....tools.models.structure_prediction.esmfold import run_esmfold


class ProteinSymmetryRingConfig(BaseConfig):
    """Configuration for protein symmetry ring constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of protomers in the ring structure (2-12 typical). Defines the oligomeric state (dimer=2, trimer=3, hexamer=6, etc.)."
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
    vectorized=False,
    concatenate=True,
    gpu_required=True
)
def protein_symmetry_ring_constraint(
    input_sequence: Sequence,
    config: ProteinSymmetryRingConfig
) -> float:
    """
    Constrain a protein to form a symmetric ring-like multimeric structure.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing n_replications, all_to_all_protomer_symmetry, and esmfold_config parameters.

    Returns:
        Constraint score based on standard deviation of inter-protomer distances.
        Lower values indicate more symmetric ring-like arrangements.

    Examples:
        Evaluating ring symmetry:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> config = ESMFoldConfig(verbose=True)
        >>> cfg = ProteinSymmetryRingConfig(n_replications=6, all_to_all_protomer_symmetry=False, esmfold_config=config)
        >>> score = protein_symmetry_ring_constraint(seq, config=cfg)  # Hexameric ring
    """
    from biotite.structure import get_chains

    # Prepare replicated sequence for multimer prediction
    replicated_sequence = ":".join([input_sequence.sequence] * config.n_replications)

    # Create ESMFoldInput and ESMFoldConfig
    esmfold_input = ESMFoldInput(sequences=replicated_sequence)
    esmfold_config = (
        config.esmfold_config if config.esmfold_config is not None else ESMFoldConfig()
    )

    # Run ESMFold prediction (caching handled transparently by decorator)
    output = run_esmfold(inputs=esmfold_input, config=esmfold_config)

    # Store results in metadata
    input_sequence._metadata.update({
        "avg_plddt": output.avg_plddt,
        "ptm": output.ptm,
        "pdb_output": output.structure_pdb_output,
        "esmfolded_sequence": replicated_sequence,
    })

    atom_array = pdb_file_to_atomarray(StringIO(output.structure_pdb_output))

    centroids = []
    for chain_id in get_chains(atom_array):
        chain_backbone = get_backbone_atoms(
            atom_array[atom_array.chain_id == chain_id]
        ).coord
        centroids.append(get_centroid(chain_backbone))

    assert len(centroids) == config.n_replications
    centroids = np.vstack(centroids)

    distance_func = (
        pairwise_distances if config.all_to_all_protomer_symmetry else adjacent_distances
    )
    return float(np.std(distance_func(centroids)))