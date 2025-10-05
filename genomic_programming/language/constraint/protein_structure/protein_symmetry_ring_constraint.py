"""
Protein symmetry ring constraint for symmetric multimeric structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import numpy as np
from pydantic import Field

from ...base import Sequence
from ...base.config import BaseConfig
from ..registry import ConstraintRegistry
from ....schemas import ESMFoldKwargs
from ....utils import (
    adjacent_distances,
    get_backbone_atoms,
    get_centroid,
    pairwise_distances,
    pdb_file_to_atomarray,
)
from ..utils import run_esmfold


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
    esmfold_kwargs: Optional[ESMFoldKwargs] = Field(
        default=None,
        description="Advanced ESMFold configuration parameters. Leave as None to use defaults."
    )


@ConstraintRegistry.register(
    key="protein-symmetry-ring",
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
        config: Configuration containing n_replications, all_to_all_protomer_symmetry, and esmfold_kwargs parameters.

    Returns:
        Constraint score based on standard deviation of inter-protomer distances.
        Lower values indicate more symmetric ring-like arrangements.

    Examples:
        Evaluating ring symmetry:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> cfg = ProteinSymmetryRingConfig(n_replications=6, all_to_all_protomer_symmetry=False, esmfold_kwargs=kwargs)
        >>> score = protein_symmetry_ring_constraint(seq, config=cfg)  # Hexameric ring
    """
    from biotite.structure import get_chains

    run_esmfold(input_sequence, config.n_replications, config.esmfold_kwargs)

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))

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