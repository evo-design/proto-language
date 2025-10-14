"""
Protein symmetry ring constraint for symmetric multimeric structures.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import numpy as np
from pydantic import Field

from ...base import Sequence
from proto_language.base_config import BaseConfig
from ..registry import ConstraintRegistry
from ....tools.models.structure_prediction.esmfold import ESMFoldConfig
from ....utils import (
    adjacent_distances,
    get_backbone_atoms,
    get_centroid,
    pairwise_distances,
    pdb_file_to_atomarray,
)
from ....tools.models.structure_prediction.esmfold import run_esmfold
from ....tools.tool_cache import ToolCache


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
        config: Configuration containing n_replications, all_to_all_protomer_symmetry, and esmfold_config parameters.

    Returns:
        Constraint score based on standard deviation of inter-protomer distances.
        Lower values indicate more symmetric ring-like arrangements.

    Examples:
        Evaluating ring symmetry:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldConfig(verbose=True)
        >>> cfg = ProteinSymmetryRingConfig(n_replications=6, all_to_all_protomer_symmetry=False, esmfold_config=kwargs)
        >>> score = protein_symmetry_ring_constraint(seq, config=cfg)  # Hexameric ring
    """
    from biotite.structure import get_chains

    # Create or copy ESMFold config
    if config.esmfold_config is None:
        esmfold_config = ESMFoldConfig()
    else:
        esmfold_config = ESMFoldConfig(**config.esmfold_config.model_dump(exclude={'sequences'}))
    
    # Extract config params for caching
    config_params = esmfold_config.model_dump(exclude={'sequences'})
    
    # Check cache
    cached_results = ToolCache.get_cached_results(
        input_sequence, "esmfold", n_replications=config.n_replications, **config_params
    )
    if not cached_results:
        # Run ESMFold prediction
        replicated_sequence = ":".join([input_sequence.sequence] * config.n_replications)
        esmfold_config.sequences = replicated_sequence
        output = run_esmfold(esmfold_config)
        
        # Store and cache results
        results = {
            "avg_plddt": output.avg_plddt,
            "ptm": output.ptm,
            "pdb_output": output.structure_pdb_output,
            "esmfolded_sequence": replicated_sequence,
        }
        ToolCache.cache_results(
            input_sequence, "esmfold", results,
            n_replications=config.n_replications, **config_params
        )
        input_sequence._metadata.update(results)
    else:
        input_sequence._metadata.update(cached_results)

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