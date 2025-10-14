"""
Protein globularity constraint for compact protein structures.
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
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
)
from ....tools.models.structure_prediction.esmfold import run_esmfold
from ....tools.tool_cache import ToolCache


class ProteinGlobularityConfig(BaseConfig):
    """Configuration for protein globularity constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Advanced ESMFold configuration parameters. Leave as None to use defaults."
    )


@ConstraintRegistry.register(
    key="protein-globularity",
    config=ProteinGlobularityConfig,
    description="Encourage compact, globular protein structures",
    vectorized=False,
    concatenate=True,
    gpu_required=True
)
def protein_globularity_constraint(
    input_sequence: Sequence,
    config: ProteinGlobularityConfig
) -> float:
    """
    Encourage compact, globular protein structures.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing n_replications and esmfold_config parameters.

    Returns:
        Constraint score based on standard deviation of distances from backbone atoms to centroid.
        Lower values indicate more compact, globular structures.

    Examples:
        Evaluating protein globularity:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldConfig(verbose=True)
        >>> cfg = ProteinGlobularityConfig(n_replications=1, esmfold_config=kwargs)
        >>> score = protein_globularity_constraint(seq, config=cfg)
    """
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
    backbone = get_backbone_atoms(atom_array).coord
    return float(np.std(distances_to_centroid(backbone)))