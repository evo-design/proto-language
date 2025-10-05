"""
Protein globularity constraint for compact protein structures.
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
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
)
from ..utils import run_esmfold


class ProteinGlobularityConfig(BaseConfig):
    """Configuration for protein globularity constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers."
    )
    esmfold_kwargs: Optional[ESMFoldKwargs] = Field(
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
        config: Configuration containing n_replications and esmfold_kwargs parameters.

    Returns:
        Constraint score based on standard deviation of distances from backbone atoms to centroid.
        Lower values indicate more compact, globular structures.

    Examples:
        Evaluating protein globularity:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> cfg = ProteinGlobularityConfig(n_replications=1, esmfold_kwargs=kwargs)
        >>> score = protein_globularity_constraint(seq, config=cfg)
    """
    run_esmfold(input_sequence, config.n_replications, config.esmfold_kwargs)

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))
    backbone = get_backbone_atoms(atom_array).coord
    return float(np.std(distances_to_centroid(backbone)))