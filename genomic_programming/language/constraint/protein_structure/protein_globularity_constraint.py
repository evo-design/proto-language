"""
Protein globularity constraint for compact protein structures.
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
    distances_to_centroid,
    get_backbone_atoms,
    pdb_file_to_atomarray,
)
from ....tools.models.structure_prediction.esmfold import run_esmfold


class ProteinGlobularityConfig(BaseConfig):
    """Configuration for protein globularity constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Advanced ESMFold configuration parameters. Leave as None to use defaults. Sequences are handled separately via ESMFoldInput.",
    )


@ConstraintRegistry.register(
    key="protein-globularity",
    label="Protein Globularity",
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
        >>> config = ESMFoldConfig(verbose=True)
        >>> cfg = ProteinGlobularityConfig(n_replications=1, esmfold_config=config)
        >>> score = protein_globularity_constraint(seq, config=cfg)
    """
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
    backbone = get_backbone_atoms(atom_array).coord
    return float(np.std(distances_to_centroid(backbone)))
