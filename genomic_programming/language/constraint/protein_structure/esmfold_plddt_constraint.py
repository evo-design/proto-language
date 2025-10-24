"""
ESMFold pLDDT constraint for protein structure quality evaluation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ...core import Sequence
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.models.structure_prediction.esmfold import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
)


class ESMFoldPLDDTConfig(BaseConfig):
    """Configuration for ESMFold pLDDT constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers, 2+ for oligomers (dimers, trimers, etc.). Higher values increase computational cost."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Optional ESMFold configuration (residue_idx_offset, chain_linker, verbose). If None, uses defaults.",
    )


@ConstraintRegistry.register(
    key="esmfold-plddt",
    label="ESMFold pLDDT Score",
    config=ESMFoldPLDDTConfig,
    description="Evaluate protein structure quality using ESMFold predicted LDDT score",
    vectorized=False,
    concatenate=True,
    gpu_required=True
)
def esmfold_plddt_constraint(
    input_sequence: Sequence,
    config: ESMFoldPLDDTConfig
) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted LDDT (pLDDT) score.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing n_replications and esmfold_config parameters.

    Returns:
        Constraint score where 0.0 indicates perfect structure confidence (pLDDT = 1.0)
        and higher values indicate lower structure confidence.

    Examples:
        Evaluating protein structure confidence:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> # Using defaults:
        >>> cfg = ESMFoldPLDDTConfig()
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
        >>> # With custom ESMFold parameters:
        >>> esmfold_cfg = ESMFoldConfig(residue_idx_offset=256, verbose=True)
        >>> cfg = ESMFoldPLDDTConfig(n_replications=2, esmfold_config=esmfold_cfg)
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
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

    return 1.0 - output.avg_plddt
