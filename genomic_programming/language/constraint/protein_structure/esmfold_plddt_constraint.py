"""
ESMFold pLDDT constraint for protein structure quality evaluation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ...base import Sequence
from ...base.config import BaseConfig
from ..registry import ConstraintRegistry
from ....schemas import ESMFoldKwargs
from ..utils import run_esmfold


class ESMFoldPLDDTConfig(BaseConfig):
    """Configuration for ESMFold pLDDT constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers, 2+ for oligomers (dimers, trimers, etc.). Higher values increase computational cost."
    )
    esmfold_kwargs: Optional[ESMFoldKwargs] = Field(
        default=None,
        description="Advanced ESMFold configuration parameters (residue_idx_offset, chain_linker, verbose). Leave as None to use defaults."
    )


@ConstraintRegistry.register(
    key="esmfold-plddt",
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
        config: Configuration containing n_replications and esmfold_kwargs parameters.

    Returns:
        Constraint score where 0.0 indicates perfect structure confidence (pLDDT = 1.0)
        and higher values indicate lower structure confidence.

    Examples:
        Evaluating protein structure confidence:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> # Using defaults:
        >>> cfg = ESMFoldPLDDTConfig()
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
        >>> # With custom args:
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> cfg = ESMFoldPLDDTConfig(n_replications=2, esmfold_kwargs=kwargs)
        >>> score = esmfold_plddt_constraint(seq, config=cfg)
    """

    run_esmfold(input_sequence, config.n_replications, config.esmfold_kwargs)
    return 1.0 - input_sequence._metadata["avg_plddt"]