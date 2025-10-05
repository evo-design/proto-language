"""
ESMFold pTM constraint for protein structure quality evaluation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ...base import Sequence
from ...base.config import BaseConfig
from ..registry import ConstraintRegistry
from ....schemas import ESMFoldKwargs
from ..utils import run_esmfold


class ESMFoldPTMConfig(BaseConfig):
    """Configuration for ESMFold pTM constraint."""
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
    key="esmfold-ptm",
    config=ESMFoldPTMConfig,
    description="Evaluate protein structure quality using ESMFold predicted TM-score",
    vectorized=False,
    concatenate=True,
    gpu_required=True
)
def esmfold_ptm_constraint(
    input_sequence: Sequence,
    config: ESMFoldPTMConfig
) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted TM-score (pTM).

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration containing n_replications and esmfold_kwargs parameters.

    Returns:
        Constraint score where 0.0 indicates perfect structure quality (pTM = 1.0)
        and higher values indicate lower structure quality.

    Examples:
        Evaluating protein structure quality:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> cfg = ESMFoldPTMConfig(n_replications=1, esmfold_kwargs=kwargs)
        >>> score = esmfold_ptm_constraint(seq, config=cfg)
    """

    run_esmfold(input_sequence, config.n_replications, config.esmfold_kwargs)
    return 1.0 - input_sequence._metadata["ptm"]