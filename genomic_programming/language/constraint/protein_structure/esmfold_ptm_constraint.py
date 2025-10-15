"""
ESMFold pTM constraint for protein structure quality evaluation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ...core import Sequence
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.models.structure_prediction.esmfold import run_esmfold, ESMFoldConfig


class ESMFoldPTMConfig(BaseConfig):
    """Configuration for ESMFold pTM constraint."""
    n_replications: int = Field(
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for multimeric structure prediction. Use 1 for monomers, 2+ for oligomers (dimers, trimers, etc.). Higher values increase computational cost."
    )
    esmfold_config: Optional[ESMFoldConfig] = Field(
        default=None,
        description="Optional ESMFold configuration (residue_idx_offset, chain_linker, verbose). If None, uses defaults. Sequences field will be set programmatically from the input sequence."
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
        config: Configuration containing n_replications and esmfold_config parameters.

    Returns:
        Constraint score where 0.0 indicates perfect structure quality (pTM = 1.0)
        and higher values indicate lower structure quality.

    Examples:
        Evaluating protein structure quality:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> # Using defaults:
        >>> cfg = ESMFoldPTMConfig()
        >>> score = esmfold_ptm_constraint(seq, config=cfg)
        >>> # With custom ESMFold parameters:
        >>> esmfold_cfg = ESMFoldConfig(verbose=True, residue_idx_offset=256)
        >>> cfg = ESMFoldPTMConfig(n_replications=2, esmfold_config=esmfold_cfg)
        >>> score = esmfold_ptm_constraint(seq, config=cfg)
    """
    # Create or copy ESMFold config
    if config.esmfold_config is None:
        esmfold_config = ESMFoldConfig()
    else:
        # Copy to avoid mutating the input
        esmfold_config = ESMFoldConfig(**config.esmfold_config.model_dump(exclude={'sequences'}))

    # Prepare replicated sequence for multimer prediction
    replicated_sequence = ":".join([input_sequence.sequence] * config.n_replications)
    esmfold_config.sequences = replicated_sequence

    # Run ESMFold prediction (caching handled transparently by decorator)
    output = run_esmfold(esmfold_config)

    # Store results in metadata
    input_sequence._metadata.update({
        "avg_plddt": output.avg_plddt,
        "ptm": output.ptm,
        "pdb_output": output.structure_pdb_output,
        "esmfolded_sequence": replicated_sequence,
    })

    return 1.0 - output.ptm