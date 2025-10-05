"""
Promoter strength constraint using Barrick Lab Promoter Calculator.
"""

from __future__ import annotations

from typing import List, Optional

from promoter_calculator.wrapper import promoter_calculator
from pydantic import Field

from ...base import Sequence
from ...base.config import BaseConfig
from ..registry import ConstraintRegistry


class PromoterStrengthConfig(BaseConfig):
    """Configuration for promoter strength constraint."""
    add_context: bool = Field(default=False, description="If True, adds flanking nucleotides to short sequences to meet calculator length minimums")
    context_length: int = Field(default=10, description="Number of 'A' nucleotides to add on each end when add_context=True")
    threads: int = Field(default=8, ge=1, description="Number of threads for parallel processing of promoter calculations")
    verbosity: int = Field(default=0, ge=0, description="Verbosity level for promoter calculator output (0=quiet, higher=more verbose)")
    circular: bool = Field(default=False, description="If True, treat sequences as circular for promoter detection across ends")
    batch_size: Optional[int] = Field(default=None, description="Max sequences per batch (None=process all together). Use for memory control.")
    scoring_type: str = Field(default="dG", description="Score type to use: 'dG' (binding free energy) or 'tx_rate' (transcription rate)")


@ConstraintRegistry.register(
    key="promoter-strength",
    config=PromoterStrengthConfig,
    description="Evaluate promoter strength using Barrick Lab Promoter Calculator",
    vectorized=True,
    concatenate=True
)
def promoter_strength_constraint(
    sequences: List[Sequence],
    config: PromoterStrengthConfig
) -> List[float]:
    """
    Run Barrick Lab Promoter Calculator and return a [0,1] penalty score.

    Caches the full promoter_calculator output in each Sequence's metadata
    under the "promoter_strength" key.

    Penalty scheme:
        For tx_rate scoring:
        - Tx_rate < 1500: weak (penalty = 1.0)
        - 1500–5000: moderate (linear from 1.0 → 0.5)
        - > 5000: strong (linear from 0.5 → 0.0, capped at 10,000)
        
        For dG scoring:
        - dG > -1.5: weak (penalty = 1.0)
        - dG between -1.5 and -3.0: moderate (linear scale from 1.0 → 0.5)
        - dG < -3.0: strong (linear from 0.5 → 0.0, capped at -5.0)

    Args:
        sequences: List of DNA Sequences to evaluate.
        config: Configuration containing scoring parameters and processing options.

    Returns:
        List of penalty scores (0.0 = strong promoter, 1.0 = weak/no promoter).
    """

    # Clean all sequences
    processed_sequences = []
    for seq_obj in sequences:
        s = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        if config.add_context:
            s = ("A" * config.context_length) + s + ("A" * config.context_length)
        processed_sequences.append(s)

    penalties: List[float] = []

    # Process in batches if batch_size is specified, otherwise all at once
    if config.batch_size and config.batch_size < len(processed_sequences):
        # Process in batches
        all_results = []
        for i in range(0, len(processed_sequences), config.batch_size):
            batch = processed_sequences[i : i + config.batch_size]
            batch_results = []
            for seq in batch:
                res = (
                    promoter_calculator(
                        seq, threads=config.threads, verbosity=config.verbosity, circular=config.circular
                    )
                    or []
                )
                batch_results.append(res)
            all_results.extend(batch_results)
    else:
        try:
            # Attempt batch processing
            all_results = promoter_calculator(
                processed_sequences,
                threads=config.threads,
                verbosity=config.verbosity,
                circular=config.circular,
            )
            if not isinstance(all_results[0], list):
                raise NotImplementedError("Batch processing format not recognized")
        except (TypeError, AttributeError, NotImplementedError):
            all_results = []
            for seq in processed_sequences:
                res = (
                    promoter_calculator(
                        seq, threads=config.threads, verbosity=config.verbosity, circular=config.circular
                    )
                    or []
                )
                all_results.append(res)

    # Process results for each sequence
    for seq_obj, res in zip(sequences, all_results):
        # Keep only + strand
        res = [r for r in res if getattr(r, "strand", "+") == "+"]

        if not res:
            penalty = 1.0
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "reason": "no_promoter_found",
                "raw_output": [],
            }
            penalties.append(penalty)
            continue

        if config.scoring_type == "tx_rate":
            # Extract tx_rate
            tx_rate = max(float(r.Tx_rate) for r in res if hasattr(r, "Tx_rate"))

            # Penalty mapping
            if tx_rate < 1500.0:
                penalty = 1.0
            elif tx_rate <= 5000.0:
                penalty = 1.0 - 0.5 * ((tx_rate - 1000.0) / (5000.0 - 1000.0))
            else:
                penalty = 0.5 - min((tx_rate - 5000.0) / 5000.0 * 0.5, 0.5)
                penalty = max(0.0, penalty)

            # Store metadata
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "tx_rate": tx_rate,
                "raw_output": [r.__dict__ for r in res],
            }
            penalties.append(penalty)
        else:
            dG = min(float(r.dG_total) for r in res if hasattr(r, "dG_total"))
            if dG >= 0:
                penalty = 1.0
            elif dG > -1.5:
                penalty = 1.0
            elif dG >= -3.0:
                penalty = 1.0 - 0.5 * ((dG + 1.5) / -1.5)
            else:
                normalized = (dG - (-3.0)) / (-5.0 - (-3.0))
                penalty = 0.5 * (1 - normalized**2)
                penalty = max(0.0, min(0.5, penalty))

            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "dG_rate": dG,
                "raw_output": [r.__dict__ for r in res],
            }
            penalties.append(penalty)
    
    return penalties
