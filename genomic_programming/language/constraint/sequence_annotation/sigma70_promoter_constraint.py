"""
σ70 promoter strength constraint for evaluating promoter strength.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from pydantic import Field

from ...core import Sequence
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry


class Sigma70PromoterConfig(BaseConfig):
    """Configuration for σ70 promoter strength constraint."""
    consensus_35: str = Field(default="TTGACA", description="-35 box consensus sequence (6 bp, typically TTGACA for E. coli σ70)")
    consensus_10: str = Field(default="TATAAT", description="-10 box consensus sequence (6 bp Pribnow box, typically TATAAT for E. coli σ70)")
    probs_35: List[float] = Field(default=[0.69, 0.79, 0.61, 0.56, 0.54, 0.54], description="Position-specific conservation probabilities for -35 box (6 values). From RegulonDB.")
    probs_10: List[float] = Field(default=[0.77, 0.76, 0.60, 0.61, 0.56, 0.82], description="Position-specific conservation probabilities for -10 box (6 values). From RegulonDB.")
    optimal_spacer: int = Field(default=17, description="Optimal spacer length between -35 and -10 boxes in base pairs (typically 17±1 bp)")
    spacer_sigma: float = Field(default=1.5, description="Standard deviation for spacer length penalty. Lower values = stricter spacing requirement.")
    spacer_weight: float = Field(default=0.3, description="Weight (0-1) for spacer penalty in total score. Higher = spacing more important.")
    gamma: float = Field(default=0.1, description="PWM score exponent for non-linearity. Lower values = more sensitive to mismatches.")
    k_opt: int = Field(default=8, description="Optimal number of matches to consensus (out of 12 total positions)")
    match_sigma: float = Field(default=2.0, description="Standard deviation for match count penalty")
    match_weight: float = Field(default=0.3, description="Weight (0-1) for match count penalty in total score")
    min_spacer: int = Field(default=14, description="Minimum acceptable spacer length in bp")
    max_spacer: int = Field(default=20, description="Maximum acceptable spacer length in bp")


@ConstraintRegistry.register(
    key="sigma70-promoter",
    config=Sigma70PromoterConfig,
    description="Evaluate σ70 promoter strength for DNA sequences",
    vectorized=True,
    concatenate=True
)
def sigma70_promoter_constraint(
    sequences: List[Sequence],
    config: Sigma70PromoterConfig
) -> List[float]:
    """
    Evaluate σ70 promoter strength for DNA sequences.
    Results are cached in each Sequence's metadata under key 'sigma70'.

    Args:
        sequences: List of DNA Sequences to evaluate.
        config: Configuration containing all promoter scoring parameters.

    Returns:
        List of constraint scores (0.0 = best promoter, 1.0 = worst).
    """

    CONS_35 = config.consensus_35.upper()
    CONS_10 = config.consensus_10.upper()
    PROBS_35 = np.array(config.probs_35)
    PROBS_10 = np.array(config.probs_10)
    max_pwm = np.prod(PROBS_35) * np.prod(PROBS_10)

    def _score_promoter(box35: str, box10: str, spacer_len: int):
        prob_35 = np.prod(
            [
                prob if b == c else (1.0 - prob)
                for b, c, prob in zip(box35, CONS_35, PROBS_35)
            ]
        )
        prob_10 = np.prod(
            [
                prob if b == c else (1.0 - prob)
                for b, c, prob in zip(box10, CONS_10, PROBS_10)
            ]
        )
        raw_pwm = prob_35 * prob_10
        normalized_pwm = (raw_pwm / max_pwm) if max_pwm > 0 else 0
        pwm_score = normalized_pwm ** config.gamma
        pwm_penalty = 1.0 - pwm_score

        total_matches = sum(a == c for a, c in zip(box35, CONS_35)) + sum(
            a == c for a, c in zip(box10, CONS_10)
        )
        match_dev = (total_matches - config.k_opt) / config.match_sigma
        match_penalty = 1.0 - math.exp(-(match_dev**2))

        spacer_dev = (spacer_len - config.optimal_spacer) / config.spacer_sigma
        spacer_penalty = 1.0 - math.exp(-(spacer_dev**2))

        box_penalty = (1 - config.match_weight) * pwm_penalty + config.match_weight * match_penalty
        total_penalty = (1 - config.spacer_weight) * box_penalty + config.spacer_weight * spacer_penalty

        return max(0.0, min(1.0, total_penalty)), {
            "pwm_penalty": pwm_penalty,
            "match_penalty": match_penalty,
            "spacer_penalty": spacer_penalty,
            "total_matches": total_matches,
            "spacer_len": spacer_len,
        }

    penalties: List[float] = []

    for seq_obj in sequences:
        seq = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        seq_len = len(seq)

        best_score, best_info = 1.0, {}
        if seq_len < 12:
            best_score, best_info = 1.0, {"reason": "too_short"}
        elif seq_len <= 32:  # treat as fixed promoter
            spacer_len = seq_len - 12
            if config.min_spacer <= spacer_len <= config.max_spacer:
                box35, box10 = seq[0:6], seq[-6:]
                best_score, best_info = _score_promoter(box35, box10, spacer_len)
                best_info.update({"pos": 0, "box35": box35, "box10": box10})
            else:
                best_score, best_info = 1.0, {"reason": "invalid_spacer"}
        else:  # scan for best
            for spacer_len in range(config.min_spacer, config.max_spacer + 1):
                promoter_len = 12 + spacer_len
                if promoter_len > seq_len:
                    continue
                for pos in range(seq_len - promoter_len + 1):
                    box35 = seq[pos : pos + 6]
                    box10 = seq[pos + 6 + spacer_len : pos + 12 + spacer_len]
                    score, info = _score_promoter(box35, box10, spacer_len)
                    if score < best_score:
                        best_score, best_info = score, {
                            **info,
                            "pos": pos,
                            "box35": box35,
                            "box10": box10,
                        }

        # Cache results in metadata
        seq_obj._metadata["sigma70"] = {
            "sigma70_score": best_score,
            **best_info,
        }
        penalties.append(best_score)

    return penalties
