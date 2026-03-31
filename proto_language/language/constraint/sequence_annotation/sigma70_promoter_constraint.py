"""
proto_language/language/constraint/sequence_annotation/sigma70_promoter_constraint.py

sigma-70 promoter similarity constraint for evaluating promoter similarity.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence


class Sigma70PromoterConfig(BaseConfig):
    """Configuration for sigma-70 promoter similarity constraint.

    This class defines configuration parameters for evaluating bacterial promoter
    similarity using a position weight matrix (PWM) model of E. coli sigma-70 promoters.
    The model scores promoter elements based on similarity to consensus sequences
    for the -35 and -10 boxes, the spacer distance between them, and
    the total number of matches to consensus. This approach is based on RegulonDB
    experimental data for E. coli sigma-70-dependent promoters.

    The scoring combines three components:
    1. PWM score: Similarity to consensus sequences weighted by conservation
    2. Match count: Number of exact matches to consensus (out of 12 positions)
    3. Spacer length: Distance between -35 and -10 boxes

    Attributes:
        consensus_35 (str): Consensus sequence for the -35 box.
            This is the upstream promoter element recognized by the sigma-70 subunit
            of RNA polymerase, typically located ~35 bp upstream of the transcription
            start site. The canonical E. coli sigma-70 consensus is "TTGACA". Must be
            exactly 6 nucleotides. Default: "TTGACA".

        consensus_10 (str): Consensus sequence for the -10 box.
            This is the downstream promoter element, typically located ~10 bp
            upstream of the transcription start site. The canonical E. coli sigma-70
            consensus is "TATAAT". Must be exactly 6 nucleotides. Default: "TATAAT".

        probs_35 (list[float]): Position-specific conservation probabilities for
            the -35 box (6 values, one per position). These values represent the
            frequency of the consensus base at each position in experimentally
            validated E. coli promoters from RegulonDB. Must be exactly 6 values
            between 0.0 and 1.0. Default: [0.69, 0.79, 0.61, 0.56, 0.54, 0.54].

        probs_10 (list[float]): Position-specific conservation probabilities for
            the -10 box (6 values, one per position). These represent conservation
            at each -10 box position. Must be exactly 6 values between 0.0
            and 1.0. Default: [0.77, 0.76, 0.60, 0.61, 0.56, 0.82].

        optimal_spacer (int): Optimal spacer length between -35 and -10 boxes in
            base pairs. For E. coli sigma-70 promoters, this is typically 17±1 bp.
            Must be a positive integer. Default: 17.

        spacer_sigma (float): Standard deviation for spacer length penalty calculation.
            Controls how strictly the spacer length is enforced. Lower values
            (e.g., 1.0) make the constraint very sensitive to spacer deviations,
            while higher values (e.g., 3.0) are more permissive. The penalty uses
            a Gaussian function: penalty = 1 - exp(-((spacer - optimal)/sigma)²).
            Must be positive. Default: 1.5.

        spacer_weight (float): Weight (0.0-1.0) for spacer penalty in the total
            score. Controls the relative importance of correct spacer length vs.
            box sequence quality. For example, 0.3 means spacer contributes 30%
            to the final score, while box sequences contribute 70%. Higher values
            make spacer length more critical. Default: 0.3.

        gamma (float): PWM score exponent for non-linear sensitivity. Applied as
            score = (PWM_probability)^gamma. Lower values (e.g., 0.05-0.1) make
            the score very sensitive to mismatches (exponential drop-off), while
            higher values (e.g., 0.5-1.0) are more permissive. Typical range:
            0.05-0.2. Default: 0.1.

        k_opt (int): Optimal number of matches to consensus sequences (out of 12
            total positions: 6 in -35 box + 6 in -10 box). Strong promoters
            typically have 8-10 matches. This parameter centers the match penalty
            around the expected number of matches. Must be between 0 and 12.
            Default: 8.

        match_sigma (float): Standard deviation for match count penalty. Controls
            sensitivity to deviations from k_opt. Lower values penalize deviations
            more strongly. The penalty uses a Gaussian: penalty = 1 - exp(-((matches - k_opt)/sigma)²).
            Default: 2.0.

        match_weight (float): Weight (0.0-1.0) for match count penalty within the
            box scoring component (before combining with spacer). This balances
            PWM-based scoring (weighted position-specific probabilities) against
            simple match counting. For example, 0.3 means matches contribute 30%
            and PWM contributes 70% to box score. Default: 0.3.

        min_spacer (int): Minimum acceptable spacer length in base pairs. Promoters
            with spacers shorter than this are not evaluated (assigned penalty 1.0).
            Typical range: 14-16 bp for sigma-70 promoters. Default: 14.

        max_spacer (int): Maximum acceptable spacer length in base pairs. Promoters
            with spacers longer than this are not evaluated. Typical range: 19-21 bp
            for sigma-70 promoters. Default: 20.

    Note:
        The constraint scans sequences to find the best-scoring promoter within
        the allowed spacer range. For sequences ≤32 bp, it treats the entire
        sequence as a single promoter (first 6 bp = -35, last 6 bp = -10). For
        longer sequences, it scans all possible positions.

        The final penalty combines three components:
        1. **Box penalty** = (1 - match_weight) * PWM_penalty + match_weight * match_penalty
        2. **Total penalty** = (1 - spacer_weight) * box_penalty + spacer_weight * spacer_penalty
    """
    consensus_35: str = ConfigField(
        title="Consensus -35 Box",
        default="TTGACA",
        description="-35 box consensus sequence (6 bp, typically TTGACA for E. coli sigma-70)",
        advanced=True,
    )
    consensus_10: str = ConfigField(
        title="Consensus -10 Box",
        default="TATAAT",
        description="-10 box consensus sequence (6 bp Pribnow box, typically TATAAT for E. coli sigma-70)",
        advanced=True,
    )
    probs_35: List[float] = ConfigField(
        title="Conservation Probs -35 Box",
        default=[0.69, 0.79, 0.61, 0.56, 0.54, 0.54],
        description="Position-specific conservation probabilities for -35 box (6 values). From RegulonDB.",
        advanced=True,
    )
    probs_10: List[float] = ConfigField(
        title="Conservation Probs -10 Box",
        default=[0.77, 0.76, 0.60, 0.61, 0.56, 0.82],
        description="Position-specific conservation probabilities for -10 box (6 values). From RegulonDB.",
        advanced=True,
    )
    optimal_spacer: int = ConfigField(
        title="Optimal Spacer",
        default=17,
        description="Optimal spacer length between -35 and -10 boxes in base pairs (typically 17±1 bp)",
    )
    spacer_sigma: float = ConfigField(
        title="Spacer Standard Deviation",
        default=1.5,
        description="Standard deviation for spacer length penalty. Lower values = stricter spacing requirement.",
        advanced=True,
    )
    spacer_weight: float = ConfigField(
        title="Spacer Weight",
        default=0.3,
        description="Weight (0-1) for spacer penalty in total score. Higher = spacing more important.",
        advanced=True,
    )
    gamma: float = ConfigField(
        title="PWM Score Exponent",
        default=0.1,
        description="PWM score exponent for non-linearity. Lower values = more sensitive to mismatches.",
        advanced=True,
    )
    k_opt: int = ConfigField(
        title="Optimal Number of Matches",
        default=8,
        description="Optimal number of matches to consensus (out of 12 total positions)",
        advanced=True,
    )
    match_sigma: float = ConfigField(
        title="Match Count Standard Deviation",
        default=2.0,
        description="Standard deviation for match count penalty",
        advanced=True,
    )
    match_weight: float = ConfigField(
        title="Match Count Weight",
        default=0.3,
        description="Weight (0-1) for match count penalty in total score",
        advanced=True,
    )
    min_spacer: int = ConfigField(
        title="Min Acceptable Spacer Length",
        default=14,
        description="Minimum acceptable spacer length in bp",
        advanced=True,
    )
    max_spacer: int = ConfigField(
        title="Max Acceptable Spacer Length",
        default=20,
        description="Maximum acceptable spacer length in bp",
        advanced=True,
    )


@constraint(
    key="sigma70-promoter",
    label="Sigma70 Promoter Strength",
    config=Sigma70PromoterConfig,
    description="Evaluate sigma-70 promoter similarity for DNA sequences",
    tools_called=[],
    category="sequence annotation",
    supported_sequence_types=["dna"],
    num_input_sequences_per_tuple=1,
)
def sigma70_promoter_constraint(input_sequences: List[Tuple[Sequence, ...]], config: Sigma70PromoterConfig) -> List[float]:
    """Evaluate E. coli sigma-70 promoter similarity using PWM-based scoring.

    This constraint function evaluates bacterial promoter similarity by scanning
    DNA sequences for sigma-70-dependent promoter elements. It identifies putative
    -35 and -10 boxes, scores them based on similarity to consensus
    sequences weighted by position-specific conservation probabilities, evaluates
    the spacer distance between them, and combines these scores into an overall
    promoter similarity prediction.

    The scoring model is based on RegulonDB experimental data for E. coli sigma-70
    promoters and uses three components:
    1. **PWM score**: Position weight matrix score based on conservation probabilities
    2. **Match count**: Simple count of consensus matches (out of 12 positions)
    3. **Spacer length**: Deviation from optimal 17 bp spacer

    The function scans sequences to find the best-scoring promoter configuration
    within the allowed spacer range [min_spacer, max_spacer]. For short sequences
    (≤32 bp), it treats the entire sequence as a fixed promoter. For longer
    sequences, it exhaustively scans all positions.

    Args:
        sequences: List of DNA sequences to evaluate. Sequences
            should contain potential promoter regions. For best results, use
            sequences 50-100+ bp that may contain -35 and -10 boxes with appropriate
            spacing. Shorter sequences (12-32 bp) are treated as fixed promoters.

        config (Sigma70PromoterConfig): Configuration object containing consensus
            sequences, conservation probabilities, spacer parameters, and scoring
            weights. Uses E. coli sigma-70 defaults if not specified.
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.

    Returns:
        List[float]: Constraint scores for each sequence, ranging from 0.0 (perfect
            promoter, exact consensus with optimal spacer) to 1.0 (poor/no promoter).
            The score represents a penalty, so lower values indicate stronger
            predicted promoters. Scores combine PWM similarity, match count, and
            spacer length penalties.

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary under the key ``sigma70``
        with the following fields:

        **For valid promoters found:**
        - ``sigma70_score``: Float overall penalty score (0.0-1.0)
        - ``pos``: Integer start position of the -35 box in the sequence
        - ``box35``: String sequence of the -35 box (6 bp)
        - ``box10``: String sequence of the -10 box (6 bp)
        - ``spacer_len``: Integer spacer length between boxes (bp)
        - ``total_matches``: Integer total matches to consensus (out of 12)
        - ``pwm_penalty``: Float PWM-based penalty component (0.0-1.0)
        - ``match_penalty``: Float match count penalty component (0.0-1.0)
        - ``spacer_penalty``: Float spacer length penalty component (0.0-1.0)

        **For sequences too short (<12 bp):**
        - ``sigma70_score``: Float 1.0 (maximum penalty)
        - ``reason``: String "too_short"

        **For sequences with invalid spacer (12-32 bp range):**
        - ``sigma70_score``: Float 1.0 (maximum penalty)
        - ``reason``: String "invalid_spacer"

    Examples:
        Evaluating a canonical sigma-70 promoter:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> # Strong promoter with consensus -35 (TTGACA) and -10 (TATAAT) boxes
        >>> promoter_seq = Sequence(
        ...     "TTGACAATGATACTTAGATTCACTTATAATACTAGTAG",  # 17 bp spacer
        ...     "dna"
        ... )
        >>> config = Sigma70PromoterConfig()  # Use defaults
        >>> scores = sigma70_promoter_constraint([promoter_seq], config)
        >>> print(scores[0])  # e.g., 0.08 (strong promoter)
        >>> metadata = promoter_seq._metadata["sigma70"]
        >>> print(f"-35: {metadata['box35']}, -10: {metadata['box10']}")  # TTGACA, TATAAT
        >>> print(f"Matches: {metadata['total_matches']}/12")  # e.g., 11/12
        >>> print(f"Spacer: {metadata['spacer_len']} bp")  # 17
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

    for (seq_obj,) in input_sequences:
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
