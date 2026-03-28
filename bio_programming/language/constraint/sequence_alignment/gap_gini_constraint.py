"""
proto_language/language/constraint/sequence_alignment/gap_gini_constraint.py

Computes a Gini coefficient on the gap run-length distribution of pairwise
alignments to detect truncation artifacts where gaps are concentrated in one
region rather than distributed evenly.

For each pair of sequences the constraint:
  1. Aligns them with MAFFT (pairwise mode).
  2. Center-crops to the middle 80% and strips end gaps.
  3. Computes the Gini coefficient of gap run-lengths in both aligned sequences.
  4. Returns the max Gini across both sequences as the pair's score.

A low Gini (< 0.1) indicates evenly distributed gaps (good alignment);
a high Gini (>= 0.3) suggests concentrated gaps (truncation artifact).
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

import numpy as np

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY

logger = logging.getLogger(__name__)


# ============================================================================
# Internal utilities (ported from the original gap-gini tool)
# ============================================================================


def _gini(x: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of values."""
    if len(x) == 0 or np.mean(x) == 0:
        return 0.0
    diffsum = 0.0
    for i in range(len(x) - 1):
        diffsum += np.sum(np.abs(x[i] - x[i + 1 :]))
    return float(diffsum / (len(x) ** 2 * np.mean(x)))


def _gap_runs(seq: str) -> List[int]:
    """Compute run lengths of consecutive gap characters in a sequence."""
    if not seq:
        return []

    runs = []
    prev = None
    run_len = 1

    for ind, char in enumerate(seq):
        if prev is None:
            prev = char
            continue

        if char == "-" and prev == "-":
            run_len += 1
        elif char != "-" and prev == "-":
            runs.append(run_len)
            run_len = 1
        else:
            runs.append(1)

        if ind == len(seq) - 1:
            runs.append(run_len)

        prev = char

    return runs


def _gap_gini_single(al1: str, al2: str) -> float:
    """Compute gap Gini score for a single pairwise alignment.

    Returns the max Gini coefficient across both aligned sequences.

    Args:
        al1 (str): First aligned sequence string.
        al2 (str): Second aligned sequence string.
    """
    al1_runs = np.array(_gap_runs(al1))
    al2_runs = np.array(_gap_runs(al2))

    gini1 = _gini(al1_runs) if len(al1_runs) > 0 else 0.0
    gini2 = _gini(al2_runs) if len(al2_runs) > 0 else 0.0

    return max(gini1, gini2)


def _trim_alignment(al1: str, al2: str) -> Tuple[str | None, str | None]:
    """Center-crop to 80% and strip end gaps (matches evocas9 pipeline).

    Returns (trimmed_al1, trimmed_al2) or (None, None) if no overlap remains.

    Args:
        al1 (str): First aligned sequence string.
        al2 (str): Second aligned sequence string.
    """
    align_len = len(al1)
    start, end = int(0.1 * align_len), int(0.9 * align_len)
    al1, al2 = al1[start:end], al2[start:end]

    def _end_gaps(seq: str) -> Tuple[int | None, int | None]:
        first = next((i for i, c in enumerate(seq) if c != "-"), None)
        last = next((i for i, c in enumerate(reversed(seq)) if c != "-"), None)
        return first, last

    g1_start, g1_end = _end_gaps(al1)
    g2_start, g2_end = _end_gaps(al2)
    if g1_start is None or g2_start is None:
        return None, None

    trim_start = max(g1_start, g2_start)
    trim_end = max(g1_end, g2_end)
    al1 = al1[trim_start : len(al1) - trim_end]
    al2 = al2[trim_start : len(al2) - trim_end]
    if len(al1) == 0:
        return None, None

    return al1, al2


def _gap_gini_from_fasta(alignment_fasta: str) -> float:
    """Compute gap Gini score from a FASTA-formatted pairwise alignment string."""
    sequences = re.findall(
        r"^[^>].*?(?=(?:^>|\Z))", alignment_fasta, re.MULTILINE | re.DOTALL
    )
    if len(sequences) != 2:
        raise ValueError(
            f"Expected 2 sequences in pairwise alignment, got {len(sequences)}"
        )
    al1, al2 = [seq.replace("\n", "") for seq in sequences]
    return _gap_gini_single(al1, al2)


# ============================================================================
# Config
# ============================================================================


class GapGiniConfig(BaseConfig):
    """Configuration for the alignment gap Gini constraint.

    The Gini coefficient measures inequality in the distribution of gap
    run-lengths within a pairwise alignment.  A value near 0 means gaps are
    evenly distributed; a value near 1 means they are concentrated in a
    single run (truncation artifact).

    Attributes:
        max_gap_gini (float): Maximum acceptable gap Gini score (0-1). Alignments
            with a Gini above this threshold are penalized.  Paper default
            is 0.1.
        trim_alignment (bool): Whether to center-crop to 80% and strip end gaps
            before computing the Gini.  Matches the evocas9 pipeline.
    """

    max_gap_gini: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Max Gap Gini",
        description=(
            "Maximum acceptable gap Gini score (0-1). "
            "Alignments above this are penalized."
        ),
        examples=[0.1, 0.3],
    )
    trim_alignment: bool = ConfigField(
        default=True,
        title="Trim Alignment",
        description=(
            "Center-crop to 80% and strip end gaps before computing "
            "the Gini coefficient."
        ),
        advanced=True,
    )


# ============================================================================
# Constraint
# ============================================================================


@constraint(
    key="gap-gini",
    label="Alignment Gap Gini",
    config=GapGiniConfig,
    description=(
        "Compute gap-distribution Gini coefficient for pairwise protein "
        "alignments (MAFFT). Low values indicate evenly distributed gaps; "
        "high values indicate truncation artifacts."
    ),
    tools_called=["mafft-align"],
    category="sequence_alignment",
    supported_sequence_types=["protein"],
    num_input_sequences_per_tuple=2,
)
def gap_gini_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: GapGiniConfig,
) -> List[float]:
    """Score pairwise protein alignments by gap-distribution Gini coefficient.

    For each (query, reference) pair the function:
      1. Aligns the two protein sequences with MAFFT.
      2. Optionally trims (center-crop 80%, strip end gaps).
      3. Computes gap run-length Gini for both sequences; takes the max.
      4. Returns 0.0 if gap_gini <= max_gap_gini, else scales linearly to 1.0.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of 2-tuples ``(query_seq, reference_seq)`` where
            each element is a :class:`Sequence` of type ``"protein"``.
        config (GapGiniConfig): :class:`GapGiniConfig` with ``max_gap_gini`` threshold and
            ``trim_alignment`` flag.

    Returns:
        list[float]: List of scores (one per pair). 0.0 = gap distribution
            acceptable, up to 1.0 = worst violation.
    """
    from proto_tools.tools.sequence_alignment.mafft import (
        MafftConfig,
        MafftInput,
        run_mafft_align,
    )

    scores: List[float] = []

    for query_seq, ref_seq in input_sequences:
        query_str = query_seq.sequence
        ref_str = ref_seq.sequence

        # --- MAFFT pairwise alignment ---
        try:
            align_result = run_mafft_align(
                MafftInput(sequences=[query_str, ref_str]),
                MafftConfig(),
            )
        except Exception:
            logger.exception(
                "MAFFT alignment failed for pair (len %d, len %d)",
                len(query_str),
                len(ref_str),
            )
            query_seq._metadata["gap_gini"] = None
            query_seq._metadata["gap_gini_error"] = True
            scores.append(MAX_ENERGY)
            continue

        if not align_result.msa or len(align_result.msa) < 2:
            logger.warning("MAFFT returned no alignment; penalizing pair")
            query_seq._metadata["gap_gini"] = None
            scores.append(MAX_ENERGY)
            continue

        al1, al2 = align_result.msa[0], align_result.msa[1]

        # --- Optional trimming ---
        if config.trim_alignment:
            al1, al2 = _trim_alignment(al1, al2)
            if al1 is None:
                # No overlap after trimming — treat as 0.0 (no gaps)
                query_seq._metadata["gap_gini"] = 0.0
                scores.append(MIN_ENERGY)
                continue

        # --- Gini computation ---
        gini_score = _gap_gini_single(al1, al2)
        query_seq._metadata["gap_gini"] = gini_score

        # --- Scoring ---
        if gini_score <= config.max_gap_gini:
            scores.append(MIN_ENERGY)
        else:
            # Linear penalty: deviation above threshold scaled to [0, 1]
            # max_gap_gini=0.1, gini=0.5 → (0.5-0.1)/(1.0-0.1)=0.44
            penalty = (gini_score - config.max_gap_gini) / (
                1.0 - config.max_gap_gini
            )
            scores.append(min(MAX_ENERGY, penalty))

    return scores
