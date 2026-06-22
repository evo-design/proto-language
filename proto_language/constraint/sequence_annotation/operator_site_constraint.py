"""Repressor-operator presence constraint for promoter occlusion design.

This module scores whether a bacterial promoter carries an inverted-repeat
(dyad-symmetric) repressor operator positioned to sterically occlude RNA
polymerase. An operator candidate is a pair of half-sites of >= ``min_half_site``
bp separated by a gap of <= ``max_gap`` bp whose first half-site matches the
reverse complement of the second with <= ``max_mismatch`` mismatches (Hamming
distance between half-site 1 and the reverse complement of half-site 2). A
promoter is "occluded" only if at least one such operator overlaps the -35 box,
the -10 box, or the transcription start site (TSS) by >= ``min_overlap`` bp.

The -35 / -10 boxes are located by the best consensus match (same anchoring used
by ``sigma70_promoter_constraint``); the TSS window is taken immediately
downstream of the -10 box. Score is 0.0 when an occluding operator is present and
rises toward 1.0 as the best occluding candidate drifts further from a valid
operator, so the constraint works both as a hard presence filter
(``threshold=0.5``) and as a soft optimization term that pulls a sampler toward
forming an occluding operator.

Examples:
    >>> from proto_language.core import Sequence
    >>> # -35 TTGACA, 17 bp spacer, -10 TATAAT, with a palindromic operator over -10.
    >>> seq = "TTGACA" + "T" * 17 + "TATAAT" + "G" + "ATTACGTACGTAAT"
    >>> cfg = OperatorSiteConfig()
    >>> out = operator_site_constraint([(Sequence(seq, "dna"),)], cfg)
    >>> out[0].score <= 1.0
    True
"""

from pydantic import model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField

_BOX_LENGTH = 6
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    """Reverse complement over the DNA alphabet (non-ACGT bases pass through)."""
    return seq.translate(_COMPLEMENT)[::-1]


def _hamming(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings."""
    return sum(x != y for x, y in zip(a, b, strict=True))


def _consensus_matches(box: str, consensus: str) -> int:
    """Number of positions where ``box`` matches ``consensus``."""
    return sum(x == y for x, y in zip(box, consensus, strict=False))


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Overlap in bp between half-open intervals [a_start, a_end) and [b_start, b_end)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


class OperatorSiteConfig(BaseConfig):
    """Configuration for the repressor-operator presence constraint.

    Locates the -35 / -10 boxes by best consensus match, defines a TSS window
    just downstream of the -10 box, then scans for inverted-repeat operators
    (two half-sites separated by a short gap whose first half-site matches the
    reverse complement of the second) and checks whether any occludes a promoter
    element. Lower score = an occluding operator is present.

    Attributes:
        min_half_site (int): Minimum half-site length in bp for an inverted-repeat
            operator candidate. Each operator is two half-sites of at least this
            length. Default 7.
        max_gap (int): Maximum gap in bp between the two half-sites of an operator
            (the operator spacer). Default 1.
        max_mismatch (int): Maximum mismatches allowed between half-site 1 and the
            reverse complement of half-site 2 (Hamming distance), i.e. how
            imperfect the dyad symmetry may be. Default 1.
        min_overlap (int): Minimum overlap in bp between an operator and a promoter
            element (-35 box, -10 box, or TSS window) for the operator to count as
            occluding. Default 3.
        consensus_35 (str): Consensus -35 box used to anchor the promoter (6 bp).
            Default "TTGACA".
        consensus_10 (str): Consensus -10 box used to anchor the promoter (6 bp).
            Default "TATAAT".
        promoter_min_spacer (int): Minimum -35/-10 spacer in bp considered when
            anchoring the promoter. Default 14.
        promoter_max_spacer (int): Maximum -35/-10 spacer in bp considered when
            anchoring the promoter. Default 20.
        tss_window (int): Width in bp of the TSS/initiation window placed
            immediately downstream of the -10 box, treated as an occlusion target.
            Default 6.
    """

    min_half_site: int = ConfigField(
        default=7,
        ge=2,
        title="Min Half-Site Length",
        description="Minimum half-site length (bp) for an inverted-repeat operator candidate.",
    )
    max_gap: int = ConfigField(
        default=1,
        ge=0,
        title="Max Half-Site Gap",
        description="Maximum gap (bp) between the two operator half-sites.",
    )
    max_mismatch: int = ConfigField(
        default=1,
        ge=0,
        title="Max Half-Site Mismatch",
        description="Max Hamming mismatches between half-site 1 and reverse complement of half-site 2.",
    )
    min_overlap: int = ConfigField(
        default=3,
        ge=1,
        title="Min Occlusion Overlap",
        description="Minimum overlap (bp) between an operator and a -35/-10/TSS element to occlude it.",
    )
    consensus_35: str = ConfigField(
        default="TTGACA",
        title="Consensus -35 Box",
        description="Consensus -35 box (6 bp) used to anchor the promoter.",
    )
    consensus_10: str = ConfigField(
        default="TATAAT",
        title="Consensus -10 Box",
        description="Consensus -10 box (6 bp) used to anchor the promoter.",
    )
    promoter_min_spacer: int = ConfigField(
        default=14,
        ge=1,
        title="Min Promoter Spacer",
        description="Minimum -35/-10 spacer (bp) considered when anchoring the promoter.",
    )
    promoter_max_spacer: int = ConfigField(
        default=20,
        ge=1,
        title="Max Promoter Spacer",
        description="Maximum -35/-10 spacer (bp) considered when anchoring the promoter.",
    )
    tss_window: int = ConfigField(
        default=6,
        ge=1,
        title="TSS Window Width",
        description="Width (bp) of the TSS window placed just downstream of the -10 box.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "OperatorSiteConfig":
        if len(self.consensus_35) != _BOX_LENGTH or len(self.consensus_10) != _BOX_LENGTH:
            raise ValueError(f"consensus_35 and consensus_10 must each be {_BOX_LENGTH} bp.")
        if self.promoter_min_spacer > self.promoter_max_spacer:
            raise ValueError(
                f"promoter_min_spacer ({self.promoter_min_spacer}) must be <= "
                f"promoter_max_spacer ({self.promoter_max_spacer})."
            )
        return self


def _anchor_promoter(seq: str, config: OperatorSiteConfig) -> tuple[int, int] | None:
    """Return (box35_start, box10_start) of the best consensus promoter, or None if too short."""
    cons_35 = config.consensus_35.upper()
    cons_10 = config.consensus_10.upper()
    seq_len = len(seq)
    best: tuple[int, int] | None = None
    best_matches = -1
    for spacer in range(config.promoter_min_spacer, config.promoter_max_spacer + 1):
        promoter_len = 2 * _BOX_LENGTH + spacer
        if promoter_len > seq_len:
            continue
        for pos in range(seq_len - promoter_len + 1):
            box35 = seq[pos : pos + _BOX_LENGTH]
            box10_start = pos + _BOX_LENGTH + spacer
            box10 = seq[box10_start : box10_start + _BOX_LENGTH]
            matches = _consensus_matches(box35, cons_35) + _consensus_matches(box10, cons_10)
            if matches > best_matches:
                best_matches = matches
                best = (pos, box10_start)
    return best


def _targets(box35_start: int, box10_start: int, config: OperatorSiteConfig) -> list[tuple[str, int, int]]:
    """Occlusion-target intervals [start, end) for the -35 box, -10 box, and TSS window."""
    box10_end = box10_start + _BOX_LENGTH
    return [
        ("minus35", box35_start, box35_start + _BOX_LENGTH),
        ("minus10", box10_start, box10_end),
        ("tss", box10_end, box10_end + config.tss_window),
    ]


def _best_occluding_operator(
    seq: str, targets: list[tuple[str, int, int]], config: OperatorSiteConfig
) -> dict[str, object] | None:
    """Scan inverted-repeat operators overlapping a target; return the lowest-mismatch one.

    Considers only operators whose span overlaps some target by >= ``min_overlap`` bp,
    then returns the candidate with the fewest half-site mismatches (ties broken toward
    longer half-sites and larger overlap). Returns None when no operator can possibly
    occlude a target (e.g. the sequence is too short).
    """
    seq_len = len(seq)
    best: dict[str, object] | None = None
    best_key: tuple[int, int, int] = (10**9, 0, 0)  # (mismatch, -half_site, -overlap) minimized
    max_half = seq_len // 2
    for half in range(config.min_half_site, max_half + 1):
        for gap in range(config.max_gap + 1):
            operator_len = 2 * half + gap
            if operator_len > seq_len:
                continue
            for start in range(seq_len - operator_len + 1):
                end = start + operator_len
                occlusion = max(
                    ((name, _overlap(start, end, t_start, t_end)) for name, t_start, t_end in targets),
                    key=lambda item: item[1],
                )
                if occlusion[1] < config.min_overlap:
                    continue
                half1 = seq[start : start + half]
                half2 = seq[start + half + gap : end]
                mismatch = _hamming(half1, _reverse_complement(half2))
                key = (mismatch, -half, -occlusion[1])
                if key < best_key:
                    best_key = key
                    best = {
                        "start": start,
                        "end": end,
                        "half_site": half,
                        "gap": gap,
                        "mismatch": mismatch,
                        "occludes": occlusion[0],
                        "overlap": occlusion[1],
                    }
                    if mismatch == 0 and half == config.min_half_site:
                        # A perfect minimal operator already satisfies presence; keep scanning
                        # only matters for richer metadata, so stop early on a clean hit.
                        return best
    return best


@constraint(
    key="operator-site",
    label="Operator Site Occlusion",
    config=OperatorSiteConfig,
    description="Score presence of an inverted-repeat operator occluding the -35/-10 box or TSS",
    tools_called=[],
    category="sequence_annotation",
    supported_sequence_types=["dna"],
)
def operator_site_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: OperatorSiteConfig
) -> list[ConstraintOutput]:
    """Score whether a promoter carries an occluding inverted-repeat operator.

    Each sequence is anchored to its best consensus -35/-10 promoter, the -35 box,
    -10 box, and a downstream TSS window are taken as occlusion targets, and the
    sequence is scanned for inverted-repeat operators (two half-sites of length
    >= ``min_half_site`` separated by a gap <= ``max_gap`` whose first half-site
    matches the reverse complement of the second within ``max_mismatch``
    mismatches). The best operator that overlaps a target by >= ``min_overlap`` bp
    determines the score.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Single-sequence tuples to
            evaluate (DNA). Sequences too short to contain a promoter receive the
            maximum penalty.
        config (OperatorSiteConfig): Validated configuration controlling operator
            geometry, mismatch tolerance, occlusion overlap, and promoter anchoring.

    Returns:
        list[ConstraintOutput]: One result per sequence. ``score`` is 0.0 when an
            occluding operator (mismatch <= ``max_mismatch``) is present and rises
            toward 1.0 as the best occluding candidate drifts away from a valid
            operator (always >= 0.5 when no valid operator occludes a target, so
            ``threshold=0.5`` acts as a presence filter). ``metadata`` carries an
            ``operator`` dict with the located boxes and best-operator details
            (``None`` when no candidate can occlude a target).
    """
    results: list[ConstraintOutput] = []
    normalizer = max(1, config.min_half_site)

    for (seq_obj,) in input_sequences:
        seq = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        anchor = _anchor_promoter(seq, config)
        if anchor is None:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"operator": {"reason": "too_short"}}))
            continue

        box35_start, box10_start = anchor
        targets = _targets(box35_start, box10_start, config)
        best = _best_occluding_operator(seq, targets, config)

        if best is None:
            score = MAX_ENERGY
            present = False
        else:
            mismatch = best["mismatch"]
            assert isinstance(mismatch, int)  # noqa: S101 -- mypy type narrowing
            if mismatch <= config.max_mismatch:
                score = MIN_ENERGY
                present = True
            else:
                # Absent: keep score >= 0.5 so threshold=0.5 filters cleanly, but grade by
                # how close the best occluding candidate is to a valid operator.
                score = min(MAX_ENERGY, 0.5 + 0.5 * (mismatch - config.max_mismatch) / normalizer)
                present = False

        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "operator": {
                        "present": present,
                        "box35_start": box35_start,
                        "box10_start": box10_start,
                        "best_operator": best,
                    }
                },
            )
        )

    return results
