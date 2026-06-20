"""TargetScan canonical seed-match miRNA site constraint.

This module scores miRNA target-site burden by TargetScan's canonical *seed-match*
site definitions (Lewis et al. 2005; Agarwal et al. 2015): a site is anchored on
the 6-mer complementary to miRNA seed positions 2-7, and is classified by whether
the match extends to position 8 (m8) and whether an ``A`` sits opposite position 1
(A1):

    * ``8mer``    -- m8 match AND A1 == 'A' (strongest)
    * ``7mer-m8`` -- m8 match, A1 != 'A'
    * ``7mer-A1`` -- 6-mer seed match, A1 == 'A', no m8 match
    * ``6mer``    -- 6-mer seed match only (weakest)

It is the deterministic, sequence-only half of TargetScan (site typing), with no
external binary or model -- complementary to the thermodynamic miRanda scan in
``mirna_specificity_constraint``. Scoring both lets an optimizer install / avoid
sites that BOTH callers agree on. This does not reproduce TargetScan's full
context++ efficacy score.

Examples:
    >>> from proto_language.core import Sequence
    >>> # miR-1 seed (positions 2-8): a UTR with an 8mer site scores low (sites present).
    >>> cfg = TargetScanSiteConfig(mirna_queries=["UGGAAUGUAAAGAAGUAUGUAU"], direction="maximize")
    >>> out = targetscan_site_constraint([(Sequence("AAACATTCCAAA", "dna"),)], cfg)
    >>> out[0].score < 1.0
    True
"""

from typing import Any, Literal

from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField

_SITE_TYPES = ("8mer", "7mer-m8", "7mer-A1", "6mer")
_DEFAULT_SITE_TYPE_WEIGHTS = {"8mer": 1.0, "7mer-m8": 0.8, "7mer-A1": 0.5, "6mer": 0.3}
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _to_dna(sequence: str) -> str:
    """Uppercase and fold RNA (U) onto the DNA alphabet (T)."""
    return sequence.strip().upper().replace("U", "T")


def _revcomp(dna: str) -> str:
    """Reverse complement of a DNA string."""
    return dna.translate(_COMPLEMENT)[::-1]


class TargetScanSiteConfig(BaseConfig):
    """Configuration for TargetScan canonical seed-match site scoring.

    Attributes:
        mirna_queries (list[str]): miRNA query sequences (>= 8 nt); ``U``/``T`` are
            interchangeable. Seed positions 2-8 define the site.
        mirna_ids (list[str] | None): Optional identifiers matching ``mirna_queries``.
        mirna_weights (list[float] | None): Optional nonnegative per-miRNA expression
            weights matching ``mirna_queries``.
        direction (Literal['maximize', 'minimize']): Whether predicted site burden
            should be high (install sites) or low (escape sites).
        repression_threshold (float): Aggregated site-burden score treated as saturating.
        site_type_weights (dict[str, float]): Per-site-type strengths for
            ``8mer``/``7mer-m8``/``7mer-A1``/``6mer``.
        include_6mer (bool): Whether to count the weak ``6mer`` site type.
    """

    mirna_queries: list[str] = ConfigField(
        title="miRNA Queries",
        description="miRNA query sequences (>= 8 nt) whose seed (positions 2-8) defines target sites.",
    )
    mirna_ids: list[str] | None = ConfigField(
        default=None,
        title="miRNA IDs",
        description="Optional identifiers for miRNA queries.",
    )
    mirna_weights: list[float] | None = ConfigField(
        default=None,
        title="miRNA Weights",
        description="Optional nonnegative per-miRNA expression weights matching the miRNA query order.",
    )
    direction: Literal["maximize", "minimize"] = ConfigField(
        default="maximize",
        title="Direction",
        description="Whether predicted miRNA site burden should be high or low.",
    )
    repression_threshold: float = ConfigField(
        default=2.0,
        gt=0.0,
        title="Repression Threshold",
        description="Aggregated site-burden score treated as saturating success.",
    )
    site_type_weights: dict[str, float] = ConfigField(
        default_factory=lambda: dict(_DEFAULT_SITE_TYPE_WEIGHTS),
        title="Site Type Weights",
        description="Per-site-type strengths for 8mer/7mer-m8/7mer-A1/6mer.",
    )
    include_6mer: bool = ConfigField(
        default=True,
        title="Include 6mer",
        description="Whether to count weak 6mer sites (set False to require >= 7mer).",
    )

    @field_validator("mirna_queries", mode="before")
    @classmethod
    def _normalize_queries(cls, queries: list[str] | str) -> list[str]:
        if isinstance(queries, str):
            queries = [queries]
        normalized = [_to_dna(query) for query in queries if query and query.strip()]
        if not normalized:
            raise ValueError("mirna_queries cannot be empty.")
        for query in normalized:
            invalid = sorted(set(query) - set("ACGTN"))
            if invalid:
                raise ValueError(f"miRNA query contains invalid bases: {invalid}.")
            if len(query) < 8:
                raise ValueError(f"miRNA query '{query}' is shorter than 8 nt; cannot define a seed site.")
        return normalized

    @field_validator("mirna_ids")
    @classmethod
    def _validate_ids(cls, ids: list[str] | None) -> list[str] | None:
        if ids is None:
            return None
        normalized = [value.strip() for value in ids]
        if any(not value for value in normalized):
            raise ValueError("mirna_ids cannot contain empty values.")
        return normalized

    @field_validator("mirna_weights")
    @classmethod
    def _validate_weights(cls, weights: list[float] | None) -> list[float] | None:
        if weights is None:
            return None
        normalized = [float(value) for value in weights]
        if any(value < 0.0 for value in normalized):
            raise ValueError("mirna_weights cannot contain negative values.")
        return normalized

    @field_validator("site_type_weights")
    @classmethod
    def _validate_site_type_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(weights) - set(_SITE_TYPES))
        if unknown:
            raise ValueError(f"site_type_weights has unknown site types {unknown}; allowed: {list(_SITE_TYPES)}.")
        if any(float(value) < 0.0 for value in weights.values()):
            raise ValueError("site_type_weights cannot contain negative values.")
        merged = dict(_DEFAULT_SITE_TYPE_WEIGHTS)
        merged.update({key: float(value) for key, value in weights.items()})
        return merged

    @model_validator(mode="after")
    def _validate_parallel_lengths(self) -> "TargetScanSiteConfig":
        if self.mirna_ids is not None and len(self.mirna_ids) != len(self.mirna_queries):
            raise ValueError("mirna_ids must match mirna_queries length.")
        if self.mirna_weights is not None and len(self.mirna_weights) != len(self.mirna_queries):
            raise ValueError("mirna_weights must match mirna_queries length.")
        return self


def _find_sites(target_dna: str, mirna_dna: str, include_6mer: bool) -> list[dict[str, Any]]:
    """Return canonical TargetScan seed-match sites of ``mirna_dna`` in ``target_dna``.

    Sites are anchored on the 6-mer complementary to miRNA positions 2-7 and typed by
    the m8 match (base 5' of the anchor) and the A1 nucleotide (base 3' of the anchor).
    """
    # miRNA seed: positions 2-7 (anchor) and position 8 (m8). 1-based -> 0-based slices.
    seed_2_7 = mirna_dna[1:7]
    m8_base = mirna_dna[7]
    core6 = _revcomp(seed_2_7)  # mRNA 6-mer complementary to seed positions 2-7
    m8_match_base = _complement_base(m8_base)  # base 5' of the anchor that pairs miRNA pos 8

    sites: list[dict[str, Any]] = []
    start = target_dna.find(core6)
    while start != -1:
        m8 = start > 0 and target_dna[start - 1] == m8_match_base
        a1 = (start + 6) < len(target_dna) and target_dna[start + 6] == "A"
        if m8 and a1:
            site_type = "8mer"
        elif m8:
            site_type = "7mer-m8"
        elif a1:
            site_type = "7mer-A1"
        else:
            site_type = "6mer"
        if include_6mer or site_type != "6mer":
            # Report the anchor as 1-indexed inclusive coordinates.
            sites.append({"site_type": site_type, "start": start + 1, "end": start + 6})
        start = target_dna.find(core6, start + 1)
    return sites


def _complement_base(base: str) -> str:
    """DNA complement of a single base (N -> N)."""
    return base.translate(_COMPLEMENT) if base in "ACGT" else "N"


@constraint(
    key="targetscan-site",
    label="TargetScan Seed Site",
    config=TargetScanSiteConfig,
    description="Score miRNA target-site burden by TargetScan canonical seed-match site typing (6mer/7mer/8mer).",
    uses_gpu=False,
    tools_called=[],
    category="sequence_annotation",
    supported_sequence_types=["dna", "rna"],
)
def targetscan_site_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: TargetScanSiteConfig,
) -> list[ConstraintOutput]:
    """Score miRNA seed-match site burden using TargetScan's canonical site definitions.

    For each target sequence and each miRNA, canonical seed-match sites are detected and
    summed with per-site-type and per-miRNA weights into an aggregate burden, bounded by
    ``repression_threshold``. ``direction='maximize'`` rewards installing sites (low score
    when burden is high); ``direction='minimize'`` rewards escaping them.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples
            (DNA or RNA target/UTR) to score.
        config (TargetScanSiteConfig): Validated configuration.

    Returns:
        list[ConstraintOutput]: One result per sequence. ``score`` is in ``[0, 1]``
            (0.0 = objective fully satisfied). ``metadata`` carries the aggregated
            ``targetscan_repression_score``, ``targetscan_num_sites``, and a per-site
            list with miRNA id, site type, weighted strength, and 1-indexed coordinates.
    """
    if not input_sequences:
        return []

    weight_ids = config.mirna_ids or [f"mirna_{idx}" for idx in range(len(config.mirna_queries))]
    mirna_weights = dict(zip(weight_ids, config.mirna_weights or [], strict=False))
    default_weight = 1.0

    results: list[ConstraintOutput] = []
    for (sequence,) in input_sequences:
        target_dna = _to_dna(sequence.sequence)
        site_records: list[dict[str, Any]] = []
        repression = 0.0
        for mirna_dna, mirna_id in zip(config.mirna_queries, weight_ids, strict=True):
            expression_weight = mirna_weights.get(mirna_id, default_weight)
            for site in _find_sites(target_dna, mirna_dna, config.include_6mer):
                strength = config.site_type_weights[site["site_type"]] * expression_weight
                repression += strength
                site_records.append({"mirna_id": mirna_id, **site, "strength": strength})

        bounded = min(max(repression, 0.0), config.repression_threshold) / config.repression_threshold
        score = 1.0 - bounded if config.direction == "maximize" else bounded
        score = min(MAX_ENERGY, score)

        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "targetscan_repression_score": repression,
                    "targetscan_specificity_score": score,
                    "targetscan_direction": config.direction,
                    "targetscan_num_sites": len(site_records),
                    "mirna_ids": config.mirna_ids,
                    "targetscan_sites": site_records or None,
                },
            )
        )

    return results
