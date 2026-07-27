"""Codon Adaptation Index (CAI) constraint for coding-sequence codon optimality.

CAI ([Sharp and Li, 1987](https://doi.org/10.1093/nar/15.3.1281)) measures how well a coding
sequence's codon usage matches a reference set of (typically highly expressed) genes:

    CAI = exp( (1/L) * sum_k ln w(c_k) )

where ``w(c)`` is the relative adaptiveness of codon ``c`` — its reference frequency divided by
the frequency of the most-used synonymous codon for the same amino acid — and the geometric mean
runs over the ``L`` scorable codons (single-codon families Met/Trp and stop codons are excluded,
following the standard definition). CAI is in ``(0, 1]``; closer to 1 is more optimally adapted.

The reference relative-adaptiveness table is supplied explicitly (``reference_weights``) or derived
from a reference coding-sequence set (``reference_sequences``); no organism table is assumed.
"""

import math
from collections import defaultdict
from typing import Literal

from pydantic import model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY, sigmoid_score
from proto_language.utils.base import BaseConfig, ConfigField

_MIN_FLOAT32_NORMAL = 1.1754943508222875e-38

# Relative-adaptiveness floor for codons absent from the reference set (avoids log(0)).
_W_FLOOR = 1e-3

# The standard genetic code in the DNA alphabet (sense codons + stops). Single-codon families
# (Met/ATG, Trp/TGG) and stops (``*``) are excluded from the CAI geometric mean.
STANDARD_GENETIC_CODE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}  # fmt: skip

# Codons excluded from the CAI product: stops plus the single-codon amino acids Met and Trp.
_EXCLUDED_CODONS = frozenset({"ATG", "TGG"} | {c for c, aa in STANDARD_GENETIC_CODE.items() if aa == "*"})


def _normalize_cds(sequence: str) -> str:
    """Uppercase and map RNA ``U`` to DNA ``T`` for codon lookup."""
    return sequence.upper().replace("U", "T")


def relative_adaptiveness_from_sequences(sequences: list[str]) -> dict[str, float]:
    """Derive per-codon relative adaptiveness ``w`` from a reference coding-sequence set.

    For each amino acid, ``w(codon) = count(codon) / max_synonymous_count``; codons whose whole
    family is unobserved get ``w = 1`` (neutral), and observed-but-zero codons get a small floor.

    Args:
        sequences (list[str]): Reference coding sequences (codon-aligned; RNA accepted).

    Returns:
        dict[str, float]: Codon (DNA) -> relative adaptiveness in ``(0, 1]``.
    """
    counts: dict[str, int] = defaultdict(int)
    for seq in sequences:
        cds = _normalize_cds(seq)
        for i in range(0, len(cds) - 2, 3):
            codon = cds[i : i + 3]
            if STANDARD_GENETIC_CODE.get(codon, "*") != "*":
                counts[codon] += 1

    aa_to_codons: dict[str, list[str]] = defaultdict(list)
    for codon, aa in STANDARD_GENETIC_CODE.items():
        if aa != "*":
            aa_to_codons[aa].append(codon)

    weights: dict[str, float] = {}
    for codons in aa_to_codons.values():
        max_count = max(counts[c] for c in codons)
        for c in codons:
            if max_count == 0:
                weights[c] = 1.0
            else:
                w = counts[c] / max_count
                weights[c] = w if w > 0.0 else _W_FLOOR
    return weights


class CodonAdaptationIndexConfig(BaseConfig):
    """Configuration for the Codon Adaptation Index (CAI) constraint.

    Exactly one reference source must be given: an explicit ``reference_weights`` table, or a
    ``reference_sequences`` set from which relative adaptiveness is derived.

    Attributes:
        reference_weights (dict[str, float] | None): Codon -> relative adaptiveness ``w`` in
            ``(0, 1]`` (DNA or RNA codons; ``U`` maps to ``T``).
        reference_sequences (list[str] | None): Reference coding sequences to derive ``w`` from.
        direction (Literal['max', 'min']): ``"max"`` rewards high CAI (codon-optimized),
            ``"min"`` rewards low CAI.
        sigmoid_center (float): CAI value mapped to energy 0.5.
        sigmoid_scale (float): Positive width of the sigmoid transform in CAI units.
    """

    reference_weights: dict[str, float] | None = ConfigField(
        default=None,
        title="Reference Weights",
        description="Codon -> relative adaptiveness in (0,1]; provide this or reference_sequences.",
    )
    reference_sequences: list[str] | None = ConfigField(
        default=None,
        title="Reference Sequences",
        description="Reference coding sequences to derive relative adaptiveness from.",
    )
    direction: Literal["max", "min"] = ConfigField(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') the codon adaptation index.",
    )
    sigmoid_center: float = ConfigField(
        default=0.7,
        allow_inf_nan=False,
        title="Sigmoid Center",
        description="CAI value mapped to energy 0.5 (tunable; CAI is in (0, 1]).",
    )
    sigmoid_scale: float = ConfigField(
        default=0.15,
        ge=_MIN_FLOAT32_NORMAL,
        allow_inf_nan=False,
        title="Sigmoid Scale",
        description="Positive float32-safe width of the sigmoid transform; larger is gentler.",
    )

    @model_validator(mode="after")
    def validate_reference(self) -> "CodonAdaptationIndexConfig":
        """Require exactly one reference source and validate the weights table."""
        has_weights = self.reference_weights is not None
        has_sequences = bool(self.reference_sequences)
        if has_weights == has_sequences:
            raise ValueError("provide exactly one of reference_weights or reference_sequences")
        if has_weights:
            for codon, w in self.reference_weights.items():  # type: ignore[union-attr]
                key = _normalize_cds(codon)
                if len(key) != 3 or STANDARD_GENETIC_CODE.get(key) in (None, "*"):
                    raise ValueError(f"reference_weights key {codon!r} is not a sense codon")
                if not (0.0 < w <= 1.0):
                    raise ValueError(f"reference_weights[{codon!r}] must be in (0, 1]; got {w}")
        return self

    def resolved_weights(self) -> dict[str, float]:
        """Return the codon -> relative-adaptiveness table (normalized DNA codon keys)."""
        if self.reference_weights is not None:
            return {_normalize_cds(c): float(w) for c, w in self.reference_weights.items()}
        return relative_adaptiveness_from_sequences(self.reference_sequences or [])


def _codon_adaptation_index(sequence: str, weights: dict[str, float]) -> float | None:
    """Geometric-mean CAI over scorable codons; ``None`` if none are scorable."""
    cds = _normalize_cds(sequence)
    log_sum = 0.0
    count = 0
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i : i + 3]
        if STANDARD_GENETIC_CODE.get(codon, "*") == "*" or codon in _EXCLUDED_CODONS:
            continue
        w = max(weights.get(codon, _W_FLOOR), _W_FLOOR)
        log_sum += math.log(w)
        count += 1
    if count == 0:
        return None
    return math.exp(log_sum / count)


@constraint(
    key="codon-adaptation-index",
    label="Codon Adaptation Index",
    config=CodonAdaptationIndexConfig,
    description="Drive a coding sequence toward high (or low) codon adaptation index (CAI).",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna"],
)
def codon_adaptation_index_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: CodonAdaptationIndexConfig
) -> list[ConstraintOutput]:
    """Score coding-sequence proposals by Codon Adaptation Index.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-sequence tuple per proposal.
        config (CodonAdaptationIndexConfig): Validated CAI configuration.

    Returns:
        list[ConstraintOutput]: One result per sequence. ``score`` is in ``[0.0, 1.0]`` (lower is
            better in the requested direction); ``metadata`` carries the raw ``cai``.
    """
    if not input_sequences:
        return []

    weights = config.resolved_weights()
    results: list[ConstraintOutput] = []
    for (seq,) in input_sequences:
        cai = _codon_adaptation_index(seq.sequence, weights)
        if cai is None:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"cai": None}))
            continue
        sigmoid = sigmoid_score(cai, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        score = (1.0 - sigmoid) if config.direction == "max" else sigmoid
        results.append(
            ConstraintOutput(
                score=min(MAX_ENERGY, max(MIN_ENERGY, score)),
                metadata={"cai": cai, "direction": config.direction},
            )
        )
    return results
