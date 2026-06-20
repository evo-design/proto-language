"""Dinucleotide-composition match constraint for DNA/RNA realism scoring.

This module scores how closely a sequence's dinucleotide (2-mer) frequency
distribution matches a fixed reference profile, e.g. measured natural-3'UTR
dinucleotide statistics. Unlike ``kmer_frequency_constraint`` (which checks each
observed k-mer against a single global ``[min, max]`` band), this constraint
compares the *whole* observed distribution to a reference distribution and
returns a single composition distance, so designs are pulled toward natural
nucleotide-pair usage rather than away from a band.

Examples:
    >>> from proto_language.core import Sequence
    >>> ref = {"AA": 0.25, "CC": 0.25, "GG": 0.25, "TT": 0.25}
    >>> cfg = DinucleotideCompositionConfig(reference_frequencies=ref)
    >>> out = dinucleotide_composition_constraint([(Sequence("AACCGGTT", "dna"),)], cfg)
    >>> round(out[0].score, 3)  # total-variation distance from the reference profile
    0.429
"""

from typing import Literal

from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField

_VALID_BASES = frozenset("ACGT")


def _canonical(text: str) -> str:
    """Uppercase and fold RNA (U) onto the DNA alphabet (T)."""
    return text.upper().replace("U", "T")


class DinucleotideCompositionConfig(BaseConfig):
    """Configuration for the dinucleotide-composition match constraint.

    Scores the distance between a sequence's observed dinucleotide-frequency
    distribution and a reference distribution. The reference is supplied as a
    mapping of dinucleotide -> frequency (e.g. derived from natural sequences);
    RNA (``U``) keys are folded onto the DNA alphabet (``T``) and the reference is
    renormalized to sum to 1.

    Attributes:
        reference_frequencies (dict[str, float]): Target dinucleotide-frequency
            profile. Keys are length-2 strings over ``A/C/G/T`` (``U`` accepted and
            folded to ``T``); values are nonnegative and need not sum to 1 (they are
            renormalized). Dinucleotides absent from the mapping are treated as
            reference frequency 0.0.
        distance_metric (Literal['total_variation', 'l2']): Distance between the
            observed and reference distributions. ``total_variation`` is
            ``0.5 * sum|p - q|`` (in ``[0, 1]``); ``l2`` is the Euclidean distance
            normalized by ``sqrt(2)`` (also in ``[0, 1]``).
        scale (float): Multiplier applied to the raw distance before clamping to
            ``[0, 1]``. Values > 1 sharpen the penalty; 1.0 leaves the distance
            unchanged.
    """

    reference_frequencies: dict[str, float] = ConfigField(
        title="Reference Dinucleotides",
        description="Target dinucleotide -> frequency mapping (natural composition profile).",
    )
    distance_metric: Literal["total_variation", "l2"] = ConfigField(
        default="total_variation",
        title="Distance Metric",
        description="Distribution distance: 'total_variation' (0.5*sum|p-q|) or normalized 'l2'.",
    )
    scale: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Scale",
        description="Multiplier applied to the raw distance before clamping to [0, 1].",
    )

    @field_validator("reference_frequencies", mode="before")
    @classmethod
    def _normalize_reference(cls, reference: object) -> dict[str, float]:
        if not isinstance(reference, dict) or not reference:
            raise ValueError("reference_frequencies must be a non-empty mapping.")
        normalized: dict[str, float] = {}
        for key, value in reference.items():
            dimer = _canonical(str(key))
            if len(dimer) != 2 or any(base not in _VALID_BASES for base in dimer):
                raise ValueError(f"Invalid dinucleotide key {key!r}; expected length-2 over A/C/G/T/U.")
            number = float(value)
            if number < 0.0:
                raise ValueError(f"reference_frequencies[{key!r}] must be nonnegative, got {number}.")
            # Fold duplicate keys (e.g. 'AU' and 'AT') by summing.
            normalized[dimer] = normalized.get(dimer, 0.0) + number
        total = sum(normalized.values())
        if total <= 0.0:
            raise ValueError("reference_frequencies must contain at least one positive value.")
        return {dimer: number / total for dimer, number in normalized.items()}

    @model_validator(mode="after")
    def _validate(self) -> "DinucleotideCompositionConfig":
        if not self.reference_frequencies:
            raise ValueError("reference_frequencies cannot be empty after normalization.")
        return self


def _observed_frequencies(sequence: str) -> dict[str, float]:
    """Overlapping dinucleotide frequencies over canonical A/C/G/T bases."""
    canonical = _canonical(sequence)
    counts: dict[str, int] = {}
    total = 0
    for i in range(len(canonical) - 1):
        dimer = canonical[i : i + 2]
        if dimer[0] in _VALID_BASES and dimer[1] in _VALID_BASES:
            counts[dimer] = counts.get(dimer, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {dimer: count / total for dimer, count in counts.items()}


def _distance(observed: dict[str, float], reference: dict[str, float], metric: str) -> float:
    keys = set(observed) | set(reference)
    if metric == "l2":
        squared = sum((observed.get(k, 0.0) - reference.get(k, 0.0)) ** 2 for k in keys)
        return float((squared**0.5) / (2.0**0.5))
    # total_variation
    return float(0.5 * sum(abs(observed.get(k, 0.0) - reference.get(k, 0.0)) for k in keys))


@constraint(
    key="dinucleotide-composition",
    label="Dinucleotide Composition",
    config=DinucleotideCompositionConfig,
    description="Match a sequence's dinucleotide-frequency distribution to a reference composition profile.",
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna", "rna"],
)
def dinucleotide_composition_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: DinucleotideCompositionConfig
) -> list[ConstraintOutput]:
    """Score the distance between observed and reference dinucleotide composition.

    The observed overlapping-dinucleotide frequency distribution of each sequence
    is compared to ``config.reference_frequencies`` using the configured distance
    metric. Lower scores mean the sequence's nucleotide-pair usage is closer to the
    reference (natural) composition.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples
            to evaluate (DNA or RNA). Sequences with fewer than two valid bases
            receive the maximum penalty.
        config (DinucleotideCompositionConfig): Validated configuration carrying the
            reference profile, ``distance_metric``, and ``scale``.

    Returns:
        list[ConstraintOutput]: One result per sequence. ``score`` is in ``[0, 1]``
            (0.0 = observed composition matches the reference). ``metadata`` carries
            ``dinucleotide_distance`` (the raw, pre-scale distance) and
            ``dinucleotide_frequencies`` (the observed distribution, or ``None`` when
            the sequence is too short).

    Examples:
        >>> from proto_language.core import Sequence
        >>> ref = {"AT": 0.6, "TA": 0.4}  # matches "ATATAT" exactly -> perfect score
        >>> cfg = DinucleotideCompositionConfig(reference_frequencies=ref)
        >>> out = dinucleotide_composition_constraint([(Sequence("ATATAT", "dna"),)], cfg)
        >>> round(out[0].score, 3)
        0.0
    """
    results: list[ConstraintOutput] = []

    for (seq,) in input_sequences:
        observed = _observed_frequencies(seq.sequence)
        if not observed:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY, metadata={"dinucleotide_distance": 1.0, "dinucleotide_frequencies": None}
                )
            )
            continue

        distance = _distance(observed, config.reference_frequencies, config.distance_metric)
        score = min(MAX_ENERGY, max(MIN_ENERGY, config.scale * distance))
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "dinucleotide_distance": distance,
                    "dinucleotide_frequencies": {k: float(v) for k, v in sorted(observed.items())},
                },
            )
        )

    return results
