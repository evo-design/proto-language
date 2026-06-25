"""Melting temperature (Tm) constraint for DNA oligonucleotide design.

Computes the predicted melting temperature of a DNA sequence and penalizes
sequences whose Tm falls outside a user-specified range. Two established
empirical methods are supported: the Wallace rule (preferred for short oligos)
and the GC-content formula (preferred for longer sequences).

Examples:
    >>> from proto_language.core import Sequence
    >>> seq = Sequence("ATCGATCGATCG", "dna")
    >>> cfg = MeltingTemperatureConfig(min_tm=40.0, max_tm=60.0)
    >>> result = melting_temperature_constraint([(seq,)], config=cfg)
    >>> result[0].score  # 0.0 — Tm is within range
"""

import logging
import math

from pydantic import model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, calculate_range_deviation
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

# Wallace rule crossover point (sequences ≤ this length use the Wallace rule)
_WALLACE_CUTOFF = 13


class MeltingTemperatureConfig(BaseConfig):
    """Configuration for the melting temperature (Tm) constraint.

    Two empirical methods are available:

    - ``"wallace"``: Tm = 2·(A+T) + 4·(G+C), in °C. Recommended for short oligos
      (≤ 13 nt) where the dominant contributors are simply base-pair stacking and
      hydrogen bonding. No salt correction is applied.
    - ``"gc_based"``: Tm = 81.5 + 16.6·log₁₀([Na⁺] mM/1000) + 0.41·%GC - 675/N,
      in °C (Bolton & McCarthy / Marmur-Doty-Wetmur approximation for duplexes
      > 13 bp). Requires a ``salt_mm`` value; defaults to 50 mM NaCl.
    - ``"auto"`` (default): Selects the Wallace rule for sequences ≤ 13 nt and the
      GC-based formula for longer sequences, matching BioPython's ``Tm_Wallace`` /
      ``Tm_GC`` behaviour.

    Attributes:
        min_tm (float): Minimum acceptable Tm in degrees Celsius. Sequences with
            predicted Tm below this value are penalized. Typical oligo targets lie
            between 50 °C and 65 °C; short oligos (< 10 nt) may target 30-50 °C.
        max_tm (float): Maximum acceptable Tm in degrees Celsius. Sequences with
            predicted Tm above this value are penalized.
        method (str): Tm calculation method: ``"auto"``, ``"wallace"``, or
            ``"gc_based"``. Defaults to ``"auto"``.
        salt_mm (float): Monovalent salt concentration in millimolar, used only
            by the ``"gc_based"`` method. Represents the [Na⁺] equivalent of the
            hybridisation buffer. Typical values: 50 mM (standard PCR), 0 mM
            (water — will produce lower predicted Tm), 200 mM (high-salt buffer).
            Defaults to 50.0 mM.
    """

    min_tm: float = ConfigField(
        title="Minimum Tm (°C)",
        description="Minimum acceptable melting temperature in degrees Celsius.",
        ge=-100.0,
        le=200.0,
        examples=[50.0],
    )
    max_tm: float = ConfigField(
        title="Maximum Tm (°C)",
        description="Maximum acceptable melting temperature in degrees Celsius.",
        ge=-100.0,
        le=200.0,
        examples=[65.0],
    )
    method: str = ConfigField(
        default="auto",
        title="Tm Calculation Method",
        description=(
            'Tm calculation method: "auto" (Wallace for ≤13 nt, GC-based for longer), '
            '"wallace" (2·(A+T) + 4·(G+C)), or "gc_based" (81.5 + 16.6·log₁₀([Na⁺]) '
            "+ 0.41·%GC - 675/N)."
        ),
    )
    salt_mm: float = ConfigField(
        default=50.0,
        title="Salt Concentration (mM)",
        description=(
            "Monovalent salt concentration in millimolar used by the gc_based method. "
            "Defaults to 50 mM (standard PCR conditions). Ignored by the wallace method."
        ),
        gt=0.0,
        examples=[50.0],
    )

    @model_validator(mode="after")
    def validate_config(self) -> "MeltingTemperatureConfig":
        """Ensure min_tm <= max_tm and method is valid."""
        if self.min_tm > self.max_tm:
            raise ValueError(
                f"min_tm ({self.min_tm}) must be <= max_tm ({self.max_tm})"
            )
        valid_methods = {"auto", "wallace", "gc_based"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {sorted(valid_methods)}, got '{self.method}'"
            )
        return self


def _tm_wallace(sequence: str) -> float:
    """Compute Tm using the Wallace rule: 2·(A+T) + 4·(G+C).

    Applies to short oligonucleotides (≤ 13 nt). Uses upper-cased sequence
    and treats both DNA (T) and RNA (U) correctly (U is not AT/GC and is
    ignored — sequences should be DNA only for reliable results).

    Args:
        sequence: Upper-cased DNA sequence string.

    Returns:
        Predicted Tm in degrees Celsius.
    """
    at_count = sequence.count("A") + sequence.count("T")
    gc_count = sequence.count("G") + sequence.count("C")
    return 2.0 * at_count + 4.0 * gc_count


def _tm_gc_based(sequence: str, salt_mm: float) -> float:
    """Compute Tm using the GC-content / Marmur-Doty approximation.

    Formula (Bolton & McCarthy 1962 / Wetmur 1991):
        Tm = 81.5 + 16.6·log₁₀([Na⁺]) + 0.41·%GC - 675/N

    where [Na⁺] is in molar units, %GC is 0-100, and N is the sequence length.

    Args:
        sequence: Upper-cased DNA sequence string (length > 0).
        salt_mm: Monovalent salt concentration in millimolar.

    Returns:
        Predicted Tm in degrees Celsius.
    """
    n = len(sequence)
    gc_count = sequence.count("G") + sequence.count("C")
    gc_pct = 100.0 * gc_count / n
    salt_molar = salt_mm / 1000.0
    return 81.5 + 16.6 * math.log10(salt_molar) + 0.41 * gc_pct - 675.0 / n


def _compute_tm(sequence: str, config: MeltingTemperatureConfig) -> float:
    """Select and apply the appropriate Tm formula.

    Args:
        sequence: Upper-cased DNA sequence (non-empty).
        config: Validated configuration.

    Returns:
        Predicted Tm in degrees Celsius.
    """
    n = len(sequence)
    method = config.method
    if method == "auto":
        method = "wallace" if n <= _WALLACE_CUTOFF else "gc_based"
    if method == "wallace":
        return _tm_wallace(sequence)
    return _tm_gc_based(sequence, config.salt_mm)


@constraint(
    key="melting-temperature",
    label="Melting Temperature",
    config=MeltingTemperatureConfig,
    description=(
        "Enforce predicted melting temperature (Tm) within a specified range. "
        "Uses the Wallace rule for short oligos (≤ 13 nt) and the GC-content "
        "formula for longer sequences when method='auto'."
    ),
    tools_called=[],
    category="sequence_composition",
    supported_sequence_types=["dna"],
)
def melting_temperature_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: MeltingTemperatureConfig,
) -> list[ConstraintOutput]:
    """Enforce predicted melting temperature within a specified range.

    Computes the predicted Tm for each DNA sequence using an empirical formula
    and penalises sequences whose Tm falls outside [min_tm, max_tm]. The penalty
    scales linearly with the degree of deviation.

    Two formulae are available (controlled by ``config.method``):

    - **Wallace rule** (``"wallace"`` or ``"auto"`` with N ≤ 13 nt):
      ``Tm = 2·(A+T) + 4·(G+C)`` °C — fast, counts-based estimate suitable for
      short oligos where nearest-neighbour stacking contributes uniformly.
    - **GC-content formula** (``"gc_based"`` or ``"auto"`` with N > 13 nt):
      ``Tm = 81.5 + 16.6·log₁₀([Na⁺] M) + 0.41·%GC - 675/N`` °C — Bolton &
      McCarthy/Wetmur approximation, accurate for typical PCR/hybridisation
      conditions.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples.
            Each tuple contains one DNA sequence. Empty sequences receive the
            maximum penalty.
        config (MeltingTemperatureConfig): Validated configuration with
            ``min_tm``, ``max_tm``, ``method``, and ``salt_mm`` fields.

    Returns:
        list[ConstraintOutput]: One result per sequence. A score of 0.0 means
        the predicted Tm falls within [min_tm, max_tm]. Higher scores indicate
        greater deviation, scaling linearly with distance from the acceptable
        range (capped at 1.0). The ``metadata`` field carries:

        - ``tm``: Predicted Tm in °C (float).
        - ``method_used``: The formula applied — ``"wallace"`` or ``"gc_based"``
          (str). Useful for debugging when ``config.method`` is ``"auto"``.

    Examples:
        Targeting a standard PCR primer Tm window:

        >>> from proto_language.core import Sequence
        >>> seq = Sequence("GCTAGCTAGCTAGCTA", "dna")
        >>> cfg = MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0)
        >>> result = melting_temperature_constraint([(seq,)], config=cfg)
        >>> result[0].score  # 0.0 if Tm is within range
    """
    results = []

    for (seq,) in input_sequences:
        if len(seq.sequence) == 0:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"tm": 0.0, "method_used": "none"},
                )
            )
            continue

        upper = seq.sequence.upper()
        n = len(upper)

        # Determine which formula was actually used
        effective_method = config.method
        if effective_method == "auto":
            effective_method = "wallace" if n <= _WALLACE_CUTOFF else "gc_based"

        tm = _compute_tm(upper, config)
        deviation = calculate_range_deviation(tm, config.min_tm, config.max_tm, epsilon=1.0)

        results.append(
            ConstraintOutput(
                score=min(MAX_ENERGY, deviation),
                metadata={"tm": tm, "method_used": effective_method},
            )
        )

    return results
