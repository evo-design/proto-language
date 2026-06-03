"""Per-position matrices keyed by sequence vocabulary: one-hot encoding, peak-probability metric, logit-bias builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import field_validator, model_validator

from proto_language.utils.base import BaseConfig, ConfigField
from proto_language.utils.serialization import is_plain_int

if TYPE_CHECKING:
    from proto_language.core.segment import Segment
    from proto_language.core.sequence import Sequence


def one_hot_protein_matrix(sequence: str) -> list[list[float]]:
    """Return an exact (1.0, 0.0) one-hot matrix in ``PROTEIN_AMINO_ACIDS`` order.

    Each row has ``1.0`` at the target amino acid and ``0.0`` everywhere else.
    Use this when encoding a discrete protein sequence for a tool that expects a
    probability matrix or one-hot input.

    Args:
        sequence (str): Protein sequence; each character must be in ``PROTEIN_AMINO_ACIDS``.

    Returns:
        list[list[float]]: One-hot matrix with shape ``(len(sequence), 20)``.
    """
    from proto_language.core.sequence import PROTEIN_AMINO_ACIDS

    aa_index = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
    n = len(PROTEIN_AMINO_ACIDS)
    rows: list[list[float]] = []
    for aa in sequence:
        row = [0.0] * n
        row[aa_index[aa]] = 1.0
        rows.append(row)
    return rows


def mean_peak_probability(matrix: np.ndarray, positions: list[int] | None = None) -> float:
    """Return the mean per-row peak probability, optionally restricted to ``positions``."""
    rows = matrix if positions is None else matrix[positions]
    return float(np.mean(np.max(rows, axis=-1)))


class SequenceLogitBiasConfig(BaseConfig):
    """Configuration for alphabet-neutral sequence logit biases.

    The config describes common per-position bias patterns without asking users
    to manually construct an ``L x vocab`` matrix. The vocabulary is resolved
    from the assigned segment, so the same fields work for DNA, RNA, and protein
    generators that consume fixed single-character alphabets.

    Attributes:
        reference_sequence (str | None): Optional reference sequence whose
            symbols can receive ``reference_bias``.
        reference_bias (float | None): Additive bias applied to the reference
            symbol at each position, except ``unbiased_positions``.
        unbiased_positions (list[int] | None): Zero-based positions excluded
            from ``reference_bias``. Also used as the default
            ``excluded_positions`` when ``excluded_symbols`` is set.
        excluded_symbols (list[str] | None): Symbols to penalize. Each entry
            must be a single character in the segment vocabulary.
        excluded_positions (list[int] | None): Zero-based positions where
            ``excluded_symbols`` are penalized. Defaults to
            ``unbiased_positions`` when set, else all positions.
        exclusion_penalty (float): Additive penalty for each excluded symbol.
        raw_matrix (list[list[float]] | None): Pre-computed ``(L, |vocab|)`` additive
            bias; summed with the declarative primitives. Escape hatch for arbitrary
            per-(position, symbol) biases (e.g. MSA-derived PSSMs).
    """

    reference_sequence: str | None = ConfigField(
        default=None,
        title="Reference Sequence",
        description="Optional reference sequence used for per-position symbol biasing.",
    )
    reference_bias: float | None = ConfigField(
        default=None,
        title="Reference Bias",
        description="Additive bias applied to each reference symbol outside unbiased_positions.",
    )
    unbiased_positions: list[int] | None = ConfigField(
        default=None,
        title="Unbiased Positions",
        description="Position indices excluded from the reference-symbol bias (Python-style zero-based).",
    )
    excluded_symbols: list[str] | None = ConfigField(
        default=None,
        title="Excluded Symbols",
        description="Single-character sequence symbols to penalize; validated against the segment vocabulary.",
    )
    excluded_positions: list[int] | None = ConfigField(
        default=None,
        title="Excluded Positions",
        description="Zero-based positions where excluded symbols are penalized; defaults to unbiased positions, else all.",
    )
    exclusion_penalty: float = ConfigField(
        default=-1e6,
        title="Exclusion Penalty",
        description="Additive logit applied to excluded symbols. Defaults to a large negative (effectively forbid).",
    )
    raw_matrix: list[list[float]] | None = ConfigField(
        default=None,
        title="Raw Bias Matrix",
        description="Pre-computed additive logit bias (rows=positions, cols=vocab). Summed with declarative bias terms.",
    )

    @field_validator("unbiased_positions", "excluded_positions")
    @classmethod
    def _validate_positions(cls, value: list[int] | None) -> list[int] | None:
        """Validate position lists that are independent of segment length."""
        if value is None:
            return None
        if not value:
            raise ValueError("position lists must be None or non-empty; got [].")
        invalid = [position for position in value if not is_plain_int(position)]
        if invalid:
            raise ValueError(f"positions must be integers; got {invalid}.")
        negative = [position for position in value if position < 0]
        if negative:
            raise ValueError(f"positions must be non-negative; got {negative}.")
        return value

    @field_validator("excluded_symbols")
    @classmethod
    def _validate_excluded_symbols(cls, value: list[str] | None) -> list[str] | None:
        """Validate excluded symbols before segment-specific vocabulary checks."""
        if value is None:
            return None
        if not value:
            raise ValueError("excluded_symbols must be None or non-empty; got [].")
        invalid = [symbol for symbol in value if len(symbol) != 1]
        if invalid:
            raise ValueError(f"excluded_symbols entries must be single-character symbols; got {invalid}.")
        return value

    @field_validator("raw_matrix")
    @classmethod
    def _validate_raw_matrix(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        """Reject non-rectangular / non-finite matrices; segment-shape check happens at build time."""
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"raw_matrix must be a rectangular 2-D matrix of floats: {exc}") from exc
        if arr.ndim != 2:
            raise ValueError(f"raw_matrix must be 2-D, got shape {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("raw_matrix must contain only finite values.")
        return value

    @model_validator(mode="after")
    def _validate_reference_bias_config(self) -> SequenceLogitBiasConfig:
        """Validate field combinations whose meaning is independent of the segment."""
        if self.reference_bias is not None and self.reference_sequence is None:
            raise ValueError("reference_sequence is required when reference_bias is set.")
        if self.excluded_positions is not None and self.excluded_symbols is None:
            raise ValueError("excluded_symbols is required when excluded_positions is set.")
        return self

    def _validate_against_vocab(self, *, sequence_length: int, vocab: set[str]) -> None:
        """Shared validator for any source that resolves to ``(sequence_length, vocab)``."""
        for field_name, positions in (
            ("unbiased_positions", self.unbiased_positions),
            ("excluded_positions", self.excluded_positions),
        ):
            if positions is None:
                continue
            out_of_range = [p for p in positions if p >= sequence_length]
            if out_of_range:
                raise ValueError(f"{field_name} {out_of_range} are >= sequence_length ({sequence_length}).")

        if self.reference_bias is not None:
            assert self.reference_sequence is not None  # noqa: S101 -- model validator requires it
            if len(self.reference_sequence) != sequence_length:
                raise ValueError(
                    f"reference_sequence length {len(self.reference_sequence)} does not match sequence length "
                    f"{sequence_length}."
                )
            invalid = sorted(set(self.reference_sequence) - vocab)
            if invalid:
                raise ValueError(f"reference_sequence contains symbols {invalid} outside vocabulary {sorted(vocab)}.")

        if self.excluded_symbols is not None:
            invalid = sorted(set(self.excluded_symbols) - vocab)
            if invalid:
                raise ValueError(f"excluded_symbols {invalid} are not in vocabulary {sorted(vocab)}.")

        if self.raw_matrix is not None:
            arr = np.asarray(self.raw_matrix, dtype=np.float64)
            expected = (sequence_length, len(vocab))
            if arr.shape != expected:
                raise ValueError(f"raw_matrix shape {arr.shape} does not match expected {expected}.")


def build_sequence_logit_bias_matrix(config: SequenceLogitBiasConfig | None, segment: Segment) -> np.ndarray | None:
    """Build an additive logit-bias matrix for a segment.

    Args:
        config (SequenceLogitBiasConfig | None): Declarative bias configuration.
            ``None`` disables declarative biasing.
        segment (Segment): Segment whose length and ordered vocabulary define
            the output matrix shape.

    Returns:
        np.ndarray | None: Bias matrix with shape ``(L, |vocab|)``, or ``None``
            when the config is unset or has no numeric effect.

    Raises:
        ValueError: If the reference sequence length, positions, or symbols do
            not match the assigned segment.
    """
    if config is None:
        return None
    return _build_matrix(config, sequence_length=segment.sequence_length, vocab=segment.ordered_vocab())


def build_sequence_logit_bias_matrix_from_sequence(
    config: SequenceLogitBiasConfig | None, sequence: Sequence
) -> np.ndarray | None:
    """Build an additive logit-bias matrix from a ``Sequence``.

    Sibling to ``build_sequence_logit_bias_matrix`` for constraint callers
    that only see ``Sequence`` instances rather than a ``Segment``.

    Args:
        config (SequenceLogitBiasConfig | None): Declarative bias configuration.
            ``None`` disables declarative biasing.
        sequence (Sequence): Sequence whose length and ordered vocabulary define
            the output matrix shape.

    Returns:
        np.ndarray | None: Bias matrix with shape ``(L, |vocab|)``, or ``None``
            when the config is unset or has no numeric effect.

    Raises:
        ValueError: If the reference sequence length, positions, or symbols do
            not match the sequence.
    """
    if config is None:
        return None
    return _build_matrix(
        config,
        sequence_length=len(sequence),
        vocab=sequence.ordered_vocab(),
    )


def _build_matrix(config: SequenceLogitBiasConfig, *, sequence_length: int, vocab: list[str]) -> np.ndarray | None:
    """Validate ``config`` against ``vocab`` and build the ``(L, |vocab|)`` bias matrix."""
    config._validate_against_vocab(sequence_length=sequence_length, vocab=set(vocab))

    vocab_index = {symbol: index for index, symbol in enumerate(vocab)}
    matrix = np.zeros((sequence_length, len(vocab)), dtype=np.float64)

    unbiased_positions = sorted(set(config.unbiased_positions)) if config.unbiased_positions else []

    if config.reference_bias is not None:
        assert config.reference_sequence is not None  # noqa: S101 -- model validator requires it
        unbiased = set(unbiased_positions)
        for position, symbol in enumerate(config.reference_sequence):
            if position not in unbiased:
                matrix[position, vocab_index[symbol]] += config.reference_bias

    if config.excluded_symbols is not None:
        if config.excluded_positions is not None:
            excluded_positions = sorted(set(config.excluded_positions))
        elif config.unbiased_positions is not None:
            excluded_positions = unbiased_positions
        else:
            excluded_positions = list(range(sequence_length))

        if excluded_positions:
            excluded_indices = [vocab_index[symbol] for symbol in config.excluded_symbols]
            matrix[np.ix_(excluded_positions, excluded_indices)] += config.exclusion_penalty

    if config.raw_matrix is not None:
        matrix += np.asarray(config.raw_matrix, dtype=np.float64)

    return matrix if np.any(matrix) else None
