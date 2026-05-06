"""Utilities for building sequence-type-aware logit-bias matrices."""

from __future__ import annotations

import numpy as np
from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core.segment import Segment
from proto_language.utils.helpers import is_plain_int


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
        description="Zero-based positions excluded from the reference-sequence bias.",
    )
    excluded_symbols: list[str] | None = ConfigField(
        default=None,
        title="Excluded Symbols",
        description="Single-character sequence symbols to penalize; validated against the segment vocabulary.",
    )
    excluded_positions: list[int] | None = ConfigField(
        default=None,
        title="Excluded Positions",
        description=(
            "Zero-based positions where excluded_symbols are penalized. "
            "Defaults to unbiased_positions when set, else all positions."
        ),
        advanced=True,
    )
    exclusion_penalty: float = ConfigField(
        default=-1e6,
        title="Exclusion Penalty",
        description="Additive logit penalty for excluded symbols.",
        advanced=True,
        hidden=True,
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

    @model_validator(mode="after")
    def _validate_reference_bias_config(self) -> SequenceLogitBiasConfig:
        """Validate field combinations whose meaning is independent of the segment."""
        if self.reference_bias is not None and self.reference_sequence is None:
            raise ValueError("reference_sequence is required when reference_bias is set.")
        if self.excluded_positions is not None and self.excluded_symbols is None:
            raise ValueError("excluded_symbols is required when excluded_positions is set.")
        return self

    def validate_against_segment(self, segment: Segment) -> None:
        """Validate cross-field constraints that depend on the segment's vocabulary and length.

        Position bounds, reference-sequence length match, and vocab membership for
        ``reference_sequence`` and ``excluded_symbols`` all require segment context
        and so cannot run as Pydantic field validators. Call once before building
        a bias matrix; raises on the first violation.

        Args:
            segment (Segment): Segment whose ordered vocab and sequence length
                the config must agree with.

        Raises:
            ValueError: If any position, length, or symbol fails the check.
        """
        vocab = set(segment.ordered_vocab())
        sequence_length = segment.sequence_length

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
                    f"reference_sequence length {len(self.reference_sequence)} does not match segment length "
                    f"{sequence_length}."
                )
            invalid = sorted(set(self.reference_sequence) - vocab)
            if invalid:
                raise ValueError(
                    f"reference_sequence contains symbols {invalid} outside segment vocabulary {sorted(vocab)}."
                )

        if self.excluded_symbols is not None:
            invalid = sorted(set(self.excluded_symbols) - vocab)
            if invalid:
                raise ValueError(f"excluded_symbols {invalid} are not in segment vocabulary {sorted(vocab)}.")


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

    config.validate_against_segment(segment)

    vocab = segment.ordered_vocab()
    vocab_index = {symbol: index for index, symbol in enumerate(vocab)}
    sequence_length = segment.sequence_length
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

    return matrix if np.any(matrix) else None


def combine_logit_biases(
    raw_logit_bias: np.ndarray | list[list[float]] | None,
    sequence_logit_bias: np.ndarray | None,
) -> np.ndarray | None:
    """Combine raw and declarative logit biases.

    Args:
        raw_logit_bias (np.ndarray | list[list[float]] | None): Existing
            explicit bias matrix supplied by advanced callers.
        sequence_logit_bias (np.ndarray | None): Bias matrix built from
            ``SequenceLogitBiasConfig``.

    Returns:
        np.ndarray | None: Additive combined bias matrix, or ``None`` when both
            inputs are unset.
    """
    if raw_logit_bias is None:
        return sequence_logit_bias
    raw = np.asarray(raw_logit_bias, dtype=float)
    if sequence_logit_bias is None:
        return raw
    combined: np.ndarray = raw + sequence_logit_bias
    return combined
