"""Tests for the MPNN perplexity filter constraint."""

import math

import pytest

from proto_language.language.constraint.sequence_scoring.mpnn_perplexity_constraint import (
    GENERATOR_KEY,
    MpnnPerplexityConfig,
    mpnn_perplexity_constraint,
)
from proto_language.language.core import ConstraintOutput, Sequence


def _make_proposals(perplexities: list[float]) -> list[tuple[Sequence, ...]]:
    out: list[tuple[Sequence, ...]] = []
    for p in perplexities:
        seq = Sequence(sequence="ACDEFG", sequence_type="protein")
        seq._generator_metadata[GENERATOR_KEY] = {"perplexity": p}
        out.append((seq,))
    return out


def _scores(results: list[ConstraintOutput]) -> list[float]:
    return [r.score for r in results]


def test_top_k_selects_lowest() -> None:
    results = mpnn_perplexity_constraint(_make_proposals([5.0, 1.0, 3.0, 2.0, 4.0]), MpnnPerplexityConfig(top_k=3))
    scores = _scores(results)
    accepted = sorted(p for p, s in zip([5.0, 1.0, 3.0, 2.0, 4.0], scores, strict=True) if s == 0.0)
    assert accepted == [1.0, 2.0, 3.0]
    assert all(math.isinf(s) for s in scores if s != 0.0)


def test_top_k_ge_n_accepts_all() -> None:
    assert _scores(mpnn_perplexity_constraint(_make_proposals([3.0, 1.0]), MpnnPerplexityConfig(top_k=5))) == [
        0.0,
        0.0,
    ]


def test_ties_accept_exactly_k() -> None:
    results = mpnn_perplexity_constraint(_make_proposals([2.0, 2.0, 2.0, 2.0, 5.0]), MpnnPerplexityConfig(top_k=3))
    assert sum(1 for r in results if r.score == 0.0) == 3


def test_raw_mode() -> None:
    results = mpnn_perplexity_constraint(_make_proposals([5.0, 1.0, 3.0]), MpnnPerplexityConfig())
    assert _scores(results) == [5.0, 1.0, 3.0]
    assert all(r.metadata["perplexity"] == r.score for r in results)


def test_missing_generator_metadata_raises() -> None:
    seq = Sequence(sequence="ACDEFG", sequence_type="protein")
    with pytest.raises(ValueError, match=r"proteinmpnn\.perplexity"):
        mpnn_perplexity_constraint([(seq,)], MpnnPerplexityConfig(top_k=3))
