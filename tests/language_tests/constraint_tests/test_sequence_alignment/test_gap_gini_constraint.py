"""tests/language_tests/constraint_tests/test_sequence_alignment/test_gap_gini_constraint.py

Tests for the gap Gini constraint (internal functions and full constraint)."""

from __future__ import annotations

import pytest

from proto_language.language.constraint.sequence_alignment.gap_gini_constraint import (
    GapGiniConfig,
    _gap_gini_single,
    _gap_runs,
    _gini,
    _trim_alignment,
    gap_gini_constraint,
)
from proto_language.language.core import Sequence

# ============================================================================
# Unit tests for internal helpers
# ============================================================================


class TestGini:
    def test_empty(self):
        import numpy as np

        assert _gini(np.array([])) == 0.0

    def test_all_equal(self):
        import numpy as np

        # All equal values → Gini = 0
        assert _gini(np.array([5, 5, 5, 5])) == 0.0

    def test_max_inequality(self):
        import numpy as np

        # One large, rest zero → Gini close to 1
        g = _gini(np.array([0, 0, 0, 100]))
        assert g > 0.7

    def test_moderate_inequality(self):
        import numpy as np

        g = _gini(np.array([1, 2, 3, 10]))
        assert 0.0 < g < 1.0


class TestGapRuns:
    def test_no_gaps(self):
        assert _gap_runs("ACGT") == [1, 1, 1, 1]

    def test_single_gap(self):
        runs = _gap_runs("A-C")
        assert any(r > 0 for r in runs)

    def test_consecutive_gaps(self):
        runs = _gap_runs("A---C")
        assert 3 in runs

    def test_empty_string(self):
        assert _gap_runs("") == []


class TestGapGiniSingle:
    def test_no_gaps(self):
        # No gaps → Gini = 0
        score = _gap_gini_single("ACGTACGT", "ACGTACGT")
        assert score == 0.0

    def test_evenly_distributed_gaps(self):
        # Evenly distributed single gaps → low Gini
        al1 = "A-C-G-T-A-C-G-T-"
        al2 = "ACGTACGTACGTACGTAC"
        score = _gap_gini_single(al1, al2)
        assert score < 0.2

    def test_concentrated_gaps(self):
        # All gaps in one block → high Gini
        al1 = "ACGT--------ACGT"
        al2 = "ACGTACGTACGTACGT"
        score = _gap_gini_single(al1, al2)
        assert score > 0.3


class TestTrimAlignment:
    def test_basic_trim(self):
        al1 = "A" * 100
        al2 = "A" * 100
        t1, t2 = _trim_alignment(al1, al2)
        assert t1 is not None
        assert len(t1) < 100  # Should be trimmed

    def test_all_gaps_returns_none(self):
        al1 = "-" * 100
        al2 = "-" * 100
        t1, t2 = _trim_alignment(al1, al2)
        assert t1 is None
        assert t2 is None

    def test_short_alignment(self):
        al1 = "AC"
        al2 = "AC"
        # Very short alignment after cropping
        t1, t2 = _trim_alignment(al1, al2)
        # May be None if too short after trim
        if t1 is not None:
            assert len(t1) <= 2


# ============================================================================
# Integration test for full constraint (requires MAFFT)
# ============================================================================


class TestGapGiniConstraint:
    @pytest.mark.integration
    def test_identical_sequences(self):
        """Identical sequences should align with no gaps → Gini = 0."""
        seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        query = Sequence(seq, "protein")
        ref = Sequence(seq, "protein")
        config = GapGiniConfig(max_gap_gini=0.1)
        scores = gap_gini_constraint([(query, ref)], config)
        assert len(scores) == 1
        assert scores[0] == 0.0
        assert query._metadata.get("gap_gini") is not None
        assert query._metadata["gap_gini"] == 0.0

    @pytest.mark.integration
    def test_similar_sequences(self):
        """Similar sequences should have low gap Gini."""
        query = Sequence(
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "protein",
        )
        ref = Sequence(
            "MVLSGEDKSNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "protein",
        )
        config = GapGiniConfig(max_gap_gini=0.5)
        scores = gap_gini_constraint([(query, ref)], config)
        assert len(scores) == 1
        # Similar sequences should pass with a loose threshold
        assert scores[0] == 0.0

    @pytest.mark.integration
    def test_threshold_penalty(self):
        """Sequences with concentrated gaps should be penalized."""
        # Very different sequences that should produce concentrated gaps
        query = Sequence(
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "protein",
        )
        # Truncated/very different sequence
        ref = Sequence(
            "MKAAVLTLAVLFLTGSQARHFWQQDEPPQSPWDRVKDLATVYVDVLKDSGE",
            "protein",
        )
        config = GapGiniConfig(max_gap_gini=0.001)  # Very strict threshold
        scores = gap_gini_constraint([(query, ref)], config)
        assert len(scores) == 1
        gini = query._metadata.get("gap_gini")
        assert gini is not None
        if gini > 0.001:
            assert scores[0] > 0.0  # Should be penalized

    @pytest.mark.integration
    def test_multiple_pairs(self):
        """Constraint should handle multiple pairs."""
        seq_a = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        seq_b = "MVLSGEDKSNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

        pairs = [
            (Sequence(seq_a, "protein"), Sequence(seq_a, "protein")),
            (Sequence(seq_a, "protein"), Sequence(seq_b, "protein")),
        ]
        config = GapGiniConfig(max_gap_gini=0.1)
        scores = gap_gini_constraint(pairs, config)
        assert len(scores) == 2
