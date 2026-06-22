"""tests/language_tests/constraint_tests/test_sequence_annotation/test_operator_site_constraint.py."""

import pytest
from pydantic import ValidationError

from proto_language.constraint import operator_site_constraint
from proto_language.constraint.sequence_annotation.operator_site_constraint import (
    OperatorSiteConfig,
    _anchor_promoter,
    _hamming,
    _reverse_complement,
)
from proto_language.core import Constraint, Segment

# A canonical -35 box + 17 bp spacer; an operator appended here overlaps the -10/TSS region.
_PROMOTER_PREFIX = "TTGACA" + "A" * 17


def _palindrome(half: str, gap: str = "") -> str:
    """Perfect inverted repeat: half-site + gap + reverse complement of half-site."""
    return half + gap + _reverse_complement(half)


def _score(sequence: str, **cfg_kwargs) -> tuple[float, dict]:
    config = OperatorSiteConfig(**cfg_kwargs)
    out = operator_site_constraint([(Segment(sequence=sequence, sequence_type="dna").proposal_sequences[0],)], config)
    return out[0].score, out[0].metadata["operator"]


class TestReverseComplementHelpers:
    def test_reverse_complement(self):
        assert _reverse_complement("GAATTCC") == "GGAATTC"

    def test_hamming(self):
        assert _hamming("ACGT", "ACGA") == 1

    def test_anchor_finds_consensus_boxes(self):
        seq = "TTGACA" + "A" * 17 + "TATAAT" + "C" * 10
        anchor = _anchor_promoter(seq, OperatorSiteConfig())
        assert anchor == (0, 23)  # -35 at 0, -10 at 23 (17 bp spacer)


class TestOperatorPresence:
    def test_perfect_operator_present(self):
        # 14 bp perfect inverted repeat appended after the spacer overlaps the -10/TSS region.
        seq = _PROMOTER_PREFIX + _palindrome("GAATTCC") + "A" * 20
        score, meta = _score(seq)
        assert score == pytest.approx(0.0)
        assert meta["present"] is True
        assert meta["best_operator"]["mismatch"] == 0
        assert meta["best_operator"]["occludes"] in {"minus35", "minus10", "tss"}

    def test_no_operator_absent_and_penalized(self):
        seq = _PROMOTER_PREFIX + "TATAAT" + "ACACACACACAC" + "A" * 10
        score, meta = _score(seq)
        assert meta["present"] is False
        assert score >= 0.5  # absent always scores >= 0.5 so threshold=0.5 filters cleanly

    def test_one_mismatch_tolerated(self):
        # Introduce a single mismatch into an otherwise-perfect operator half-site.
        operator = list(_palindrome("GAATTCC"))
        operator[0] = "T" if operator[0] != "T" else "A"  # one half-site mismatch vs the dyad
        seq = _PROMOTER_PREFIX + "".join(operator) + "A" * 20
        score, meta = _score(seq, max_mismatch=1)
        assert meta["present"] is True
        assert score == pytest.approx(0.0)

    def test_two_mismatch_rejected_at_default(self):
        operator = list(_palindrome("GAATTCC"))
        operator[0] = "C"
        operator[1] = "C"  # two mismatches vs the perfect dyad
        seq = _PROMOTER_PREFIX + "".join(operator) + "A" * 20
        _, meta = _score(seq, max_mismatch=1)
        assert meta["present"] is False

    def test_gap_one_operator_detected(self):
        seq = _PROMOTER_PREFIX + _palindrome("GAATTCC", gap="A") + "A" * 20
        _, meta = _score(seq, max_gap=1)
        assert meta["present"] is True
        assert meta["best_operator"]["gap"] == 1

    def test_gap_one_rejected_when_max_gap_zero(self):
        seq = _PROMOTER_PREFIX + _palindrome("GAATTCC", gap="A") + "A" * 20
        _, meta = _score(seq, max_gap=0)
        # The gapped dyad is no longer a valid operator with max_gap=0.
        assert meta["present"] is False

    def test_non_occluding_operator_is_absent(self):
        # Strong promoter at the 5' end; a perfect operator far downstream that occludes nothing.
        strong = "TTGACA" + "A" * 17 + "TATAAT"
        seq = strong + "C" * 30 + _palindrome("GAATTCC")
        _, meta = _score(seq)
        assert meta["present"] is False

    def test_min_overlap_controls_occlusion(self):
        seq = _PROMOTER_PREFIX + _palindrome("GAATTCC") + "A" * 20
        # Requiring a larger overlap than the operator achieves drops presence.
        _, meta = _score(seq, min_overlap=12)
        assert meta["present"] is False


class TestEdgeCases:
    def test_too_short_sequence(self):
        score, meta = _score("ACGT")
        assert score == pytest.approx(1.0)
        assert meta["reason"] == "too_short"

    def test_wrong_sequence_type(self):
        segment = Segment(sequence="MVLSPADKTNVK", sequence_type="protein")
        config = OperatorSiteConfig()
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(inputs=[segment], function=operator_site_constraint, function_config=config)

    def test_metadata_propagation(self):
        segment = Segment(sequence=_PROMOTER_PREFIX + _palindrome("GAATTCC") + "A" * 20, sequence_type="dna")
        Constraint(inputs=[segment], function=operator_site_constraint, function_config=OperatorSiteConfig()).evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["operator_site_constraint"]["data"]
        assert data["operator"]["present"] is True
        assert data["operator"]["best_operator"]["half_site"] >= 7

    def test_threshold_filter_behaviour(self):
        present_seq = _PROMOTER_PREFIX + _palindrome("GAATTCC") + "A" * 20
        absent_seq = _PROMOTER_PREFIX + "TATAAT" + "ACACACACACAC" + "A" * 10
        present_seg = Segment(sequence=present_seq, sequence_type="dna")
        absent_seg = Segment(sequence=absent_seq, sequence_type="dna")
        # threshold=0.5: score <= 0.5 passes; only the operator-bearing promoter should pass.
        assert Constraint(
            inputs=[present_seg],
            function=operator_site_constraint,
            function_config=OperatorSiteConfig(),
            threshold=0.5,
        ).evaluate()[0]
        assert not Constraint(
            inputs=[absent_seg],
            function=operator_site_constraint,
            function_config=OperatorSiteConfig(),
            threshold=0.5,
        ).evaluate()[0]


class TestConfigValidation:
    def test_bad_consensus_length_raises(self):
        with pytest.raises(ValidationError, match="6 bp"):
            OperatorSiteConfig(consensus_35="TTGAC")

    def test_spacer_order_raises(self):
        with pytest.raises(ValidationError, match="promoter_min_spacer"):
            OperatorSiteConfig(promoter_min_spacer=20, promoter_max_spacer=14)

    def test_negative_mismatch_raises(self):
        with pytest.raises(ValidationError):
            OperatorSiteConfig(max_mismatch=-1)
