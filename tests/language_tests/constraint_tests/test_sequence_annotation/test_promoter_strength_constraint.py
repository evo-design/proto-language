"""
tests/language_tests/constraint_tests/test_sequence_annotation/test_promoter_strength_constraint.py

promoter_calculator is mocked; it accepts a single sequence string
and returns a flat list of result objects (one per detected promoter).
"""

from unittest.mock import Mock, patch

import pytest

from proto_language.language.constraint import (
    ConstraintRegistry,
    promoter_strength_constraint,
)
from proto_language.language.constraint.sequence_annotation.promoter_strength_constraint import (
    PromoterStrengthConfig,
)
from proto_language.language.core import Constraint, Segment

PATCH_TARGET = (
    "proto_language.language.constraint.sequence_annotation"
    ".promoter_strength_constraint.promoter_calculator"
)


def _mock_result(dG_total=-4.0, Tx_rate=15000.0, strand="+"):
    result = Mock()
    result.dG_total = dG_total
    result.Tx_rate = Tx_rate
    result.strand = strand
    result.__dict__ = {"dG_total": dG_total, "Tx_rate": Tx_rate, "strand": strand}
    return result


def _evaluate(segment, config, mock_return):
    """Patch promoter_calculator with return_value and evaluate."""
    with patch(PATCH_TARGET) as mock_calc:
        mock_calc.return_value = mock_return
        scores = Constraint(
            inputs=[segment],
            function=promoter_strength_constraint,
            function_config=config,
        ).evaluate()
    return scores, mock_calc


class TestPromoterStrengthConstraint:

    @pytest.mark.parametrize(
        "dG, lo, hi",
        [
            (0.5, 1.0, 1.0),      # positive dG
            (-1.0, 1.0, 1.0),     # weak (above -1.5 threshold)
            (-2.0, 0.5, 1.0),     # moderate (-1.5 to -3.0)
            (-4.0, 0.0, 0.5),     # strong (below -3.0)
            (-10.0, 0.0, 0.0),    # very strong, clamped to 0.0
        ],
    )
    def test_dG_scoring(self, dG, lo, hi):
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="dG")
        scores, _ = _evaluate(segment, config, [_mock_result(dG_total=dG)])
        assert lo <= scores[0] <= hi

    @pytest.mark.parametrize(
        "tx_rate, lo, hi",
        [
            (1000.0, 1.0, 1.0),   # weak (below 3000)
            (8000.0, 0.0, 1.0),   # moderate (3000-10000)
            (30000.0, 0.0, 0.5),  # strong (above 20000)
        ],
    )
    def test_tx_rate_scoring(self, tx_rate, lo, hi):
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="tx_rate")
        scores, _ = _evaluate(segment, config, [_mock_result(Tx_rate=tx_rate)])
        assert lo <= scores[0] <= hi

    @pytest.mark.parametrize("return_value", [[], None])
    def test_no_promoter_found(self, return_value):
        """Empty or None results -> penalty 1.0 with no_promoter_found metadata."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        scores, _ = _evaluate(segment, PromoterStrengthConfig(), return_value)
        assert scores == [1.0]
        data = segment.proposal_sequences[0]._constraints_metadata
        meta = data["promoter_strength_constraint"]["data"]["promoter_strength"]
        assert meta["penalty"] == 1.0
        assert meta["reason"] == "no_promoter_found"

    def test_metadata_structure(self):
        """Verify constraint metadata is propagated correctly."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="dG")
        _evaluate(segment, config, [_mock_result(dG_total=-4.0)])

        data = segment.proposal_sequences[0]._constraints_metadata
        meta = data["promoter_strength_constraint"]["data"]["promoter_strength"]
        assert meta["dG_rate"] == -4.0
        assert "penalty" in meta
        assert "raw_output" in meta

    def test_minus_strand_filtered(self):
        """Only + strand results are used; - strand alone -> no promoter."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="dG")

        plus = _mock_result(dG_total=-2.0, strand="+")
        minus = _mock_result(dG_total=-5.0, strand="-")

        # Only minus strand -> no promoter
        scores, _ = _evaluate(segment, config, [minus])
        assert scores == [1.0]

        # Mixed strands -> uses only plus (dG=-2.0, moderate)
        segment2 = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        scores, _ = _evaluate(segment2, config, [plus, minus])
        assert 0.5 < scores[0] < 1.0

    def test_add_context(self):
        """add_context pads with flanking 'A' nucleotides."""
        segment = Segment(sequence="ATCG", sequence_type="dna")
        config = PromoterStrengthConfig(add_context=True, context_length=5)
        _, mock_calc = _evaluate(segment, config, [_mock_result(dG_total=-2.0)])

        processed_seq = mock_calc.call_args[0][0]
        assert processed_seq == "AAAAA" + "ATCG" + "AAAAA"

    def test_registry_integration(self):
        spec = ConstraintRegistry.get("promoter-strength")
        assert spec.label == "Promoter Strength"
        assert "dna" in spec.supported_sequence_types

        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        constraint = ConstraintRegistry.create("promoter-strength", [segment], {})
        assert isinstance(constraint, Constraint)
