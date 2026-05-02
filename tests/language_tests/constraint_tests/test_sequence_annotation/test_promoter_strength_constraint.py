"""Tests for the promoter_strength constraint."""

from unittest.mock import patch

import pytest
from proto_tools import (
    PromoterCalculatorOutput,
    PromoterCalculatorSequenceResult,
    PromoterPrediction,
)

from proto_language.language.constraint import (
    ConstraintRegistry,
    promoter_strength_constraint,
)
from proto_language.language.constraint.sequence_annotation.promoter_strength_constraint import (
    PromoterStrengthConfig,
)
from proto_language.language.core import Constraint, Segment

PATCH_TARGET = (
    "proto_language.language.constraint.sequence_annotation.promoter_strength_constraint.run_promoter_calculator"
)


def _prediction(dG_total: float = -4.0, Tx_rate: float = 15000.0, strand: str = "+") -> PromoterPrediction:
    return PromoterPrediction(
        tss_name=f"{'Fwd' if strand == '+' else 'Rev'}45",
        tss=45,
        strand=strand,
        dG_total=dG_total,
        Tx_rate=Tx_rate,
        promoter_sequence="A" * 50,
        length=50,
        UP_position=[0, 10],
        hex35_position=[10, 16],
        spacer_position=[16, 33],
        hex10_position=[33, 39],
        disc_position=[39, 45],
    )


def _output(predictions: list[PromoterPrediction]) -> PromoterCalculatorOutput:
    return PromoterCalculatorOutput(
        results=[PromoterCalculatorSequenceResult(sequence_id="seq_0", predictions=predictions)],
    )


def _evaluate(segment: Segment, config: PromoterStrengthConfig, predictions: list[PromoterPrediction]):
    with patch(PATCH_TARGET) as mock_call:
        mock_call.return_value = _output(predictions)
        scores = Constraint(
            inputs=[segment],
            function=promoter_strength_constraint,
            function_config=config,
        ).evaluate()
    return scores, mock_call


class TestPromoterStrengthConstraint:
    @pytest.mark.parametrize(
        "dG, expected",
        [
            (0.5, 1.0),  # positive dG
            (-1.0, 1.0),  # weak (above -1.5 threshold)
            (-2.0, 5 / 6),  # moderate (-1.5 to -3.0)
            (-4.0, 0.375),  # strong (below -3.0)
            (-10.0, 0.0),  # very strong, clamped to 0.0
        ],
    )
    def test_dG_scoring(self, dG, expected):
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="dG")
        scores, _ = _evaluate(segment, config, [_prediction(dG_total=dG)])
        assert scores[0] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "tx_rate, expected",
        [
            (1000.0, 1.0),  # weak (below 3000)
            (3000.0, 1.0),  # lower moderate boundary
            (8000.0, 0.6428571428571428),  # moderate (3000-10000)
            (10000.0, 0.5),  # upper moderate boundary
            (20000.0, 0.0),  # upper strong boundary
            (30000.0, 0.0),  # clamped to zero above 20000
        ],
    )
    def test_tx_rate_scoring(self, tx_rate, expected):
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="tx_rate")
        scores, _ = _evaluate(segment, config, [_prediction(Tx_rate=tx_rate)])
        assert scores[0] == pytest.approx(expected)

    def test_no_promoter_found(self):
        """Empty predictions -> penalty 1.0 with no_promoter_found metadata."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        scores, _ = _evaluate(segment, PromoterStrengthConfig(), [])
        assert scores == [1.0]
        data = segment.proposal_sequences[0]._constraints_metadata
        meta = data["promoter_strength_constraint"]["data"]["promoter_strength"]
        assert meta["penalty"] == 1.0
        assert meta["reason"] == "no_promoter_found"

    def test_metadata_structure(self):
        """Verify constraint metadata is propagated correctly."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="dG")
        _evaluate(segment, config, [_prediction(dG_total=-4.0)])

        data = segment.proposal_sequences[0]._constraints_metadata
        meta = data["promoter_strength_constraint"]["data"]["promoter_strength"]
        assert meta["dG_rate"] == -4.0
        assert "penalty" in meta
        assert "raw_output" in meta

    def test_minus_strand_filtered(self):
        """Only + strand contributes to the score."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = PromoterStrengthConfig(scoring_type="dG")
        plus = _prediction(dG_total=-2.0, strand="+")
        minus = _prediction(dG_total=-5.0, strand="-")

        scores, _ = _evaluate(segment, config, [minus])
        assert scores == [1.0]

        segment2 = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        scores, _ = _evaluate(segment2, config, [plus, minus])
        assert scores[0] == pytest.approx(5 / 6)

    def test_add_context(self):
        """add_context pads with flanking 'A' nucleotides before dispatch."""
        segment = Segment(sequence="ATCG", sequence_type="dna")
        config = PromoterStrengthConfig(add_context=True, context_length=5)
        _, mock_call = _evaluate(segment, config, [_prediction(dG_total=-2.0)])

        sent_inputs = mock_call.call_args[0][0]
        assert sent_inputs.sequences == ["AAAAA" + "ATCG" + "AAAAA"]

    def test_registry_integration(self):
        spec = ConstraintRegistry.get("promoter-strength")
        assert spec.label == "Promoter Strength"
        assert "dna" in spec.supported_sequence_types

        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        constraint = ConstraintRegistry.create("promoter-strength", [segment], {})
        assert isinstance(constraint, Constraint)


@pytest.mark.integration
def test_run_against_real_wrapper_lac_uv5():
    """End-to-end through the real run_promoter_calculator subprocess."""
    seq = "A" * 20 + "AAAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACC" + "A" * 20
    segment = Segment(sequence=seq, sequence_type="dna")
    scores = Constraint(
        inputs=[segment],
        function=promoter_strength_constraint,
        function_config=PromoterStrengthConfig(scoring_type="dG"),
    ).evaluate()

    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
    meta = segment.proposal_sequences[0]._constraints_metadata
    payload = meta["promoter_strength_constraint"]["data"]["promoter_strength"]
    assert "penalty" in payload
    assert "raw_output" in payload
