"""tests/language_tests/constraint_tests/test_sequence_annotation/test_sigma70_promoter_constraint.py."""

import pytest
from pydantic import ValidationError

from proto_language.constraint import sigma70_promoter_constraint
from proto_language.constraint.sequence_annotation.sigma70_promoter_constraint import (
    Sigma70PromoterConfig,
)
from proto_language.core import Constraint, Segment


# Tests for sigma70_promoter_constraint
class TestSigma70PromoterConstraint:
    def test_ideal_promoter(self):
        """Test ideal sigma70 promoter sequence."""
        # Ideal promoter: -35 box + 17bp spacer + -10 box
        ideal = "TTGACA" + "A" * 17 + "TATAAT"
        segment = Segment(sequence=ideal, sequence_type="dna")
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            function=sigma70_promoter_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert scores[0] < 0.5  # Should have low penalty
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "sigma70" in constraints["sigma70_promoter_constraint"]["data"]
        sigma70_data = constraints["sigma70_promoter_constraint"]["data"]["sigma70"]
        assert sigma70_data["spacer_len"] == 17

    def test_poor_promoter(self):
        """Test poor promoter sequence."""
        poor = "AAAAAA" + "G" * 17 + "CCCCCC"
        segment = Segment(sequence=poor, sequence_type="dna")
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            function=sigma70_promoter_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert scores[0] > 0.4  # Should have moderate-to-high penalty
        # Check some metadata was stored
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "sigma70_promoter_constraint" in constraints

    def test_scanning_long_sequence(self):
        """Test scanning long sequence for best promoter."""
        # Embed promoter in longer sequence
        long_seq = "A" * 50 + "TTGACA" + "T" * 17 + "TATAAT" + "G" * 50
        segment = Segment(sequence=long_seq, sequence_type="dna")
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            function=sigma70_promoter_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert scores[0] < 0.5
        # Check position metadata was stored
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "sigma70" in constraints["sigma70_promoter_constraint"]["data"]
        sigma70_data = constraints["sigma70_promoter_constraint"]["data"]["sigma70"]
        assert "pos" in sigma70_data

    def test_short_sequence(self):
        """Test sequence too short for promoter."""
        short = "ATCG"
        segment = Segment(sequence=short, sequence_type="dna")
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            function=sigma70_promoter_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert scores[0] == 1.0
        # Should indicate sequence is too short
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "sigma70" in constraints["sigma70_promoter_constraint"]["data"]
        sigma70_data = constraints["sigma70_promoter_constraint"]["data"]["sigma70"]
        assert "reason" in sigma70_data
        assert sigma70_data["reason"] == "too_short"

    def test_custom_consensus_sequences(self):
        """Test with custom consensus sequences (constraint-specific config behavior)."""
        # Use custom consensus sequences
        custom_seq = "AAAAAA" + "T" * 17 + "CCCCCC"
        segment = Segment(sequence=custom_seq, sequence_type="dna")
        config = Sigma70PromoterConfig(consensus_35="AAAAAA", consensus_10="CCCCCC")

        constraint = Constraint(
            inputs=[segment],
            function=sigma70_promoter_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        # Should have low penalty with matching custom consensus
        assert scores[0] < 0.5

        # Check constraint-specific metadata
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "sigma70" in constraints["sigma70_promoter_constraint"]["data"]
        sigma70_data = constraints["sigma70_promoter_constraint"]["data"]["sigma70"]
        assert "spacer_len" in sigma70_data
        assert sigma70_data["spacer_len"] == 17

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"consensus_35": "TTGAC"},  # 5 bp
            {"consensus_10": "TATAATG"},  # 7 bp
            {"probs_35": [0.5, 0.5, 0.5, 0.5, 0.5]},  # 5 values
            {"probs_10": [0.5] * 7},  # 7 values
        ],
    )
    def test_box_length_validation(self, kwargs):
        """Consensus/probability vectors must be exactly 6 elements (else zip truncates silently)."""
        with pytest.raises(ValidationError, match="must have exactly 6 elements"):
            Sigma70PromoterConfig(**kwargs)
