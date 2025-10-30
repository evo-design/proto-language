import pytest
import sys

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import sigma70_promoter_constraint
from proto_language.language.constraint.sequence_annotation.sigma70_promoter_constraint import Sigma70PromoterConfig
from ..utils import create_segment


# Tests for sigma70_promoter_constraint
class TestSigma70PromoterConstraint:
    def test_ideal_promoter(self):
        """Test ideal sigma70 promoter sequence."""
        # Ideal promoter: -35 box + 17bp spacer + -10 box
        ideal = "TTGACA" + "A" * 17 + "TATAAT"
        segment = create_segment(ideal, SequenceType.DNA)
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sigma70_promoter_constraint,
            scoring_function_config=config,
            vectorized=True,  # This is a vectorized constraint
        )
        
        scores = constraint.evaluate()
        assert scores[0] < 0.5  # Should have low penalty
        assert "segment_0.sigma70_promoter_constraint.sigma70" in segment.candidate_sequences[0]._metadata
        sigma70_data = segment.candidate_sequences[0]._metadata["segment_0.sigma70_promoter_constraint.sigma70"]
        assert sigma70_data["spacer_len"] == 17

    def test_poor_promoter(self):
        """Test poor promoter sequence."""
        poor = "AAAAAA" + "G" * 17 + "CCCCCC"
        segment = create_segment(poor, SequenceType.DNA)
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sigma70_promoter_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        assert scores[0] > 0.4  # Should have moderate-to-high penalty
        # Check some metadata was stored
        assert any("sigma70_promoter_constraint" in key for key in segment.candidate_sequences[0]._metadata.keys())

    def test_scanning_long_sequence(self):
        """Test scanning long sequence for best promoter."""
        # Embed promoter in longer sequence
        long_seq = "A" * 50 + "TTGACA" + "T" * 17 + "TATAAT" + "G" * 50
        segment = create_segment(long_seq, SequenceType.DNA)
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sigma70_promoter_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        assert scores[0] < 0.5
        # Check position metadata was stored
        assert "segment_0.sigma70_promoter_constraint.sigma70" in segment.candidate_sequences[0]._metadata
        sigma70_data = segment.candidate_sequences[0]._metadata["segment_0.sigma70_promoter_constraint.sigma70"]
        assert "pos" in sigma70_data

    def test_short_sequence(self):
        """Test sequence too short for promoter."""
        short = "ATCG"
        segment = create_segment(short, SequenceType.DNA)
        config = Sigma70PromoterConfig()

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sigma70_promoter_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        assert scores[0] == 1.0
        # Should indicate sequence is too short
        assert "segment_0.sigma70_promoter_constraint.sigma70" in segment.candidate_sequences[0]._metadata
        sigma70_data = segment.candidate_sequences[0]._metadata["segment_0.sigma70_promoter_constraint.sigma70"]
        assert "reason" in sigma70_data
        assert sigma70_data["reason"] == "too_short"

    def test_custom_consensus_sequences(self):
        """Test with custom consensus sequences (constraint-specific config behavior)."""
        # Use custom consensus sequences
        custom_seq = "AAAAAA" + "T" * 17 + "CCCCCC"
        segment = create_segment(custom_seq, SequenceType.DNA)
        config = Sigma70PromoterConfig(consensus_35="AAAAAA", consensus_10="CCCCCC")

        constraint = Constraint(
            inputs=[segment],
            scoring_function=sigma70_promoter_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        # Should have low penalty with matching custom consensus
        assert scores[0] < 0.5
        
        # Check constraint-specific metadata
        assert "segment_0.sigma70_promoter_constraint.sigma70" in segment.candidate_sequences[0]._metadata
        sigma70_data = segment.candidate_sequences[0]._metadata["segment_0.sigma70_promoter_constraint.sigma70"]
        assert "spacer_len" in sigma70_data
        assert sigma70_data["spacer_len"] == 17