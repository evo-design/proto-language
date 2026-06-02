"""Tests for protein complexity constraint with mocked segmasker and config validation."""

from unittest.mock import patch

import pytest
from proto_tools import SegmaskerMetrics, SegmaskerOutput

from proto_language.constraint import protein_complexity_constraint
from proto_language.constraint.protein_quality.protein_complexity_constraint import (
    ProteinComplexityConfig,
)
from proto_language.core import Constraint, Segment


class TestProteinComplexityConstraint:
    """Tests for Protein Complexity constraint."""

    @pytest.mark.parametrize(
        "low_complexity_fraction, max_low_complexity, expected_score",
        [
            (0.2, 0.3, 0.0),  # Within range
            (0.4, 0.3, 0.1428571428571429),  # Above range: (0.4-0.3)/(1.0-0.3) = 0.1/0.7
            (0.0, 0.3, 0.0),  # Perfect complexity
        ],
        ids=["within_range", "above_range", "perfect"],
    )
    def test_scoring_logic(self, low_complexity_fraction, max_low_complexity, expected_score):
        """Test the scoring logic with mocked segmasker output."""
        segment = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        config = ProteinComplexityConfig(max_low_complexity=max_low_complexity)

        # Mock run_segmasker
        with patch("proto_language.constraint.protein_quality.protein_complexity_constraint.run_segmasker") as mock_seg:
            mock_output = SegmaskerOutput(
                tool_id="segmasker",
                execution_time=0.1,
                success=True,
                results=[
                    SegmaskerMetrics(
                        low_complexity_fraction=low_complexity_fraction,
                        low_complexity_count=int(low_complexity_fraction * 16),
                        sequence_length=16,
                    )
                ],
            )
            mock_seg.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_complexity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert scores[0] == pytest.approx(expected_score)

            # Check constraint-specific metadata fields
            data = segment.proposal_sequences[0]._constraints_metadata["protein_complexity_constraint"]["data"]
            assert "low_complexity_fraction" in data
            assert data["low_complexity_fraction"] == pytest.approx(low_complexity_fraction)
            # Report segmasker's exact count; drop the old hardcoded segmasker_* keys.
            assert data["low_complexity_count"] == int(low_complexity_fraction * 16)
            assert "segmasker_error" not in data
            assert "segmasker_lowercase_count" not in data

    def test_segmasker_error_handling(self):
        """Test that segmasker failures propagate from the wrapped tool."""
        segment = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        config = ProteinComplexityConfig(max_low_complexity=0.3)

        with patch("proto_language.constraint.protein_quality.protein_complexity_constraint.run_segmasker") as mock_seg:
            mock_seg.side_effect = RuntimeError("Segmasker execution failed")

            constraint = Constraint(
                inputs=[segment],
                function=protein_complexity_constraint,
                function_config=config,
            )

            with pytest.raises(RuntimeError, match="Segmasker execution failed"):
                constraint.evaluate()

    def test_wrong_sequence_type(self):
        """Test that DNA/RNA sequences raise TypeError at construction (centralized validation)."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        config = ProteinComplexityConfig(max_low_complexity=0.3)

        with pytest.raises(TypeError, match="does not support sequence type 'dna'"):
            Constraint(
                inputs=[segment],
                function=protein_complexity_constraint,
                function_config=config,
            )
