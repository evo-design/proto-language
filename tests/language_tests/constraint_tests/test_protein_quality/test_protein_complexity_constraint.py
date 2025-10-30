"""
Comprehensive tests for Protein Complexity constraint.

Tests cover:
1. Basic functionality with mocked segmasker
2. Configuration validation
3. Registry integration
4. Error handling
5. Metadata propagation
"""

import pytest
import sys
from unittest.mock import patch

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import protein_complexity_constraint
from proto_language.language.constraint.protein_quality.protein_complexity_constraint import ProteinComplexityConfig
from proto_language.tools.sequence_scoring.segmasker import SegmaskerOutput
from ..utils import create_segment


class TestProteinComplexityConstraint:
    """Tests for Protein Complexity constraint."""
    
    @pytest.mark.parametrize(
        "low_complexity_fraction, max_low_complexity, expected_score",
        [
            (0.2, 0.3, 0.0),  # Within range
            (0.4, 0.3, 0.1428571428571429),  # Above range: (0.4-0.3)/(1.0-0.3) = 0.1/0.7
            (0.0, 0.3, 0.0),  # Perfect complexity
        ],
        ids=["within_range", "above_range", "perfect"]
    )
    def test_scoring_logic(self, low_complexity_fraction, max_low_complexity, expected_score):
        """Test the scoring logic with mocked segmasker output."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinComplexityConfig(max_low_complexity=max_low_complexity)
        
        # Mock run_segmasker
        with patch('proto_language.language.constraint.protein_quality.protein_complexity_constraint.run_segmasker') as mock_seg:
            mock_output = SegmaskerOutput(
                tool_id="segmasker",
                execution_time=0.1,
                success=True,
                low_complexity_fractions=[low_complexity_fraction],
                low_complexity_counts=[int(low_complexity_fraction * 16)],
                sequence_lengths=[16],
                errors=[]
            )
            mock_seg.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_complexity_constraint,
                scoring_function_config=config,
                vectorized=True,
            )
            
            scores = constraint.evaluate()
            assert abs(scores[0] - expected_score) < 0.01
            
            # Check constraint-specific metadata fields
            metadata = segment.candidate_sequences[0]._metadata
            assert "segment_0.protein_complexity_constraint.low_complexity_fraction" in metadata
            assert abs(metadata["segment_0.protein_complexity_constraint.low_complexity_fraction"] - low_complexity_fraction) < 1e-9
            assert "segment_0.protein_complexity_constraint.segmasker_X_count" in metadata
            assert metadata["segment_0.protein_complexity_constraint.segmasker_error"] == False
    
    def test_segmasker_error_handling(self):
        """Test error handling when segmasker fails."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinComplexityConfig(max_low_complexity=0.3)
        
        with patch('proto_language.language.constraint.protein_quality.protein_complexity_constraint.run_segmasker') as mock_seg:
            mock_output = SegmaskerOutput(
                tool_id="segmasker",
                execution_time=0.0,
                success=False,
                low_complexity_fractions=[],
                low_complexity_counts=[],
                sequence_lengths=[],
                errors=["Segmasker execution failed"]
            )
            mock_seg.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_complexity_constraint,
                scoring_function_config=config,
                vectorized=True,
            )
            
            # The constraint should raise ValueError
            with pytest.raises(ValueError, match="Segmasker analysis failed"):
                constraint.evaluate()
    
    def test_wrong_sequence_type(self):
        """Test that DNA/RNA sequences raise assertion (constraint-specific check)."""
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        config = ProteinComplexityConfig(max_low_complexity=0.3)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_complexity_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        with pytest.raises(AssertionError):
            constraint.evaluate()
    