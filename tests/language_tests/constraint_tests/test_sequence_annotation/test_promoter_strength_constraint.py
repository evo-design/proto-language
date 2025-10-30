"""
Comprehensive tests for Promoter Strength constraint.

Tests cover:
1. Configuration validation
2. Basic functionality with mocked promoter calculator
3. Scoring schemes (tx_rate vs dG)
4. Batch processing
5. Registry integration
6. Context addition
7. Error handling

Note: Actual promoter_calculator execution is mocked to avoid dependencies.
"""

import pytest
import sys
from unittest.mock import patch, Mock

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import promoter_strength_constraint
from proto_language.language.constraint.sequence_annotation.promoter_strength_constraint import PromoterStrengthConfig
from ..utils import create_segment


class TestPromoterStrengthConstraint:
    """Tests for Promoter Strength constraint."""
    
    def test_no_promoter_found(self):
        """Test scoring when no promoter is found."""
        segment = create_segment("ATCGATCGATCG", SequenceType.DNA)
        config = PromoterStrengthConfig()
        
        # Mock promoter_calculator returning empty list for each sequence
        with patch('proto_language.language.constraint.sequence_annotation.promoter_strength_constraint.promoter_calculator') as mock_calc:
            mock_calc.return_value = [[]]  # List of lists, one empty list per sequence
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=promoter_strength_constraint,
                scoring_function_config=config,
                vectorized=True
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] == 1.0  # No promoter -> penalty = 1.0
    
    def test_tx_rate_scoring_weak(self):
        """Test tx_rate scoring for weak promoter."""
        segment = create_segment("ATCGATCGATCG", SequenceType.DNA)
        config = PromoterStrengthConfig(scoring_type="tx_rate")
        
        # Mock promoter with low tx_rate
        mock_result = Mock()
        mock_result.Tx_rate = 1000.0
        mock_result.strand = "+"
        
        with patch('proto_language.language.constraint.sequence_annotation.promoter_strength_constraint.promoter_calculator') as mock_calc:
            mock_calc.return_value = [mock_result]
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=promoter_strength_constraint,
                scoring_function_config=config,
                vectorized=True
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] == 1.0  # tx_rate < 1500 -> penalty = 1.0
    
    def test_tx_rate_scoring_strong(self):
        """Test tx_rate scoring for strong promoter."""
        segment = create_segment("ATCGATCGATCG", SequenceType.DNA)
        config = PromoterStrengthConfig(scoring_type="tx_rate")
        
        # Mock promoter with high tx_rate
        mock_result = Mock()
        mock_result.Tx_rate = 8000.0
        mock_result.strand = "+"
        
        with patch('proto_language.language.constraint.sequence_annotation.promoter_strength_constraint.promoter_calculator') as mock_calc:
            mock_calc.return_value = [mock_result]
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=promoter_strength_constraint,
                scoring_function_config=config,
                vectorized=True
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] <= 0.5  # High tx_rate -> low penalty
    
    def test_dG_scoring_weak(self):
        """Test dG scoring for weak promoter."""
        segment = create_segment("ATCGATCGATCG", SequenceType.DNA)
        config = PromoterStrengthConfig(scoring_type="dG")
        
        # Mock promoter with weak binding (high dG)
        mock_result = Mock()
        mock_result.dG_total = 0.0
        mock_result.strand = "+"
        
        with patch('proto_language.language.constraint.sequence_annotation.promoter_strength_constraint.promoter_calculator') as mock_calc:
            mock_calc.return_value = [mock_result]
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=promoter_strength_constraint,
                scoring_function_config=config,
                vectorized=True
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] == 1.0  # dG >= 0 -> penalty = 1.0
    
    def test_dG_scoring_strong(self):
        """Test dG scoring for strong promoter and constraint-specific metadata."""
        segment = create_segment("ATCGATCGATCG", SequenceType.DNA)
        config = PromoterStrengthConfig(scoring_type="dG")
        
        # Mock promoter with strong binding (low dG)
        mock_result = Mock()
        mock_result.dG_total = -4.0
        mock_result.strand = "+"
        mock_result.__dict__ = {"dG_total": -4.0, "strand": "+"}
        
        with patch('proto_language.language.constraint.sequence_annotation.promoter_strength_constraint.promoter_calculator') as mock_calc:
            mock_calc.return_value = [[mock_result]]
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=promoter_strength_constraint,
                scoring_function_config=config,
                vectorized=True
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] <= 0.5  # dG < -3.0 -> low penalty
            
            # Check constraint-specific metadata fields
            metadata = segment.candidate_sequences[0]._metadata
            metadata_key = "segment_0.promoter_strength_constraint.promoter_strength"
            assert metadata_key in metadata
            assert "penalty" in metadata[metadata_key]
            assert "dG_rate" in metadata[metadata_key]
    
    def test_add_context(self):
        """Test that add_context parameter adds flanking sequence."""
        segment = create_segment("ATCG", SequenceType.DNA)
        config = PromoterStrengthConfig(add_context=True, context_length=5)
        
        mock_result = Mock()
        mock_result.dG_total = -2.0
        mock_result.strand = "+"
        
        with patch('proto_language.language.constraint.sequence_annotation.promoter_strength_constraint.promoter_calculator') as mock_calc:
            mock_calc.return_value = [[mock_result]]  # List of lists for vectorized constraint
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=promoter_strength_constraint,
                scoring_function_config=config,
                vectorized=True
            )
            
            constraint.evaluate()
            
            # Verify the sequence passed to calculator has context added
            call_args = mock_calc.call_args
            # For vectorized constraint, it receives a list of sequences
            processed_seq = call_args[0][0][0]
            # Should be "AAAAA" + "ATCG" + "AAAAA" = "AAAAAA TCGAAAAA"
            assert processed_seq.startswith("AAAAA")
            assert processed_seq.endswith("AAAAA")
            assert "ATCG" in processed_seq
    
