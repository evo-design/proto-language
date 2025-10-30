"""
Comprehensive tests for overall_protein_quality_constraint.
"""

import pytest
import sys

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import overall_protein_quality_constraint
from proto_language.language.constraint.protein_quality.overall_protein_quality_constraint import OverallProteinQualityConfig, ProteinQualitySubConfig
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import SequenceLengthConfig
from proto_language.language.constraint.protein_quality.protein_diversity_constraint import ProteinDiversityConfig
from proto_language.language.constraint.protein_quality.protein_repetitiveness_constraint import ProteinRepetitivenessConfig
from ..utils import create_segment


class TestOverallProteinQualityConstraint:
    """Unit tests for overall_protein_quality_constraint."""
    
    def test_protein_input_high_quality(self):
        """Test with high quality protein sequence."""
        segment = create_segment("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF", SequenceType.PROTEIN)
        
        # Only check length - protein is 48 amino acids
        sub_config = ProteinQualitySubConfig(
            length=SequenceLengthConfig(target_length=50),  # Close to actual length of 48
            quality_threshold=0.5
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=overall_protein_quality_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert scores[0] == 0.0  # High quality
    
    def test_protein_input_low_quality_length(self):
        """Test with protein that violates length constraint."""
        segment = create_segment("MVLSP", SequenceType.PROTEIN)
        
        sub_config = ProteinQualitySubConfig(
            length=SequenceLengthConfig(target_length=20),  # Much longer than actual length of 5
            quality_threshold=0.1
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=overall_protein_quality_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert scores[0] > 0.0  # Low quality due to short length
    
    def test_protein_input_multiple_checks(self):
        """Test with multiple quality checks and constraint-specific metadata."""
        segment = create_segment("MVLSPADKTNVKAAWGKVGAHAGEYGAEAL", SequenceType.PROTEIN)
        
        sub_config = ProteinQualitySubConfig(
            length=SequenceLengthConfig(target_length=30),
            diversity=ProteinDiversityConfig(min_diversity=0.3),
            quality_threshold=0.5
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=overall_protein_quality_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 1
        # Should be high quality (good length, good diversity)
        assert scores[0] < 0.5
        
        # Check constraint-specific metadata fields
        assert any("overall_protein_quality_constraint" in key for key in segment.candidate_sequences[0]._metadata.keys())
        assert any("avg_constraint_score" in key for key in segment.candidate_sequences[0]._metadata.keys())
        assert any("is_high_quality" in key for key in segment.candidate_sequences[0]._metadata.keys())
    
    def test_protein_input_repetitive(self):
        """Test with repetitive protein."""
        segment = create_segment("AAAAAAAAAAAAAAAA", SequenceType.PROTEIN)
        
        sub_config = ProteinQualitySubConfig(
            repetitiveness=ProteinRepetitivenessConfig(max_repetitiveness=0.3),
            diversity=ProteinDiversityConfig(min_diversity=0.2),
            quality_threshold=0.1
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=overall_protein_quality_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert scores[0] > 0.0  # Low quality
    
    def test_config_validation_no_subchecks(self):
        """Test that config requires at least one sub-check (constraint-specific validation)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            sub_config = ProteinQualitySubConfig(quality_threshold=0.1)
            config = OverallProteinQualityConfig(
                protein_quality_config=sub_config
            )
