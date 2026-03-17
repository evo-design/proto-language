"""
Comprehensive tests for overall_protein_quality_constraint.
"""

import pytest

from proto_language.language.constraint import overall_protein_quality_constraint
from proto_language.language.constraint.protein_quality.overall_protein_quality_constraint import (
    OverallProteinQualityConfig,
    ProteinQualitySubConfig,
)
from proto_language.language.core import Constraint, Segment


class TestOverallProteinQualityConstraint:
    """Unit tests for overall_protein_quality_constraint."""

    def test_protein_input_high_quality(self):
        """Test with high quality protein sequence."""
        segment = Segment(sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF", sequence_type="protein")

        # Only check length - protein is 46 amino acids
        sub_config = ProteinQualitySubConfig(
            enable_length=True,
            length_target_length=50,  # Close to actual length of 46
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )

        constraint = Constraint(
            inputs=[segment],
            function=overall_protein_quality_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert scores[0] < 0.1  # Near-perfect: deviation is |46-50|/50 = 0.08

    def test_protein_input_low_quality_length(self):
        """Test with protein that violates length constraint."""
        segment = Segment(sequence="MVLSP", sequence_type="protein")

        sub_config = ProteinQualitySubConfig(
            enable_length=True,
            length_target_length=20,  # Much longer than actual length of 5
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )

        constraint = Constraint(
            inputs=[segment],
            function=overall_protein_quality_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert scores[0] > 0.0  # Low quality due to short length

    def test_protein_input_multiple_checks(self):
        """Test with multiple quality checks and constraint-specific metadata."""
        segment = Segment(sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEAL", sequence_type="protein")

        sub_config = ProteinQualitySubConfig(
            enable_length=True,
            length_target_length=30,
            enable_diversity=True,
            diversity_min_diversity=0.3,
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )

        constraint = Constraint(
            inputs=[segment],
            function=overall_protein_quality_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        # Should be high quality (good length, good diversity)
        assert scores[0] < 0.5

        # Check constraint-specific metadata fields (nested under constraints)
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "overall_protein_quality_constraint" in constraints
        constraint_data = constraints["overall_protein_quality_constraint"]
        assert "avg_constraint_score" in constraint_data["data"]

    def test_protein_input_repetitive(self):
        """Test with repetitive protein."""
        segment = Segment(sequence="AAAAAAAAAAAAAAAA", sequence_type="protein")

        sub_config = ProteinQualitySubConfig(
            enable_repetitiveness=True,
            repetitiveness_max_repetitiveness=0.3,
            enable_diversity=True,
            diversity_min_diversity=0.2,
        )
        config = OverallProteinQualityConfig(
            protein_quality_config=sub_config
        )

        constraint = Constraint(
            inputs=[segment],
            function=overall_protein_quality_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert scores[0] > 0.0  # Low quality

    def test_config_validation_no_subchecks(self):
        """Test that config requires at least one sub-check (constraint-specific validation)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            sub_config = ProteinQualitySubConfig()
            _ = OverallProteinQualityConfig(
                protein_quality_config=sub_config
            )
