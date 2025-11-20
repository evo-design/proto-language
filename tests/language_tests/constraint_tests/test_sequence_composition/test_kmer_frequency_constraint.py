"""
Comprehensive tests for kmer_frequency_constraint.

Tests the generalized k-mer frequency constraint that replaced
dinucleotide_frequency and tetranucleotide_usage constraints.
"""

import pytest

from proto_language.language.core import Constraint, Segment, SequenceType
from proto_language.language.constraint import kmer_frequency_constraint, ConstraintRegistry
from proto_language.language.constraint.sequence_composition.kmer_frequency_constraint import KmerFrequencyConfig


class TestKmerFrequencyConstraint:
    """Tests for k-mer frequency constraint."""

    def test_dinucleotide_frequency_mode(self):
        """Test dinucleotide frequency evaluation."""
        # ATCGATCG has AT, TC, CG, GA dinucleotides
        seq = Segment(starting_sequence_or_desired_length="ATCGATCG", sequence_type=SequenceType.DNA)

        config = KmerFrequencyConfig(
            k=2,
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.3
        )

        constraint = Constraint(
            inputs=[seq],
            function=kmer_frequency_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0

        # Check metadata
        metadata = seq.candidate_sequences[0]._metadata
        assert "segment_0.kmer_frequency_constraint.2mer_frequencies" in metadata
        freqs = metadata["segment_0.kmer_frequency_constraint.2mer_frequencies"]
        assert "AT" in freqs
        assert "CG" in freqs

    def test_tetranucleotide_usage_deviation_mode(self):
        """Test tetranucleotide usage deviation evaluation."""
        seq = Segment(starting_sequence_or_desired_length="AGCT" * 10 + "GATC" + "AGCT" * 10, sequence_type=SequenceType.DNA)

        config = KmerFrequencyConfig(
            k=4,
            scoring_mode="usage_deviation",
            specific_kmer="GATC",
            min_value=0.8,
            max_value=1.2
        )

        constraint = Constraint(
            inputs=[seq],
            function=kmer_frequency_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0

        # Check metadata
        metadata = seq.candidate_sequences[0]._metadata
        assert "segment_0.kmer_frequency_constraint.GATC_usage_deviation" in metadata
        assert "segment_0.kmer_frequency_constraint.GATC_count" in metadata

    def test_protein_kmer_frequency(self):
        """Test k-mer frequency on protein sequences."""
        seq = Segment(starting_sequence_or_desired_length="MVLSPADKTNVKAAW", sequence_type=SequenceType.PROTEIN)

        config = KmerFrequencyConfig(
            k=2,
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5
        )

        constraint = Constraint(
            inputs=[seq],
            function=kmer_frequency_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0

        metadata = seq.candidate_sequences[0]._metadata
        assert "segment_0.kmer_frequency_constraint.2mer_frequencies" in metadata

    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        seq = Segment(starting_sequence_or_desired_length="", sequence_type=SequenceType.DNA)

        config = KmerFrequencyConfig(
            k=2,
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5
        )

        constraint = Constraint(
            inputs=[seq],
            function=kmer_frequency_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 1.0  # MAX_ENERGY for empty sequence

    def test_sequence_too_short(self):
        """Test sequences shorter than k."""
        seq = Segment(starting_sequence_or_desired_length="AT", sequence_type=SequenceType.DNA)

        config = KmerFrequencyConfig(
            k=4,
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5
        )

        constraint = Constraint(
            inputs=[seq],
            function=kmer_frequency_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 1.0  # MAX_ENERGY for sequence too short

    def test_config_validation(self):
        """Test configuration validation."""
        from pydantic import ValidationError

        # min_value > max_value should fail
        with pytest.raises(ValidationError):
            KmerFrequencyConfig(
                k=2,
                scoring_mode="frequency",
                min_value=0.8,
                max_value=0.2
            )

        # specific_kmer length mismatch should fail
        with pytest.raises(ValidationError):
            KmerFrequencyConfig(
                k=2,
                scoring_mode="frequency",
                specific_kmer="ATCG",  # Length 4, but k=2
                min_value=0.0,
                max_value=0.5
            )

    def test_registry_integration(self):
        """Test that constraint is properly registered."""
        spec = ConstraintRegistry.get("kmer-frequency")
        assert spec.key == "kmer-frequency"
        assert spec.label == "K-mer Frequency"
        assert spec.batched == True
        assert spec.concatenate == True
