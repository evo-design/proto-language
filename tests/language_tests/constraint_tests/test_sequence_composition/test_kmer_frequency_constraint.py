"""
tests/language_tests/constraint_tests/test_sequence_composition/test_kmer_frequency_constraint.py

Tests the generalized k-mer frequency constraint that replaced
dinucleotide_frequency and tetranucleotide_usage constraints.
"""

import pytest

from proto_language.language.constraint import (
    ConstraintRegistry,
    kmer_frequency_constraint,
)
from proto_language.language.constraint.sequence_composition.kmer_frequency_constraint import (
    KmerFrequencyConfig,
)
from proto_language.language.core import Constraint, Segment


class TestKmerFrequencyConstraint:
    """Tests for k-mer frequency constraint."""

    def test_dinucleotide_frequency_mode(self):
        """Test dinucleotide frequency evaluation."""
        # ATCGATCG has AT, TC, CG, GA dinucleotides
        seq = Segment(sequence="ATCGATCG", sequence_type="dna")

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
        constraints = seq.proposal_sequences[0]._constraints_metadata
        assert "2mer_frequencies" in constraints["kmer_frequency_constraint"]["data"]
        freqs = constraints["kmer_frequency_constraint"]["data"]["2mer_frequencies"]
        assert "AT" in freqs
        assert "CG" in freqs

    def test_tetranucleotide_usage_deviation_mode(self):
        """Test tetranucleotide usage deviation evaluation."""
        seq = Segment(sequence="AGCT" * 10 + "GATC" + "AGCT" * 10, sequence_type="dna")

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
        constraints = seq.proposal_sequences[0]._constraints_metadata
        assert "GATC_usage_deviation" in constraints["kmer_frequency_constraint"]["data"]
        assert "GATC_count" in constraints["kmer_frequency_constraint"]["data"]

    def test_protein_kmer_frequency(self):
        """Test k-mer frequency on protein sequences."""
        seq = Segment(sequence="MVLSPADKTNVKAAW", sequence_type="protein")

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

        constraints = seq.proposal_sequences[0]._constraints_metadata
        assert "2mer_frequencies" in constraints["kmer_frequency_constraint"]["data"]

    def test_empty_sequence(self):
        """Test that zero-length segment raises ValueError."""
        with pytest.raises(ValueError, match="Segment length must be positive"):
            Segment(length=0, sequence_type="dna")

    def test_sequence_too_short(self):
        """Test sequences shorter than k."""
        seq = Segment(sequence="AT", sequence_type="dna")

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
        assert "dna" in spec.supported_sequence_types
        assert "protein" in spec.supported_sequence_types
