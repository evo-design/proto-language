"""Tests the generalized k-mer frequency constraint that evaluates all k-mers of a given length.

For specific single-kmer tests, see test_specific_kmer_constraint.py.
"""

import pytest

from proto_language.constraint import (
    ConstraintRegistry,
    kmer_frequency_constraint,
)
from proto_language.constraint.sequence_composition.kmer_frequency_constraint import (
    KmerFrequencyConfig,
)
from proto_language.core import Constraint, Segment


class TestKmerFrequencyConstraint:
    """Tests for k-mer frequency constraint."""

    def test_dinucleotide_frequency_mode(self):
        """Test dinucleotide frequency evaluation."""
        # ATCGATCG has AT, TC, CG, GA dinucleotides
        seq = Segment(sequence="ATCGATCG", sequence_type="dna")

        config = KmerFrequencyConfig(k=2, scoring_mode="frequency", min_value=0.0, max_value=0.3)

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

    def test_trinucleotide_usage_deviation_mode(self):
        """Test trinucleotide usage deviation evaluation across all k-mers."""
        seq = Segment(sequence="ATCGATCGATCGATCG" * 5, sequence_type="dna")

        config = KmerFrequencyConfig(
            k=3,
            scoring_mode="usage_deviation",
            min_value=0.5,
            max_value=2.0,
        )

        constraint = Constraint(
            inputs=[seq],
            function=kmer_frequency_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0

        constraints = seq.proposal_sequences[0]._constraints_metadata
        assert "3mer_usage_deviations" in constraints["kmer_frequency_constraint"]["data"]
        deviations = constraints["kmer_frequency_constraint"]["data"]["3mer_usage_deviations"]
        # Only observed trinucleotides are reported, never the full 4^3 space.
        assert len(deviations) == 4
        assert set(deviations) == {"ATC", "TCG", "CGA", "GAT"}

    def test_protein_kmer_frequency(self):
        """Test k-mer frequency on protein sequences."""
        seq = Segment(sequence="MVLSPADKTNVKAAW", sequence_type="protein")

        config = KmerFrequencyConfig(k=2, scoring_mode="frequency", min_value=0.0, max_value=0.5)

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

        config = KmerFrequencyConfig(k=4, scoring_mode="frequency", min_value=0.0, max_value=0.5)

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
            KmerFrequencyConfig(k=2, scoring_mode="frequency", min_value=0.8, max_value=0.2)

    def test_nonzero_min_value_does_not_saturate(self):
        """min_value > 0 must not saturate every score to MAX_ENERGY (H1)."""
        # 4 distinct dinucleotides, each ~0.21-0.26 (in-band for [0.05, 0.4]); the
        # 12 absent ones used to saturate the old full-space lower-bound check to 1.0.
        seq = Segment(sequence="ACGTACGTACGTACGTACGT", sequence_type="dna")

        config = KmerFrequencyConfig(k=2, scoring_mode="frequency", min_value=0.05, max_value=0.4)
        constraint = Constraint(inputs=[seq], function=kmer_frequency_constraint, function_config=config)

        score = constraint.evaluate()[0]
        # An honest in-band sequence must score well below the max penalty.
        assert score < 1.0
        assert score == pytest.approx(0.0, abs=1e-6)

        freqs = seq.proposal_sequences[0]._constraints_metadata["kmer_frequency_constraint"]["data"]["2mer_frequencies"]
        # Only observed dinucleotides are reported, never the full 16-mer space worth of zeros.
        assert all(v > 0.0 for v in freqs.values())

    def test_registry_integration(self):
        """Test that constraint is properly registered."""
        spec = ConstraintRegistry.get("kmer-frequency")
        assert spec.key == "kmer-frequency"
        assert spec.label == "K-mer Frequency"
        assert "dna" in spec.supported_sequence_types
        assert "protein" in spec.supported_sequence_types
