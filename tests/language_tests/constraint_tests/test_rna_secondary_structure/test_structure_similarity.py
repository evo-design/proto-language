"""
Tests for RNA secondary structure similarity constraints.
"""

from proto_language.language.core import Sequence
from proto_language.language.constraint.rna_secondary_structure.structure_similarity_constraint import (
    rna_property_similarity_constraint,
    rna_motif_similarity_constraint,
    rna_feature_similarity_constraint,
    rna_basepair_similarity_constraint,
    RNAPropertySimilarityConfig,
    RNAMotifSimilarityConfig,
    RNAFeatureSimilarityConfig,
    RNABasePairSimilarityConfig,
)


# Test sequences
HAIRPIN_SEQ = "GCGCUUUUGCGC"  # Forms a simple hairpin
SIMILAR_HAIRPIN = "GCGCAAAAGCGC"  # Similar hairpin, different loop
DIFFERENT_SEQ = "AAAAAAAAAAAAA"  # Unstructured poly-A
LONG_HAIRPIN = "GGGGCCCCUUUUGGGGCCCC"  # Longer structured sequence

EPSILON = 1e-10


class TestRNAPropertySimilarityConstraint:
    """Tests for RNA structural property similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_property_similarity_constraint(
            [Sequence(HAIRPIN_SEQ, "rna")],
            config,
        )[0]
        assert score < EPSILON

    def test_imperfect_match(self):
        """Different sequences should have score > 0."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_property_similarity_constraint(
            [Sequence(DIFFERENT_SEQ, "rna")],
            config,
        )[0]
        assert score > 0.0

    def test_similar_structures(self):
        """Similar hairpins should have low but non-zero score."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_property_similarity_constraint(
            [Sequence(SIMILAR_HAIRPIN, "rna")],
            config,
        )[0]
        assert score < 0.5  # Should be fairly similar


class TestRNAMotifSimilarityConstraint:
    """Tests for RNA structural motif similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNAMotifSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_motif_similarity_constraint(
            [Sequence(HAIRPIN_SEQ, "rna")],
            config,
        )[0]
        assert score < EPSILON

    def test_different_motifs(self):
        """Unstructured vs structured should have high score."""
        config = RNAMotifSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_motif_similarity_constraint(
            [Sequence(DIFFERENT_SEQ, "rna")],
            config,
        )[0]
        assert score > 0.5


class TestRNAFeatureSimilarityConstraint:
    """Tests for RNA feature vector similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNAFeatureSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_feature_similarity_constraint(
            [Sequence(HAIRPIN_SEQ, "rna")],
            config,
        )[0]
        assert score < EPSILON

    def test_imperfect_match(self):
        """Different sequences should have score > 0."""
        config = RNAFeatureSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_feature_similarity_constraint(
            [Sequence(DIFFERENT_SEQ, "rna")],
            config,
        )[0]
        assert score > 0.0


class TestRNABasePairSimilarityConstraint:
    """Tests for RNA base pair similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNABasePairSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_basepair_similarity_constraint(
            [Sequence(HAIRPIN_SEQ, "rna")],
            config,
        )[0]
        assert score < EPSILON

    def test_no_shared_pairs(self):
        """Unstructured vs structured should have score of 1."""
        config = RNABasePairSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        score = rna_basepair_similarity_constraint(
            [Sequence(DIFFERENT_SEQ, "rna")],
            config,
        )[0]
        assert score == 1.0

    def test_length_ratio_cutoff(self):
        """Very different lengths should return score of 1."""
        config = RNABasePairSimilarityConfig(
            reference_sequence=HAIRPIN_SEQ,
            max_length_ratio_diff=0.3,
        )
        # LONG_HAIRPIN is much longer than HAIRPIN_SEQ
        score = rna_basepair_similarity_constraint(
            [Sequence(LONG_HAIRPIN, "rna")],
            config,
        )[0]
        assert score == 1.0


class TestBatchedConstraints:
    """Test that constraints handle batched inputs correctly."""

    def test_batched_property_constraint(self):
        """Test multiple sequences in one call."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        scores = rna_property_similarity_constraint(
            [
                Sequence(HAIRPIN_SEQ, "rna"),
                Sequence(SIMILAR_HAIRPIN, "rna"),
                Sequence(DIFFERENT_SEQ, "rna"),
            ],
            config,
        )
        assert len(scores) == 3
        assert scores[0] < scores[2]  # Perfect match < different sequence
