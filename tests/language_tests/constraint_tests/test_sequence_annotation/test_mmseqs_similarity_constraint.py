"""
Comprehensive tests for mmseqs_similarity_constraint.

Tests the MMseqs2 similarity constraint for protein sequences.
"""

import pytest
from unittest.mock import patch

from proto_language.language.core import Constraint, Segment
from proto_language.language.constraint import mmseqs_similarity_constraint, ConstraintRegistry
from proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint import MMseqsSimilarityConfig
from proto_language.bio_tools.tools.gene_annotation.mmseqs import (
    MmseqsSearchProteinsConfig,
    MmseqsSearchProteinsOutput,
    MmseqsSequenceSearchResult,
    MmseqsHit,
)


class TestMMseqsSimilarityConstraint:
    """Tests for MMseqs2 similarity constraint."""

    @pytest.fixture
    def dummy_db_path(self, tmp_path):
        """Create a dummy database path."""
        db_path = tmp_path / "test_db"
        db_path.mkdir()
        return str(db_path)

    def _create_mock_output(self, results):
        """Helper to create mock output with success=True."""
        return MmseqsSearchProteinsOutput(
            metadata={},
            results=results,
            success=True,
        )

    def test_config_validation(self, dummy_db_path):
        """Test configuration validation."""
        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        assert config.min_similarity == 80.0
        assert config.max_similarity == 100.0
        assert config.mmseqs_db == dummy_db_path

    def test_with_mocked_mmseqs(self, dummy_db_path):
        """Test constraint with mocked MMseqs2 results."""
        segment = Segment(sequence="MVLSPADKTNVKAAW", sequence_type="protein")

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
            mmseqs_config=MmseqsSearchProteinsConfig(threads=1)
        )

        # Mock MMseqs2 search with new output structure
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.run_mmseqs_search_proteins') as mock_mmseqs:
            mock_mmseqs.return_value = self._create_mock_output([
                MmseqsSequenceSearchResult(
                    query_id="seq_0",
                    query_sequence="MVLSPADKTNVKAAW",
                    hits=[
                        MmseqsHit(target_id="hit1", pident=90.0, evalue=1e-10),
                    ],
                )
            ])

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            assert isinstance(scores[0], float)
            assert scores[0] >= 0.0

            # Check metadata - verify results were stored
            constraints = segment.candidate_sequences[0]._metadata["constraints"]
            assert "mmseqs_results" in constraints["mmseqs_similarity_constraint"]["data"]
            # Should have 1 hit from our mock
            results = constraints["mmseqs_similarity_constraint"]["data"]["mmseqs_results"]
            assert len(results) == 1
            assert results[0]["pident"] == 90.0
            assert "unique_orfs_with_hits" in constraints["mmseqs_similarity_constraint"]["data"]

    def test_no_hits_scenario(self, dummy_db_path):
        """Test when no MMseqs2 hits are found."""
        segment = Segment(sequence="MVLSP", sequence_type="protein")

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        # Mock MMseqs2 with no results
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.run_mmseqs_search_proteins') as mock_mmseqs:
            mock_mmseqs.return_value = self._create_mock_output([
                MmseqsSequenceSearchResult(
                    query_id="seq_0",
                    query_sequence="MVLSP",
                    hits=[],  # No hits
                )
            ])

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            # Score depends on implementation - MAX_ENERGY (1.0) for no hits
            assert isinstance(scores[0], float)
            assert scores[0] == 1.0  # MAX_ENERGY

    def test_registry_integration(self):
        """Test that constraint is properly registered."""
        spec = ConstraintRegistry.get("mmseqs-gene-similarity")
        assert spec.key == "mmseqs-gene-similarity"
        assert spec.label == "Gene/Protein Similarity"  # Actual label in registry
        assert "protein" in spec.supported_sequence_types

    def test_dna_sequence_with_orf_prediction(self, dummy_db_path):
        """Test that DNA sequences work via ORF prediction."""
        segment = Segment(sequence="ATGGTGCTGAGCCCGGCGGACAAG", sequence_type="dna")

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        # Mock MMseqs2 with results - note: for DNA, ORF prediction happens first
        # and might produce zero proteins if no ORFs are found in short sequences
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.run_mmseqs_search_proteins') as mock_mmseqs:
            # This test might not even call mmseqs if no ORFs are predicted
            # But if it does, we mock the response
            mock_mmseqs.return_value = self._create_mock_output([
                MmseqsSequenceSearchResult(
                    query_id="seq_0",
                    query_sequence="MVLSPADKTN",  # Hypothetical translated protein
                    hits=[
                        MmseqsHit(target_id="hit1", pident=85.0, evalue=1e-10),
                    ],
                )
            ])

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            assert isinstance(scores[0], float)
            # DNA sequences are supported via ORF prediction
            assert scores[0] >= 0.0

    def test_multiple_hits_per_sequence(self, dummy_db_path):
        """Test handling of multiple hits per sequence."""
        segment = Segment(sequence="MVLSPADKTNVKAAW", sequence_type="protein")

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        # Mock MMseqs2 with multiple hits for the single protein
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.run_mmseqs_search_proteins') as mock_mmseqs:
            mock_mmseqs.return_value = self._create_mock_output([
                MmseqsSequenceSearchResult(
                    query_id="seq_0",
                    query_sequence="MVLSPADKTNVKAAW",
                    hits=[
                        MmseqsHit(target_id="hit1", pident=95.0, evalue=1e-50),
                        MmseqsHit(target_id="hit2", pident=85.0, evalue=1e-30),
                        MmseqsHit(target_id="hit3", pident=75.0, evalue=1e-20),  # Below threshold
                    ],
                )
            ])

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            assert isinstance(scores[0], float)

            # Check metadata shows correct hit counts
            constraints = segment.candidate_sequences[0]._metadata["constraints"]
            assert constraints["mmseqs_similarity_constraint"]["data"]["total_orfs_with_hits"] == 3
            # 2 hits are within range (85 and 95), 1 is below (75)
            assert constraints["mmseqs_similarity_constraint"]["data"]["orfs_with_acceptable_similarity"] == 2

    def test_multiple_candidates_in_segment(self, dummy_db_path):
        """Test constraint with multiple candidate sequences in a single segment."""
        # Create a segment with multiple candidates
        segment = Segment(sequence="MVLSPADKTN", sequence_type="protein")
        # Add another candidate sequence
        from proto_language.language.core import Sequence
        segment.candidate_sequences.append(Sequence("MKLLVVAAAA", "protein"))

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        # Mock MMseqs2 with results for both proteins
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.run_mmseqs_search_proteins') as mock_mmseqs:
            # The constraint evaluates candidates one at a time in batched mode
            # but mmseqs_similarity_constraint processes all proteins at once
            # So mock needs to handle calls for each candidate
            def mock_side_effect(inputs, config):
                # Return appropriate number of results based on input
                results = [
                    MmseqsSequenceSearchResult(
                        query_id=f"seq_{i}",
                        query_sequence=seq,
                        hits=[MmseqsHit(target_id=f"hit_{i}", pident=90.0, evalue=1e-10)],
                    )
                    for i, seq in enumerate(inputs.query_sequences)
                ]
                return MmseqsSearchProteinsOutput(metadata={}, results=results, success=True)
            
            mock_mmseqs.side_effect = mock_side_effect

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            # Should have 2 scores, one for each candidate
            assert len(scores) == 2
            assert all(isinstance(s, float) for s in scores)

    def test_score_within_acceptable_range(self, dummy_db_path):
        """Test that scores are 0 when hits are within acceptable range."""
        segment = Segment(sequence="MVLSPADKTNVKAAW", sequence_type="protein")

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.run_mmseqs_search_proteins') as mock_mmseqs:
            mock_mmseqs.return_value = self._create_mock_output([
                MmseqsSequenceSearchResult(
                    query_id="seq_0",
                    query_sequence="MVLSPADKTNVKAAW",
                    hits=[
                        MmseqsHit(target_id="hit1", pident=90.0, evalue=1e-10),  # Within range
                    ],
                )
            ])

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            # Score should be 0.0 (MIN_ENERGY) when all hits are within range
            assert scores[0] == 0.0

    def test_score_outside_acceptable_range(self, dummy_db_path):
        """Test that scores are non-zero when hits are outside acceptable range."""
        segment = Segment(sequence="MVLSPADKTNVKAAW", sequence_type="protein")

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.run_mmseqs_search_proteins') as mock_mmseqs:
            mock_mmseqs.return_value = self._create_mock_output([
                MmseqsSequenceSearchResult(
                    query_id="seq_0",
                    query_sequence="MVLSPADKTNVKAAW",
                    hits=[
                        MmseqsHit(target_id="hit1", pident=50.0, evalue=1e-10),  # Below min
                    ],
                )
            ])

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            # Score should be > 0 when hits are outside range
            assert scores[0] > 0.0
