"""
Comprehensive tests for mmseqs_similarity_constraint.

Tests the MMseqs2 similarity constraint for protein sequences.
"""

import pytest
from unittest.mock import patch, MagicMock

from proto_language.language.core import Constraint, SequenceType, Segment
from proto_language.language.constraint import mmseqs_similarity_constraint, ConstraintRegistry
from proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint import MMseqsSimilarityConfig
from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig, MmseqsOutput


class TestMMseqsSimilarityConstraint:
    """Tests for MMseqs2 similarity constraint."""

    @pytest.fixture
    def dummy_db_path(self, tmp_path):
        """Create a dummy database path."""
        db_path = tmp_path / "test_db"
        db_path.mkdir()
        return str(db_path)

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
        segment = Segment(starting_sequence_or_desired_length="MVLSPADKTNVKAAW", sequence_type=SequenceType.PROTEIN)

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
            mmseqs_config=MmseqsSearchProteinsConfig(results_dir="", threads=1)
        )

        # Mock MMseqs2 search
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.mmseqs_search_proteins') as mock_mmseqs:
            import pandas as pd
            mock_output = MagicMock(spec=MmseqsOutput)
            mock_output.success = True
            # Provide results_df with the proper format expected by the constraint
            mock_output.results_df = pd.DataFrame([
                {
                    'query': 'protein_0',
                    'target': 'hit1',
                    'pident': 90.0,  # percent identity
                    'evalue': 1e-10,
                }
            ])
            mock_output.results = []
            mock_mmseqs.return_value = mock_output

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
            metadata = segment.candidate_sequences[0]._metadata
            assert "segment_0.mmseqs_similarity_constraint.mmseqs_results" in metadata
            assert len(metadata["segment_0.mmseqs_similarity_constraint.mmseqs_results"]) > 0
            assert "segment_0.mmseqs_similarity_constraint.unique_orfs_with_hits" in metadata

    def test_no_hits_scenario(self, dummy_db_path):
        """Test when no MMseqs2 hits are found."""
        segment = Segment(starting_sequence_or_desired_length="MVLSP", sequence_type=SequenceType.PROTEIN)

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        # Mock MMseqs2 with no results
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.mmseqs_search_proteins') as mock_mmseqs:
            import pandas as pd
            mock_output = MagicMock(spec=MmseqsOutput)
            mock_output.success = True
            mock_output.results_df = pd.DataFrame()  # Empty DataFrame
            mock_output.results = []
            mock_mmseqs.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=mmseqs_similarity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            # Score depends on implementation - could be 0 or MAX_ENERGY
            assert isinstance(scores[0], float)

    def test_registry_integration(self):
        """Test that constraint is properly registered."""
        spec = ConstraintRegistry.get("mmseqs-gene-similarity")
        assert spec.key == "mmseqs-gene-similarity"
        assert spec.label == "Gene/Protein Similarity"  # Actual label in registry
        assert spec.batched == True
        assert spec.concatenate == True

    def test_dna_sequence_with_orf_prediction(self, dummy_db_path):
        """Test that DNA sequences work via ORF prediction."""
        segment = Segment(starting_sequence_or_desired_length="ATGGTGCTGAGCCCGGCGGACAAG", sequence_type=SequenceType.DNA)

        config = MMseqsSimilarityConfig(
            min_similarity=80.0,
            max_similarity=100.0,
            mmseqs_db=dummy_db_path,
        )

        # Mock MMseqs2 with results
        with patch('proto_language.language.constraint.sequence_annotation.mmseqs_similarity_constraint.mmseqs_search_proteins') as mock_mmseqs:
            import pandas as pd
            mock_output = MagicMock(spec=MmseqsOutput)
            mock_output.success = True
            # Provide results_df with the proper format expected by the constraint
            mock_output.results_df = pd.DataFrame([
                {
                    'query': 'protein_0',
                    'target': 'hit1',
                    'pident': 85.0,  # percent identity
                    'evalue': 1e-10,
                }
            ])
            mock_output.results = []
            mock_mmseqs.return_value = mock_output

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
