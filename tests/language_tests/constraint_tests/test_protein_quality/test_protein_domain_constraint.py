"""
Comprehensive tests for Protein Domain constraint.

Tests cover:
1. Configuration validation
2. Protein sequence handling
3. DNA sequence handling (with Prodigal)
4. Keyword matching logic (any vs all)
5. Registry integration
6. Metadata propagation
7. Error handling
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from Bio import SeqIO
from proto_language.language.core import Constraint, Segment
from proto_language.language.constraint import protein_domain_constraint
from proto_language.language.constraint.protein_quality.protein_domain_constraint import ProteinDomainConfig

TEST_HMM = (
    Path(__file__).parent.parent.parent.parent / "dummy_data" / "test_multiple_hmm.hmm"
)

# Load test sequence and properly close the file handle
TEST_FASTA_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "dummy_data"
    / "test_sequences_for_pyhmmer.fasta"
)
with open(TEST_FASTA_PATH, "r") as fasta_file:
    sequence_iterator = SeqIO.parse(fasta_file, "fasta")
    SAMPLE_SEQUENCE = [str(seq.seq) for seq in sequence_iterator][0]

class TestProteinDomainConstraint:
    """Tests for Protein Domain constraint."""

    def test_config_required_fields(self):
        """Test that required config fields must be provided (constraint-specific validation)."""
        # hmm_db is required
        with pytest.raises(Exception):  # Pydantic ValidationError
            ProteinDomainConfig(keywords=["kinase"])

        # keywords is required
        with pytest.raises(Exception):  # Pydantic ValidationError
            ProteinDomainConfig(hmm_db="/path/to/db.hmm")

    def test_scoring_algorithm_matching_domain(self):
        """Test protein sequence with matching domain and metadata."""
        segment = Segment(sequence=SAMPLE_SEQUENCE, sequence_type="protein"
        )
        config = ProteinDomainConfig(hmm_db=str(TEST_HMM), keywords=["kinase"])

        constraint = Constraint(
            inputs=[segment],
            function=protein_domain_constraint,
            function_config=config,
        )
        scores = constraint.evaluate()

        assert (
            scores[0] == 0.0
        ), f"Keyword should be found, score should be 0.0, but got {scores[0]}"

        # Check constraint-specific metadata
        constraints = segment.candidate_sequences[0]._metadata["constraints"]
        assert (
            "domain_keywords_found" in constraints["protein_domain_constraint"]["data"]
        ), f"Metadata should contain domain keywords found, but got {constraints}"
        assert (
            "kinase"
            in constraints["protein_domain_constraint"]["data"]["domain_keywords_found"]
        ), f"Keyword should be found in metadata, but got {constraints['protein_domain_constraint']['data']['domain_keywords_found']}"

    def test_protein_sequence_without_matching_domain(self):
        """Test protein sequence without matching domain."""
        segment = Segment(sequence=SAMPLE_SEQUENCE, sequence_type="protein")
        config = ProteinDomainConfig(hmm_db=str(TEST_HMM), keywords=["helicase"])

        # Run the constraint
        constraint = Constraint(
            inputs=[segment],
            function=protein_domain_constraint,
            function_config=config,
        )

        # Evaluate the constraint
        scores = constraint.evaluate()
        assert (
            scores[0] == 1.0
        ), f"Keyword should NOT be found, score should be 1.0, but got {scores[0]}"

        # Check constraint-specific metadata
        constraints = segment.candidate_sequences[0]._metadata["constraints"]
        assert "domain_keywords_found" in constraints["protein_domain_constraint"]["data"]
        assert constraints["protein_domain_constraint"]["data"]["domain_keywords_found"] == []
        assert "domain_matching_hits" in constraints["protein_domain_constraint"]["data"]

    def test_match_all_keywords(self):
        """Test match_all_keywords parameter (constraint-specific config behavior)."""
        segment = Segment(sequence=SAMPLE_SEQUENCE, sequence_type="protein")
        config = ProteinDomainConfig(
            hmm_db=str(TEST_HMM),
            keywords=["kinase", "ATP-binding"],
            match_all_keywords=True,
        )

        constraint = Constraint(
            inputs=[segment],
            function=protein_domain_constraint,
            function_config=config,
        )

        scores = constraint.evaluate()
        # Only one keyword found, but need all -> score = 1.0
        assert scores[0] == 1.0

    def test_hmm_db_not_found(self):
        """Test error when HMM database doesn't exist (constraint-specific error handling)."""
        segment = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        config = ProteinDomainConfig(
            hmm_db="/nonexistent/path.hmm", keywords=["kinase"]
        )

        with patch(
            "proto_language.language.constraint.protein_quality.protein_domain_constraint.Path"
        ) as mock_path:
            mock_path_inst = Mock()
            mock_path_inst.exists.return_value = False
            mock_path.return_value = mock_path_inst

            constraint = Constraint(
                inputs=[segment],
                function=protein_domain_constraint,
                function_config=config,
            )

            with pytest.raises(ValueError, match="HMM database not found"):
                constraint.evaluate()

    def test_dna_sequence_with_proteins(self):
        """Test DNA sequence with no predicted proteins (constraint-specific edge case)."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = ProteinDomainConfig(hmm_db=str(TEST_HMM), keywords=["kinase"])

        # Mock Prodigal returning proteins with ProdigalOrf objects
        from proto_language.tools.orf_prediction.prodigal.prodigal import ProdigalOrf
        
        mock_orf = ProdigalOrf(
            parent_id="seq_0",
            orf_id="gene_1",
            strand="+",
            frame=1,
            amino_acid_sequence=SAMPLE_SEQUENCE,
            nucleotide_sequence="ATCGATCGATCG",
            amino_acid_length=len(SAMPLE_SEQUENCE),
            nucleotide_length=12,
            nucleotide_start=0,
            nucleotide_end=12,
        )
        
        mock_prodigal_output = Mock()
        mock_prodigal_output.predicted_orfs = [[mock_orf]]
        mock_prodigal_output.num_orfs_per_sequence = [1]

        with (
            patch(
                "proto_language.language.constraint.protein_quality.protein_domain_constraint.run_prodigal_prediction"
            ) as mock_prodigal,
        ):

            mock_prodigal.return_value = mock_prodigal_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_domain_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            # Proteins should be predicted, so score should be 0.0
            assert scores[0] == 0.0

    def test_dna_sequence_no_proteins(self):
        """Test DNA sequence with no predicted proteins (constraint-specific edge case)."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = ProteinDomainConfig(hmm_db=str(TEST_HMM), keywords=["kinase"])

        # Mock Prodigal returning no proteins
        mock_prodigal_output = Mock()
        mock_prodigal_output.predicted_orfs = [[]]
        mock_prodigal_output.num_orfs_per_sequence = [0]

        with patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.Path') as mock_path, \
             patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.run_prodigal_prediction') as mock_prodigal:

            mock_path_inst = Mock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value = mock_path_inst

            mock_prodigal.return_value = mock_prodigal_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_domain_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            # No proteins predicted -> score = 1.0
            assert scores[0] == 1.0
