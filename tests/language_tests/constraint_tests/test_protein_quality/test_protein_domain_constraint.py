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

Note: Actual HMMER/Prodigal execution is mocked to avoid dependencies.
"""

import pytest
import sys
from unittest.mock import patch, Mock
import pandas as pd

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType, Sequence
from proto_language.language.constraint import ConstraintRegistry, protein_domain_constraint
from proto_language.language.constraint.protein_quality.protein_domain_constraint import ProteinDomainConfig
from ..test_utils import create_segment


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
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinDomainConfig(
            hmm_db="/tmp/test.hmm",
            keywords=["kinase"]
        )
        
        # Mock the necessary functions
        mock_hits_df = pd.DataFrame({
            "description": ["Protein kinase domain"],
            "evalue": [1e-10],
            "ali_from": [1],
            "ali_to": [100]
        })
        
        with patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.Path') as mock_path, \
             patch('proto_language.language.constraint.protein_quality.protein_domain_constraint._run_hmmer') as mock_hmmscan:
            
            mock_path_inst = Mock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value = mock_path_inst
            
            mock_hmmscan.return_value = mock_hits_df
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_domain_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            assert scores[0] == 0.0  # Keyword found, score = 0.0
            
            # Check constraint-specific metadata
            metadata = segment[0]._metadata
            assert "segment_0.protein_domain_constraint.domain_keywords_found" in metadata
            assert "kinase" in metadata["segment_0.protein_domain_constraint.domain_keywords_found"]
    
    def test_protein_sequence_without_matching_domain(self):
        """Test protein sequence without matching domain."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinDomainConfig(
            hmm_db="/tmp/test.hmm",
            keywords=["helicase"]
        )
        
        # Mock hits that don't contain the keyword
        mock_hits_df = pd.DataFrame({
            "description": ["Protein kinase domain"],
            "evalue": [1e-10],
            "ali_from": [1],
            "ali_to": [100]
        })
        
        with patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.Path') as mock_path, \
             patch('proto_language.language.constraint.protein_quality.protein_domain_constraint._run_hmmer') as mock_hmmscan:
            
            mock_path_inst = Mock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value = mock_path_inst
            
            mock_hmmscan.return_value = mock_hits_df
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_domain_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            assert scores[0] == 1.0  # Keyword not found, score = 1.0
    
    def test_match_all_keywords(self):
        """Test match_all_keywords parameter (constraint-specific config behavior)."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinDomainConfig(
            hmm_db="/tmp/test.hmm",
            keywords=["kinase", "ATP-binding"],
            match_all_keywords=True
        )
        
        # Mock hits with only one keyword
        mock_hits_df = pd.DataFrame({
            "description": ["Protein kinase domain"],
            "evalue": [1e-10],
            "ali_from": [1],
            "ali_to": [100]
        })
        
        with patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.Path') as mock_path, \
             patch('proto_language.language.constraint.protein_quality.protein_domain_constraint._run_hmmer') as mock_hmmscan:
            
            mock_path_inst = Mock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value = mock_path_inst
            
            mock_hmmscan.return_value = mock_hits_df
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_domain_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            # Only one keyword found, but need all -> score = 1.0
            assert scores[0] == 1.0
    
    def test_dna_sequence_no_proteins(self):
        """Test DNA sequence with no predicted proteins (constraint-specific edge case)."""
        segment = create_segment("ATCGATCGATCG", SequenceType.DNA)
        config = ProteinDomainConfig(
            hmm_db="/tmp/test.hmm",
            keywords=["kinase"]
        )
        
        # Mock Prodigal returning no proteins
        empty_df = pd.DataFrame(columns=["id", "description", "sequence"])
        mock_prodigal_output = Mock()
        mock_prodigal_output.results_df = empty_df
        mock_prodigal_output.num_genes = 0
        
        with patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.Path') as mock_path, \
             patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.run_prodigal_prediction') as mock_prodigal:
            
            mock_path_inst = Mock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value = mock_path_inst
            
            mock_prodigal.return_value = mock_prodigal_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_domain_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            # No proteins predicted -> score = 1.0
            assert scores[0] == 1.0
    
    def test_hmm_db_not_found(self):
        """Test error when HMM database doesn't exist (constraint-specific error handling)."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinDomainConfig(
            hmm_db="/nonexistent/path.hmm",
            keywords=["kinase"]
        )
        
        with patch('proto_language.language.constraint.protein_quality.protein_domain_constraint.Path') as mock_path:
            mock_path_inst = Mock()
            mock_path_inst.exists.return_value = False
            mock_path.return_value = mock_path_inst
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_domain_constraint,
                scoring_function_config=config,
            )
            
            with pytest.raises(ValueError, match="HMM database not found"):
                constraint.evaluate()
