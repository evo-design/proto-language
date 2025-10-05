"""
Comprehensive tests for Protein Symmetry Ring constraint.

Tests cover:
1. Basic functionality with mock ESMFold output
2. Configuration validation
3. Different n_replications values
4. all_to_all_protomer_symmetry parameter
5. Registry integration
6. Metadata propagation
"""

import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path
from unittest.mock import patch
from io import StringIO

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import ConstraintRegistry, protein_symmetry_ring_constraint
from proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint import ProteinSymmetryRingConfig
from proto_language.schemas import ESMFoldKwargs
from ..test_utils import create_segment


class TestProteinSymmetryRingConstraint:
    """Tests for Protein Symmetry Ring constraint."""
    
    def test_scoring_algorithm(self):
        """Test basic constraint evaluation with mocked structure."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinSymmetryRingConfig(n_replications=3)
        
        # Mock PDB output for a symmetric ring
        mock_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00 90.00           C
ATOM      3  CA  ALA C   1       2.500   4.330   0.000  1.00 90.00           C"""
        
        with patch('proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold') as mock_run:
            def mock_esmfold(seq, n_rep, kwargs):
                seq._metadata["pdb_output"] = mock_pdb
            mock_run.side_effect = mock_esmfold
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_symmetry_ring_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0  # Score should be non-negative
    
    def test_all_to_all_protomer_symmetry_parameter(self):
        """Test all_to_all_protomer_symmetry parameter (constraint-specific config)."""
        segment = create_segment("MVLSPADK", SequenceType.PROTEIN)
        
        constraint = ConstraintRegistry.create(
            key="protein-symmetry-ring",
            segments=[segment],
            config_dict={
                "n_replications": 6,
                "all_to_all_protomer_symmetry": True
            }
        )
        
        assert constraint.scoring_function_config.n_replications == 6
        assert constraint.scoring_function_config.all_to_all_protomer_symmetry == True
    
    def test_wrong_sequence_type(self):
        """Test that DNA/RNA sequences raise errors (constraint calls ESMFold which validates)."""
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        config = ProteinSymmetryRingConfig(n_replications=3)
        
        with patch('proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold') as mock_run:
            mock_run.side_effect = ValueError("Can only run ESMFold on a protein sequence.")
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_symmetry_ring_constraint,
                scoring_function_config=config,
            )
            
            with pytest.raises(ValueError, match="Can only run ESMFold on a protein sequence"):
                constraint.evaluate()
    
    def test_n_replications_parameter(self):
        """Test that n_replications is passed correctly."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinSymmetryRingConfig(n_replications=5)
        
        # Create a mock PDB with 5 chains (A, B, C, D, E)
        mock_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00 90.00           C
ATOM      3  CA  ALA C   1       2.500   4.330   0.000  1.00 90.00           C
ATOM      4  CA  ALA D   1      -2.500   4.330   0.000  1.00 90.00           C
ATOM      5  CA  ALA E   1      -5.000   0.000   0.000  1.00 90.00           C"""
        
        with patch('proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold') as mock_run:
            def mock_esmfold(seq, n_rep, kwargs):
                seq._metadata["pdb_output"] = mock_pdb
            mock_run.side_effect = mock_esmfold
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_symmetry_ring_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify n_replications was passed
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][1] == 5
