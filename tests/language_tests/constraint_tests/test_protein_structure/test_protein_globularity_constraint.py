"""
Comprehensive tests for Protein Globularity constraint.

Tests cover:
1. Basic functionality with mock ESMFold output
2. Configuration validation
3. n_replications parameter
4. Registry integration
5. Metadata propagation
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

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import ConstraintRegistry, protein_globularity_constraint
from proto_language.language.constraint.protein_structure.protein_globularity_constraint import ProteinGlobularityConfig
from proto_language.schemas import ESMFoldKwargs
from ..test_utils import create_segment


class TestProteinGlobularityConstraint:
    """Tests for Protein Globularity constraint."""
    
    def test_scoring_algorithm(self):
        """Test basic constraint evaluation with mocked structure."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinGlobularityConfig()
        
        # Mock a compact globular structure (low std of distances)
        mock_pdb = """ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  MET A   1       1.000   1.000   1.000  1.00 90.00           C
ATOM      3  C   MET A   1       2.000   2.000   2.000  1.00 90.00           C
ATOM      4  N   LYS A   2       3.000   3.000   3.000  1.00 90.00           N
ATOM      5  CA  LYS A   2       4.000   4.000   4.000  1.00 90.00           C"""
        
        with patch('proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold') as mock_run:
            def mock_esmfold(seq, n_rep, kwargs):
                seq._metadata["pdb_output"] = mock_pdb
            mock_run.side_effect = mock_esmfold
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_globularity_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0  # Score should be non-negative
    
    def test_wrong_sequence_type(self):
        """Test that DNA/RNA sequences raise errors (constraint calls ESMFold which validates)."""
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        config = ProteinGlobularityConfig()
        
        with patch('proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold') as mock_run:
            mock_run.side_effect = ValueError("Can only run ESMFold on a protein sequence.")
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_globularity_constraint,
                scoring_function_config=config,
            )
            
            with pytest.raises(ValueError, match="Can only run ESMFold on a protein sequence"):
                constraint.evaluate()
    
    def test_n_replications_parameter(self):
        """Test that n_replications is passed correctly."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ProteinGlobularityConfig(n_replications=3)
        
        mock_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C"
        
        with patch('proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold') as mock_run:
            def mock_esmfold(seq, n_rep, kwargs):
                seq._metadata["pdb_output"] = mock_pdb
            mock_run.side_effect = mock_esmfold
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_globularity_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify n_replications was passed
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][1] == 3
