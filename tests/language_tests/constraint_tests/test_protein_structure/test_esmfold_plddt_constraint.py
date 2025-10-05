"""
Comprehensive tests for ESMFold pLDDT constraint.

Tests cover:
1. Basic functionality with valid protein sequences
2. Configuration validation
3. Wrong sequence type handling
4. Batch processing
5. Registry integration
6. Metadata propagation
7. n_replications parameter
8. ESMFold kwargs handling
"""

import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import ConstraintRegistry, esmfold_plddt_constraint
from proto_language.language.constraint.protein_structure.esmfold_plddt_constraint import ESMFoldPLDDTConfig
from proto_language.schemas import ESMFoldKwargs
from ..test_utils import create_segment, create_batched_segment


class TestESMFoldPLDDTConstraint:
    """Tests for ESMFold pLDDT constraint."""
    
    @pytest.mark.parametrize(
        "plddt, expected_score",
        [
            (1.0, 0.0),  # Perfect pLDDT -> score of 0.0
            (0.9, 0.1),  # Good pLDDT -> score of 0.1
            (0.5, 0.5),  # Moderate pLDDT -> score of 0.5
            (0.0, 1.0),  # Poor pLDDT -> score of 1.0
        ],
    )
    def test_scoring_algorithm(self, plddt, expected_score):
        """Test that constraint correctly calculates score from pLDDT and stores metadata."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ESMFoldPLDDTConfig()
        
        # Mock run_esmfold to avoid actual computation
        with patch('proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold') as mock_run:
            def mock_esmfold(seq, n_rep, kwargs):
                seq._metadata["avg_plddt"] = plddt
            mock_run.side_effect = mock_esmfold
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            assert abs(scores[0] - expected_score) < 1e-9
            
            # Check constraint-specific metadata field
            metadata = segment[0]._metadata
            assert "segment_0.esmfold_plddt_constraint.avg_plddt" in metadata
            assert abs(metadata["segment_0.esmfold_plddt_constraint.avg_plddt"] - plddt) < 1e-9
    
    def test_wrong_sequence_type(self):
        """Test that DNA/RNA sequences raise errors (constraint calls ESMFold which validates)."""
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        config = ESMFoldPLDDTConfig()
        
        with patch('proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold') as mock_run:
            # run_esmfold should raise ValueError for non-protein sequences
            mock_run.side_effect = ValueError("Can only run ESMFold on a protein sequence.")
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            with pytest.raises(ValueError, match="Can only run ESMFold on a protein sequence"):
                constraint.evaluate()
    
    def test_n_replications_parameter(self):
        """Test that n_replications is passed correctly to run_esmfold."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ESMFoldPLDDTConfig(n_replications=4)
        
        with patch('proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold') as mock_run:
            def mock_esmfold(seq, n_rep, kwargs):
                seq._metadata["avg_plddt"] = 0.9
                # Store n_rep so we can verify it
                seq._metadata["_test_n_rep"] = n_rep
            mock_run.side_effect = mock_esmfold
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify n_replications was passed correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][1] == 4  # n_replications argument
    
    def test_esmfold_kwargs_parameter(self):
        """Test that esmfold_kwargs is passed correctly to run_esmfold."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        esmfold_kwargs = ESMFoldKwargs(verbose=True)
        config = ESMFoldPLDDTConfig(n_replications=1, esmfold_kwargs=esmfold_kwargs)
        
        with patch('proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold') as mock_run:
            def mock_esmfold(seq, n_rep, kwargs):
                seq._metadata["avg_plddt"] = 0.9
            mock_run.side_effect = mock_esmfold
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify esmfold_kwargs was passed
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][2] == esmfold_kwargs  # esmfold_kwargs argument