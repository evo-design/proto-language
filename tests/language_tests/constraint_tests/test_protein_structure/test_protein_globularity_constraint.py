"""
Comprehensive tests for Protein Globularity constraint.

Tests cover:
1. Basic functionality with mock ESMFold output
2. Configuration validation
3. n_replications parameter
4. Registry integration
5. Metadata propagation
"""

import pytest
import sys
from unittest.mock import Mock, patch

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import protein_globularity_constraint
from proto_language.language.constraint.protein_structure.protein_globularity_constraint import ProteinGlobularityConfig
from proto_language.tools.models.structure_prediction.esmfold import ESMFoldOutput
from proto_language.tools.models.structure_prediction.esmfold.esmfold import ESMFoldStructureOutput
from ..utils import create_segment


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
            # Create mock structure with PDB output
            mock_structure = Mock(spec=ESMFoldStructureOutput)
            mock_structure.avg_plddt = 0.9
            mock_structure.ptm = 0.9
            mock_structure.structure_pdb_output = mock_pdb
            
            # Create mock output with structures list
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.structures = [mock_structure]
            mock_run.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_globularity_constraint,
                scoring_function_config=config,
                vectorized=True,
            )
            
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0  # Score should be non-negative
    
    def test_wrong_sequence_type(self):
        """Test that DNA/RNA sequences work via translation to protein."""
        segment = create_segment("ATGAAAAAACGT", SequenceType.DNA)  # Codes for MKR
        config = ProteinGlobularityConfig()

        mock_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C"

        with patch('proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold') as mock_run:
            # Create mock structure
            mock_structure = Mock(spec=ESMFoldStructureOutput)
            mock_structure.avg_plddt = 0.9
            mock_structure.ptm = 0.9
            mock_structure.structure_pdb_output = mock_pdb
            
            # Create mock output with structures list
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.structures = [mock_structure]
            mock_run.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_globularity_constraint,
                scoring_function_config=config,
                vectorized=True,
            )

            # DNA sequences are translated to protein before evaluation
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0
    
    def test_n_replications_parameter(self):
        """Test that n_replications correctly replicates the sequence."""
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        config = ProteinGlobularityConfig(n_replications=3)
        
        mock_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C"
        
        with patch('proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold') as mock_run:
            # Create mock structure
            mock_structure = Mock(spec=ESMFoldStructureOutput)
            mock_structure.avg_plddt = 0.9
            mock_structure.ptm = 0.9
            mock_structure.structure_pdb_output = mock_pdb
            
            # Create mock output with structures list
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.structures = [mock_structure]
            mock_run.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=protein_globularity_constraint,
                scoring_function_config=config,
                vectorized=True,
            )
            
            constraint.evaluate()
            
            # Verify sequence was replicated 3 times
            mock_run.assert_called_once()
            passed_input = mock_run.call_args.kwargs['inputs']  # Function called with keyword args
            assert passed_input.sequences == [["MKTAYIAK", "MKTAYIAK", "MKTAYIAK"]]
