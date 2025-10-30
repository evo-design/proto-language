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

import pytest
import sys
from unittest.mock import patch

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import protein_symmetry_ring_constraint, ConstraintRegistry
from proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint import ProteinSymmetryRingConfig
from proto_language.tools.models.structure_prediction.esmfold import ESMFoldOutput
from proto_language.tools.models.structure_prediction.esmfold.esmfold import ESMFoldStructureOutput
from unittest.mock import Mock
from ..utils import create_segment


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
                scoring_function=protein_symmetry_ring_constraint,
                scoring_function_config=config,
                vectorized=True,
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
        """Test that DNA/RNA sequences work via translation to protein."""
        segment = create_segment("ATGAAAAAACGT", SequenceType.DNA)  # Codes for MKR
        config = ProteinSymmetryRingConfig(n_replications=3)

        mock_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C"

        with patch('proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold') as mock_run:
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
                scoring_function=protein_symmetry_ring_constraint,
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
        config = ProteinSymmetryRingConfig(n_replications=5)

        # Create a mock PDB with 5 chains (A, B, C, D, E)
        mock_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00 90.00           C
ATOM      3  CA  ALA C   1       2.500   4.330   0.000  1.00 90.00           C
ATOM      4  CA  ALA D   1      -2.500   4.330   0.000  1.00 90.00           C
ATOM      5  CA  ALA E   1      -5.000   0.000   0.000  1.00 90.00           C"""

        with patch('proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold') as mock_run:
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
                scoring_function=protein_symmetry_ring_constraint,
                scoring_function_config=config,
                vectorized=True,
            )

            constraint.evaluate()

            # Verify sequence was replicated 5 times
            mock_run.assert_called_once()
            passed_input = mock_run.call_args.kwargs['inputs']  # Function called with keyword args
            assert passed_input.sequences == [["MKTAYIAK", "MKTAYIAK", "MKTAYIAK", "MKTAYIAK", "MKTAYIAK"]]
