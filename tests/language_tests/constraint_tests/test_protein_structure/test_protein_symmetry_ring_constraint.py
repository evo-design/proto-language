"""
Tests for Protein Symmetry Ring constraint.
"""

from unittest.mock import patch
from tests.helpers.mock_structure import MockProteinStructure
import pandas as pd
from proto_language.language.core import Constraint, SequenceType, Segment
from proto_language.language.constraint import protein_symmetry_ring_constraint
from proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint import (
    ProteinSymmetryRingConfig,
)
from proto_language.tools.structure_prediction.schemas import (
    StructurePredictionOutput,
)
from proto_language.tools.orf_prediction.prodigal import (
    ProdigalOutput,
)
from unittest.mock import Mock

mock_pdb = """ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   MET A   1       2.009   1.421   0.000  1.00 90.00           C
ATOM      4  N   LYS A   2       1.458   2.421   0.000  1.00 90.00           N
ATOM      5  CA  LYS A   2       2.009   3.771   0.000  1.00 90.00           C
ATOM      6  C   LYS A   2       1.458   4.771   0.000  1.00 90.00           C
ATOM      7  N   ARG A   3       2.009   5.771   0.000  1.00 90.00           N
ATOM      8  CA  ARG A   3       1.458   7.121   0.000  1.00 90.00           C
ATOM      9  C   ARG A   3       2.009   8.121   0.000  1.00 90.00           C
ATOM     10  N   MET B   1       5.000   0.000   0.000  1.00 90.00           N
ATOM     11  CA  MET B   1       6.458   0.000   0.000  1.00 90.00           C
ATOM     12  C   MET B   1       7.009   1.421   0.000  1.00 90.00           C
ATOM     13  N   LYS B   2       6.458   2.421   0.000  1.00 90.00           N
ATOM     14  CA  LYS B   2       7.009   3.771   0.000  1.00 90.00           C
ATOM     15  C   LYS B   2       6.458   4.771   0.000  1.00 90.00           C
ATOM     16  N   ARG B   3       7.009   5.771   0.000  1.00 90.00           N
ATOM     17  CA  ARG B   3       6.458   7.121   0.000  1.00 90.00           C
ATOM     18  C   ARG B   3       7.009   8.121   0.000  1.00 90.00           C
ATOM     19  N   MET C   1       2.500   4.330   0.000  1.00 90.00           N
ATOM     20  CA  MET C   1       3.958   4.330   0.000  1.00 90.00           C
ATOM     21  C   MET C   1       4.509   5.751   0.000  1.00 90.00           C
ATOM     22  N   LYS C   2       3.958   6.751   0.000  1.00 90.00           N
ATOM     23  CA  LYS C   2       4.509   8.101   0.000  1.00 90.00           C
ATOM     24  C   LYS C   2       3.958   9.101   0.000  1.00 90.00           C
ATOM     25  N   ARG C   3       4.509  10.101   0.000  1.00 90.00           N
ATOM     26  CA  ARG C   3       3.958  11.451   0.000  1.00 90.00           C
ATOM     27  C   ARG C   3       4.509  12.451   0.000  1.00 90.00           C"""


class TestProteinSymmetryRingConstraint:
    """Tests for Protein Symmetry Ring constraint."""

    def test_scoring_algorithm(self):
        """Test basic constraint evaluation with mocked structure."""
        segment = Segment(sequence="MKR", sequence_type="protein")
        config = ProteinSymmetryRingConfig(n_replications=3)

        with patch(
            "proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold"
        ) as mock_run:
            # Create mock structure
            mock_structure = MockProteinStructure(structure_content=mock_pdb)
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", 0.9)

            # Create mock output with structures list
            mock_output = Mock(spec=StructurePredictionOutput)
            mock_output.structures = [mock_structure]
            mock_run.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_symmetry_ring_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0  # Score should be non-negative

    def test_dna_input(self):
        """Test that DNA/RNA sequences work via translation to protein."""
        segment = Segment(sequence="ATGAAAAAACGT", sequence_type="dna")  # Codes for MKR
        config = ProteinSymmetryRingConfig(n_replications=3)

        # Mock the Prodigal output
        mock_prodigal_result = pd.DataFrame(
            {
                "protein_sequence": ["MKR"],
                "start": [1],
                "end": [9],
                "strand": [1],
                "partial": ["00"],
            }
        )
        mock_prodigal_output = Mock(spec=ProdigalOutput)
        mock_prodigal_output.results_per_sequence = [mock_prodigal_result]
        mock_prodigal_output.total_num_genes_per_sequence = [1]

        # Mock the ESMFold output
        mock_structure = MockProteinStructure(structure_content=mock_pdb)
        mock_structure.add_metric("avg_plddt", 0.9)
        mock_structure.add_metric("ptm", 0.9)

        # Create mock output with structures list
        mock_structure_prediction_output = Mock(spec=StructurePredictionOutput)
        mock_structure_prediction_output.structures = [mock_structure]

        with (
            patch(
                "proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_prodigal_prediction"
            ) as mock_prodigal,
            patch(
                "proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold"
            ) as mock_esmfold,
        ):
            # Setup mock return values
            mock_prodigal.return_value = mock_prodigal_output
            mock_esmfold.return_value = mock_structure_prediction_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_symmetry_ring_constraint,
                function_config=config,
            )

            # DNA sequences are translated to protein before evaluation
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0

    def test_n_replications_parameter(self):
        """Test that n_replications correctly replicates the sequence."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")
        segment2 = Segment(sequence="MARG", sequence_type="protein")
        segment3 = Segment(sequence="MLYS", sequence_type="protein")
        config = ProteinSymmetryRingConfig(n_replications=5)

        # Create a mock PDB with 5 chains (A, B, C, D, E)
        mock_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00 90.00           C
ATOM      3  CA  ALA C   1       2.500   4.330   0.000  1.00 90.00           C
ATOM      4  CA  ALA D   1      -2.500   4.330   0.000  1.00 90.00           C
ATOM      5  CA  ALA E   1      -5.000   0.000   0.000  1.00 90.00           C"""

        with patch(
            "proto_language.language.constraint.protein_structure.protein_symmetry_ring_constraint.run_esmfold"
        ) as mock_run:
            # Create mock structure
            mock_structure = MockProteinStructure(structure_content=mock_pdb)
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", 0.9)

            # Create mock output with structures list
            mock_output = Mock(spec=StructurePredictionOutput)
            mock_output.structures = [mock_structure]
            mock_run.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_symmetry_ring_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify sequence was replicated 5 times for each input complex
            mock_run.assert_called_once()
            passed_input = mock_run.call_args.kwargs["inputs"]
            for comp in passed_input.complexes:
                assert len(comp.chains) == 5
