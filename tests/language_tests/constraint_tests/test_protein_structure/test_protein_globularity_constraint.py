"""
tests/language_tests/constraint_tests/test_protein_structure/test_protein_globularity_constraint.py

Tests for Protein Globularity constraint.
"""

from unittest.mock import Mock, patch

from proto_tools import BFactorType, ProdigalOutput, StructurePredictionOutput

from proto_language.language.constraint import protein_globularity_constraint
from proto_language.language.constraint.protein_structure.protein_globularity_constraint import (
    ProteinGlobularityConfig,
)
from proto_language.language.core import Constraint, Segment
from tests.helpers.mock_structure import MockStructure

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

class TestProteinGlobularityConstraint:
    """tests/language_tests/constraint_tests/test_protein_structure/test_protein_globularity_constraint.py

Tests for Protein Globularity constraint."""

    def test_scoring_algorithm(self):
        """Test basic constraint evaluation with mocked structure."""
        segment = Segment(sequence="MKR", sequence_type="protein")
        config = ProteinGlobularityConfig()

        # Mock a compact globular structure (low std of distances)
        with patch('proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold') as mock_run:
            # Create mock structure with PDB output
            mock_structure = MockStructure(
                structure_content=mock_pdb,
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
                metrics={
                    "avg_plddt": 0.9,
                    "ptm": 0.9,
                },
            )

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold-prediction",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_run.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_globularity_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0  # Score should be non-negative

    def test_dna_sequence_input(self):
        """Test that DNA/RNA sequences work via translation to protein."""
        segment = Segment(sequence="ATGAAAAAACGT", sequence_type="dna")  # Codes for MKR
        config = ProteinGlobularityConfig()

        # Mock the Prodigal output with ORF objects
        from proto_tools import ORF

        mock_orf = ORF(
            parent_id="seq_0",
            orf_id="gene_1",
            strand="+",
            frame=1,
            amino_acid_sequence="MKR",
            nucleotide_sequence="ATGAAAAAACGT",
            amino_acid_length=3,
            nucleotide_length=12,
            nucleotide_start=1,
            nucleotide_end=12,
        )

        mock_prodigal_output = Mock(spec=ProdigalOutput)
        mock_prodigal_output.predicted_orfs = [[mock_orf]]
        mock_prodigal_output.num_orfs_per_sequence = [1]

        # Mock the ESMFold output
        mock_structure = MockStructure(
            structure_content=mock_pdb,
            structure_format="pdb",
            b_factor_type=BFactorType.NORMALIZED_PLDDT,
            source="esmfold-prediction",
            metrics={"avg_plddt": 0.9, "ptm": 0.9},
        )
        mock_structure_prediction_output = StructurePredictionOutput(
            tool_id="esmfold-prediction",
            execution_time=0.0,
            success=True,
            structures=[mock_structure],
            warnings=[],
            metadata={},
        )
        with (
            patch(
                "proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_prodigal_prediction"
            ) as mock_prodigal,
            patch(
                "proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold"
            ) as mock_esmfold,
        ):
            # Setup mock return values
            mock_prodigal.return_value = mock_prodigal_output
            mock_esmfold.return_value = mock_structure_prediction_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_globularity_constraint,
                function_config=config,
            )

            # DNA sequences are translated to protein before evaluation
            scores = constraint.evaluate()
            assert len(scores) == 1
            assert scores[0] >= 0.0

    def test_n_replications_parameter(self):
        """Test that n_replications correctly replicates the sequence."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")
        config = ProteinGlobularityConfig(n_replications=3)
        with patch('proto_language.language.constraint.protein_structure.protein_globularity_constraint.run_esmfold') as mock_run:
            # Create mock structure
            mock_structure = MockStructure(
                structure_content=mock_pdb,
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
                metrics={"avg_plddt": 0.9, "ptm": 0.9},
            )
            mock_structure_prediction_output = StructurePredictionOutput(
                tool_id="esmfold-prediction",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_run.return_value = mock_structure_prediction_output

            constraint = Constraint(
                inputs=[segment],
                function=protein_globularity_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify sequence was replicated 3 times
            mock_run.assert_called_once()
            passed_input = mock_run.call_args.kwargs['inputs']  # Function called with keyword args
            assert [chain.sequence for chain in passed_input.complexes[0].chains] == [
                "MKTAYIAK",
                "MKTAYIAK",
                "MKTAYIAK",
            ]
