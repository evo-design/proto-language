"""
Comprehensive tests for ESMFold constraints (pLDDT and pTM).

Tests cover:
1. Basic scoring functionality
2. Sequence replication for multimers
3. ESMFold configuration parameters
4. Caching behavior
5. Metadata storage
"""

import pytest
from unittest.mock import Mock, patch

from proto_language.language.core import Constraint, SequenceType, Segment
from proto_language.language.constraint import (
    esmfold_plddt_constraint,
    esmfold_ptm_constraint,
)
from proto_language.language.constraint.protein_structure.esmfold_confidence_constraints import (
    ESMFoldConfidenceConfig,
)
from proto_language.tools.structure_prediction import (
    ESMFoldConfig,
    StructurePredictionOutput,
)
from proto_language.tools.structures import ProteinStructure, BFactorType
from tests.helpers.mock_structure import MockProteinStructure, MOCK_PDB

class TestESMFoldPLDDTConstraint:
    """Tests for ESMFold pLDDT constraint."""

    @pytest.mark.parametrize(
        "avg_plddt, expected_score",
        [
            (1.0, 0.0),  # Perfect confidence
            (0.9, 0.1),
            (0.5, 0.5),
            (0.0, 1.0),  # No confidence
        ],
    )
    def test_scoring_calculation(self, avg_plddt, expected_score):
        """Test that constraint score = 1.0 - avg_plddt."""
        segment = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        config = ESMFoldConfidenceConfig()

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", avg_plddt)
            mock_structure.add_metric("ptm", 0.9)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_plddt_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert abs(scores[0] - expected_score) < 1e-9

    def test_sequence_replication(self):
        """Test that sequences are replicated correctly for multimers."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")
        config = ESMFoldConfidenceConfig(n_replications=3)

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", 0.9)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_plddt_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify the sequence was replicated 3 times
            # After Pydantic validation, sequences becomes a list of lists
            mock_esmfold.assert_called_once()
            passed_input = mock_esmfold.call_args.kwargs[
                "inputs"
            ]  # Function called with keyword args
            assert passed_input.complexes[0].chains == [
                "MKTAYIAK",
                "MKTAYIAK",
                "MKTAYIAK",
            ]

            assert passed_input.complexes[0].entity_types == [
                "protein",
                "protein",
                "protein",
            ]

    def test_esmfold_config_passthrough(self):
        """Test that custom ESMFold config parameters are passed through."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")

        esmfold_cfg = ESMFoldConfig(
            verbose=True, residue_idx_offset=256, chain_linker="GGGGG"
        )
        config = ESMFoldConfidenceConfig(esmfold_config=esmfold_cfg)

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", 0.9)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_plddt_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify config parameters were passed through
            passed_input = mock_esmfold.call_args.kwargs[
                "inputs"
            ]  # Function called with keyword args
            passed_config = mock_esmfold.call_args.kwargs["config"]
            assert passed_config.verbose == True
            assert passed_config.residue_idx_offset == 256
            assert passed_config.chain_linker == "GGGGG"

    def test_caching(self):
        """Test that multiple evaluations produce consistent results."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")
        config = ESMFoldConfidenceConfig()

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", 0.85)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_plddt_constraint,
                function_config=config,
            )

            # First evaluation
            scores = constraint.evaluate()

            # Second evaluation should produce the same result
            scores2 = constraint.evaluate()

            # Note: When mocking run_esmfold directly, we bypass the tool_cache decorator
            # so the mock will be called twice. The actual implementation with tool_cache
            # would only call run_esmfold once due to caching.
            # This test now just verifies consistent scoring behavior.

            # Score should be the same
            assert scores2[0] == scores[0]

    def test_metadata_storage(self):
        """Test that results are stored in sequence metadata."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")
        config = ESMFoldConfidenceConfig()

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.92)
            mock_structure.add_metric("ptm", 0.88)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_plddt_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify metadata was stored (with constraint-specific prefix)
            metadata = segment.candidate_sequences[0]._metadata
            prefix = "segment_0.esmfold_plddt_constraint."
            assert f"{prefix}avg_plddt" in metadata
            assert metadata[f"{prefix}avg_plddt"] == 0.92
            assert metadata[f"{prefix}ptm"] == 0.88
            assert metadata[f"{prefix}pdb_output"] == MOCK_PDB
            assert metadata[f"{prefix}esmfolded_sequence"] == "MKTAYIAK"


class TestESMFoldPTMConstraint:
    """Tests for ESMFold pTM constraint."""

    @pytest.mark.parametrize(
        "ptm, expected_score",
        [
            (1.0, 0.0),  # Perfect quality
            (0.9, 0.1),
            (0.5, 0.5),
            (0.0, 1.0),  # Poor quality
        ],
    )
    def test_scoring_calculation(self, ptm, expected_score):
        """Test that constraint score = 1.0 - ptm."""
        segment = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        config = ESMFoldConfidenceConfig()

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", ptm)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_ptm_constraint,
                function_config=config,
            )

            scores = constraint.evaluate()
            assert abs(scores[0] - expected_score) < 1e-9

    def test_sequence_replication(self):
        """Test that sequences are replicated correctly for multimers."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")
        config = ESMFoldConfidenceConfig(n_replications=2)

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", 0.85)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_ptm_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify the sequence was replicated 2 times
            # After Pydantic validation, sequences becomes a list of lists
            mock_esmfold.assert_called_once()
            passed_input = mock_esmfold.call_args.kwargs[
                "inputs"
            ]  # Function called with keyword args
            assert passed_input.complexes[0].chains == ["MKTAYIAK", "MKTAYIAK"]

    def test_esmfold_config_passthrough(self):
        """Test that custom ESMFold config parameters are passed through."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")

        esmfold_cfg = ESMFoldConfig(
            verbose=False, residue_idx_offset=1024, chain_linker="AAAAA"
        )
        config = ESMFoldConfidenceConfig(esmfold_config=esmfold_cfg)

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.9)
            mock_structure.add_metric("ptm", 0.85)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_ptm_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify config parameters were passed through
            passed_input = mock_esmfold.call_args.kwargs[
                "inputs"
            ]  # Function called with keyword args
            passed_config = mock_esmfold.call_args.kwargs["config"]
            assert passed_config.verbose == False
            assert passed_config.residue_idx_offset == 1024
            assert passed_config.chain_linker == "AAAAA"

    def test_metadata_storage(self):
        """Test that results are stored in sequence metadata."""
        segment = Segment(sequence="MKTAYIAK", sequence_type="protein")
        config = ESMFoldConfidenceConfig()

        with patch(
            "proto_language.language.constraint.protein_structure.esmfold_confidence_constraints.run_esmfold"
        ) as mock_esmfold:
            # Create mock structure with avg_plddt and ptm
            mock_structure = MockProteinStructure(
                structure_format="pdb",
                b_factor_type=BFactorType.NORMALIZED_PLDDT,
                source="esmfold-prediction",
            )
            mock_structure.add_metric("avg_plddt", 0.92)
            mock_structure.add_metric("ptm", 0.88)

            # Create mock output with structures list
            mock_output = StructurePredictionOutput(
                tool_id="esmfold",
                execution_time=0.0,
                success=True,
                structures=[mock_structure],
                warnings=[],
                metadata={},
            )
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                function=esmfold_ptm_constraint,
                function_config=config,
            )

            constraint.evaluate()

            # Verify metadata was stored (with constraint-specific prefix)
            metadata = segment.candidate_sequences[0]._metadata
            prefix = "segment_0.esmfold_ptm_constraint."
            assert f"{prefix}avg_plddt" in metadata
            assert metadata[f"{prefix}avg_plddt"] == 0.92
            assert metadata[f"{prefix}ptm"] == 0.88
            assert metadata[f"{prefix}pdb_output"] == MOCK_PDB
            assert metadata[f"{prefix}esmfolded_sequence"] == "MKTAYIAK"
