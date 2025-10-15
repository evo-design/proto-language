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
import sys
from unittest.mock import Mock, patch

sys.path.append(".")

from proto_language.language.core import Constraint, Sequence, SequenceType
from proto_language.language.constraint import esmfold_plddt_constraint, esmfold_ptm_constraint
from proto_language.language.constraint.protein_structure.esmfold_plddt_constraint import ESMFoldPLDDTConfig
from proto_language.language.constraint.protein_structure.esmfold_ptm_constraint import ESMFoldPTMConfig
from proto_language.tools.models.structure_prediction.esmfold import ESMFoldConfig, ESMFoldOutput
from ..test_utils import create_segment


class TestESMFoldPLDDTConstraint:
    """Tests for ESMFold pLDDT constraint."""
    
    @pytest.mark.parametrize("avg_plddt, expected_score", [
        (1.0, 0.0),  # Perfect confidence
        (0.9, 0.1),
        (0.5, 0.5),
        (0.0, 1.0),  # No confidence
    ])
    def test_scoring_calculation(self, avg_plddt, expected_score):
        """Test that constraint score = 1.0 - avg_plddt."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ESMFoldPLDDTConfig()

        with patch("proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = avg_plddt
            mock_output.ptm = 0.9
            mock_output.structure_pdb_output = "MOCK PDB"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            assert abs(scores[0] - expected_score) < 1e-9
    
    def test_sequence_replication(self):
        """Test that sequences are replicated correctly for multimers."""
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        config = ESMFoldPLDDTConfig(n_replications=3)

        with patch("proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.9
            mock_output.ptm = 0.9
            mock_output.structure_pdb_output = "MOCK PDB"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify the sequence was replicated 3 times
            # After Pydantic validation, sequences becomes a list
            mock_esmfold.assert_called_once()
            passed_config = mock_esmfold.call_args[0][0]
            assert passed_config.sequences == ["MKTAYIAK", "MKTAYIAK", "MKTAYIAK"]
    
    def test_esmfold_config_passthrough(self):
        """Test that custom ESMFold config parameters are passed through."""
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        
        esmfold_cfg = ESMFoldConfig(
            verbose=True,
            residue_idx_offset=256,
            chain_linker="GGGGG"
        )
        config = ESMFoldPLDDTConfig(esmfold_config=esmfold_cfg)

        with patch("proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.9
            mock_output.ptm = 0.9
            mock_output.structure_pdb_output = "MOCK PDB"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify config parameters were passed through
            passed_config = mock_esmfold.call_args[0][0]
            assert passed_config.verbose == True
            assert passed_config.residue_idx_offset == 256
            assert passed_config.chain_linker == "GGGGG"
    
    def test_caching(self):
        """Test that multiple evaluations produce consistent results."""
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        config = ESMFoldPLDDTConfig()

        with patch("proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.9
            mock_output.ptm = 0.85
            mock_output.structure_pdb_output = "MOCK PDB"
            mock_esmfold.return_value = mock_output

            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
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
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        config = ESMFoldPLDDTConfig()

        with patch("proto_language.language.constraint.protein_structure.esmfold_plddt_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.92
            mock_output.ptm = 0.88
            mock_output.structure_pdb_output = "MOCK PDB OUTPUT"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_plddt_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify metadata was stored (with constraint-specific prefix)
            metadata = segment[0]._metadata
            prefix = "segment_0.esmfold_plddt_constraint."
            assert f"{prefix}avg_plddt" in metadata
            assert metadata[f"{prefix}avg_plddt"] == 0.92
            assert metadata[f"{prefix}ptm"] == 0.88
            assert metadata[f"{prefix}pdb_output"] == "MOCK PDB OUTPUT"
            assert metadata[f"{prefix}esmfolded_sequence"] == "MKTAYIAK"


class TestESMFoldPTMConstraint:
    """Tests for ESMFold pTM constraint."""
    
    @pytest.mark.parametrize("ptm, expected_score", [
        (1.0, 0.0),  # Perfect quality
        (0.9, 0.1),
        (0.5, 0.5),
        (0.0, 1.0),  # Poor quality
    ])
    def test_scoring_calculation(self, ptm, expected_score):
        """Test that constraint score = 1.0 - ptm."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)
        config = ESMFoldPTMConfig()

        with patch("proto_language.language.constraint.protein_structure.esmfold_ptm_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.9
            mock_output.ptm = ptm
            mock_output.structure_pdb_output = "MOCK PDB"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_ptm_constraint,
                scoring_function_config=config,
            )
            
            scores = constraint.evaluate()
            assert abs(scores[0] - expected_score) < 1e-9
    
    def test_sequence_replication(self):
        """Test that sequences are replicated correctly for multimers."""
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        config = ESMFoldPTMConfig(n_replications=2)

        with patch("proto_language.language.constraint.protein_structure.esmfold_ptm_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.9
            mock_output.ptm = 0.85
            mock_output.structure_pdb_output = "MOCK PDB"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_ptm_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify the sequence was replicated 2 times
            # After Pydantic validation, sequences becomes a list
            mock_esmfold.assert_called_once()
            passed_config = mock_esmfold.call_args[0][0]
            assert passed_config.sequences == ["MKTAYIAK", "MKTAYIAK"]
    
    def test_esmfold_config_passthrough(self):
        """Test that custom ESMFold config parameters are passed through."""
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        
        esmfold_cfg = ESMFoldConfig(
            verbose=False,
            residue_idx_offset=1024,
            chain_linker="AAAAA"
        )
        config = ESMFoldPTMConfig(esmfold_config=esmfold_cfg)

        with patch("proto_language.language.constraint.protein_structure.esmfold_ptm_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.9
            mock_output.ptm = 0.85
            mock_output.structure_pdb_output = "MOCK PDB"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_ptm_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify config parameters were passed through
            passed_config = mock_esmfold.call_args[0][0]
            assert passed_config.verbose == False
            assert passed_config.residue_idx_offset == 1024
            assert passed_config.chain_linker == "AAAAA"
    
    def test_metadata_storage(self):
        """Test that results are stored in sequence metadata."""
        segment = create_segment("MKTAYIAK", SequenceType.PROTEIN)
        config = ESMFoldPTMConfig()

        with patch("proto_language.language.constraint.protein_structure.esmfold_ptm_constraint.run_esmfold") as mock_esmfold:
            mock_output = Mock(spec=ESMFoldOutput)
            mock_output.avg_plddt = 0.92
            mock_output.ptm = 0.88
            mock_output.structure_pdb_output = "MOCK PDB OUTPUT"
            mock_esmfold.return_value = mock_output
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=esmfold_ptm_constraint,
                scoring_function_config=config,
            )
            
            constraint.evaluate()
            
            # Verify metadata was stored (with constraint-specific prefix)
            metadata = segment[0]._metadata
            prefix = "segment_0.esmfold_ptm_constraint."
            assert f"{prefix}avg_plddt" in metadata
            assert metadata[f"{prefix}avg_plddt"] == 0.92
            assert metadata[f"{prefix}ptm"] == 0.88
            assert metadata[f"{prefix}pdb_output"] == "MOCK PDB OUTPUT"
            assert metadata[f"{prefix}esmfolded_sequence"] == "MKTAYIAK"
