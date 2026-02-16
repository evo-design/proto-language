"""
Tests for structure ensemble similarity constraint.
"""

from unittest.mock import MagicMock, patch

import pytest
from proto_tools import BioEmuConfig

from proto_language.language.constraint.protein_structure.structure_ensemble_similarity_constraint import (
    StructureEnsembleSimilarityConfig,
    structure_ensemble_rmsd_constraint,
)
from proto_language.language.core import Sequence

# Short test sequence
TEST_SEQ = "MGDIQVQVNIDDNGKNFDYTYTVTTE"

# Minimal PDB content for testing
MOCK_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2       3.000   2.000   0.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       4.000   3.000   0.000  1.00  0.00           C
END
"""

MOCK_PDB_CHAIN_B = """ATOM      1  N   ALA B   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA B   1       1.458   0.000   0.000  1.00  0.00           C
END
"""

MOCK_PDB_MULTICHAIN = MOCK_PDB.replace("END\n", "") + MOCK_PDB_CHAIN_B


class TestConfig:
    """Tests for StructureEnsembleSimilarityConfig validation."""

    def test_target_structure_required(self):
        """Must provide target_structure (it's a required field)."""
        # No target - should fail (missing required field)
        with pytest.raises(ValueError):
            StructureEnsembleSimilarityConfig()

        # Valid string - should work
        config = StructureEnsembleSimilarityConfig(
            target_structure=MOCK_PDB,
        )
        assert config.target_structure == MOCK_PDB

    def test_residue_range_validation(self):
        """Residue ranges must be valid (1-indexed, start <= end)."""
        # Start < 1 should fail
        with pytest.raises(ValueError, match="must be >= 1"):
            StructureEnsembleSimilarityConfig(
                target_structure=MOCK_PDB,
                target_residue_range=(0, 10),
            )

        # End < start should fail
        with pytest.raises(ValueError, match="must be >= start"):
            StructureEnsembleSimilarityConfig(
                target_structure=MOCK_PDB,
                candidate_residue_range=(10, 5),
            )

    def test_default_values(self):
        """Check key defaults."""
        config = StructureEnsembleSimilarityConfig(target_structure=MOCK_PDB)
        assert config.rmsd_aggregation == "min"
        assert config.inflection_point_angstroms == 3.0


class TestConstraintWithMocks:
    """Tests for the constraint function with mocked BioEmu."""

    @pytest.fixture
    def mock_bioemu(self):
        """Mock run_bioemu to avoid GPU requirement."""
        with patch(
            "proto_language.language.constraint.protein_structure."
            "structure_ensemble_similarity_constraint.run_bioemu"
        ) as mock:
            # Create mock ensemble with 3 frames
            mock_structure = MagicMock()
            mock_structure.structure_pdb = MOCK_PDB

            mock_ensemble = MagicMock()
            mock_ensemble.structures = [mock_structure] * 3

            mock_output = MagicMock()
            mock_output.ensembles = [mock_ensemble]

            mock.return_value = mock_output
            yield mock

    @pytest.fixture
    def mock_pymol_rmsd(self):
        """Mock PyMOL RMSD computation."""
        with patch(
            "proto_language.language.constraint.protein_structure."
            "structure_ensemble_similarity_constraint._compute_pymol_aligned_rmsd"
        ) as mock:
            # Return decreasing RMSDs: 3.0, 2.0, 1.0
            mock.side_effect = [
                {"rmsd": 3.0, "aligned_atoms": 10, "alignment_cycles": 1},
                {"rmsd": 2.0, "aligned_atoms": 10, "alignment_cycles": 1},
                {"rmsd": 1.0, "aligned_atoms": 10, "alignment_cycles": 1},
            ]
            yield mock

    def test_constraint_returns_score(self, mock_bioemu, mock_pymol_rmsd):
        """Basic test that constraint runs and returns a score."""
        config = StructureEnsembleSimilarityConfig(
            target_structure=MOCK_PDB,
            bioemu_config=BioEmuConfig(num_samples=3),
        )

        assert config.bioemu_config.num_samples == 3

        scores = structure_ensemble_rmsd_constraint(
            [(Sequence(TEST_SEQ, "protein"),)],
            config,
        )

        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_metadata_populated(self, mock_bioemu, mock_pymol_rmsd):
        """Check that metadata is populated on the sequence."""
        config = StructureEnsembleSimilarityConfig(
            target_structure=MOCK_PDB,
            bioemu_config=BioEmuConfig(num_samples=3),
        )

        seq = Sequence(TEST_SEQ, "protein")
        structure_ensemble_rmsd_constraint([(seq,)], config)

        # Check key metadata fields exist
        assert "ensemble_rmsd_min" in seq._metadata
        assert "ensemble_rmsd_mean" in seq._metadata
        assert "ensemble_size" in seq._metadata
        assert "pct_within_2A" in seq._metadata
        assert "pct_within_3A" in seq._metadata

        # Check values make sense (RMSDs were 3.0, 2.0, 1.0)
        assert seq._metadata["ensemble_rmsd_min"] == 1.0
        assert seq._metadata["ensemble_rmsd_mean"] == 2.0
        assert seq._metadata["ensemble_size"] == 3

    def test_candidate_residue_range_subsets_sequence(self, mock_bioemu, mock_pymol_rmsd):
        """Verify candidate_residue_range subsets sequence before BioEmu."""
        config = StructureEnsembleSimilarityConfig(
            target_structure=MOCK_PDB,
            candidate_residue_range=(5, 15),  # 11 residues
            bioemu_config=BioEmuConfig(num_samples=3),
        )

        structure_ensemble_rmsd_constraint(
            [(Sequence(TEST_SEQ, "protein"),)],
            config,
        )

        # Check that BioEmu was called with the subsequence
        call_args = mock_bioemu.call_args
        input_obj = call_args[0][0]  # First positional arg
        chain_seq = input_obj.complexes[0].chains[0].sequence

        assert len(chain_seq) == 11
        assert chain_seq == TEST_SEQ[4:15]  # 0-indexed slice

    def test_aggregation_methods(self, mock_bioemu):
        """Test different RMSD aggregation methods produce different scores."""
        # Mock returns same RMSDs for each aggregation test
        with patch(
            "proto_language.language.constraint.protein_structure."
            "structure_ensemble_similarity_constraint._compute_pymol_aligned_rmsd"
        ) as mock_rmsd:
            results = {}

            for agg in ["min", "mean", "median", "p10"]:
                # Reset mock for each run
                mock_rmsd.side_effect = [
                    {"rmsd": 1.0, "aligned_atoms": 10, "alignment_cycles": 1},
                    {"rmsd": 5.0, "aligned_atoms": 10, "alignment_cycles": 1},
                    {"rmsd": 10.0, "aligned_atoms": 10, "alignment_cycles": 1},
                ]

                config = StructureEnsembleSimilarityConfig(
                    target_structure=MOCK_PDB,
                    rmsd_aggregation=agg,
                    bioemu_config=BioEmuConfig(num_samples=3),
                )

                scores = structure_ensemble_rmsd_constraint(
                    [(Sequence(TEST_SEQ, "protein"),)],
                    config,
                )
                results[agg] = scores[0]

            # min should give best (lowest) score, mean should be worse
            assert results["min"] < results["mean"]


@pytest.mark.uses_gpu
class TestConstraintIntegration:
    """Integration tests requiring GPU (BioEmu + PyMOL)."""

    def test_full_pipeline(self):
        """End-to-end test with real BioEmu and PyMOL."""
        config = StructureEnsembleSimilarityConfig(
            target_structure=MOCK_PDB,
            bioemu_config=BioEmuConfig(num_samples=10),  # Small for speed
            verbose=True,
        )

        assert config.bioemu_config.num_samples == 10

        scores = structure_ensemble_rmsd_constraint(
            [(Sequence("AGAGAGAG", "protein"),)],  # Short sequence
            config,
        )

        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0
