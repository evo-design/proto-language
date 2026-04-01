"""Tests all structure confidence and similarity constraints with protein-ligand.

complexes using the Boltz structure predictor without MSAs.

Tests cover:
1. structure_plddt_constraint
2. structure_ptm_constraint
3. structure_iptm_constraint
4. structure_pae_constraint
5. structure_rmsd_constraint
6. structure_tmscore_constraint.
"""

import pytest

from proto_language.language.constraint import (
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
    structure_rmsd_constraint,
    structure_tmscore_constraint,
)
from proto_language.language.constraint.protein_structure.structure_confidence_constraint import (
    StructureBasedConstraintConfig,
)
from proto_language.language.constraint.protein_structure.structure_similarity_constraint import (
    StructureRMSDConfig,
    StructureTMScoreConfig,
)
from proto_language.language.core import Sequence

# ============================================================================
# Test sequences
# ============================================================================

RUVB_PROTEIN = "VERTLRPQYFKEYIGQDKVKDQLKIFIEAAKLRDEALDHTLLFGPPGLGKTTMAFVIANEMGVNLKQTSGPAIEKAGDLVAILNDLEPGDILFIDEIHRMPMAVEEVLYSAMEDYYIDIMIGAGETSRSVHLDLPPFTLVGATTRAGMLSNPLRARFGINGHMEYYELPDLTEIVERTSEIFEMTITPEAALELARRSRGTPRIANRLLKRVRDYAQIMGDGVIDDKIADQALTMLDVDHEGLDYVDQKILRTMIEMYGGGPVGLGTLSVNIAEERETVEDMYEPYLIQKGFIMRTRTGRVATAKAYEHMGYDYTRDN"
ADP_LIGAND = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO[P@](=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"

# Boltz configuration without MSAs
BOLTZ2_CONFIG = {"use_msa": False}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def protein_sequence():
    """Protein sequence fixture."""
    return Sequence(RUVB_PROTEIN, "protein")


@pytest.fixture
def ligand_sequence():
    """Ligand sequence fixture."""
    return Sequence(ADP_LIGAND, "ligand")


# ============================================================================
# Test all structure constraints with protein-ligand complex
# ============================================================================


@pytest.mark.slow
@pytest.mark.uses_gpu
class TestStructureConstraintsProteinLigandComplex:
    """Test all structure constraints with protein-ligand complexes."""

    @pytest.mark.parametrize(
        "constraint_name,constraint_fn,config_class,expected_metadata_keys",
        [
            # Confidence constraints
            (
                "plddt",
                structure_plddt_constraint,
                StructureBasedConstraintConfig,
                ["avg_plddt", "pdb_output", "structure_tool"],
            ),
            (
                "ptm",
                structure_ptm_constraint,
                StructureBasedConstraintConfig,
                ["ptm", "pdb_output", "structure_tool"],
            ),
            (
                "iptm",
                structure_iptm_constraint,
                StructureBasedConstraintConfig,
                ["iptm", "pdb_output", "structure_tool"],
            ),
            (
                "pae",
                structure_pae_constraint,
                StructureBasedConstraintConfig,
                ["avg_pae", "pdb_output", "structure_tool"],
            ),
            # Similarity constraints
            (
                "rmsd",
                structure_rmsd_constraint,
                StructureRMSDConfig,
                ["rmsd_val", "rmsd_score", "pdb_output"],
            ),
            (
                "tmscore",
                structure_tmscore_constraint,
                StructureTMScoreConfig,
                ["tm_score_raw", "tm_score_inverted", "pdb_output"],
            ),
        ],
    )
    def test_constraint_protein_ligand_complex(
        self,
        protein_sequence,
        ligand_sequence,
        constraint_name,
        constraint_fn,
        config_class,
        expected_metadata_keys,
    ):
        """Test that constraint runs end-to-end with protein-ligand complex."""
        proposals = [(protein_sequence, ligand_sequence)]

        # Create appropriate config based on constraint type
        if config_class == StructureBasedConstraintConfig:
            config = StructureBasedConstraintConfig(
                structure_tool="boltz2",
                tool_config=BOLTZ2_CONFIG,
            )
        elif config_class == StructureRMSDConfig:
            config = StructureRMSDConfig(
                target_chains=(RUVB_PROTEIN, ADP_LIGAND),
                structure_tool="boltz2",
                tool_config=BOLTZ2_CONFIG,
            )
        elif config_class == StructureTMScoreConfig:
            config = StructureTMScoreConfig(
                target_chains=(RUVB_PROTEIN, ADP_LIGAND),
                structure_tool="boltz2",
                tool_config=BOLTZ2_CONFIG,
            )

        # Run constraint
        scores = constraint_fn(proposals, config)

        # Verify score structure
        assert len(scores) == 1, f"{constraint_name}: Expected 1 score"
        assert isinstance(scores[0], float), f"{constraint_name}: Score should be float"
        assert 0.0 <= scores[0] <= 1.0, f"{constraint_name}: Score should be in [0, 1]"

        # For self-similarity tests (RMSD and TM-score), expect near-perfect scores
        if constraint_name in ["rmsd", "tmscore"]:
            assert scores[0] < 0.1, f"{constraint_name}: Self-similarity should yield score < 0.1, got {scores[0]}"

        # Verify metadata was stored (on first sequence in tuple)
        metadata = protein_sequence._metadata
        for key in expected_metadata_keys:
            assert key in metadata, f"{constraint_name}: Missing metadata key '{key}'"

        # Additional metadata checks
        if "pdb_output" in metadata:
            assert len(metadata["pdb_output"]) > 0, f"{constraint_name}: PDB output should not be empty"

        if "structure_tool" in metadata:
            assert metadata["structure_tool"] == "boltz2", f"{constraint_name}: Structure tool should be 'boltz2'"


# ============================================================================
# Test ESMFold rejection of unsupported entity types
# ============================================================================


class TestESMFoldLigandRejection:
    """Test that ESMFold properly rejects ligand inputs.

    ESMFold only supports protein sequences. This test verifies that attempting
    to use ESMFold with ligands raises appropriate validation errors.
    """

    @pytest.mark.parametrize(
        "constraint_name,constraint_fn,config_class",
        [
            # Confidence constraints
            (
                "plddt",
                structure_plddt_constraint,
                StructureBasedConstraintConfig,
            ),
            (
                "ptm",
                structure_ptm_constraint,
                StructureBasedConstraintConfig,
            ),
            (
                "iptm",
                structure_iptm_constraint,
                StructureBasedConstraintConfig,
            ),
            (
                "pae",
                structure_pae_constraint,
                StructureBasedConstraintConfig,
            ),
            # Similarity constraints
            (
                "rmsd",
                structure_rmsd_constraint,
                StructureRMSDConfig,
            ),
            (
                "tmscore",
                structure_tmscore_constraint,
                StructureTMScoreConfig,
            ),
        ],
    )
    def test_esmfold_rejects_ligand(
        self,
        protein_sequence,
        ligand_sequence,
        constraint_name,
        constraint_fn,
        config_class,
    ):
        """Test that ESMFold raises ValueError for ligand-containing complexes."""
        proposals = [(protein_sequence, ligand_sequence)]

        # Create appropriate config with ESMFold
        if config_class == StructureBasedConstraintConfig:
            config = StructureBasedConstraintConfig(
                structure_tool="esmfold",
                tool_config={},
            )
        elif config_class == StructureRMSDConfig:
            config = StructureRMSDConfig(
                target_chains=(RUVB_PROTEIN, ADP_LIGAND),
                structure_tool="esmfold",
                tool_config={},
            )
        elif config_class == StructureTMScoreConfig:
            config = StructureTMScoreConfig(
                target_chains=(RUVB_PROTEIN, ADP_LIGAND),
                structure_tool="esmfold",
                tool_config={},
            )

        # Verify that the constraint raises a ValueError
        with pytest.raises(ValueError):
            constraint_fn(proposals, config)
