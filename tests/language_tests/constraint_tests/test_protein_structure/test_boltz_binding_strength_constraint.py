"""
Tests for Boltz Binding Strength constraint.
"""

from proto_language.language.core import Segment
from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint import (
    BoltzBindingStrengthConfig,
    DEFAULT_DESIRED_HIGHER,
)
from proto_language.bio_tools.tools.structure_prediction import (
    StructurePredictionOutput,
)
from proto_language.bio_tools.entities.structures import BFactorType
from unittest.mock import patch

from tests.helpers.mock_structure import MockStructure

mock_protein_protein_ligand_structure = MockStructure(
    structure_format="cif",
    b_factor_type=BFactorType.PLDDT,
    source="boltz2-prediction",
    metrics={
        "confidence_score": 0.7753927111625671,
        "ptm": 0.8913218975067139,
        "iptm": 0.8530166149139404,
        "ligand_iptm": 0.867040753364563,
        "protein_iptm": 0.8485115766525269,
        "complex_plddt": 0.755986750125885,
        "complex_iplddt": 0.7038567662239075,
        "complex_pde": 0.34867003560066223,
        "complex_ipde": 0.5199494361877441,
        "chains_ptm": [0.9603453874588013, 0.9573644995689392, 0.8839606046676636],
        "pair_chains_iptm": [
            [0.9603453874588013, 0.8485115766525269, 0.4917132258415222],
            [0.8294212222099304, 0.9573644995689392, 0.4966408908367157],
            [0.857824444770813, 0.867040753364563, 0.8839606046676636],
        ],
    },
)
mock_protein_protein_ligand_output = StructurePredictionOutput(
    tool_id="boltz2-prediction",
    execution_time=0.0,
    success=True,
    structures=[mock_protein_protein_ligand_structure],
    warnings=[],
    metadata={},
)


mock_protein_protein_structure = MockStructure(
    structure_format="cif",
    b_factor_type=BFactorType.PLDDT,
    source="boltz2-prediction",
    metrics={
        "confidence_score": 0.7753927111625671,
        "ptm": 0.8913218975067139,
        "iptm": 0.8530166149139404,
        "protein_iptm": 0.8485115766525269,
        "complex_plddt": 0.755986750125885,
        "complex_iplddt": 0.7038567662239075,
        "complex_pde": 0.34867003560066223,
        "complex_ipde": 0.5199494361877441,
        "chains_ptm": [0.9603453874588013, 0.9573644995689392],
        "pair_chains_iptm": [
            [0.9603453874588013, 0.8485115766525269],
            [0.8294212222099304, 0.9573644995689392],
        ],
    },
)
mock_protein_protein_output = StructurePredictionOutput(
    tool_id="boltz2-prediction",
    execution_time=0.0,
    success=True,
    structures=[mock_protein_protein_structure],
    warnings=[],
    metadata={},
)

mock_monomer_structure = MockStructure(
    structure_format="cif",
    b_factor_type=BFactorType.PLDDT,
    source="boltz2-prediction",
    metrics={
        "confidence_score": 0.7753927111625671,
        "ptm": 0.8913218975067139,
        "iptm": 0.8530166149139404,
        "protein_iptm": 0.8485115766525269,
        "complex_plddt": 0.755986750125885,
        "complex_iplddt": 0.7038567662239075,
        "complex_pde": 0.34867003560066223,
        "complex_ipde": 0.5199494361877441,
        "chains_ptm": [0.9603453874588013],
        "pair_chains_iptm": [[0.9603453874588013]],
    },
)
mock_monomer_output = StructurePredictionOutput(
    tool_id="boltz2-prediction",
    execution_time=0.0,
    success=True,
    structures=[mock_monomer_structure],
    warnings=[],
    metadata={},
)


class TestBoltzBindingStrengthConstraint:
    """Tests for Boltz Binding Strength constraint."""

    def test_config_merge_overrides(self):
        """
        Test config merge overrides default values for dicts and ensures
        nested configs are set correctly.
        """
        from proto_language.bio_tools.tools.structure_prediction.boltz2 import (
            Boltz2Config,
        )

        # Ensures config merge overrides default values
        desired_higher = {
            "iptm": 0.95,
        }

        boltz2_cfg = Boltz2Config(
            recycling_steps=3,
            diffusion_samples=1,
        )
        config = BoltzBindingStrengthConfig(
            desired_higher=desired_higher, boltz2_config=boltz2_cfg
        )
        assert config.desired_higher["iptm"] == 0.95
        for key, value in DEFAULT_DESIRED_HIGHER.items():
            if key != "iptm":
                assert config.desired_higher[key] == value
            else:
                assert config.desired_higher[key] == 0.95

        # Ensures nested config is set correctly
        assert config.boltz2_config.recycling_steps == 3  # pylint: disable=no-member
        assert config.boltz2_config.diffusion_samples == 1  # pylint: disable=no-member

    def test_with_protein_protein_ligand_complex(self):
        """Test constraint with protein-protein-ligand complex."""
        protein1 = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        protein2 = Segment(sequence="MVLSEGEWQLVLHVWAK", sequence_type="protein")
        ligand_target = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        complex_list = [protein1, protein2]

        constraint = ConstraintRegistry.create(
            key="boltz2-binding-strength", segments=complex_list, config_dict={"ligands": ligand_target}
        )

        with patch(
            "proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint.run_boltz2",
            return_value=mock_protein_protein_ligand_output,
        ):

            _ = constraint.evaluate()

    def test_with_protein_protein_complex(self):
        """Test constraint with protein-protein complex."""
        protein1 = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        protein2 = Segment(sequence="MVLSEGEWQLVLHVWAK", sequence_type="protein")
        complex_list = [protein1, protein2]

        constraint = ConstraintRegistry.create(
            key="boltz2-binding-strength", segments=complex_list, config_dict={}
        )

        with patch(
            "proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint.run_boltz2",
            return_value=mock_protein_protein_output,
        ):

            _ = constraint.evaluate()

    def test_with_monomer(self):
        """Test constraint with monomer."""
        protein = Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")
        complex_list = [protein]

        constraint = ConstraintRegistry.create(
            key="boltz2-binding-strength", segments=complex_list, config_dict={}
        )

        with patch(
            "proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint.run_boltz2",
            return_value=mock_monomer_output,
        ):

            _ = constraint.evaluate()
