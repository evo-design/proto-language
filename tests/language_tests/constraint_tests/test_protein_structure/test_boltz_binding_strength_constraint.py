"""
Tests for Boltz Binding Strength constraint.
"""

import pytest

from proto_language.language.core import SequenceType, Segment
from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint import (
    BoltzBindingStrengthConfig,
    DEFAULT_DESIRED_HIGHER,
)
from proto_language.tools.structure_prediction import (
    BoltzStructure,
    StructurePredictionOutput,
)
from unittest.mock import patch


mock_protein_protein_ligand_output = StructurePredictionOutput(
    tool_id="boltz-prediction",
    execution_time=0.0,
    success=True,
    structures=[
        BoltzStructure(
            structure_cif="mock_structure.cif",
            num_residues=60,
            num_chains=3,
            confidence_score=0.7753927111625671,
            ptm=0.8913218975067139,
            iptm=0.8530166149139404,
            ligand_iptm=0.867040753364563,
            protein_iptm=0.8485115766525269,
            complex_plddt=0.755986750125885,
            complex_iplddt=0.7038567662239075,
            complex_pde=0.34867003560066223,
            complex_ipde=0.5199494361877441,
            chains_ptm=[0.9603453874588013, 0.9573644995689392, 0.8839606046676636],
            pair_chains_iptm=[
                [0.9603453874588013, 0.8485115766525269, 0.4917132258415222],
                [0.8294212222099304, 0.9573644995689392, 0.4966408908367157],
                [0.857824444770813, 0.867040753364563, 0.8839606046676636],
            ],
        )
    ],
    warnings=[],
    metadata={},
)


mock_protein_protein_output = StructurePredictionOutput(
    tool_id="boltz-prediction",
    execution_time=0.0,
    success=True,
    structures=[
        BoltzStructure(
            structure_cif="mock_structure.cif",
            num_residues=40,
            num_chains=2,
            confidence_score=0.7753927111625671,
            ptm=0.8913218975067139,
            iptm=0.8530166149139404,
            protein_iptm=0.8485115766525269,
            complex_plddt=0.755986750125885,
            complex_iplddt=0.7038567662239075,
            complex_pde=0.34867003560066223,
            complex_ipde=0.5199494361877441,
            chains_ptm=[0.9603453874588013, 0.9573644995689392],
            pair_chains_iptm=[
                [0.9603453874588013, 0.8485115766525269],
                [0.8294212222099304, 0.9573644995689392],
            ],
        )
    ],
    warnings=[],
    metadata={},
)


mock_monomer_output = StructurePredictionOutput(
    tool_id="boltz-prediction",
    execution_time=0.0,
    success=True,
    structures=[
        BoltzStructure(
            structure_cif="mock_structure.cif",
            num_residues=20,
            num_chains=1,
            confidence_score=0.7753927111625671,
            ptm=0.8913218975067139,
            iptm=0.8530166149139404,
            protein_iptm=0.8485115766525269,
            complex_plddt=0.755986750125885,
            complex_iplddt=0.7038567662239075,
            complex_pde=0.34867003560066223,
            complex_ipde=0.5199494361877441,
            chains_ptm=[0.9603453874588013],
            pair_chains_iptm=[
                [0.9603453874588013],
            ],
        )
    ],
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
        from proto_language.tools.structure_prediction.boltz import (
            BoltzConfig,
        )

        # Ensures config merge overrides default values
        desired_higher = {
            "iptm": 0.95,
        }

        boltz_cfg = BoltzConfig(
            recycling_steps=3,
            diffusion_samples=1,
        )
        config = BoltzBindingStrengthConfig(
            desired_higher=desired_higher, boltz_config=boltz_cfg
        )
        assert config.desired_higher["iptm"] == 0.95
        for key, value in DEFAULT_DESIRED_HIGHER.items():
            if key != "iptm":
                assert config.desired_higher[key] == value
            else:
                assert config.desired_higher[key] == 0.95

        # Ensures nested config is set correctly
        assert config.boltz_config.recycling_steps == 3  # pylint: disable=no-member
        assert config.boltz_config.diffusion_samples == 1  # pylint: disable=no-member

    def test_with_protein_protein_ligand_complex(self):
        """Test constraint with protein-protein-ligand complex."""
        protein1 = Segment(starting_sequence_or_desired_length="MKTAYIAKQRQISFVK", sequence_type=SequenceType.PROTEIN)
        protein2 = Segment(starting_sequence_or_desired_length="MVLSEGEWQLVLHVWAK", sequence_type=SequenceType.PROTEIN)
        target = Segment(starting_sequence_or_desired_length="N[C@@H](Cc1ccc(O)cc1)C(=O)O", sequence_type=SequenceType.LIGAND)
        complex_list = [protein1, protein2, target]

        constraint = ConstraintRegistry.create(
            key="boltz-binding-strength", segments=complex_list, config_dict={}
        )

        with patch(
            "proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint.run_boltz",
            return_value=mock_protein_protein_ligand_output,
        ):

            scores = constraint.evaluate()

    def test_with_protein_protein_complex(self):
        """Test constraint with protein-protein complex."""
        protein1 = Segment(starting_sequence_or_desired_length="MKTAYIAKQRQISFVK", sequence_type=SequenceType.PROTEIN)
        protein2 = Segment(starting_sequence_or_desired_length="MVLSEGEWQLVLHVWAK", sequence_type=SequenceType.PROTEIN)
        complex_list = [protein1, protein2]

        constraint = ConstraintRegistry.create(
            key="boltz-binding-strength", segments=complex_list, config_dict={}
        )

        with patch(
            "proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint.run_boltz",
            return_value=mock_protein_protein_output,
        ):

            scores = constraint.evaluate()

    def test_with_monomer(self):
        """Test constraint with monomer."""
        protein = Segment(starting_sequence_or_desired_length="MKTAYIAKQRQISFVK", sequence_type=SequenceType.PROTEIN)
        complex_list = [protein]

        constraint = ConstraintRegistry.create(
            key="boltz-binding-strength", segments=complex_list, config_dict={}
        )

        with patch(
            "proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint.run_boltz",
            return_value=mock_monomer_output,
        ):

            scores = constraint.evaluate()
