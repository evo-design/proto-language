"""Tests for Metal3D probability constraint helpers."""

from proto_tools.entities.structures import ResidueSelection, Structure

from proto_language.constraint.protein_structure.metal3d_probability_constraint import (
    _candidate_residues_for_structure,
)
from proto_language.constraint.protein_structure.structure_preparation import StructurePreparationConfig


def test_candidate_residues_remap_from_scaffold_chain_to_prepared_chain(temp_pdb_file):
    structure = Structure.from_file(temp_pdb_file)
    candidate_residues = ResidueSelection(chains={"X": [2]})
    preparation_config = StructurePreparationConfig(chain_ids=["X"])

    remapped = _candidate_residues_for_structure(candidate_residues, structure, preparation_config)

    assert remapped is not None
    assert remapped.chains == {"A": [2]}
