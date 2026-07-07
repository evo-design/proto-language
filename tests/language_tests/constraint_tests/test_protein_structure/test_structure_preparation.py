"""Tests for reusable protein structure preparation helpers."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from proto_tools import Structure
from proto_tools.entities.structures import ResidueSelection

from proto_language.constraint.protein_structure.structure_preparation import (
    StructurePreparationConfig,
    prepare_structures_for_proposals,
)
from proto_language.core import Sequence


def test_fampnn_pack_from_scaffold_threads_sequence_and_returns_first_pack(
    monkeypatch: pytest.MonkeyPatch, temp_pdb_file: str
) -> None:
    """FAMPNN remains available as a generic structure-preparation backend."""
    module = import_module("proto_language.constraint.protein_structure.structure_preparation")
    scaffold = Structure.from_file(temp_pdb_file)
    packed_structure = scaffold.model_copy(update={"source": "packed"})
    fixed_positions = ResidueSelection(chains={"A": [1]})
    fixed_sidechains = ResidueSelection(chains={"A": [2]})
    captured = {}

    def fake_run_fampnn_pack(*, inputs, config):
        captured["inputs"] = inputs
        captured["config"] = config
        return SimpleNamespace(packed_structures=[[packed_structure]])

    monkeypatch.setattr(module, "run_fampnn_pack", fake_run_fampnn_pack)
    config = StructurePreparationConfig(
        mode="fampnn_pack_from_scaffold",
        scaffold_structure=scaffold,
        chain_ids=["A"],
        fixed_positions=fixed_positions,
        fixed_sidechain_positions=fixed_sidechains,
    )

    prepared = prepare_structures_for_proposals(
        [(Sequence("CCCCC", sequence_type="protein"),)],
        config,
    )

    threaded_input = captured["inputs"].inputs[0]
    assert prepared == [packed_structure]
    assert threaded_input.structure.get_chain_sequence("A") == "CCCCC"
    assert threaded_input.fixed_positions is fixed_positions
    assert threaded_input.fixed_sidechain_positions is fixed_sidechains
    assert captured["config"] is config.fampnn_pack_config
