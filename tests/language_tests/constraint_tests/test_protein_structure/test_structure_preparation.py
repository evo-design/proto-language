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


def test_ligandmpnn_pack_from_scaffold_fixes_ordinal_positions_for_offset_numbering(
    monkeypatch: pytest.MonkeyPatch, temp_pdb_file: str
) -> None:
    """Generated fixed positions use ResidueSelection's 1-indexed chain positions."""
    module = import_module("proto_language.constraint.protein_structure.structure_preparation")
    scaffold = Structure.from_file(temp_pdb_file)
    for residue in scaffold.gemmi_struct[0]["A"]:
        residue.seqid.num += 100
    packed_structure = scaffold.model_copy(update={"source": "ligandmpnn-packed"})
    captured = {}

    def fake_run_ligandmpnn_sample(*, inputs, config):
        captured["inputs"] = inputs
        captured["config"] = config
        return SimpleNamespace(
            design_sets=[
                SimpleNamespace(
                    complexes=[
                        SimpleNamespace(
                            chains=[SimpleNamespace(id="A", sequence="CCCCC")],
                            structure=packed_structure,
                        )
                    ]
                )
            ]
        )

    monkeypatch.setattr(module, "run_ligandmpnn_sample", fake_run_ligandmpnn_sample)
    config = StructurePreparationConfig(
        mode="ligandmpnn_pack_from_scaffold",
        scaffold_structure=scaffold,
        chain_ids=["A"],
    )

    prepared = prepare_structures_for_proposals(
        [(Sequence("CCCCC", sequence_type="protein"),)],
        config,
    )

    threaded_input = captured["inputs"].inputs[0]
    assert prepared == [packed_structure]
    assert scaffold.get_chain_positions("A") == [101, 102, 103, 104, 105]
    assert threaded_input.fixed_positions.chains == {"A": [1, 2, 3, 4, 5]}
    assert threaded_input.structure.get_chain_positions("A") == [101, 102, 103, 104, 105]
    assert captured["config"].num_sequences_per_structure == 1
    assert captured["config"].batch_size == 1
