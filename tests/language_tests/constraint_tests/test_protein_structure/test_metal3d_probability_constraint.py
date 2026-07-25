"""Tests for Metal3D probability constraint helpers."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from proto_tools.entities.structures import ResidueSelection, Structure

from proto_language.constraint.protein_structure.metal3d_probability_constraint import (
    Metal3DProbabilityConfig,
    _candidate_residues_for_structure,
    metal3d_probability_constraint,
)
from proto_language.constraint.protein_structure.structure_preparation import StructurePreparationConfig
from proto_language.core import Sequence


class _FakeMetal3DResult(SimpleNamespace):
    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


def test_candidate_residues_remap_from_scaffold_chain_to_prepared_chain(temp_pdb_file):
    structure = Structure.from_file(temp_pdb_file)
    candidate_residues = ResidueSelection(chains={"X": [2]})
    preparation_config = StructurePreparationConfig(chain_ids=["X"])

    remapped = _candidate_residues_for_structure(candidate_residues, structure, preparation_config)

    assert remapped is not None
    assert remapped.chains == {"A": [2]}


def test_metal3d_threshold_gates_sites_but_score_rewards_higher_probability(
    monkeypatch: pytest.MonkeyPatch, temp_pdb_file
):
    structure = Structure.from_file(temp_pdb_file)
    module = import_module("proto_language.constraint.protein_structure.metal3d_probability_constraint")

    def fake_prepare_structures_for_proposals(input_sequences, config):
        return [structure for _ in input_sequences]

    def fake_run_metal3d_prediction(*, inputs, config):
        assert len(inputs.inputs) == 3
        assert config.probability_threshold == 0.5
        return SimpleNamespace(
            results=[
                _FakeMetal3DResult(
                    pmetal=0.0,
                    found=False,
                    sites=[],
                    residue_probabilities=[],
                    annotated_structure=structure,
                ),
                _FakeMetal3DResult(
                    pmetal=0.6,
                    found=True,
                    sites=[],
                    residue_probabilities=[],
                    annotated_structure=structure,
                ),
                _FakeMetal3DResult(
                    pmetal=1.0,
                    found=True,
                    sites=[],
                    residue_probabilities=[],
                    annotated_structure=structure,
                ),
            ]
        )

    monkeypatch.setattr(module, "prepare_structures_for_proposals", fake_prepare_structures_for_proposals)
    monkeypatch.setattr(module, "run_metal3d_prediction", fake_run_metal3d_prediction)

    input_sequences = [(Sequence("AGSVL", sequence_type="protein"),) for _ in range(3)]
    config = Metal3DProbabilityConfig(min_probability=0.5)
    outputs = metal3d_probability_constraint(input_sequences, config)

    assert [output.score for output in outputs] == [1.0, pytest.approx(0.4), 0.0]
    assert outputs[0].metadata["found"] is False
    assert outputs[1].metadata["found"] is True
    assert outputs[1].metadata["min_probability"] == config.min_probability
