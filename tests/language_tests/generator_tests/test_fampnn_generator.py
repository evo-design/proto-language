"""Tests for the FAMPNN generator wrapper."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from proto_tools import FAMPNNStructureInput, Structure

from proto_language.core import Segment
from proto_language.generator import FAMPNNGenerator, FAMPNNGeneratorConfig


def test_fampnn_generator_samples_sequence_and_structure(
    monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str
) -> None:
    """FAMPNN remains available as a structure-conditioned generator."""
    module = import_module("proto_language.generator.fampnn_generator")
    output_structure = Structure(structure=sample_pdb_content)
    captured = {}

    def fake_run_fampnn_sample(*, inputs, config):
        captured["inputs"] = inputs
        captured["config"] = config
        design = SimpleNamespace(
            chains=[SimpleNamespace(id="A", sequence="WWWWW")],
            designed=[True],
            metrics=SimpleNamespace(avg_psce=0.25),
            structure=output_structure,
        )
        return SimpleNamespace(design_sets=[SimpleNamespace(complexes=[design])])

    monkeypatch.setattr(module, "run_fampnn_sample", fake_run_fampnn_sample)
    generator = FAMPNNGenerator(
        FAMPNNGeneratorConfig(
            structure_inputs=FAMPNNStructureInput(
                structure=sample_pdb_content,
                chains_to_redesign=["A"],
            ),
            output_chain_id="A",
            temperature=0.2,
            batch_size=3,
        )
    )
    segment = Segment(sequence="AGSVL", sequence_type="protein")
    generator.assign(segment)

    generator.sample()

    proposal = segment.proposal_sequences[0]
    assert captured["inputs"].inputs[0].chain_ids_to_redesign == ["A"]
    assert captured["config"].num_sequences_per_structure == 1
    assert captured["config"].batch_size == 3
    assert proposal.sequence == "WWWWW"
    assert proposal.structure is output_structure
    assert proposal._generator_metadata["fampnn"] == {
        "avg_psce": 0.25,
        "full_sequence": "WWWWW",
    }
