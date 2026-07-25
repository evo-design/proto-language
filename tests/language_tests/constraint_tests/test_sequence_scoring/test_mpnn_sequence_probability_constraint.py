"""Tests for the MPNN sequence probability constraint."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from proto_tools import InverseFoldingStructureInput
from proto_tools.entities.structures import Structure

from proto_language.constraint.sequence_scoring.mpnn_sequence_probability_constraint import (
    MPNNSequenceProbabilityConfig,
    mpnn_sequence_probability_constraint,
)
from proto_language.core import Sequence


def test_ligandmpnn_sequence_probability_scores_one_minus_pmpnn(
    monkeypatch: pytest.MonkeyPatch,
    temp_pdb_file,
) -> None:
    structure = Structure.from_file(temp_pdb_file)
    module = import_module("proto_language.constraint.sequence_scoring.mpnn_sequence_probability_constraint")

    def fake_run_ligandmpnn_score(*, inputs, config):
        assert config.scoring_mode == "single_aa"
        pair = inputs.sequence_structure_pairs[0]
        assert pair.sequence == "AGSVL"
        return SimpleNamespace(
            scores=[
                SimpleNamespace(
                    avg_log_likelihood=-0.25,
                    perplexity=1.2840254166877414,
                    model_dump=lambda exclude=None: {
                        "avg_log_likelihood": -0.25,
                        "perplexity": 1.2840254166877414,
                    },
                )
            ]
        )

    monkeypatch.setattr(module, "run_ligandmpnn_score", fake_run_ligandmpnn_score)

    outputs = mpnn_sequence_probability_constraint(
        [(Sequence("AGSVL", sequence_type="protein"),)],
        config=MPNNSequenceProbabilityConfig(
            model="ligandmpnn",
            structure_inputs=InverseFoldingStructureInput(structure=structure),
            score_mode="probability_loss",
            device="cpu",
        ),
    )

    assert outputs[0].score == pytest.approx(1.0 - 0.7788007830714049)
    assert outputs[0].metadata["pmpnn"] == pytest.approx(0.7788007830714049)
    assert outputs[0].metadata["mpnn_avg_log_likelihood"] == -0.25
