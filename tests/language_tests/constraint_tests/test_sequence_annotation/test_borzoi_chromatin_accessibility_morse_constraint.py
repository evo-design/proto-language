"""Unit tests for the Borzoi chromatin accessibility Morse constraint."""

from types import SimpleNamespace

import numpy as np
from proto_tools.tools.sequence_scoring.borzoi import BORZOI_CONTEXT, BORZOI_OUTPUT_FLANK

from proto_language import borzoi_chromatin_accessibility_morse_constraint
from proto_language.constraint import BorzoiChromatinAccessibilityMorseConfig
from proto_language.core import Sequence


def test_borzoi_default_tracks_follow_organism():
    assert BorzoiChromatinAccessibilityMorseConfig().borzoi_output_tracks == [1901]
    assert BorzoiChromatinAccessibilityMorseConfig(organism="mouse").borzoi_output_tracks == [741]


def test_borzoi_constraint_batches_ensemble_predictions(monkeypatch):
    config = BorzoiChromatinAccessibilityMorseConfig(
        borzoi_output_tracks=[0, 1],
        pattern=".",
        dot_bp=1,
        dash_bp=1,
        batch_size=3,
    )
    captured = {}

    def fake_run_borzoi_ensemble(tool_input, tool_config):
        captured["tool_input"] = tool_input
        captured["tool_config"] = tool_config
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    predictions=np.ones((4, 2, window.target_range.end + 2), dtype=np.float32),
                    output_resolution=1.0,
                    output_start=0,
                )
                for window in tool_input.sequences
            ]
        )

    monkeypatch.setitem(
        borzoi_chromatin_accessibility_morse_constraint.__globals__,
        "run_borzoi_ensemble",
        fake_run_borzoi_ensemble,
    )

    (output,) = borzoi_chromatin_accessibility_morse_constraint(
        [(Sequence("AA", "dna"), Sequence("CG", "dna"), Sequence("TT", "dna"))],
        config,
    )

    assert len(captured["tool_input"].sequences[0].sequence) == BORZOI_CONTEXT
    assert captured["tool_input"].sequences[0].target_range.start == BORZOI_OUTPUT_FLANK
    assert captured["tool_input"].sequences[0].target_range.end == BORZOI_OUTPUT_FLANK + 2
    assert captured["tool_config"].batch_size == 3
    assert output.metadata_recipient == "Target"
    assert output.metadata["chromatin_accessibility_morse_model"] == "borzoi"
