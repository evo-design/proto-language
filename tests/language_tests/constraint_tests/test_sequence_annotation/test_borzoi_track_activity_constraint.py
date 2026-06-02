"""Unit tests for the Borzoi track activity constraint."""

from types import SimpleNamespace

import numpy as np
import pytest
from proto_tools.tools.sequence_scoring.borzoi import BORZOI_CONTEXT, BORZOI_OUTPUT_FLANK

from proto_language import borzoi_track_activity_constraint
from proto_language.constraint import BorzoiTrackActivityConfig, BorzoiTrackInterval
from proto_language.core import Sequence


def test_borzoi_track_activity_batches_ensemble_predictions(monkeypatch):
    config = BorzoiTrackActivityConfig(
        borzoi_output_tracks=[2, 4],
        score_intervals=[BorzoiTrackInterval(start_bp=0, end_bp=2)],
        activity_threshold=10.0,
        batch_size=3,
    )
    captured = {}

    def fake_run_borzoi_ensemble(tool_input, tool_config):
        captured["tool_input"] = tool_input
        captured["tool_config"] = tool_config
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    predictions=np.full((4, 2, BORZOI_CONTEXT), 5.0, dtype=np.float32),
                    output_resolution=1.0,
                    output_start=0,
                )
            ]
        )

    monkeypatch.setitem(borzoi_track_activity_constraint.__globals__, "run_borzoi_ensemble", fake_run_borzoi_ensemble)

    (output,) = borzoi_track_activity_constraint(
        [(Sequence("AA", "dna"), Sequence("CG", "dna"), Sequence("TT", "dna"))],
        config,
    )

    assert len(captured["tool_input"].sequences[0].sequence) == BORZOI_CONTEXT
    assert captured["tool_input"].sequences[0].target_range.start == BORZOI_OUTPUT_FLANK
    assert captured["tool_input"].sequences[0].target_range.end == BORZOI_OUTPUT_FLANK + 2
    assert captured["tool_config"].batch_size == 3
    assert captured["tool_config"].output_tracks == [2, 4]
    assert output.score == 0.5
    assert output.metadata["borzoi_track_activity"] == 5.0


def test_borzoi_track_activity_minimize_direction(monkeypatch):
    config = BorzoiTrackActivityConfig(direction="minimize", activity_threshold=20.0)

    def fake_run_borzoi_ensemble(tool_input, tool_config):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    predictions=np.full((4, 1, BORZOI_CONTEXT), 5.0, dtype=np.float32),
                    output_resolution=1.0,
                    output_start=0,
                )
            ]
        )

    monkeypatch.setitem(borzoi_track_activity_constraint.__globals__, "run_borzoi_ensemble", fake_run_borzoi_ensemble)

    (output,) = borzoi_track_activity_constraint(
        [(Sequence("AA", "dna"), Sequence("CG", "dna"), Sequence("TT", "dna"))],
        config,
    )

    assert output.score == 0.25


def test_borzoi_track_activity_rejects_out_of_range_interval(monkeypatch):
    config = BorzoiTrackActivityConfig(score_intervals=[BorzoiTrackInterval(start_bp=0, end_bp=3)])

    def fake_run_borzoi_ensemble(tool_input, tool_config):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    predictions=np.ones((4, 1, BORZOI_CONTEXT), dtype=np.float32),
                    output_resolution=1.0,
                    output_start=0,
                )
            ]
        )

    monkeypatch.setitem(borzoi_track_activity_constraint.__globals__, "run_borzoi_ensemble", fake_run_borzoi_ensemble)

    with pytest.raises(ValueError, match="exceeds target length"):
        borzoi_track_activity_constraint(
            [(Sequence("AA", "dna"), Sequence("CG", "dna"), Sequence("TT", "dna"))],
            config,
        )
