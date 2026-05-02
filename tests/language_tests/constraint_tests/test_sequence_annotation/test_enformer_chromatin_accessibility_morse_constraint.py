"""Unit tests for the Enformer chromatin accessibility Morse constraint."""

from types import SimpleNamespace

import numpy as np
from proto_tools.tools.sequence_scoring.enformer import ENFORMER_CONTEXT, ENFORMER_OUTPUT_FLANK

from proto_language.language.constraint import EnformerChromatinAccessibilityMorseConfig
from proto_language.language.constraint.sequence_annotation import enformer_chromatin_accessibility_morse_constraint
from proto_language.language.constraint.sequence_annotation.chromatin_accessibility_morse_utils import (
    build_binary_pattern_for_target,
    compute_morse_windows,
)
from proto_language.language.core import Sequence


def test_enformer_default_tracks_follow_organism():
    assert EnformerChromatinAccessibilityMorseConfig().enformer_output_tracks == [121]
    assert EnformerChromatinAccessibilityMorseConfig(organism="mouse").enformer_output_tracks == [11]


def test_morse_layout_marks_expected_windows():
    config = EnformerChromatinAccessibilityMorseConfig(
        enformer_output_tracks=[0],
        pattern=". .",
        dot_bp=16,
        dash_bp=16,
        intra_symbol_gap_bp=0,
        inter_letter_gap_bp=16,
    )

    highs, lows = compute_morse_windows(
        pattern=config.pattern,
        pattern_start_bp=config.pattern_start_bp,
        dot_bp=config.dot_bp,
        dash_bp=config.dash_bp,
        intra_symbol_gap_bp=config.intra_symbol_gap_bp,
        inter_letter_gap_bp=config.inter_letter_gap_bp,
    )
    pattern = build_binary_pattern_for_target(highs, target_num_bins=4, resolution=16.0)

    assert highs == [(0, 16), (32, 48)]
    assert lows == [(16, 32)]
    assert np.array_equal(pattern, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_enformer_constraint_batches_prepared_sequences(monkeypatch):
    config = EnformerChromatinAccessibilityMorseConfig(
        enformer_output_tracks=[0],
        pattern=".",
        dot_bp=1,
        dash_bp=1,
        batch_size=4,
    )
    captured = {}

    def fake_run_enformer(tool_input, tool_config):
        captured["tool_input"] = tool_input
        captured["tool_config"] = tool_config
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    prediction=np.ones((target_range.end + 2, 1), dtype=np.float32),
                    output_resolution=1.0,
                    output_start=0,
                )
                for target_range in tool_input.target_ranges
            ]
        )

    monkeypatch.setitem(
        enformer_chromatin_accessibility_morse_constraint.__globals__,
        "run_enformer",
        fake_run_enformer,
    )

    outputs = enformer_chromatin_accessibility_morse_constraint(
        [
            (Sequence("AA", "dna"), Sequence("CG", "dna"), Sequence("TT", "dna")),
            (Sequence("GG", "dna"), Sequence("TA", "dna"), Sequence("CC", "dna")),
        ],
        config,
    )

    assert len(captured["tool_input"].sequences) == 2
    assert len(captured["tool_input"].sequences[0]) == ENFORMER_CONTEXT
    assert captured["tool_input"].sequences[0][ENFORMER_OUTPUT_FLANK - 2 : ENFORMER_OUTPUT_FLANK + 4] == "AACGTT"
    assert captured["tool_input"].target_ranges[0].start == ENFORMER_OUTPUT_FLANK
    assert captured["tool_input"].target_ranges[0].end == ENFORMER_OUTPUT_FLANK + 2
    assert captured["tool_config"].batch_size == 4
    assert [output.metadata_recipient for output in outputs] == ["Target", "Target"]
    assert outputs[0].metadata["chromatin_accessibility_morse_model"] == "enformer"
