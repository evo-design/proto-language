"""Unit tests for shared chromatin accessibility MORSE helpers."""

import numpy as np
import pytest

from proto_language.language.constraint.sequence_annotation.chromatin_accessibility_morse_utils import (
    score_morse_signal,
)


def test_score_morse_signal_combines_pattern_contrast_and_window_terms():
    output = score_morse_signal(
        model_name="test",
        raw_signal=np.array([0.0, 2.0, 0.5, 0.0], dtype=np.float32),
        target_start=0,
        target_end=40,
        pattern=".",
        pattern_start_bp=10,
        pattern_normalization="global_max",
        contrast_margin=1.0,
        contrast_weight=0.5,
        raw_amplitude_weight=0.1,
        high_window_reward_weight=0.2,
        low_window_penalty_weight=0.3,
        window_stat_transform="identity",
        highs=[(10, 20)],
        lows=[(20, 30)],
        resolution=10.0,
        output_start=0,
    )

    assert output.score == pytest.approx(-0.3458333333333333)
    assert output.metadata["chromatin_accessibility_morse_model"] == "test"
    assert output.metadata["chromatin_accessibility_morse_raw_amplitude"] == 2.0
    assert output.metadata["chromatin_accessibility_morse_high_window_mean"] == 2.0
    assert output.metadata["chromatin_accessibility_morse_low_window_mean"] == 0.5
