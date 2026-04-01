"""Unit tests for AlphaGenome interval-track sequence-annotation constraint."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from proto_language.language.constraint import (
    ConstraintRegistry,
    alphagenome_interval_track_constraint,
)
from proto_language.language.constraint.sequence_annotation.alphagenome_interval_track_constraint import (
    AlphaGenomeIntervalTrackConfig,
)
from proto_language.language.core import Constraint, Segment


class _DummyAlphaGenomePredictOutput:
    """Minimal object matching AlphaGenome predict tool output contract."""

    def __init__(self, matrix: np.ndarray, key: str = "rna_seq"):
        self.result = {
            "predictions": {
                key: {
                    "values": matrix.tolist(),
                }
            }
        }


class _DummyAlphaGenomePredictBatchOutput:
    """Minimal batched AlphaGenome output with per-sequence results."""

    def __init__(self, outputs: list[_DummyAlphaGenomePredictOutput]):
        self.results = outputs


TEST_SEQUENCE_LENGTH = 16_384
TEST_INTERVALS = [(2_000, 4_000), (12_000, 14_000)]


def test_interval_track_config_validation():
    """Config should reject malformed intervals and empty ontology terms."""
    with pytest.raises(ValidationError):
        AlphaGenomeIntervalTrackConfig(
            intervals=[],
            ontology_terms=["CL:0002319"],
        )

    with pytest.raises(ValidationError):
        AlphaGenomeIntervalTrackConfig(
            intervals=[(10, 10)],
            ontology_terms=["CL:0002319"],
        )

    with pytest.raises(ValidationError):
        AlphaGenomeIntervalTrackConfig(
            intervals=[(0, 10)],
            ontology_terms=[],
        )

    with pytest.raises(ValidationError):
        AlphaGenomeIntervalTrackConfig(
            intervals=[(0, 10)],
            ontology_terms=["CL:0002319"],
            maximize_sigmoid_scale=0.0,
        )

    with pytest.raises(ValidationError):
        AlphaGenomeIntervalTrackConfig(
            intervals=[(0, 10)],
            ontology_terms=["CL:0002319"],
            minimize_threshold_value=0.0,
        )


def test_interval_track_scoring_maximize_with_sigmoid():
    """Maximize mode should map interval mean through thresholded sigmoid objective."""
    segment = Segment(sequence="A" * TEST_SEQUENCE_LENGTH, sequence_type="dna")

    matrix = np.arange(10, dtype=float).reshape(10, 1)

    cfg = AlphaGenomeIntervalTrackConfig(
        intervals=TEST_INTERVALS,
        ontology_terms=["CL:0002319"],
        direction="maximize",
        maximize_inflection_value=5.0,
        maximize_sigmoid_scale=1.0,
    )

    with patch(
        "proto_language.language.constraint.sequence_annotation.alphagenome_interval_track_constraint.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput([_DummyAlphaGenomePredictOutput(matrix, key="rna-seq")]),
    ):
        constraint = Constraint(
            inputs=[segment],
            function=alphagenome_interval_track_constraint,
            function_config=cfg,
        )
        scores = constraint.evaluate()

    assert len(scores) == 1
    expected_mean = np.mean([1.0, 2.0, 7.0, 8.0])  # mapped rows for the two intervals
    expected_sigmoid = 1.0 / (1.0 + np.exp(-(expected_mean - 5.0)))
    expected_score = 1.0 - expected_sigmoid
    assert abs(scores[0] - expected_score) < 1e-6


def test_interval_track_scoring_minimize_returns_thresholded_linear_score():
    """Minimize mode should clip and linearly normalize interval mean to [0, 1]."""
    segment = Segment(sequence="A" * TEST_SEQUENCE_LENGTH, sequence_type="dna")

    matrix = np.arange(10, dtype=float).reshape(10, 1)
    cfg = AlphaGenomeIntervalTrackConfig(
        intervals=TEST_INTERVALS,
        ontology_terms=["EFO:0002067"],
        direction="minimize",
        minimize_threshold_value=10.0,
    )

    with patch(
        "proto_language.language.constraint.sequence_annotation.alphagenome_interval_track_constraint.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput([_DummyAlphaGenomePredictOutput(matrix)]),
    ):
        constraint = Constraint(
            inputs=[segment],
            function=alphagenome_interval_track_constraint,
            function_config=cfg,
        )
        scores = constraint.evaluate()
        clipped_constraint = Constraint(
            inputs=[segment],
            function=alphagenome_interval_track_constraint,
            function_config=AlphaGenomeIntervalTrackConfig(
                intervals=TEST_INTERVALS,
                ontology_terms=["EFO:0002067"],
                direction="minimize",
                minimize_threshold_value=1.0,
            ),
        )
        clipped_scores = clipped_constraint.evaluate()

    assert len(scores) == 1
    assert abs(scores[0] - (np.mean([1.0, 2.0, 7.0, 8.0]) / 10.0)) < 1e-6
    assert len(clipped_scores) == 1
    assert clipped_scores[0] == 1.0


def test_interval_track_metadata_population():
    """Constraint metadata should include interval, signal, and score details."""
    segment = Segment(sequence="A" * TEST_SEQUENCE_LENGTH, sequence_type="dna")
    matrix = np.full((10, 2), 3.5, dtype=float)

    cfg = AlphaGenomeIntervalTrackConfig(
        intervals=[(2_000, 14_000)],
        ontology_terms=["CL:0002319"],
        direction="maximize",
        maximize_inflection_value=5.0,
    )

    with patch(
        "proto_language.language.constraint.sequence_annotation.alphagenome_interval_track_constraint.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput([_DummyAlphaGenomePredictOutput(matrix)]),
    ):
        constraint = Constraint(
            inputs=[segment],
            function=alphagenome_interval_track_constraint,
            function_config=cfg,
        )
        constraint.evaluate()

    constraint_data = segment.proposal_sequences[0]._constraints_metadata["alphagenome_interval_track_constraint"][
        "data"
    ]
    expected_keys = {
        "intervals",
        "ontology_terms",
        "requested_output",
        "direction",
        "interval_mean_signal",
        "maximize_inflection_value",
        "maximize_sigmoid_scale",
        "maximize_sigmoid_value",
        "minimize_threshold_value",
        "minimize_clipped_signal",
        "alphagenome_interval_track_score",
    }
    assert expected_keys.issubset(set(constraint_data.keys()))
    assert constraint_data["intervals"] == [[2_000, 14_000]]
    assert constraint_data["ontology_terms"] == ["CL:0002319"]


def test_interval_track_prediction_timeout_passed_to_tool():
    """Configured prediction timeout should be forwarded to AlphaGenome tool config."""
    segment = Segment(sequence="A" * TEST_SEQUENCE_LENGTH, sequence_type="dna")
    matrix = np.zeros((10, 1), dtype=float)
    observed_timeouts: list[int] = []

    def _mock_predict(inputs, config, instance=None):
        observed_timeouts.append(int(config.timeout))
        return _DummyAlphaGenomePredictBatchOutput([_DummyAlphaGenomePredictOutput(matrix)])

    cfg = AlphaGenomeIntervalTrackConfig(
        intervals=[(0, 1_000)],
        ontology_terms=["CL:0002319"],
        prediction_timeout=1234,
    )

    with patch(
        "proto_language.language.constraint.sequence_annotation.alphagenome_interval_track_constraint.run_alphagenome_predict_sequences",
        side_effect=_mock_predict,
    ):
        constraint = Constraint(
            inputs=[segment],
            function=alphagenome_interval_track_constraint,
            function_config=cfg,
        )
        constraint.evaluate()

    assert observed_timeouts == [1234]


def test_interval_track_interval_out_of_bounds_errors():
    """Interval extending beyond sequence length should fail loudly."""
    segment = Segment(sequence="A" * TEST_SEQUENCE_LENGTH, sequence_type="dna")
    matrix = np.zeros((10, 1), dtype=float)
    cfg = AlphaGenomeIntervalTrackConfig(
        intervals=[(0, 17_000)],
        ontology_terms=["CL:0002319"],
    )

    with patch(
        "proto_language.language.constraint.sequence_annotation.alphagenome_interval_track_constraint.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput([_DummyAlphaGenomePredictOutput(matrix)]),
    ):
        constraint = Constraint(
            inputs=[segment],
            function=alphagenome_interval_track_constraint,
            function_config=cfg,
        )
        with pytest.raises(ValueError, match="exceeds sequence length"):
            constraint.evaluate()


def test_interval_track_constraint_registry_exposed():
    """Constraint should be discoverable via registry."""
    spec = ConstraintRegistry.get("alphagenome-interval-track")
    assert spec.function == alphagenome_interval_track_constraint
