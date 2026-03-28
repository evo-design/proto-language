"""tests/language_tests/constraint_tests/test_rna_splicing/test_alphagenome_splice_site_usage_constraint.py

The constraint accepts three-part input tuples (left_flank, intron_core,
right_flank), integrates them into a genomic context via cassette insertion,
and scores splice-site usage with AlphaGenome."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from proto_language.language.constraint import (
    ConstraintRegistry,
    alphagenome_splice_site_usage,
)
from proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage import (
    AlphaGenomeSpliceSiteUsageConfig,
    _integrate_cassette_into_context,
)
from proto_language.language.core import Constraint, Segment, Sequence


class _DummyAlphaGenomePredictOutput:
    def __init__(
        self,
        matrix: np.ndarray,
        metadata_records: list[dict] | None = None,
    ):
        payload = {
            "values": matrix.tolist(),
            "resolution": 1,
            "interval": None,
            "uns": None,
        }
        if metadata_records is not None:
            payload["metadata"] = {"records": metadata_records}
        self.result = {"predictions": {"splice_site_usage": payload}}


class _DummyAlphaGenomePredictBatchOutput:
    def __init__(self, outputs: list[_DummyAlphaGenomePredictOutput]):
        self.results = outputs


# Test dimensions: genomic context must match AlphaGenome supported lengths.
TARGET_LENGTH = 1000  # left_flank + intron_core + right_flank
LEFT_FLANK_LEN = 200
INTRON_CORE_LEN = 600
RIGHT_FLANK_LEN = 200
CASSETTE_LEFT_LEN = 500
CASSETTE_RIGHT_LEN = 500
GENOMIC_CONTEXT_LEN = 16_384  # Smallest supported AlphaGenome context length


def _make_config(**overrides) -> AlphaGenomeSpliceSiteUsageConfig:
    """Create a test config with all required fields populated."""
    defaults = dict(
        genomic_context="A" * GENOMIC_CONTEXT_LEN,
        cassette_left_context="C" * CASSETTE_LEFT_LEN,
        cassette_right_context="G" * CASSETTE_RIGHT_LEN,
        ontology_terms=["EFO:0002067"],
        splice_pos=[100, 200],
        direction="max",
        strand="positive",
    )
    defaults.update(overrides)
    return AlphaGenomeSpliceSiteUsageConfig(**defaults)


def _compute_absolute_pos(config: AlphaGenomeSpliceSiteUsageConfig) -> list[int]:
    """Compute absolute splice positions in the integrated sequence."""
    cassette_len = len(config.cassette_left_context) + TARGET_LENGTH + len(config.cassette_right_context)
    insert_start = (len(config.genomic_context) - cassette_len) // 2
    cassette_offset = insert_start + len(config.cassette_left_context)
    return [cassette_offset + pos for pos in config.splice_pos]


def _make_3part_segments():
    """Create three Segment objects summing to TARGET_LENGTH."""
    left = Segment(sequence="A" * LEFT_FLANK_LEN, sequence_type="dna")
    intron = Segment(sequence="T" * INTRON_CORE_LEN, sequence_type="dna")
    right = Segment(sequence="G" * RIGHT_FLANK_LEN, sequence_type="dna")
    return left, intron, right


def _make_3part_sequences():
    """Create three Sequence objects summing to TARGET_LENGTH."""
    left = Sequence("A" * LEFT_FLANK_LEN, sequence_type="dna")
    intron = Sequence("T" * INTRON_CORE_LEN, sequence_type="dna")
    right = Sequence("G" * RIGHT_FLANK_LEN, sequence_type="dna")
    return left, intron, right


# --- Registration ---


def test_ag_ssu_registered_with_three_part():
    spec = ConstraintRegistry.get("alphagenome-splice-site-usage")
    assert spec.num_input_sequences_per_tuple == 3
    assert spec.function == alphagenome_splice_site_usage


# --- Config validation ---


def test_ag_ssu_config_validation():
    with pytest.raises(ValidationError):
        _make_config(ontology_terms=[])

    with pytest.raises(ValidationError):
        _make_config(splice_pos=[])


def test_ag_ssu_wrong_sequence_type():
    with pytest.raises(TypeError, match="does not support sequence type"):
        ConstraintRegistry.create(
            key="alphagenome-splice-site-usage",
            segments=[
                Segment(sequence="MKTAY", sequence_type="protein"),
                Segment(sequence="ACDEF", sequence_type="protein"),
                Segment(sequence="GHIKL", sequence_type="protein"),
            ],
            config_dict=_make_config().model_dump(),
        )


# --- Scoring ---


def test_ag_ssu_scoring_positive_strand_max_and_min():
    left, intron, right = _make_3part_segments()
    config = _make_config(splice_pos=[100, 200])
    abs_pos = _compute_absolute_pos(config)

    matrix = np.zeros((GENOMIC_CONTEXT_LEN, 4), dtype=float)
    matrix[abs_pos[0], :] = np.asarray([0.8, 0.1, 0.6, 0.2], dtype=float)
    matrix[abs_pos[1], :] = np.asarray([1.0, 0.3, 0.2, 0.4], dtype=float)
    metadata = [
        {"name": "track_0", "strand": "+"},
        {"name": "track_1", "strand": "-"},
        {"name": "track_2", "strand": "+"},
        {"name": "track_3", "strand": "-"},
    ]
    # Positive strand columns: 0 and 2
    expected_raw = float(np.mean([0.8, 0.6, 1.0, 0.2]))

    with patch(
        "proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput(
            [_DummyAlphaGenomePredictOutput(matrix, metadata_records=metadata)]
        ),
    ):
        max_constraint = Constraint(
            inputs=[left, intron, right],
            function=alphagenome_splice_site_usage,
            function_config=_make_config(splice_pos=[100, 200], direction="max"),
        )
        min_constraint = Constraint(
            inputs=[left, intron, right],
            function=alphagenome_splice_site_usage,
            function_config=_make_config(splice_pos=[100, 200], direction="min"),
        )
        max_scores = max_constraint.evaluate()
        min_scores = min_constraint.evaluate()

    assert len(max_scores) == len(min_scores) == 1
    assert abs(max_scores[0] - (1.0 - expected_raw)) < 1e-6
    assert abs(min_scores[0] - expected_raw) < 1e-6


def test_ag_ssu_missing_strand_metadata_fails_for_positive_filter():
    left, intron, right = _make_3part_segments()
    matrix = np.zeros((GENOMIC_CONTEXT_LEN, 2), dtype=float)

    with patch(
        "proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput(
            [_DummyAlphaGenomePredictOutput(matrix, metadata_records=None)]
        ),
    ):
        constraint = Constraint(
            inputs=[left, intron, right],
            function=alphagenome_splice_site_usage,
            function_config=_make_config(splice_pos=[100], direction="min"),
        )
        with pytest.raises(ValueError, match="metadata is missing"):
            constraint.evaluate()


# --- Out of bounds ---


def test_ag_ssu_out_of_bounds_splice_pos_errors():
    left, intron, right = _make_3part_segments()
    matrix = np.zeros((GENOMIC_CONTEXT_LEN, 2), dtype=float)
    metadata = [
        {"name": "plus_0", "strand": "+"},
        {"name": "plus_1", "strand": "+"},
    ]

    with patch(
        "proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput(
            [_DummyAlphaGenomePredictOutput(matrix, metadata_records=metadata)]
        ),
    ):
        constraint = Constraint(
            inputs=[left, intron, right],
            function=alphagenome_splice_site_usage,
            function_config=_make_config(splice_pos=[TARGET_LENGTH], direction="min"),
        )
        with pytest.raises(ValueError, match="out of bounds"):
            constraint.evaluate()


# --- Metadata propagation ---


def test_ag_ssu_metadata_propagated_to_all_three_segments():
    left, intron, right = _make_3part_segments()
    config = _make_config(splice_pos=[100])
    abs_pos = _compute_absolute_pos(config)

    matrix = np.zeros((GENOMIC_CONTEXT_LEN, 3), dtype=float)
    matrix[abs_pos[0], :] = np.asarray([0.2, 0.4, 0.6], dtype=float)
    metadata = [
        {"name": "plus_0", "strand": "+"},
        {"name": "minus_1", "strand": "-"},
        {"name": "plus_2", "strand": "+"},
    ]

    with patch(
        "proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput(
            [_DummyAlphaGenomePredictOutput(matrix, metadata_records=metadata)]
        ),
    ):
        constraint = Constraint(
            inputs=[left, intron, right],
            function=alphagenome_splice_site_usage,
            function_config=config,
        )
        constraint.evaluate()

    expected_keys = {
        "ontology_terms",
        "splice_pos",
        "direction",
        "strand",
        "selected_track_count",
        "selected_track_names",
        "selected_track_strands",
        "alphagenome_splice_site_usage_raw",
        "alphagenome_splice_site_usage_score",
    }

    for seg in (left, intron, right):
        constraint_data = seg.proposal_sequences[0]._constraints_metadata[
            "alphagenome_splice_site_usage"
        ]["data"]
        assert expected_keys.issubset(set(constraint_data.keys()))
        assert constraint_data["selected_track_names"] == ["plus_0", "plus_2"]
        assert constraint_data["selected_track_strands"] == ["+", "+"]


# --- Batched inference ---


def test_ag_ssu_batched_prediction_uses_single_call():
    left_a, intron_a, right_a = _make_3part_sequences()
    left_b, intron_b, right_b = _make_3part_sequences()

    config = _make_config(splice_pos=[100], direction="min")
    abs_pos = _compute_absolute_pos(config)

    metadata = [
        {"name": "track_0", "strand": "+"},
        {"name": "track_1", "strand": "-"},
    ]
    matrix_a = np.zeros((GENOMIC_CONTEXT_LEN, 2), dtype=float)
    matrix_b = np.zeros((GENOMIC_CONTEXT_LEN, 2), dtype=float)
    matrix_a[abs_pos[0], :] = np.asarray([0.9, 0.2], dtype=float)
    matrix_b[abs_pos[0], :] = np.asarray([0.3, 0.8], dtype=float)
    expected = [0.9, 0.3]

    with patch(
        "proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage.run_alphagenome_predict_sequences",
        return_value=_DummyAlphaGenomePredictBatchOutput(
            [
                _DummyAlphaGenomePredictOutput(matrix_a, metadata_records=metadata),
                _DummyAlphaGenomePredictOutput(matrix_b, metadata_records=metadata),
            ]
        ),
    ) as mock_batch:
        scores = alphagenome_splice_site_usage(
            [(left_a, intron_a, right_a), (left_b, intron_b, right_b)],
            config,
        )

    assert scores == pytest.approx(expected, abs=1e-6)
    mock_batch.assert_called_once()


# --- Cassette integration ---


def test_integrate_cassette_into_context():
    genomic = "A" * 100
    cassette = "G" * 20
    integrated, insert_start = _integrate_cassette_into_context(genomic, cassette)
    assert len(integrated) == 100
    assert insert_start == 40
    assert integrated[40:60] == "G" * 20
    assert integrated[:40] == "A" * 40
    assert integrated[60:] == "A" * 40


def test_integrate_cassette_too_large():
    with pytest.raises(ValueError, match="exceeds context length"):
        _integrate_cassette_into_context("A" * 10, "G" * 20)
