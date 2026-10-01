"""Tests for the taggability constraint."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_language import TaggabilityConfig
from proto_language.classifiers import ClassifierEndpointError, ClassifierOutput
from proto_language.constraint import ConstraintRegistry, taggability_constraint
from proto_language.core import Segment, Sequence
from proto_language.utils import MAX_ENERGY

PATCH_TARGET = "proto_language.constraint.protein_tagging.taggability_constraint.ClassifierRegistry"

PROTEIN = "MKTAYIAKQRQISFVKSHFSRQ"
OTHER_PROTEIN = "MDDDIAALVVDNGSGMCKAGF"


def _classifier_output(taggability: float, percentile: float | None = None, **extra) -> ClassifierOutput:
    """Build a classifier output shaped like the endpoint's response."""
    return ClassifierOutput(
        score=taggability,
        metadata={
            "taggability": taggability,
            "percentile": percentile,
            "model_version": "taggability_6b_terminal_logreg",
            "truncated": False,
            **extra,
        },
    )


def _patched_registry(outputs: list[ClassifierOutput] | None = None, side_effect=None):
    """Patch the registry the constraint uses so no classifier or network runs."""
    patcher = patch(PATCH_TARGET)
    mock_registry = patcher.start()
    bound = mock_registry.create.return_value
    if side_effect is not None:
        bound.predict.side_effect = side_effect
    else:
        bound.predict.return_value = outputs
    return patcher, bound


def _inputs(*sequences: str) -> list[tuple[Sequence, ...]]:
    """Wrap protein strings as single-slot proposal tuples."""
    return [(Sequence(seq, "protein"),) for seq in sequences]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_taggability_is_registered() -> None:
    spec = ConstraintRegistry.get("taggability")
    assert spec.category == "protein_tagging"
    assert spec.supported_sequence_types == ["protein"]
    assert spec.uses_gpu is False
    assert spec.mode == "discrete"


def test_taggability_takes_a_single_protein_slot() -> None:
    """One slot named Protein keeps callers from passing a joined construct."""
    spec = ConstraintRegistry.get("taggability")
    labels = [slot if isinstance(slot, str) else slot.label for slot in spec.input_labels]
    assert labels == ["Protein"]


# ---------------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "taggability, expected_energy",
    [
        (0.25, 0.75),
        (0.38, 0.62),
        (0.64, 0.36),
        (0.83, 0.17),
        (0.93, 0.07),
        (0.97, 0.03),
    ],
)
def test_raw_mode_energy_is_one_minus_taggability(taggability: float, expected_energy: float) -> None:
    patcher, _ = _patched_registry([_classifier_output(taggability)])
    try:
        results = taggability_constraint(_inputs(PROTEIN), TaggabilityConfig(score_mode="raw"))
    finally:
        patcher.stop()

    assert results[0].score == pytest.approx(expected_energy)


@pytest.mark.parametrize(
    "taggability, percentile, expected_energy",
    [
        (0.25, 0.05, 0.95),
        (0.38, 0.10, 0.90),
        (0.64, 0.25, 0.75),
        (0.83, 0.50, 0.50),
        (0.93, 0.75, 0.25),
        (0.97, 0.90, 0.10),
    ],
)
def test_percentile_mode_energy_is_one_minus_percentile(
    taggability: float, percentile: float, expected_energy: float
) -> None:
    """The vendor's published score-to-percentile table, used as fixtures."""
    patcher, _ = _patched_registry([_classifier_output(taggability, percentile)])
    try:
        results = taggability_constraint(_inputs(PROTEIN), TaggabilityConfig(score_mode="percentile"))
    finally:
        patcher.stop()

    assert results[0].score == pytest.approx(expected_energy)


def test_percentile_mode_without_percentile_scores_worst() -> None:
    patcher, _ = _patched_registry([_classifier_output(0.9, percentile=None)])
    try:
        results = taggability_constraint(_inputs(PROTEIN), TaggabilityConfig(score_mode="percentile"))
    finally:
        patcher.stop()

    assert results[0].score == MAX_ENERGY


def test_default_score_mode_is_raw() -> None:
    assert TaggabilityConfig().score_mode == "raw"


def test_energy_is_clamped_into_unit_range() -> None:
    patcher, _ = _patched_registry([_classifier_output(1.5), _classifier_output(-0.5)])
    try:
        results = taggability_constraint(_inputs(PROTEIN, OTHER_PROTEIN), TaggabilityConfig())
    finally:
        patcher.stop()

    assert results[0].score == 0.0
    assert results[1].score == MAX_ENERGY


# ---------------------------------------------------------------------------
# Inputs, metadata, and error handling
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty() -> None:
    assert taggability_constraint([], TaggabilityConfig()) == []


def test_only_the_protein_sequence_is_sent() -> None:
    patcher, bound = _patched_registry([_classifier_output(0.9)])
    try:
        taggability_constraint(_inputs(PROTEIN), TaggabilityConfig())
    finally:
        patcher.stop()

    bound.predict.assert_called_once_with([PROTEIN])


def test_metadata_carries_raw_percentile_and_provenance() -> None:
    patcher, _ = _patched_registry([_classifier_output(0.83, 0.50)])
    try:
        results = taggability_constraint(_inputs(PROTEIN), TaggabilityConfig(score_mode="raw"))
    finally:
        patcher.stop()

    metadata = results[0].metadata
    assert metadata["taggability"] == pytest.approx(0.83)
    assert metadata["percentile"] == pytest.approx(0.50)
    assert metadata["model_version"] == "taggability_6b_terminal_logreg"
    assert metadata["taggability_energy"] == pytest.approx(0.17)
    assert metadata["score_mode"] == "raw"


def test_invalid_sequence_scores_worst_without_sinking_the_batch() -> None:
    outputs = [
        _classifier_output(0.9),
        ClassifierOutput(score=0.0, metadata={"taggability_invalid": True, "taggability_error_detail": "too short"}),
        _classifier_output(0.8),
    ]
    patcher, _ = _patched_registry(outputs)
    try:
        results = taggability_constraint(_inputs(PROTEIN, "MKT", OTHER_PROTEIN), TaggabilityConfig())
    finally:
        patcher.stop()

    assert results[0].score == pytest.approx(0.1)
    assert results[1].score == MAX_ENERGY
    assert results[1].metadata["taggability_invalid"] is True
    assert results[2].score == pytest.approx(0.2)


def test_endpoint_failure_scores_batch_worst_when_not_strict() -> None:
    patcher, _ = _patched_registry(side_effect=ClassifierEndpointError("endpoint down"))
    try:
        results = taggability_constraint(_inputs(PROTEIN, OTHER_PROTEIN), TaggabilityConfig(strict=False))
    finally:
        patcher.stop()

    assert [r.score for r in results] == [MAX_ENERGY, MAX_ENERGY]
    assert all(r.metadata["classifier_error"] is True for r in results)
    assert "endpoint down" in results[0].metadata["classifier_error_message"]


def test_endpoint_failure_raises_when_strict() -> None:
    patcher, _ = _patched_registry(side_effect=ClassifierEndpointError("endpoint down"))
    try:
        with pytest.raises(ClassifierEndpointError, match="endpoint down"):
            taggability_constraint(_inputs(PROTEIN), TaggabilityConfig(strict=True))
    finally:
        patcher.stop()


# ---------------------------------------------------------------------------
# Config validation and framework integration
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_score_mode() -> None:
    with pytest.raises(ValidationError):
        TaggabilityConfig(score_mode="percentiles")


def test_config_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValidationError):
        TaggabilityConfig(timeout_s=0.0)


def test_config_rejects_negative_retries() -> None:
    with pytest.raises(ValidationError):
        TaggabilityConfig(max_retries=-1)


def test_classifier_config_dict_omits_constraint_only_fields() -> None:
    config_dict = TaggabilityConfig(score_mode="percentile", strict=True).classifier_config_dict()
    assert "score_mode" not in config_dict
    assert "strict" not in config_dict
    assert set(config_dict) == {"base_url", "api_key_env", "timeout_s", "max_retries", "cache_size"}


def test_wrong_sequence_type_is_rejected() -> None:
    dna_segment = Segment(length=30, sequence_type="dna", label="dna_segment")
    with pytest.raises(TypeError, match="does not support sequence type"):
        ConstraintRegistry.create("taggability", [dna_segment], {})


def test_metadata_propagates_to_proposal_sequences() -> None:
    segment = Segment(length=len(PROTEIN), sequence_type="protein", label="protein_segment")
    segment.proposal_sequences = [Sequence(PROTEIN, "protein")]
    constraint = ConstraintRegistry.create("taggability", [segment], {})

    patcher, _ = _patched_registry([_classifier_output(0.83, 0.50)])
    try:
        scores = constraint.evaluate()
    finally:
        patcher.stop()

    assert scores[0] == pytest.approx(0.17)
    data = segment.proposal_sequences[0].metadata["constraints"][constraint.label]["data"]
    assert data["taggability"] == pytest.approx(0.83)
    assert data["percentile"] == pytest.approx(0.50)
