"""Tests for the classifier registry."""

import pytest

from proto_language.classifiers import (
    Classifier,
    ClassifierOutput,
    ClassifierRegistry,
    GFPTaggabilityConfig,
    classifier,
)
from proto_language.utils.base import BaseConfig, ConfigField


class _DummyConfig(BaseConfig):
    """Config for the dummy classifier used in registry tests.

    Attributes:
        offset (float): Constant added to every returned score.
    """

    offset: float = ConfigField(
        default=0.0,
        title="Offset",
        description="Constant added to every returned score.",
    )


@pytest.fixture
def clean_registry():
    """Snapshot and restore the registry so test registrations do not leak."""
    snapshot = ClassifierRegistry.snapshot()
    yield
    ClassifierRegistry.restore(snapshot)


def test_gfp_taggability_is_registered() -> None:
    spec = ClassifierRegistry.get("gfp-taggability")
    assert spec.category == "protein_tagging"
    assert spec.supported_sequence_types == ["protein"]
    assert spec.output_range == (0.0, 1.0)
    assert spec.endpoint_env_var == "FLUORESCE_API_URL"
    assert spec.uses_gpu is False


def test_get_unknown_key_lists_available() -> None:
    with pytest.raises(ValueError, match="Unknown classifier"):
        ClassifierRegistry.get("does-not-exist")


def test_register_and_create_roundtrip(clean_registry) -> None:
    @classifier(
        key="dummy-classifier",
        label="Dummy",
        config=_DummyConfig,
        description="Return a constant score for every sequence.",
        output_description="Constant score",
        supported_sequence_types=["protein"],
    )
    def dummy_classifier(sequences: list[str], config: _DummyConfig) -> list[ClassifierOutput]:
        return [ClassifierOutput(score=config.offset) for _ in sequences]

    bound = ClassifierRegistry.create("dummy-classifier", {"offset": 0.25})
    assert isinstance(bound, Classifier)
    assert bound.predict_scores(["MKT", "MKA"]) == [0.25, 0.25]


def test_register_rejects_duplicate_key(clean_registry) -> None:
    with pytest.raises(ValueError, match="already registered"):

        @classifier(
            key="gfp-taggability",
            label="Duplicate",
            config=_DummyConfig,
            description="Duplicate registration should fail.",
            output_description="Constant score",
            supported_sequence_types=["protein"],
        )
        def duplicate_classifier(sequences: list[str], config: _DummyConfig) -> list[ClassifierOutput]:
            return []


def test_register_rejects_empty_sequence_types(clean_registry) -> None:
    with pytest.raises(ValueError, match="supported_sequence_types must be non-empty"):

        @classifier(
            key="no-types-classifier",
            label="No Types",
            config=_DummyConfig,
            description="Registration without sequence types should fail.",
            output_description="Constant score",
            supported_sequence_types=[],
        )
        def no_types_classifier(sequences: list[str], config: _DummyConfig) -> list[ClassifierOutput]:
            return []


def test_create_reformats_validation_error() -> None:
    with pytest.raises(ValueError, match="classifier 'gfp-taggability' config invalid"):
        ClassifierRegistry.create("gfp-taggability", {"timeout_s": -1.0})


def test_create_rejects_unknown_config_field() -> None:
    with pytest.raises(ValueError, match="config invalid"):
        ClassifierRegistry.create("gfp-taggability", {"not_a_field": 1})


def test_get_schema_exposes_titles_and_descriptions() -> None:
    schema = ClassifierRegistry.get_schema("gfp-taggability")
    for name, prop in schema["properties"].items():
        assert prop.get("title"), f"{name} is missing a title"


def test_predict_returns_empty_for_empty_input() -> None:
    bound = ClassifierRegistry.create("gfp-taggability", {})
    assert bound.predict([]) == []
    assert bound.predict_scores([]) == []


def test_predict_raises_on_length_mismatch(clean_registry) -> None:
    @classifier(
        key="short-classifier",
        label="Short",
        config=_DummyConfig,
        description="Return fewer results than inputs.",
        output_description="Constant score",
        supported_sequence_types=["protein"],
    )
    def short_classifier(sequences: list[str], config: _DummyConfig) -> list[ClassifierOutput]:
        return [ClassifierOutput(score=0.0)]

    bound = ClassifierRegistry.create("short-classifier", {})
    with pytest.raises(ValueError, match="returned 1 results for 2 sequences"):
        bound.predict(["MKT", "MKA"])


def test_snapshot_restore_round_trips(clean_registry) -> None:
    before = ClassifierRegistry.count()
    ClassifierRegistry.unregister("gfp-taggability")
    assert ClassifierRegistry.count() == before - 1


def test_taggability_config_defaults() -> None:
    config = GFPTaggabilityConfig()
    assert config.base_url is None
    assert config.api_key_env == "FLUORESCE_KEY"
    assert config.max_retries == 1
    assert config.cache_size == 1024
