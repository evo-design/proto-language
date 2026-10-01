"""Trained predictors that map sequences to continuous scores."""

from proto_language.classifiers.base import (
    Classifier,
    ClassifierEndpointError,
    ClassifierError,
    ClassifierInvalidSequenceError,
    ClassifierOutput,
)
from proto_language.classifiers.classifier_registry import (
    ClassifierRegistry,
    ClassifierSpec,
    classifier,
)
from proto_language.classifiers.protein_tagging import (
    GFPTaggabilityConfig,
    gfp_taggability_classifier,
)

__all__ = [
    "Classifier",
    "ClassifierEndpointError",
    "ClassifierError",
    "ClassifierInvalidSequenceError",
    "ClassifierOutput",
    "ClassifierRegistry",
    "ClassifierSpec",
    "GFPTaggabilityConfig",
    "classifier",
    "gfp_taggability_classifier",
]
