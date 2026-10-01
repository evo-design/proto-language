"""Taggability classifier: predicts whether a fluorescent knock-in tag will succeed.

Scores a bare protein sequence through a remote endpoint that embeds it with
ESM-C 6B and applies a logistic regression trained on 1,757 OpenCell
split-mNeonGreen knock-in attempts. The native score is ``P(success)`` in
``[0, 1]``; the endpoint also reports where that score falls among its reference
proteins, which matters because the raw scale is not centred on 0.5.

The model is trained on bare proteins, so callers must pass the protein alone —
not a tag/linker construct, whose extra residues shift the pooled embedding off
the training distribution.

Examples:
    >>> classifier = ClassifierRegistry.create("gfp-taggability", {})  # doctest: +SKIP
    >>> classifier.predict_scores(["MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIV"])  # doctest: +SKIP
    [0.9772]
"""

import logging
from typing import Any

from proto_language.classifiers.base import (
    ClassifierEndpointError,
    ClassifierInvalidSequenceError,
    ClassifierOutput,
)
from proto_language.classifiers.classifier_registry import classifier
from proto_language.classifiers.endpoints import ResponseCache, score_many
from proto_language.utils.base import BaseConfig, ConfigField

__all__ = ["GFPTaggabilityConfig", "gfp_taggability_classifier"]

logger = logging.getLogger(__name__)

TAGGABILITY_BASE_URL_ENV = "FLUORESCE_API_URL"
"""Environment variable naming the taggability endpoint base URL."""

TAGGABILITY_API_KEY_ENV = "FLUORESCE_KEY"
"""Environment variable holding the taggability endpoint bearer token."""

TAGGABILITY_SCORE_PATH = "/v1/score"
"""Endpoint path that scores a single protein sequence."""


class GFPTaggabilityConfig(BaseConfig):
    """Configuration for the remote taggability classifier.

    The served deployment exposes one fixed model and takes no model-selection
    parameters, so the returned ``model_version`` is recorded as metadata rather
    than requested: scores are only comparable within a single model version.

    Attributes:
        base_url (str | None): Endpoint base URL. When None, read from the
            ``FLUORESCE_API_URL`` environment variable.
        api_key_env (str): Environment variable holding the bearer token.
        timeout_s (float): Per-request timeout in seconds. The default absorbs the
            one-time model load on the first request after a restart.
        max_retries (int): Additional attempts after a server error or timeout.
        cache_size (int): Retained cached responses keyed by sequence; 0 disables.
    """

    base_url: str | None = ConfigField(
        default=None,
        title="Endpoint Base URL",
        description="Taggability endpoint base URL. When unset, read from FLUORESCE_API_URL.",
    )
    api_key_env: str = ConfigField(
        default=TAGGABILITY_API_KEY_ENV,
        title="API Key Env Var",
        description="Environment variable holding the endpoint bearer token.",
    )
    timeout_s: float = ConfigField(
        default=60.0,
        title="Request Timeout",
        description="Per-request timeout in seconds; absorbs the cold-start model load.",
        gt=0.0,
    )
    max_retries: int = ConfigField(
        default=1,
        title="Max Retries",
        description="Additional attempts after a server error or timeout.",
        ge=0,
    )
    cache_size: int = ConfigField(
        default=1024,
        title="Cache Size",
        description="Retained cached responses keyed by sequence; 0 disables caching.",
        ge=0,
    )


def _output_from_payload(payload: dict[str, Any]) -> ClassifierOutput:
    """Build a :class:`ClassifierOutput` from one endpoint response body.

    Args:
        payload (dict[str, Any]): Decoded response body for a single sequence.

    Returns:
        ClassifierOutput: Native taggability score plus provenance metadata.

    Raises:
        ClassifierEndpointError: If the response omits the taggability score.
    """
    if "taggability" not in payload:
        raise ClassifierEndpointError(f"Taggability response missing 'taggability' field: {sorted(payload)}")

    sequence_info = payload.get("sequence") or {}
    truncated = bool(sequence_info.get("truncated", False))
    ambiguous_residues = list(sequence_info.get("ambiguous_residues") or [])

    if truncated:
        logger.warning(
            "Taggability input was middle-truncated to %s residues; the score reflects the truncated sequence.",
            sequence_info.get("scored_length"),
        )
    if ambiguous_residues:
        logger.warning(
            "Taggability input contains non-standard residues %s absent from the training set.",
            ambiguous_residues,
        )

    return ClassifierOutput(
        score=float(payload["taggability"]),
        metadata={
            "taggability": float(payload["taggability"]),
            "percentile": payload.get("percentile"),
            "model_version": payload.get("model_version"),
            "scored_length": sequence_info.get("scored_length"),
            "truncated": truncated,
            "ambiguous_residues": ambiguous_residues or None,
            "elapsed_ms": payload.get("elapsed_ms"),
        },
    )


def _invalid_output(error: ClassifierInvalidSequenceError) -> ClassifierOutput:
    """Represent an endpoint-rejected sequence as a zero-score output.

    Args:
        error (ClassifierInvalidSequenceError): Rejection raised for this sequence.

    Returns:
        ClassifierOutput: Score 0.0 flagged as invalid, so wrapping constraints can
            penalize this proposal without discarding the rest of the batch.
    """
    return ClassifierOutput(
        score=0.0,
        metadata={
            "taggability": None,
            "percentile": None,
            "taggability_invalid": True,
            "taggability_error_detail": error.detail or str(error),
        },
    )


@classifier(
    key="gfp-taggability",
    label="Taggability",
    config=GFPTaggabilityConfig,
    description="Predict P(fluorescent knock-in tag succeeds) for a protein via a remote endpoint.",
    output_description="P(knock-in tagging succeeds) under the OpenCell split-mNeonGreen library",
    uses_gpu=False,
    category="protein_tagging",
    supported_sequence_types=["protein"],
    output_range=(0.0, 1.0),
    endpoint_env_var=TAGGABILITY_BASE_URL_ENV,
)
def gfp_taggability_classifier(
    sequences: list[str],
    config: GFPTaggabilityConfig,
) -> list[ClassifierOutput]:
    """Score bare protein sequences for fluorescent-tag compatibility.

    Args:
        sequences (list[str]): Bare protein sequences, without tag or linker residues.
        config (GFPTaggabilityConfig): Validated configuration object.

    Returns:
        list[ClassifierOutput]: One output per input, order-aligned. ``score`` is
            ``P(success)`` in ``[0, 1]``; sequences the endpoint rejects score 0.0
            and carry ``taggability_invalid`` in metadata.

    Raises:
        ClassifierEndpointError: If the endpoint is unreachable, unauthorized, or
            fails after exhausting retries.
    """
    if not sequences:
        return []

    cache = _cache_for(config)
    payloads = score_many(
        sequences,
        base_url=config.base_url,
        base_url_env=TAGGABILITY_BASE_URL_ENV,
        api_key_env=config.api_key_env,
        path=TAGGABILITY_SCORE_PATH,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
        cache=cache,
    )

    return [
        _invalid_output(payload)
        if isinstance(payload, ClassifierInvalidSequenceError)
        else _output_from_payload(payload)
        for payload in payloads
    ]


_CACHES: dict[tuple[str, str], ResponseCache] = {}


def _cache_for(config: GFPTaggabilityConfig) -> ResponseCache | None:
    """Return the response cache for ``config``'s endpoint, creating it on first use.

    Caches are keyed by endpoint identity rather than config identity: a response
    depends only on the sequence and the endpoint serving it, so two configs that
    differ solely in timeout or retry budget can safely share one. Keying on
    endpoint also avoids reusing a cache across distinct deployments.

    Args:
        config (GFPTaggabilityConfig): Validated configuration object.

    Returns:
        ResponseCache | None: Cache for this endpoint, or None when disabled.
    """
    if config.cache_size <= 0:
        return None
    cache_key = (config.base_url or "", config.api_key_env)
    cache = _CACHES.get(cache_key)
    if cache is None or cache.max_size != config.cache_size:
        cache = ResponseCache(max_size=config.cache_size)
        _CACHES[cache_key] = cache
    return cache
