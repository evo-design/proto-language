"""Tests for the taggability classifier and its HTTP endpoint client.

No test performs real network I/O: ``requests.Session.post`` is patched at the
module the client calls it from.
"""

from importlib import import_module
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from proto_language.classifiers import (
    ClassifierEndpointError,
    ClassifierRegistry,
    GFPTaggabilityConfig,
    gfp_taggability_classifier,
)
from proto_language.classifiers.endpoints import ResponseCache, resolve_base_url, score_many

SESSION_TARGET = "proto_language.classifiers.endpoints.session"

BASE_URL = "https://taggability.invalid"
BETA_ACTIN_SCORE = 0.9772
BETA_ACTIN_PERCENTILE = 0.936


def _response(status_code: int = 200, payload: dict[str, Any] | None = None) -> MagicMock:
    """Build a mock ``requests.Response`` with the given status and JSON body."""
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.json.return_value = payload if payload is not None else {}
    response.text = ""
    return response


def _score_payload(
    taggability: float = BETA_ACTIN_SCORE,
    percentile: float | None = BETA_ACTIN_PERCENTILE,
    truncated: bool = False,
    ambiguous: list[str] | None = None,
) -> dict[str, Any]:
    """Build a successful ``/v1/score`` response body."""
    return {
        "taggability": taggability,
        "percentile": percentile,
        "model_version": "taggability_6b_terminal_logreg",
        "sequence": {
            "length": 375,
            "scored_length": 375,
            "truncated": truncated,
            "ambiguous_residues": ambiguous or [],
        },
        "elapsed_ms": 56,
    }


@pytest.fixture
def endpoint_env(monkeypatch):
    """Populate the endpoint URL and API key environment variables."""
    monkeypatch.setenv("FLUORESCE_API_URL", BASE_URL)
    monkeypatch.setenv("FLUORESCE_KEY", "test-token-not-a-real-credential")


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset the module-level response caches between tests.

    Resolved via ``import_module`` because the package re-exports the predict
    function under the same name as its module, which shadows attribute access.
    """
    module = import_module("proto_language.classifiers.protein_tagging.gfp_taggability_classifier")

    module._CACHES.clear()
    yield
    module._CACHES.clear()


# ---------------------------------------------------------------------------
# URL and credential resolution
# ---------------------------------------------------------------------------


def test_resolve_base_url_prefers_config_over_env(monkeypatch) -> None:
    monkeypatch.setenv("FLUORESCE_API_URL", "https://from-env.invalid")
    assert resolve_base_url("https://from-config.invalid", "FLUORESCE_API_URL") == "https://from-config.invalid"


def test_resolve_base_url_strips_trailing_slash(monkeypatch) -> None:
    monkeypatch.delenv("FLUORESCE_API_URL", raising=False)
    assert resolve_base_url("https://x.invalid/", "FLUORESCE_API_URL") == "https://x.invalid"


def test_resolve_base_url_error_names_config_and_env(monkeypatch) -> None:
    monkeypatch.delenv("FLUORESCE_API_URL", raising=False)
    with pytest.raises(ClassifierEndpointError, match=r"base_url.*FLUORESCE_API_URL"):
        resolve_base_url(None, "FLUORESCE_API_URL")


def test_missing_api_key_raises_naming_env_var(monkeypatch) -> None:
    monkeypatch.setenv("FLUORESCE_API_URL", BASE_URL)
    monkeypatch.delenv("FLUORESCE_KEY", raising=False)
    with pytest.raises(ClassifierEndpointError, match="FLUORESCE_KEY"):
        gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig())


# ---------------------------------------------------------------------------
# Status-code semantics
# ---------------------------------------------------------------------------


def test_successful_score_maps_payload_to_output(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(200, _score_payload())

    with patch(SESSION_TARGET, return_value=mock_session):
        outputs = gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig())

    assert len(outputs) == 1
    assert outputs[0].score == pytest.approx(BETA_ACTIN_SCORE)
    assert outputs[0].metadata["percentile"] == pytest.approx(BETA_ACTIN_PERCENTILE)
    assert outputs[0].metadata["model_version"] == "taggability_6b_terminal_logreg"
    assert outputs[0].metadata["truncated"] is False


def test_request_sends_bearer_token_and_single_sequence(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(200, _score_payload())

    with patch(SESSION_TARGET, return_value=mock_session):
        gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig())

    _, kwargs = mock_session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-token-not-a-real-credential"
    assert kwargs["json"] == {"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}


def test_401_raises_without_retry(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(401, {"detail": "bad token"})

    with patch(SESSION_TARGET, return_value=mock_session):
        with pytest.raises(ClassifierEndpointError, match="401"):
            gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig(max_retries=3))

    assert mock_session.post.call_count == 1


def test_422_yields_invalid_output_without_retry(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(422, {"detail": "sequence shorter than 20 aa"})

    with patch(SESSION_TARGET, return_value=mock_session):
        outputs = gfp_taggability_classifier(["MKT"], GFPTaggabilityConfig(max_retries=3))

    assert mock_session.post.call_count == 1
    assert outputs[0].metadata["taggability_invalid"] is True
    assert outputs[0].metadata["taggability_error_detail"] == "sequence shorter than 20 aa"


def test_500_retries_once_then_raises(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(500, {"detail": "boom"})

    with patch(SESSION_TARGET, return_value=mock_session):
        with pytest.raises(ClassifierEndpointError, match="500"):
            gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig(max_retries=1))

    assert mock_session.post.call_count == 2


def test_500_then_success_recovers(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.side_effect = [_response(500, {}), _response(200, _score_payload())]

    with patch(SESSION_TARGET, return_value=mock_session):
        outputs = gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig(max_retries=1))

    assert mock_session.post.call_count == 2
    assert outputs[0].score == pytest.approx(BETA_ACTIN_SCORE)


def test_timeout_retries_then_raises(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.side_effect = requests.Timeout("timed out")

    with patch(SESSION_TARGET, return_value=mock_session):
        with pytest.raises(ClassifierEndpointError, match="timed out"):
            gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig(max_retries=2))

    assert mock_session.post.call_count == 3


def test_missing_taggability_field_raises(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(200, {"percentile": 0.5})

    with patch(SESSION_TARGET, return_value=mock_session):
        with pytest.raises(ClassifierEndpointError, match="missing 'taggability'"):
            gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig())


# ---------------------------------------------------------------------------
# Batching, ordering, and caching
# ---------------------------------------------------------------------------


def test_results_preserve_input_order(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.side_effect = [
        _response(200, _score_payload(taggability=0.10)),
        _response(200, _score_payload(taggability=0.50)),
        _response(200, _score_payload(taggability=0.90)),
    ]

    with patch(SESSION_TARGET, return_value=mock_session):
        outputs = gfp_taggability_classifier(
            ["MKTAYIAKQRQISFVKSHFA", "MKTAYIAKQRQISFVKSHFB", "MKTAYIAKQRQISFVKSHFC"],
            GFPTaggabilityConfig(cache_size=0),
        )

    assert [o.score for o in outputs] == [pytest.approx(0.10), pytest.approx(0.50), pytest.approx(0.90)]


def test_one_invalid_sequence_does_not_sink_the_batch(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.side_effect = [
        _response(200, _score_payload(taggability=0.80)),
        _response(422, {"detail": "sequence shorter than 20 aa"}),
        _response(200, _score_payload(taggability=0.60)),
    ]

    with patch(SESSION_TARGET, return_value=mock_session):
        outputs = gfp_taggability_classifier(
            ["MKTAYIAKQRQISFVKSHFA", "MKT", "MKTAYIAKQRQISFVKSHFC"],
            GFPTaggabilityConfig(cache_size=0),
        )

    assert outputs[0].score == pytest.approx(0.80)
    assert outputs[1].metadata["taggability_invalid"] is True
    assert outputs[2].score == pytest.approx(0.60)


def test_cache_avoids_a_second_request(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(200, _score_payload())
    config = GFPTaggabilityConfig(cache_size=16)

    with patch(SESSION_TARGET, return_value=mock_session):
        gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], config)
        gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], config)

    assert mock_session.post.call_count == 1


def test_cache_size_zero_disables_caching(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(200, _score_payload())
    config = GFPTaggabilityConfig(cache_size=0)

    with patch(SESSION_TARGET, return_value=mock_session):
        gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], config)
        gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], config)

    assert mock_session.post.call_count == 2


def test_response_cache_evicts_oldest_entry() -> None:
    cache = ResponseCache(max_size=2)
    cache.put("a", {"taggability": 0.1})
    cache.put("b", {"taggability": 0.2})
    cache.put("c", {"taggability": 0.3})

    assert cache.get("a") is None
    assert cache.get("b") == {"taggability": 0.2}
    assert cache.get("c") == {"taggability": 0.3}


def test_score_many_returns_empty_for_empty_input() -> None:
    assert (
        score_many(
            [],
            base_url=None,
            base_url_env="FLUORESCE_API_URL",
            api_key_env="FLUORESCE_KEY",
            path="/v1/score",
            timeout_s=1.0,
            max_retries=0,
        )
        == []
    )


# ---------------------------------------------------------------------------
# Diagnostics and the external predictor contract
# ---------------------------------------------------------------------------


def test_truncation_and_ambiguous_residues_surface_in_metadata(endpoint_env) -> None:
    mock_session = MagicMock()
    mock_session.post.return_value = _response(200, _score_payload(truncated=True, ambiguous=["X"]))

    with patch(SESSION_TARGET, return_value=mock_session):
        outputs = gfp_taggability_classifier(["MKTAYIAKQRQISFVKSHFSRQ"], GFPTaggabilityConfig())

    assert outputs[0].metadata["truncated"] is True
    assert outputs[0].metadata["ambiguous_residues"] == ["X"]


def test_predict_scores_matches_external_predictor_contract(endpoint_env) -> None:
    """``predict_scores`` must be a batched ``Callable[[list[str]], list[float]]``."""
    mock_session = MagicMock()
    mock_session.post.side_effect = [
        _response(200, _score_payload(taggability=0.42)),
        _response(200, _score_payload(taggability=0.84)),
    ]

    bound = ClassifierRegistry.create("gfp-taggability", {"cache_size": 0})
    with patch(SESSION_TARGET, return_value=mock_session):
        scores = bound.predict_scores(["MKTAYIAKQRQISFVKSHFA", "MKTAYIAKQRQISFVKSHFB"])

    assert isinstance(scores, list)
    assert all(isinstance(score, float) for score in scores)
    assert scores == [pytest.approx(0.42), pytest.approx(0.84)]
