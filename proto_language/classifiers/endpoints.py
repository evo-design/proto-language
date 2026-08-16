"""Shared HTTP client for classifiers backed by a remote scoring endpoint.

This is the only module in ``proto_language`` that performs network I/O. It owns
credential resolution, status-code semantics, bounded retries, connection reuse,
and an optional per-call-site response cache, so individual classifiers stay a
thin mapping from response JSON to :class:`ClassifierOutput`.

Endpoint base URLs are never hard-coded as defaults: hosted inference URLs are
frequently ephemeral, so a baked-in default would silently rot. A URL resolves
from an explicit config value, then from the environment, then raises.

Examples:
    >>> resolve_base_url(None, "FLUORESCE_API_URL")  # doctest: +SKIP
    'https://example.invalid'
"""

import logging
import os
from collections import OrderedDict
from typing import Any

import requests

from proto_language.classifiers.base import (
    ClassifierEndpointError,
    ClassifierInvalidSequenceError,
)

__all__ = [
    "ResponseCache",
    "post_score",
    "resolve_base_url",
    "score_many",
    "session",
]

logger = logging.getLogger(__name__)

_SESSION: requests.Session | None = None


def session() -> requests.Session:
    """Return the module-level :class:`requests.Session`, creating it on first use.

    A single session keeps the TLS connection warm across the sequential scoring
    loop, which is where most of the per-request wall time would otherwise go.

    Returns:
        requests.Session: Shared session for all classifier endpoint calls.
    """
    global _SESSION  # noqa: PLW0603 -- one lazily created connection pool per process
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def resolve_base_url(base_url: str | None, env_var: str) -> str:
    """Resolve the endpoint base URL from config, then environment.

    Args:
        base_url (str | None): Explicit base URL from configuration, if any.
        env_var (str): Environment variable to fall back to.

    Returns:
        str: Base URL with any trailing slash removed.

    Raises:
        ClassifierEndpointError: If neither source supplies a URL.
    """
    resolved = base_url or os.environ.get(env_var)
    if not resolved:
        raise ClassifierEndpointError(
            f"No classifier endpoint configured. Set the 'base_url' config field or the {env_var} environment variable."
        )
    return resolved.rstrip("/")


def _resolve_api_key(api_key_env: str) -> str:
    """Read the bearer token from the environment.

    Args:
        api_key_env (str): Environment variable holding the bearer token.

    Returns:
        str: The bearer token.

    Raises:
        ClassifierEndpointError: If the variable is unset or empty.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ClassifierEndpointError(
            f"No classifier API key found. Set the {api_key_env} environment variable to the bearer token."
        )
    return api_key


def _error_detail(response: requests.Response) -> str:
    """Extract the endpoint's ``detail`` field, falling back to the raw body."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:200]


def post_score(
    sequence: str,
    *,
    base_url: str,
    api_key: str,
    path: str,
    timeout_s: float,
    max_retries: int,
) -> dict[str, Any]:
    """POST one sequence to the scoring endpoint and return the decoded response.

    Retry policy follows the endpoint contract: server errors and timeouts are
    transient and retried up to ``max_retries`` times; authentication failures and
    sequence rejections are terminal and raise immediately.

    Args:
        sequence (str): Sequence to score.
        base_url (str): Endpoint base URL, without a trailing slash.
        api_key (str): Bearer token.
        path (str): Endpoint path, e.g. ``"/v1/score"``.
        timeout_s (float): Per-request timeout in seconds.
        max_retries (int): Additional attempts after a transient failure.

    Returns:
        dict[str, Any]: Decoded JSON response body.

    Raises:
        ClassifierInvalidSequenceError: If the endpoint rejects the sequence (422).
        ClassifierEndpointError: On authentication failure, exhausted retries, or
            an undecodable response.
    """
    url = f"{base_url}{path}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            response = session().post(url, headers=headers, json={"sequence": sequence}, timeout=timeout_s)
        except requests.Timeout as exc:
            last_error = f"request timed out after {timeout_s}s"
            logger.warning("Classifier endpoint timeout (attempt %d/%d)", attempt + 1, max_retries + 1)
            if attempt == max_retries:
                raise ClassifierEndpointError(f"Classifier endpoint {url} failed: {last_error}") from exc
            continue
        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning("Classifier endpoint transport error (attempt %d/%d)", attempt + 1, max_retries + 1)
            if attempt == max_retries:
                raise ClassifierEndpointError(f"Classifier endpoint {url} failed: {last_error}") from exc
            continue

        if response.status_code == 401:
            raise ClassifierEndpointError(
                f"Classifier endpoint {url} rejected the bearer token (401). Check the API key environment variable."
            )
        if response.status_code == 422:
            detail = _error_detail(response)
            raise ClassifierInvalidSequenceError(f"Classifier endpoint rejected the sequence (422): {detail}", detail)
        if response.status_code >= 500:
            last_error = f"HTTP {response.status_code}: {_error_detail(response)}"
            logger.warning("Classifier endpoint server error (attempt %d/%d)", attempt + 1, max_retries + 1)
            if attempt == max_retries:
                raise ClassifierEndpointError(f"Classifier endpoint {url} failed: {last_error}")
            continue
        if not response.ok:
            raise ClassifierEndpointError(
                f"Classifier endpoint {url} returned HTTP {response.status_code}: {_error_detail(response)}"
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ClassifierEndpointError(f"Classifier endpoint {url} returned a non-JSON response.") from exc
        return payload

    raise ClassifierEndpointError(f"Classifier endpoint {url} failed: {last_error}")


class ResponseCache:
    """Bounded insertion-ordered cache of endpoint responses, keyed by sequence.

    Hit rates are near zero while an optimizer mutates the scored sequence every
    proposal. The cache earns its keep on multi-stage programs, which reuse the
    same construct objects across optimizers, and on re-runs that vary only a
    seed or an unrelated constraint.

    Attributes:
        max_size (int): Maximum retained entries; ``0`` disables caching entirely.

    Examples:
        >>> cache = ResponseCache(max_size=2)
        >>> cache.put("MKT", {"taggability": 0.5})
        >>> cache.get("MKT")
        {'taggability': 0.5}
    """

    def __init__(self, max_size: int) -> None:
        """Create a cache retaining at most ``max_size`` entries.

        Args:
            max_size (int): Maximum retained entries; ``0`` disables caching.
        """
        self.max_size = max_size
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached response for ``key``, or None on a miss."""
        if self.max_size <= 0:
            return None
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store ``value`` under ``key``, evicting the oldest entry when full."""
        if self.max_size <= 0:
            return
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)


def score_many(
    sequences: list[str],
    *,
    base_url: str | None,
    base_url_env: str,
    api_key_env: str,
    path: str,
    timeout_s: float,
    max_retries: int,
    cache: ResponseCache | None = None,
) -> list[dict[str, Any] | ClassifierInvalidSequenceError]:
    """Score sequences one at a time, preserving input order.

    The endpoint accepts a single sequence per request and processes requests
    serially, so this loops rather than fanning out concurrently. A sequence the
    endpoint rejects yields its :class:`ClassifierInvalidSequenceError` in place
    of a response, letting callers penalize that one input and keep the rest.

    Args:
        sequences (list[str]): Sequences to score.
        base_url (str | None): Explicit base URL, or None to read the environment.
        base_url_env (str): Environment variable naming the base URL.
        api_key_env (str): Environment variable holding the bearer token.
        path (str): Endpoint path, e.g. ``"/v1/score"``.
        timeout_s (float): Per-request timeout in seconds.
        max_retries (int): Additional attempts after a transient failure.
        cache (ResponseCache | None): Optional response cache.

    Returns:
        list[dict[str, Any] | ClassifierInvalidSequenceError]: One entry per input
            sequence, order-aligned; rejections appear as the raised error.

    Raises:
        ClassifierEndpointError: If the endpoint is unreachable, unauthorized, or
            fails after exhausting retries.
    """
    if not sequences:
        return []

    resolved_base_url = resolve_base_url(base_url, base_url_env)
    api_key = _resolve_api_key(api_key_env)

    results: list[dict[str, Any] | ClassifierInvalidSequenceError] = []
    for sequence in sequences:
        cached = cache.get(sequence) if cache is not None else None
        if cached is not None:
            results.append(cached)
            continue
        try:
            payload = post_score(
                sequence,
                base_url=resolved_base_url,
                api_key=api_key,
                path=path,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
        except ClassifierInvalidSequenceError as exc:
            results.append(exc)
            continue
        if cache is not None:
            cache.put(sequence, payload)
        results.append(payload)
    return results
