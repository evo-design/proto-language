"""
proto_language/storage/resolver.py

Utilities for resolving and caching GCS file references to local paths.

Used primarily for downloading pre-existing GCS-hosted databases (e.g., MMseqs DBs)
to a local cache directory for tool execution.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# storage volume mount point (set via environment variable or default)
VOLUME_PATH = Path(os.environ.get("STORAGE_VOLUME_PATH", "/data"))


def get_cache_path(reference: str) -> Path:
    """Get the path in the storage volume for a given reference."""
    cache_key = hashlib.sha256(reference.encode()).hexdigest()[:32]
    return VOLUME_PATH / cache_key


def download_gcs_file(gcs_url: str, destination: Path) -> Path:
    """Download a file from GCS (gs:// or gcs://) to a local path.

    Uses google-cloud-storage's Blob.from_string for robust URL parsing.
    Falls back to an anonymous client for public buckets.

    Args:
        gcs_url (str): Google Cloud Storage URL to download from.
        destination (Path): Local file path to write the downloaded file.
    """
    if not gcs_url.startswith(("gs://", "gcs://")):
        raise ValueError(f"Invalid GCS URL: {gcs_url!r} (expected gs://bucket/path)")

    # Convert gcs:// to gs:// for google-cloud-storage compatibility
    if gcs_url.startswith("gcs://"):
        gcs_url = gcs_url.replace("gcs://", "gs://", 1)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading from GCS: %s to %s", gcs_url, destination)

    from google.cloud import storage

    # Prefer authenticated client; fall back to anonymous for public buckets.
    try:
        client = storage.Client()
        logger.info("Using authenticated GCS client")
    except Exception:
        logger.info("Using anonymous GCS client for public bucket access")
        client = storage.Client.create_anonymous_client()

    try:
        blob = storage.Blob.from_string(gcs_url, client=client)
    except Exception as e:
        raise ValueError(f"Invalid GCS URL {gcs_url!r}: {e}") from e

    try:
        blob.download_to_filename(str(destination))
        if destination.exists():
            size_mb = destination.stat().st_size / (1024 * 1024)
            logger.info("Successfully downloaded %.2f MB to %s", size_mb, destination)
        else:
            logger.info("Downloaded to %s", destination)
    except Exception as e:
        logger.error("Failed to download from GCS: %s", e)
        raise RuntimeError(f"Failed to download {gcs_url}: {e}") from e

    return destination


def resolve_file(reference: str) -> Path:
    """Resolve a file reference to a local path.

    Supports:
    - gcs://bucket/path/to/file - Google Cloud Storage
    - gs://bucket/path/to/file - Google Cloud Storage (alternative prefix)
    - /absolute/path - Local paths (for development only)

    Args:
        reference (str): File reference string (local path or GCS URL).

    Returns:
        Path: Path to the file (local path or cached in storage volume).
    """
    # Handle local paths (for development)
    if reference.startswith("/") and Path(reference).exists():
        return Path(reference)

    # Get the path in the volume where this should be cached
    cache_path = get_cache_path(reference)

    # If it already exists in the volume, use it
    if cache_path.exists():
        logger.info(f"Found in volume: {reference} at {cache_path}")
        return cache_path

    # Handle GCS paths
    if reference.startswith(("gcs://", "gs://")):
        download_gcs_file(reference, cache_path)
        logger.info(f"Downloaded to volume: {cache_path}")
    else:
        raise ValueError(
            f"Unsupported reference: {reference}. "
            "Only gcs://, gs://, or absolute local paths are supported"
        )

    return cache_path


def resolve_paths(value: Any) -> Any:
    """Recursively resolve any cloud file paths in a value.

    Args:
        value (Any): Value or nested structure potentially containing file references.

    Examples:
        >>> resolve_paths("gcs://bucket/database.tar.gz")
        "/data/a1b2c3d4"

        >>> resolve_paths({"database": "gcs://bucket/db.tar.gz", "threads": 4})
        {"database": "/data/e5f6g7h8", "threads": 4}
    """
    if isinstance(value, str):
        if value.startswith(("gcs://", "gs://")):
            return str(resolve_file(value))
        return value
    elif isinstance(value, dict):
        return {k: resolve_paths(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_paths(item) for item in value]
    else:
        return value
