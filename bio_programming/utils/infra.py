"""
Infrastructure utilities for proto-language.

This module provides utilities for file resolution and cloud storage (GCS).
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# FILE RESOLUTION AND CLOUD STORAGE UTILITIES
# =============================================================================

# storage volume mount point (set via environment variable or default)
# In the deploy platform, you mount volumes at specific paths like /data or /cache
VOLUME_PATH = Path(os.environ.get("STORAGE_VOLUME_PATH", "/data"))


def get_cache_path(reference: str) -> Path:
    """Get the path in the storage volume for a given reference."""
    # Create a short, readable cache key
    cache_key = hashlib.md5(reference.encode()).hexdigest()[:16]
    return VOLUME_PATH / cache_key


def download_gcs_file(gcs_url: str, destination: Path) -> Path:
    """
    Download a file from GCS (gs:// or gcs://) to a local path.
    Uses google-cloud-storage's Blob.from_string for robust URL parsing.
    Falls back to an anonymous client for public buckets.
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

    # Parse the URL (now always gs://)
    try:
        blob = storage.Blob.from_string(gcs_url, client=client)
    except Exception as e:
        raise ValueError(f"Invalid GCS URL {gcs_url!r}: {e}") from e

    # Download
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
    """
    Resolve a file reference to a local path in the storage volume.

    Supports:
    - gcs://bucket/path/to/file - Google Cloud Storage
    - gs://bucket/path/to/file - Google Cloud Storage (alternative prefix)
    - /absolute/path - Local paths (for development only)

    Returns:
        Path to the file in the storage volume
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
        # Download from GCS using the Python client
        download_gcs_file(reference, cache_path)
        logger.info(f"Downloaded to volume: {cache_path}")
    else:
        raise ValueError(
            f"Unsupported reference: {reference}. Only gcs:// or gs:// paths are supported"
        )

    return cache_path


def resolve_paths(value: Any) -> Any:
    """
    Recursively resolve any cloud file paths in a value.

    Examples:
        >>> resolve_paths("gcs://bucket/database.tar.gz")
        "/data/a1b2c3d4"

        >>> resolve_paths({"database": "gcs://bucket/db.tar.gz", "threads": 4})
        {"database": "/data/e5f6g7h8", "threads": 4}
    """
    if isinstance(value, str):
        # Check if it's a cloud path (GCS only)
        if any(value.startswith(p) for p in ["gcs://", "gs://"]):
            return str(resolve_file(value))
        return value
    elif isinstance(value, dict):
        return {k: resolve_paths(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_paths(item) for item in value]
    else:
        return value


def upload_to_gcs(content: bytes, bucket_name: str, blob_path: str) -> str:
    """
    Upload content to Google Cloud Storage.

    Args:
        content: The content to upload as bytes.
        bucket_name: Name of the GCS bucket.
        blob_path: Path within the bucket where the content will be stored.

    Returns:
        The gs:// URL of the uploaded blob.

    Raises:
        RuntimeError: If upload fails.
    """
    logger.info("Uploading to GCS: gs://%s/%s", bucket_name, blob_path)

    from google.cloud import storage

    try:
        client = storage.Client()
    except Exception:
        raise RuntimeError(
            "Failed to initialize GCS client. Ensure you have valid credentials."
        )

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(content)

        gcs_url = f"gs://{bucket_name}/{blob_path}"
        size_kb = len(content) / 1024
        logger.info("Successfully uploaded %.2f KB to %s", size_kb, gcs_url)
        return gcs_url
    except Exception as e:
        logger.error("Failed to upload to GCS: %s", e)
        raise RuntimeError(f"Failed to upload to gs://{bucket_name}/{blob_path}: {e}") from e
