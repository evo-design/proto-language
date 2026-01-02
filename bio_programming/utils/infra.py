"""
Infrastructure utilities for proto-language.

This module provides utilities for managing compute resources and file storage,
including GPU selection (local/cloud), cloud storage access (GCS), and caching.
"""
from __future__ import annotations
import os
import hashlib
import logging
from pathlib import Path
from typing import Any, Union

import torch
from google.cloud import storage

logger = logging.getLogger(__name__)

# =============================================================================
# COMPUTE AND GPU UTILITIES
# =============================================================================


def number_of_available_gpus() -> int:
    """Returns the number of available GPUs."""
    return torch.cuda.device_count()


def use_cloud_gpu() -> bool:
    """
    Smart GPU selection: try local GPU first, fall back to cloud.

    Returns:
        bool: True if should use cloud, False if should use local GPU.

    Environment Variables:
        USE_CLOUD: Set to "true" to force cloud, "false" to force local
                   If not set, automatically chooses based on GPU availability
    """
    # Check if user explicitly set preference
    use_cloud_env = os.getenv("USE_CLOUD")
    if use_cloud_env is not None:
        return use_cloud_env.lower() == "true"

    # Auto-detect: try local GPU first, fall back to cloud
    if _is_local_gpu_available():
        return False
    elif _is_cloud_available():
        print("Local GPU not available, falling back to cloud")
        return True
    else:
        raise RuntimeError(
            "Neither local GPU nor cloud is available. "
            "Please either:\n"
            "1. Ensure you have CUDA available locally\n"
            "2. Set up cloud (cloud token new)\n"
            "3. Set USE_CLOUD=true to force cloud execution"
        )


def _is_local_gpu_available() -> bool:
    """Check if local GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_cloud_available() -> bool:
    """Check if cloud is available and configured."""
    try:
        import cloud

        # Try creating a simple app to test authentication
        cloud.App("test-auth")
        return True
    except (ImportError, Exception) as e:
        print(f"cloud not available: {e}")
        return False


def is_gpu_available() -> bool:
    """Check if any GPU is available (local CUDA or cloud)."""
    return _is_local_gpu_available() or _is_cloud_available()


def get_default_device() -> str:
    """Get the default device to use for computation."""
    if is_gpu_available():
        return "cuda"
    else:
        return "cpu"


def get_device_string(device_str_or_int: Union[int, str, torch.device]) -> str:
    """
    Returns the string representation of the GPU specified by an integer index.
    """
    # If the device is a torch.device, get the string representation
    if isinstance(device_str_or_int, torch.device):
        device_str_or_int = str(device_str_or_int)

    # If we have a string
    if isinstance(device_str_or_int, str):
        # If it's just "cuda", return "cuda:0"
        if device_str_or_int == "cuda":
            return "cuda:0"

        if device_str_or_int == "cpu":
            return "cpu"

        # Otherwise, ensure it parses correctly to a single integer
        try:
            device_int = parse_cuda_device_index(device_str_or_int)
            return f"cuda:{device_int}"
        except ValueError:
            raise ValueError(f"Invalid device string: {device_str_or_int}")

    # If it's an integer, return the string representation
    elif isinstance(device_str_or_int, int):
        return f"cuda:{device_str_or_int}"

    else:
        raise ValueError(f"Invalid device: {device_str_or_int}")


def parse_cuda_device_index(device_string: str):
    """Returns the integer index of the GPU specified by a cuda device string."""
    # If the device is not a cuda device string, raise an error
    if not device_string.startswith("cuda"):
        raise ValueError("Device string must start with 'cuda'")

    # If the device is "cuda", return 0
    if device_string == "cuda":
        return 0

    # Otherwise, return the integer index of the GPU
    device_string = device_string.replace("cuda:", "")
    device_int = int(device_string)

    if device_int >= number_of_available_gpus():
        raise ValueError(
            f"Device index {device_int} is greater than the number of available GPUs ({number_of_available_gpus()})"
        )

    return device_int


def determine_visible_devices(device: Union[int, str]) -> str:
    """
    Returns a string corresponding to the CUDA_VISIBLE_DEVICES environment variable
    for a given device.
    """
    # If we are using the CPU, set no devices to be visible
    if device == "cpu":
        return ""

    # If CUDA is specified, but no number is provided, set the first device to be visible
    elif device == "cuda":
        return "0"

    # If CUDA is specified with a number, set the specified device to be visible
    elif device.startswith("cuda:"):
        return device.replace("cuda:", "")

    else:
        try:
            device_int = int(device)
            if device_int >= number_of_available_gpus():
                raise ValueError(
                    f"Device index {device_int} is greater than the number of available GPUs ({number_of_available_gpus()})"
                )
            return device
        except ValueError:
            raise ValueError(f"Invalid device: {device}")


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
