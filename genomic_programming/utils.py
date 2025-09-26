from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from io import StringIO
import numpy as np
import os
from typing import Union, Any, Dict, Optional
import hashlib
from pathlib import Path
import logging
from google.cloud import storage
from enum import Enum

logger = logging.getLogger(__name__)


def propagate_metadata(
    source_metadata: Dict[str, Any], 
    target_metadata: Dict[str, Any], 
    prefix: Optional[str] = None
) -> None:
    """
    Utility function to propagate metadata from source to target, filtering out system keys.
    
    Args:
        source_metadata: Metadata from scored sequence
        target_metadata: Target metadata dictionary to receive the metadata
        prefix: Optional prefix for metadata keys (e.g. "promoter.esmfold_constraint")
    """
    # Sequence and sequence_length not be propagated since they are populated dynamically by the Sequence class
    system_keys = {"sequence", "sequence_length"}
    for key, value in source_metadata.items():
        if key not in system_keys:
            final_key = f"{prefix}.{key}" if prefix else key
            target_metadata[final_key] = value


class SequenceType(Enum):
    """Enumeration of supported biological sequence types."""

    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"


class ConstraintType(Enum):
    """Enumeration of constraint evaluation strategies for multiple inputs."""

    CONTIGUOUS = "contiguous"  # Concatenate sequences before evaluation
    DISJOINT = "disjoint"  # Evaluate sequences separately as a group


def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    return PDBFile.read(pdb_path).get_structure(model=1)


def get_atomarray_in_residue_range(atoms: AtomArray, start: int, end: int) -> AtomArray:
    return atoms[np.logical_and(atoms.res_id >= start, atoms.res_id < end)]


def _is_Nx3(array: np.ndarray) -> bool:
    return len(array.shape) == 2 and array.shape[1] == 3


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(m, axis=-1)
    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


def adjacent_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates - np.roll(coordinates, shift=1, axis=0)
    return np.linalg.norm(m, axis=-1)


def get_centroid(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    return coordinates.mean(axis=0).reshape(1, 3)


def distances_to_centroid(coordinates: np.ndarray) -> np.ndarray:
    """
    Computes the distances from each of the coordinates to the
    centroid of all coordinates.
    """
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    centroid = get_centroid(coordinates)
    m = coordinates - centroid
    return np.linalg.norm(m, axis=-1)


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[
        (atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")
    ]


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices of the top-k values in the scores vector.

    Args:
        scores (np.ndarray): 1D array of scores.
        k (int): number of top elements to return.

    Returns:
        np.ndarray: Array of indices of the top-k scores.
    """
    # np.argpartition is more efficient than sorting the entire array
    # when we only need the top-k elements
    if k >= len(scores):
        # If k is larger than the array length, return all indices in sorted order
        return np.argsort(scores)[::-1]

    # Get indices of top-k elements
    # The negative sign is because we want the largest values (descending order)
    top_k_idx = np.argpartition(scores, -k)[-k:]

    # Sort these top-k indices by their corresponding values (highest first)
    top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

    return top_k_idx


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
        cloud.App('test-auth')
        return True
    except (ImportError, Exception) as e:
        print(f"cloud not available: {e}")
        return False


def is_gpu_available() -> bool:
    """Check if any GPU is available (local CUDA or cloud)."""
    return _is_local_gpu_available() or _is_cloud_available()


########################
## File resolver utils #
########################

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
        raise ValueError(f"Unsupported reference: {reference}. Only gcs:// or gs:// paths are supported")
    
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
