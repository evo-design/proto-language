from biotite.structure import AtomArray
from biotite.structure.atoms import AtomArray
from biotite.structure.io.pdb import PDBFile
from io import StringIO
import numpy as np
import os
from scipy.special import softmax
from typing import Union


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
