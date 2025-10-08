"""
Compute and GPU utilities for proto-language.

This module provides utilities for managing compute resources,
including local GPU and cloud GPU selection.
"""

import os
import torch
from typing import Union


def number_of_available_gpus() -> int:
    """
    Returns the number of available GPUs.
    """
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
        cloud.App('test-auth')
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
        return "cuda:0"
    else:
        return "cpu"


def get_device_string(device_str_or_int: Union[int, str, torch.device]):
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
    """
    Returns the integer index of the GPU specified by a cuda device string.
    """

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


def determine_visible_devices(device: str) -> str:
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
        raise ValueError(f"Invalid device: {device}")
