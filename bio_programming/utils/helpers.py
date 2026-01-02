"""
Shared helper utilities for proto-language.

This module provides utilities for metadata management and structural/geometric
calculations used across the proto-language framework.
"""
from __future__ import annotations
import numpy as np
import random
import subprocess
from typing import Any, Dict, List, Optional


# =============================================================================
# CONSTRAINT SCORING UTILITIES
# =============================================================================

# Constraint scoring constants
MIN_ENERGY = 0.0
MAX_ENERGY = 1.0
LOG_BASE = 2

# GC content constants (0-100%)
MIN_GC_CONTENT = 0.0
MAX_GC_CONTENT = 100.0


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """
    Validate that a value falls within the specified range.

    Args:
        value: The value to validate.
        min_val: Minimum acceptable value (inclusive).
        max_val: Maximum acceptable value (inclusive).
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If value is outside the specified range.
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def calculate_range_deviation(actual: float, min_val: float, max_val: float) -> float:
    """
    Calculate deviation from acceptable range for general constraints.

    Args:
        actual: The actual measured value.
        min_val: Minimum acceptable value.
        max_val: Maximum acceptable value.

    Returns:
        Range deviation score where 0.0 indicates the value is within range
        and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / min_val)
    else:
        return min(MAX_ENERGY, (actual - max_val) / max_val)


def calculate_percentage_range_deviation(
    actual: float, min_val: float, max_val: float
) -> float:
    """
    Calculate deviation from acceptable range for percentage-based constraints (0-100%).

    Args:
        actual: The actual measured percentage value.
        min_val: Minimum acceptable percentage.
        max_val: Maximum acceptable percentage.

    Returns:
        Percentage range deviation score where 0.0 indicates the value is within range
        and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, 1))
    else:
        return min(MAX_ENERGY, (actual - max_val) / max(100 - max_val, 1))


def calculate_normalized_deviation(actual: float, target: float) -> float:
    """
    Calculate normalized deviation from target value for target-based constraints.

    Args:
        actual: The actual measured value.
        target: The desired target value.

    Returns:
        Normalized deviation score where 0.0 indicates perfect match
        and higher values indicate greater deviation from target.
    """
    return min(MAX_ENERGY, abs(actual - target) / max(target, 1))


def sigmoid_score(
    metric: float,
    inflection: float,
    slope: float = 3.0,
) -> float:
    """
    Squeezes a non-negative metric (i.e., >= 0) into a 0-1 score using a sigmoid
    function.

    Args:
        metric: A non-negative metric value.
        inflection: The value of the original metric where the transformed score
            would be 0.5.
        slope: The steepness of the curve. Default: 3.0.

    Returns:
        float: Score between 0.0 (good/low) and 1.0 (bad/high).
    """
    if metric < 0:
        raise ValueError(f"Input metric value cannot be negative, found {metric}")

    # 1 / (1 + e^(-k(x - x0)))
    # We want low metric -> 0 and high metric -> 1.
    # The standard sigmoid 1/(1+e^-x) goes 0->1 as x increases.
    # We use slope * (metric - inflection).
    return 1.0 / (1.0 + np.exp(-slope * (metric - inflection)))


# =============================================================================
# METADATA UTILITIES
# =============================================================================


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


# =============================================================================
# TOOL UTILITIES
# =============================================================================


def mask_k(
    sequence: str, k: int, mask_str: str = "_", fixed_indices: List[int] = None
) -> str:
    """
    Mask k random positions of a sequence.

    Args:
        sequence (str): The sequence to mask.
        k (int): The number of positions to mask.
        mask_str (str): The string of characters that replace sequence characters
            in masked positions.
        fixed_indices (List[int]): The indices of the positions that are fixed and
            should not be masked.
    """
    if k > len(sequence):
        raise ValueError("k cannot be greater than the length of the sequence")

    # Create a list of the sequence
    sequence_list = list(sequence)

    if fixed_indices is None:
        fixed_indices = []

    # Create a list of maskable indices
    maskable_indices = [i for i in range(len(sequence)) if i not in fixed_indices]

    # Randomly select k positions to mask
    positions = random.sample(maskable_indices, k)

    # Mask the selected positions
    for position in positions:
        sequence_list[position] = mask_str

    # Convert the list back to a string
    return "".join(sequence_list)


def mask_p(
    sequence: str, p: float, mask_str: str = "_", fixed_indices: List[int] = None
) -> str:
    """
    Mask a random fraction of positions in a sequence.

    Args:
        sequence (str): The sequence to mask.
        p (float): The fraction of positions to mask.
        mask_str (str): The string of characters that replace sequence characters
            in masked positions.

    Returns:
        str: The masked sequence.
    """
    if p > 1 or p < 0:
        raise ValueError("p must be between 0 and 1")

    if fixed_indices is None:
        fixed_indices = []

    # Determine how many positions are designable
    num_designable_positions = len(sequence) - len(fixed_indices)

    # Determine the number of positions to mask
    k = max(1, int(p * num_designable_positions))

    # Mask the sequence
    masked_sequence = mask_k(sequence, k, mask_str, fixed_indices)

    return masked_sequence


def mask_assigned_positions(
    sequence: str, inds_to_mask: list[int], mask_str: str = "_"
) -> str:
    """
    Returns a masked version of the sequence where the positions in inds_to_mask
    are replaced with the mask_str.

    Args:
        sequence (str): The sequence to mask.
        inds_to_mask (list[int]): The indices of the positions to mask. (0-indexed)
        mask_str (str): The string of characters that replace sequence characters
            in masked positions.

    Returns:
        str: The masked sequence.
    """
    # Create a list of the sequence
    sequence_list = list(sequence)

    # Mask the assigned positions
    for ind in inds_to_mask:
        sequence_list[ind] = mask_str

    # Convert the list back to a string
    return "".join(sequence_list)


def run_subprocess_command(cmd: List[str], tool_name: str) -> None:
    """
    Run subprocess command with error handling.

    Args:
        cmd: Command and arguments to execute.
        tool_name: Name of the tool being executed for error messages.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{tool_name} failed (exit {proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
