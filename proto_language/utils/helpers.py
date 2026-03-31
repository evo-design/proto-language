"""
proto_language/utils/helpers.py

This module provides utilities for metadata management and structural/geometric
calculations used across the proto-language framework.
"""

from __future__ import annotations

import math
import random
import subprocess
from typing import List, Optional

import numpy as np

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


def filter_inf_nan_scores(score: float) -> float | None:
    """Convert inf/nan to None for JSON compatibility."""
    if math.isinf(score) or math.isnan(score):
        return None
    return score


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """
    Validate that a value falls within the specified range.

    Args:
        value (float): The value to validate.
        min_val (float): Minimum acceptable value (inclusive).
        max_val (float): Maximum acceptable value (inclusive).
        name (str): Name of the parameter for error messages.

    Raises:
        ValueError: If value is outside the specified range.
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def calculate_range_deviation(actual: float, min_val: float, max_val: float) -> float:
    """
    Calculate deviation from acceptable range for general constraints.

    Args:
        actual (float): The actual measured value.
        min_val (float): Minimum acceptable value.
        max_val (float): Maximum acceptable value.

    Returns:
        float: Range deviation score where 0.0 indicates the value is within range
            and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, 1))
    else:
        return min(MAX_ENERGY, (actual - max_val) / max(max_val, 1))


def calculate_percentage_range_deviation(
    actual: float, min_val: float, max_val: float
) -> float:
    """
    Calculate deviation from acceptable range for percentage-based constraints (0-100%).

    Args:
        actual (float): The actual measured percentage value.
        min_val (float): Minimum acceptable percentage.
        max_val (float): Maximum acceptable percentage.

    Returns:
        float: Percentage range deviation score where 0.0 indicates the value is within range
            and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, 1))
    else:
        return min(MAX_ENERGY, (actual - max_val) / max(100 - max_val, 1))


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate the GC content percentage of a DNA/RNA sequence.

    Args:
        sequence (str): DNA or RNA sequence string.

    Returns:
        float: GC content as a percentage (0-100).
    """
    if not sequence:
        return 0.0

    sequence_upper = sequence.upper()
    gc_count = sequence_upper.count("G") + sequence_upper.count("C")
    return 100.0 * gc_count / len(sequence)


def calculate_normalized_deviation(actual: float, target: float) -> float:
    """
    Calculate normalized deviation from target value for target-based constraints.

    Args:
        actual (float): The actual measured value.
        target (float): The desired target value.

    Returns:
        float: Normalized deviation score where 0.0 indicates perfect match
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
        metric (float): A non-negative metric value.
        inflection (float): The value of the original metric where the transformed score
            would be 0.5.
        slope (float): The steepness of the curve. Default: 3.0.

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


def inverse_sigmoid_score(
    score: float,
    inflection: float,
    slope: float = 3.0,
) -> float:
    """
    Inverts the sigmoid_score function to recover the original metric from a 0-1
    score using the **logit function**. Helps to recover the original metric from a
    0-1 score.

    Args:
        score (float): A score value strictly between 0.0 and 1.0.
        inflection (float): The value of the original metric where the transformed score
                    is 0.5.
        slope (float): The steepness of the curve. Default: 3.0.

    Returns:
        float: The recovered metric value.
    """
    # The sigmoid function has asymptotes at 0 and 1, so exact 0.0 or 1.0
    # scores correspond to -infinity and +infinity metrics respectively.
    if score <= 0.0 or score >= 1.0:
        raise ValueError(f"Input score must be strictly between 0 and 1, found {score}")

    if slope == 0:
        raise ValueError("Slope cannot be zero for inversion.")

    # Mathematical derivation:
    # y = 1 / (1 + e^(-k(x - x0)))
    # 1/y = 1 + e^(-k(x - x0))
    # (1 - y) / y = e^(-k(x - x0))
    # ln((1 - y) / y) = -slope * (metric - inflection)
    # -1/slope * ln((1 - y) / y) = metric - inflection
    #
    # Using the property -ln(a/b) = ln(b/a):
    # metric = inflection + (1/slope) * ln(y / (1 - y))

    return inflection + (np.log(score / (1.0 - score)) / slope)


# =============================================================================
# TOOL UTILITIES
# =============================================================================


def mask_k(sequence: str, k: int, mask_str: str = "_", fixed_indices: List[int] = None) -> str:
    """
    Mask k random positions of a sequence.

    Args:
        sequence (str): The sequence to mask.
        k (int): The number of positions to mask.
        mask_str (str): The string of characters that replace sequence characters
            in masked positions.
        fixed_indices (list[int]): The indices of the positions that are fixed and
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
        fixed_indices (list[int]): Sequence positions that should remain unchanged during mutation.

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


def run_subprocess_command(cmd: List[str], tool_name: str) -> subprocess.CompletedProcess:
    """
    Run subprocess command with error handling.

    Args:
        cmd (list[str]): Command and arguments to execute.
        tool_name (str): Name of the tool being executed for error messages.

    Returns:
        subprocess.CompletedProcess: CompletedProcess object with stdout/stderr accessible.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{tool_name} failed (exit {proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def resolve_sequence_ids(sequences: List[str], ids: Optional[List[str]]) -> List[str]:
    """Resolve sequence identifiers, using provided IDs or generating defaults.

    Args:
        sequences (list[str]): List of sequences to generate IDs for.
        ids (list[str] | None): Optional list of user-provided sequence identifiers.

    Returns:
        list[str]: List of sequence identifiers (provided IDs or seq_0, seq_1, ...).

    Raises:
        ValueError: If ids length doesn't match sequences length.
    """
    if ids is not None:
        if len(ids) != len(sequences):
            raise ValueError(
                f"sequence_ids length ({len(ids)}) must match sequences length ({len(sequences)})"
            )
        return ids
    return [f"seq_{i}" for i in range(len(sequences))]
