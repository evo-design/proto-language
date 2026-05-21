"""This module provides utilities for metadata management and structural/geometric.

calculations used across the proto-language framework.
"""

import json
import math
import subprocess
from typing import Any

import numpy as np
import pydantic


def format_pydantic_error(e: pydantic.ValidationError, prefix: str) -> str:
    """Reformat a Pydantic ValidationError as a one-line ``<prefix> — <field>: <msg> [got=<input>]; ...``.

    Each per-field error becomes ``loc.path: msg`` plus the rejected ``input`` when present
    and short, so an LLM agent reading the error sees both *which* field broke and *what*
    value was rejected.
    """
    parts: list[str] = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "__root__"
        msg = err["msg"]
        item = f"{loc}: {msg}"
        bad = err.get("input")
        if bad is not None:
            preview = repr(bad)
            if len(preview) > 80:
                preview = preview[:77] + "..."
            item += f" [got={preview}]"
        parts.append(item)
    return f"{prefix} — {'; '.join(parts)}"


def make_json_safe(obj: Any) -> Any:
    """Recursively convert metadata to JSON-safe values, replacing NaN/Inf with None."""
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, pydantic.BaseModel):
        return make_json_safe(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {_make_json_safe_dict_key(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [make_json_safe(v) for v in obj]
    return obj


def _make_json_safe_dict_key(key: Any) -> str:
    """Convert metadata dict keys to strings that can be encoded as JSON object keys."""
    if isinstance(key, str):
        return key
    safe_key = make_json_safe(key)
    try:
        return json.dumps(safe_key, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(safe_key)


def is_plain_int(value: object) -> bool:
    """Return True for ints while excluding bool, which subclasses int."""
    return isinstance(value, int) and not isinstance(value, bool)


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
    """Validate that a value falls within the specified range.

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


def calculate_range_deviation(actual: float, min_val: float, max_val: float, epsilon: float = 1) -> float:
    """Calculate deviation from acceptable range for general constraints.

    Args:
        actual (float): The actual measured value.
        min_val (float): Minimum acceptable value.
        max_val (float): Maximum acceptable value.
        epsilon (float): Floor for the denominator to avoid division by zero.
            Use 1 (default) for integer-scale values, 1e-9 for fractional values.

    Returns:
        float: Range deviation score where 0.0 indicates the value is within range
            and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    if actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, epsilon))
    return min(MAX_ENERGY, (actual - max_val) / max(max_val, epsilon))


def calculate_percentage_range_deviation(actual: float, min_val: float, max_val: float) -> float:
    """Calculate deviation from acceptable range for percentage-based constraints (0-100%).

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
    if actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, 1))
    return min(MAX_ENERGY, (actual - max_val) / max(100 - max_val, 1))


def calculate_gc_content(sequence: str) -> float:
    """Calculate the GC content percentage of a DNA/RNA sequence.

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
    """Calculate normalized deviation from target value for target-based constraints.

    Args:
        actual (float): The actual measured value.
        target (float): The desired target value.

    Returns:
        float: Normalized deviation score where 0.0 indicates perfect match
            and higher values indicate greater deviation from target.
    """
    return min(MAX_ENERGY, abs(actual - target) / max(target, 1))


def one_hot_protein_matrix(sequence: str) -> list[list[float]]:
    """Return an exact (1.0, 0.0) one-hot matrix in ``PROTEIN_AMINO_ACIDS`` order.

    Each row has ``1.0`` at the target amino acid and ``0.0`` everywhere else.
    Use this when encoding a discrete protein sequence for a tool that expects a
    probability matrix or one-hot input.

    Args:
        sequence (str): Protein sequence; each character must be in ``PROTEIN_AMINO_ACIDS``.

    Returns:
        list[list[float]]: One-hot matrix with shape ``(len(sequence), 20)``.
    """
    from proto_language.language.core.sequence import PROTEIN_AMINO_ACIDS

    aa_index = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
    n = len(PROTEIN_AMINO_ACIDS)
    rows: list[list[float]] = []
    for aa in sequence:
        row = [0.0] * n
        row[aa_index[aa]] = 1.0
        rows.append(row)
    return rows


def softmax(matrix: np.ndarray) -> np.ndarray:
    """Compute numerically stable row-wise softmax."""
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(shifted)
    result = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy scalar arithmetic for mypy
    return result


def mean_peak_probability(matrix: np.ndarray, positions: list[int] | None = None) -> float:
    """Return the mean per-row peak probability, optionally restricted to ``positions``."""
    rows = matrix if positions is None else matrix[positions]
    return float(np.mean(np.max(rows, axis=-1)))


def sigmoid_score(
    metric: float,
    inflection: float,
    slope: float = 3.0,
) -> float:
    """Squeezes a metric into a 0-1 score using a sigmoid function.

    Args:
        metric (float): A metric value.
        inflection (float): The value of the original metric where the transformed score
            would be 0.5.
        slope (float): The steepness of the curve. Default: 3.0.

    Returns:
        float: Score between 0.0 (good/low) and 1.0 (bad/high).
    """
    # 1 / (1 + e^(-k(x - x0)))  # noqa: ERA001
    # We want low metric -> 0 and high metric -> 1.
    # The standard sigmoid 1/(1+e^-x) goes 0->1 as x increases.
    # We use slope * (metric - inflection).
    scaled = slope * (metric - inflection)
    if scaled >= 0.0:
        z = math.exp(-scaled)
        return 1.0 / (1.0 + z)
    z = math.exp(scaled)
    return z / (1.0 + z)


def inverse_sigmoid_score(
    score: float,
    inflection: float,
    slope: float = 3.0,
) -> float:
    """Inverts the sigmoid_score function to recover the original metric from a 0-1.

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
    # y = 1 / (1 + e^(-k(x - x0)))  # noqa: ERA001
    # 1/y = 1 + e^(-k(x - x0))
    # (1 - y) / y = e^(-k(x - x0))
    # ln((1 - y) / y) = -slope * (metric - inflection)
    # -1/slope * ln((1 - y) / y) = metric - inflection
    #
    # Using the property -ln(a/b) = ln(b/a):
    # metric = inflection + (1/slope) * ln(y / (1 - y))  # noqa: ERA001

    return float(inflection + (np.log(score / (1.0 - score)) / slope))


# =============================================================================
# TOOL UTILITIES
# =============================================================================


def run_subprocess_command(cmd: list[str], tool_name: str) -> subprocess.CompletedProcess[str]:
    """Run subprocess command with error handling.

    Args:
        cmd (list[str]): Command and arguments to execute.
        tool_name (str): Name of the tool being executed for error messages.

    Returns:
        subprocess.CompletedProcess[str]: CompletedProcess object with stdout/stderr accessible.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(
            f"{tool_name} failed (exit {proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def resolve_sequence_ids(sequences: list[str], ids: list[str] | None) -> list[str]:
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
            raise ValueError(f"sequence_ids length ({len(ids)}) must match sequences length ({len(sequences)})")
        return ids
    return [f"seq_{i}" for i in range(len(sequences))]
