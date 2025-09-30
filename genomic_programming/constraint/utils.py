"""
Shared utility functions for constraint validation and scoring.
"""

from __future__ import annotations
import itertools
import tempfile
from io import StringIO
from pathlib import Path
from typing import Optional, Union, List, Optional, Union, Literal, Dict, Any, Tuple
import numpy as np
import pandas as pd
import math
from collections import Counter
import os
import subprocess
from promoter_calculator.wrapper import promoter_calculator
from ..base import *
from ..utils import resolve_paths
from ..tool_cache import ToolCache
from ..schemas import ESMFoldKwargs, ORFipyKwargs, MMseqsKwargs
from ..tools.orf_prediction.prodigal import run_prodigal
from ..tools.orf_prediction.orfipy import run_orfipy, parse_orfipy_results_to_df
from ..tools.gene_annotation.mmseqs import (
    run_mmseqs_search_proteins,
)
from ..tools.gene_annotation.blast import calculate_segmasker_score
from ..tools.gene_annotation.hmmer import run_hmmscan
from ..tools.structure_prediction.esmfold import predict_structure_esmfold
from ..tools.structure_prediction.boltz import predict_structure_boltz2


# Valid nucleotides for different sequence types
DNA_NUCLEOTIDES = "ATCG"
RNA_NUCLEOTIDES = "AUCG"

# Constraint scoring constants
MIN_ENERGY = 0.0
MAX_ENERGY = 1.0
LOG_BASE = 2

# GC content constants (0-100%)
MIN_GC_CONTENT = 0.0
MAX_GC_CONTENT = 100.0


def _validate_required_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that all required configuration keys are present.

    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required configuration keys.

    Raises:
        ValueError: If any required keys are missing from the configuration.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

def _validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
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


def _calculate_range_deviation(actual: float, min_val: float, max_val: float) -> float:
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


def _calculate_percentage_range_deviation(
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