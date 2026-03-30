"""
tests/language_tests/constraint_tests/utils.py

This module provides helper functions, mock scoring functions, and fixtures
used across multiple constraint test files. It does NOT contain actual unit tests.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest

from proto_language.language.core import (
    Sequence,
)

# =============================================================================
# MOCK SCORING FUNCTIONS FOR TESTING CONSTRAINT EVALUATION
# =============================================================================

def mock_single_input_scoring_function(
    input_sequences: List[Tuple[Sequence, ...]],
    config=None
) -> List[float]:
    """
    Mock scoring function for testing single-input constraints.

    Returns scores based on the fraction of 'T' characters in each sequence.
    Also adds metadata to each sequence to demonstrate metadata propagation.

    Args:
        input_sequences: List of single-sequence tuples to score
        config: Optional configuration (unused in mock)

    Returns:
        List of scores as fractions of T characters (0.0 to 1.0)
    """
    scores = []
    for (sequence,) in input_sequences:
        score = sequence.sequence.count("T") / len(sequence)
        # Add metadata to demonstrate propagation
        sequence._metadata["t_count"] = sequence.sequence.count("T")
        sequence._metadata["total_length"] = len(sequence)
        sequence._metadata["t_fraction"] = score
        scores.append(score)
    return scores

# Set attributes that would normally be set by registry decorator
mock_single_input_scoring_function._constraint_config_class = None
mock_single_input_scoring_function._constraint_supported_sequence_types = ["dna", "rna", "protein"]


def mock_multi_input_scoring_function(
    input_sequences: List[Tuple[Sequence, ...]],
    config=None
) -> List[float]:
    """
    Mock scoring function for testing single-input batched constraints.

    Returns scores based on the fraction of 'T' characters in each sequence.
    Also adds metadata to each sequence to demonstrate metadata propagation.

    Args:
        input_sequences: List of single-sequence tuples to score
        config: Optional configuration (unused in mock)

    Returns:
        List of scores as fractions of T characters (0.0 to 1.0)
    """
    scores = []
    for (sequence,) in input_sequences:
        score = sequence.sequence.count("T") / len(sequence)
        # Add metadata to demonstrate propagation
        sequence._metadata["t_count"] = sequence.sequence.count("T")
        sequence._metadata["total_length"] = len(sequence)
        sequence._metadata["t_fraction"] = score
        scores.append(score)
    return scores

# Set attributes that would normally be set by registry decorator
mock_multi_input_scoring_function._constraint_config_class = None
mock_multi_input_scoring_function._constraint_supported_sequence_types = ["dna", "rna", "protein"]


def mock_multi_input_scoring_function_disjoint(
    input_sequences: List[Tuple[Sequence, ...]],
    config=None
) -> List[float]:
    """
    Mock scoring function for testing multi-input disjoint constraints.

    Expects a list of tuples of two sequences each and returns scores based on:
    - Fraction of 'T' in the first sequence of each tuple
    - Fraction of 'C' in the second sequence of each tuple
    Average of these two fractions is returned for each tuple.

    Args:
        input_sequences: List of tuples, each containing two Sequence objects
        config: Optional configuration (unused in mock)

    Returns:
        List of average scores
    """
    scores = []
    for sequence_tuple in input_sequences:
        t_percent = sequence_tuple[0].sequence.count("T") / len(sequence_tuple[0])
        c_percent = sequence_tuple[1].sequence.count("C") / len(sequence_tuple[1])
        scores.append((t_percent + c_percent) / 2)

        # Add metadata
        sequence_tuple[0]._metadata["t_percent"] = t_percent
        sequence_tuple[1]._metadata["c_percent"] = c_percent

    return scores

# Set attributes that would normally be set by registry decorator
mock_multi_input_scoring_function_disjoint._constraint_config_class = None
mock_multi_input_scoring_function_disjoint._constraint_supported_sequence_types = ["dna", "rna", "protein"]


def mock_dna_only_scoring_function(
    input_sequences: List[Tuple[Sequence, ...]],
    config=None
) -> List[float]:
    """
    Mock scoring function for testing type-restricted constraints.
    Only supports DNA sequences.
    """
    return [0.5 for _ in input_sequences]

# Set attributes that would normally be set by registry decorator
mock_dna_only_scoring_function._constraint_config_class = None
mock_dna_only_scoring_function._constraint_supported_sequence_types = ["dna"]  # DNA only


def mock_protein_only_scoring_function(
    input_sequences: List[Tuple[Sequence, ...]],
    config=None
) -> List[float]:
    """
    Mock scoring function for testing type-restricted constraints.
    Only supports protein sequences.
    """
    return [0.5 for _ in input_sequences]

# Set attributes that would normally be set by registry decorator
mock_protein_only_scoring_function._constraint_config_class = None
mock_protein_only_scoring_function._constraint_supported_sequence_types = ["protein"]  # Protein only


# =============================================================================
# TEST DATA PATHS AND FIXTURES
# =============================================================================

TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_DB_PATH = TEST_DATA_DIR / "test_proteins_database.faa"


@pytest.fixture(scope="module")
def dummy_db_path():
    """
    Fixture providing path to dummy protein database for testing.
    Module-scoped to avoid recreating for each test.
    """
    return str(PROTEIN_DB_PATH)


@pytest.fixture
def temp_dir():
    """
    Fixture providing a temporary directory for test files.
    Automatically cleaned up after test completion.

    Yields:
        Path object pointing to temporary directory
    """
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)
