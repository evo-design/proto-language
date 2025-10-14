import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)


# Helper functions
def create_segment(sequence: str, seq_type: SequenceType = SequenceType.DNA) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


def create_batched_segment(
    sequences: List[str], seq_type: SequenceType = SequenceType.DNA
) -> Segment:
    """Helper to create a Segment with a batch of sequences."""
    segment = Segment(sequence=sequences[0], sequence_type=seq_type)
    segment.create_batch(len(sequences))
    for i, seq_str in enumerate(sequences):
        segment.batch_sequences[i].sequence = seq_str
    return segment


# Mock scoring functions
def mock_single_input_scoring_function(sequence: Sequence, config=None) -> float:
    """
    Mock scoring function that takes in a single sequence and returns a score
    corresponding to the number of T characters in the sequence
    """
    score = sequence.sequence.count("T") / len(sequence)
    # Add metadata to demonstrate propagation
    sequence._metadata["t_count"] = sequence.sequence.count("T")
    sequence._metadata["total_length"] = len(sequence)
    sequence._metadata["t_fraction"] = score
    return score


def mock_multi_input_scoring_function(sequences: List[Sequence], config=None) -> List[float]:
    """
    Mock scoring function that takes in a list of sequences and returns a list of scores
    corresponding to the number of T characters in each sequence
    """
    scores = []
    for sequence in sequences:
        score = sequence.sequence.count("T") / len(sequence)
        # Add metadata to demonstrate propagation
        sequence._metadata["t_count"] = sequence.sequence.count("T")
        sequence._metadata["total_length"] = len(sequence)
        sequence._metadata["t_fraction"] = score
        scores.append(score)
    return scores


def mock_single_input_scoring_function_disjoint(
    sequence_tuple: Tuple[Sequence, Sequence],
    config=None
) -> float:
    """
    Mock scoring function that takes in a tuple of sequences and returns a score
    corresponding to the number of T characters in the sequences. Expects two sequences in the tuple.
    """
    # Compute percent of T in first and percent of C in second
    t_percent = sequence_tuple[0].sequence.count("T") / len(sequence_tuple[0])
    c_percent = sequence_tuple[1].sequence.count("C") / len(sequence_tuple[1])
    # Add metadata
    sequence_tuple[0]._metadata["t_percent"] = t_percent
    sequence_tuple[1]._metadata["c_percent"] = c_percent

    score = (t_percent + c_percent) / 2
    return score


def mock_multi_input_scoring_function_disjoint(
    sequence_tuples: List[Tuple[Sequence, Sequence]],
    config=None
) -> float:
    """
    Mock scoring function that takes in a tuple of sequences and returns a score
    corresponding to the number of T characters in the sequences. Expects two sequences in the tuple.
    """
    scores = []
    for sequence_tuple in sequence_tuples:
        t_percent = sequence_tuple[0].sequence.count("T") / len(sequence_tuple[0])
        c_percent = sequence_tuple[1].sequence.count("C") / len(sequence_tuple[1])
        scores.append((t_percent + c_percent) / 2)
        sequence_tuple[0]._metadata["t_percent"] = t_percent
        sequence_tuple[1]._metadata["c_percent"] = c_percent
    return scores


# Test data file paths
TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_DB_PATH = TEST_DATA_DIR / "test_proteins_database.faa"
DNA_FASTA_PATH = TEST_DATA_DIR / "test_dna_sequences.fna"
ORFIPY_AA_PATH = TEST_DATA_DIR / "test_orfipy_aa.faa"
ORFIPY_NT_PATH = TEST_DATA_DIR / "test_orfipy_nt.fna"
M8_RESULTS_PATH = TEST_DATA_DIR / "test_mmseqs_results.m8"


def get_test_sequences_with_real_hits():
    """Returns DNA sequences that should produce hits against our dummy database."""
    # These sequences correspond to the test data files we created
    sequences = []
    with open(DNA_FASTA_PATH, "r") as f:
        current_seq = ""
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append(current_seq)
    return sequences


@pytest.fixture(scope="module")
def dummy_db_path():
    return str(PROTEIN_DB_PATH)


# Sample data for constraint tests
SAMPLE_ORFIPY_AA_FASTA = """>dna_seq_1_ORF.1 [0-180](+)
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK*
>dna_seq_2_ORF.1 [0-540](+)
MKALIVLGLVLLSVTVQGKVFGRCELAAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL*
"""

SAMPLE_ORFIPY_NT_FASTA = """>dna_seq_1_ORF.1 [0-180](+)
ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGGTGGGCGCGCACGCCGGCGAATATGGCGCAGAAGCCTTGGAAAGAATGTTTTTGAGCTTTCCAACCACCAAGACCTATTTCCCACACTTTGATTTGAGCCACGGCAGCGCACAGGTGAAAGGCCACGGCAAA
>dna_seq_2_ORF.1 [0-540](+)
ATGAAAGCCTTGATCGTGTTGGGCTTGGTGTTGTTGAGCGTGACCGTGCAGGGCAAAGTGTTCGGCAGATGCGAATTGGCCGCAGCCGCAATGAAGAGACACGGCTTGGATAACTACAGAGGCTACAGCTTGGGCAACTGGGTGTGCGCAGCAAAGTTTGAAAGCAACTTCAACACACAGGCCACCAACAGAAACACCGATGGCAGCACCGATTATGGCATCTTGCAGATCAACAGCAGATGGTGGTGCAACGATGGCAGAACCCCAGGCAGCAGAAACTTGTGCAACATCCCATGCAGCGCCTTGTTGAGCAGCGATATTACCGCAAGCGTGAACTGCGCAAAGAAAATCGTGAGCGATGGCAACGGCATGAACGCATGGGTGGCATGGAGAAACAGATGCAAAGGCACCGATGTGCAGGCATGGATCAGAGGCTGCAGATTGTAA
"""

SAMPLE_M8_OUTPUT = """protein_seq_1	test_protein_1	95.2	1.5e-35
protein_seq_2	test_protein_2	87.3	2.1e-28
protein_seq_3	test_protein_5	100.0	1.0e-3
protein_seq_4	test_protein_1	98.1	3.2e-42
"""


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


def setup_test_files(temp_dir: Path, sequence: str) -> dict:
    """Set up test files for orfipy and mmseqs tests using real files."""
    # Create input DNA file
    dna_file = temp_dir / "input.fna"
    dna_file.write_text(f">test_seq\n{sequence}\n")

    # Create orfipy output directory and files
    orfipy_dir = temp_dir / "orfipy_output"
    orfipy_dir.mkdir()

    # Use real test data files
    shutil.copy(ORFIPY_AA_PATH, orfipy_dir / "orfipy_aa.faa")
    shutil.copy(ORFIPY_NT_PATH, orfipy_dir / "orfipy_nt.fna")

    # Create mmseqs output file
    mmseqs_file = temp_dir / "mmseqs_results.m8"
    shutil.copy(M8_RESULTS_PATH, mmseqs_file)

    return {
        "dna_file": dna_file,
        "orfipy_dir": orfipy_dir,
        "mmseqs_file": mmseqs_file,
    }


# Check if orfipy is available
try:
    import subprocess

    subprocess.run(["orfipy", "--help"], capture_output=True, check=True)
    ORFIPY_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    ORFIPY_AVAILABLE = False


# =============================================================================
# UNIT TESTS FOR CONSTRAINT UTILITY FUNCTIONS
# =============================================================================

from proto_language.utils import (
    validate_range,
    calculate_range_deviation,
    calculate_percentage_range_deviation,
    calculate_normalized_deviation,
    MIN_ENERGY,
    MAX_ENERGY,
)


class TestValidateRange:
    """Tests for validate_range() utility function."""
    
    def test_value_within_range(self):
        """Test that values within range pass validation."""
        # Should not raise any exception
        validate_range(50.0, 0.0, 100.0, "test_param")
        validate_range(0.0, 0.0, 100.0, "test_param")
        validate_range(100.0, 0.0, 100.0, "test_param")
        validate_range(-5.0, -10.0, 10.0, "test_param")
    
    def test_value_below_range(self):
        """Test that values below range raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be between"):
            validate_range(-1.0, 0.0, 100.0, "test_param")
        
        with pytest.raises(ValueError, match="gc_content must be between"):
            validate_range(-10.0, 0.0, 100.0, "gc_content")
    
    def test_value_above_range(self):
        """Test that values above range raise ValueError."""
        with pytest.raises(ValueError, match="test_param must be between"):
            validate_range(101.0, 0.0, 100.0, "test_param")
        
        with pytest.raises(ValueError, match="protein_len must be between"):
            validate_range(1000.0, 0.0, 500.0, "protein_len")
    
    def test_edge_cases(self):
        """Test edge cases for validation."""
        # Boundary values should pass
        validate_range(0.0, 0.0, 0.0, "zero_range")
        
        # Negative ranges
        validate_range(-50.0, -100.0, 0.0, "negative_range")


class TestCalculateRangeDeviation:
    """Tests for calculate_range_deviation() utility function."""
    
    def test_value_within_range(self):
        """Test that values within range have zero deviation."""
        assert calculate_range_deviation(50.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_range_deviation(40.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_range_deviation(60.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_range_deviation(55.5, 50.0, 60.0) == MIN_ENERGY
    
    def test_value_below_range(self):
        """Test deviation calculation for values below range."""
        # actual=30, min=40, max=60 -> deviation = (40-30)/40 = 0.25
        assert abs(calculate_range_deviation(30.0, 40.0, 60.0) - 0.25) < 1e-9
        
        # actual=0, min=40, max=60 -> deviation = (40-0)/40 = 1.0
        assert abs(calculate_range_deviation(0.0, 40.0, 60.0) - 1.0) < 1e-9
        
        # actual=10, min=50, max=100 -> deviation = (50-10)/50 = 0.8
        assert abs(calculate_range_deviation(10.0, 50.0, 100.0) - 0.8) < 1e-9
    
    def test_value_above_range(self):
        """Test deviation calculation for values above range."""
        # actual=70, min=40, max=60 -> deviation = (70-60)/60 = 0.166...
        assert abs(calculate_range_deviation(70.0, 40.0, 60.0) - (10.0/60.0)) < 1e-9
        
        # actual=120, min=40, max=60 -> deviation = (120-60)/60 = 1.0
        assert abs(calculate_range_deviation(120.0, 40.0, 60.0) - 1.0) < 1e-9
        
        # actual=200, min=50, max=100 -> deviation = (200-100)/100 = 1.0 (capped)
        deviation = calculate_range_deviation(200.0, 50.0, 100.0)
        assert abs(deviation - 1.0) < 1e-9
    
    def test_capping_at_max_energy(self):
        """Test that deviation is capped at MAX_ENERGY."""
        # Very large deviation
        deviation = calculate_range_deviation(0.0, 100.0, 200.0)
        assert deviation <= MAX_ENERGY
        
        deviation = calculate_range_deviation(1000.0, 10.0, 20.0)
        assert deviation <= MAX_ENERGY
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero minimum
        assert calculate_range_deviation(5.0, 10.0, 20.0) == 0.5
        
        # Exact match at boundaries
        assert calculate_range_deviation(10.0, 10.0, 20.0) == MIN_ENERGY
        assert calculate_range_deviation(20.0, 10.0, 20.0) == MIN_ENERGY


class TestCalculatePercentageRangeDeviation:
    """Tests for calculate_percentage_range_deviation() utility function."""
    
    def test_value_within_range(self):
        """Test that values within range have zero deviation."""
        assert calculate_percentage_range_deviation(50.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_percentage_range_deviation(40.0, 40.0, 60.0) == MIN_ENERGY
        assert calculate_percentage_range_deviation(60.0, 40.0, 60.0) == MIN_ENERGY
    
    def test_value_below_range(self):
        """Test deviation for values below range."""
        # actual=30, min=40, max=60 -> deviation = (40-30)/max(40,1) = 0.25
        assert abs(calculate_percentage_range_deviation(30.0, 40.0, 60.0) - 0.25) < 1e-9
        
        # actual=0, min=50, max=70 -> deviation = (50-0)/max(50,1) = 1.0
        assert abs(calculate_percentage_range_deviation(0.0, 50.0, 70.0) - 1.0) < 1e-9
        
        # Edge case: min_val is 0 (use 1 as denominator)
        # actual=0, min=0, max=50 -> deviation = (0-0)/max(0,1) = 0.0
        assert calculate_percentage_range_deviation(0.0, 0.0, 50.0) == MIN_ENERGY
    
    def test_value_above_range(self):
        """Test deviation for values above range."""
        # actual=70, min=40, max=60 -> deviation = (70-60)/max(100-60,1) = 10/40 = 0.25
        assert abs(calculate_percentage_range_deviation(70.0, 40.0, 60.0) - 0.25) < 1e-9
        
        # actual=100, min=40, max=60 -> deviation = (100-60)/max(100-60,1) = 40/40 = 1.0
        assert abs(calculate_percentage_range_deviation(100.0, 40.0, 60.0) - 1.0) < 1e-9
        
        # Edge case: max_val is 100 (denominator becomes 1)
        # actual=105, min=40, max=100 -> deviation = (105-100)/max(0,1) = 5/1 = 1.0 (capped)
        deviation = calculate_percentage_range_deviation(105.0, 40.0, 100.0)
        assert deviation <= MAX_ENERGY
    
    def test_capping_at_max_energy(self):
        """Test that deviation is capped at MAX_ENERGY."""
        # Very large deviation
        deviation = calculate_percentage_range_deviation(0.0, 90.0, 95.0)
        assert deviation <= MAX_ENERGY
        
        deviation = calculate_percentage_range_deviation(100.0, 5.0, 10.0)
        assert deviation <= MAX_ENERGY
    
    def test_edge_cases(self):
        """Test edge cases specific to percentage ranges."""
        # min_val = 0
        assert calculate_percentage_range_deviation(5.0, 0.0, 50.0) == MIN_ENERGY
        
        # max_val = 100
        assert calculate_percentage_range_deviation(95.0, 50.0, 100.0) == MIN_ENERGY
        
        # Full range 0-100
        assert calculate_percentage_range_deviation(50.0, 0.0, 100.0) == MIN_ENERGY


class TestCalculateNormalizedDeviation:
    """Tests for calculate_normalized_deviation() utility function."""
    
    def test_exact_match(self):
        """Test deviation when actual matches target."""
        assert calculate_normalized_deviation(50.0, 50.0) == MIN_ENERGY
        assert calculate_normalized_deviation(0.0, 0.0) == MIN_ENERGY
        assert calculate_normalized_deviation(100.0, 100.0) == MIN_ENERGY
    
    def test_deviation_below_target(self):
        """Test deviation when actual is below target."""
        # actual=40, target=50 -> deviation = |40-50|/max(50,1) = 10/50 = 0.2
        assert abs(calculate_normalized_deviation(40.0, 50.0) - 0.2) < 1e-9
        
        # actual=25, target=100 -> deviation = |25-100|/max(100,1) = 75/100 = 0.75
        assert abs(calculate_normalized_deviation(25.0, 100.0) - 0.75) < 1e-9
    
    def test_deviation_above_target(self):
        """Test deviation when actual is above target."""
        # actual=60, target=50 -> deviation = |60-50|/max(50,1) = 10/50 = 0.2
        assert abs(calculate_normalized_deviation(60.0, 50.0) - 0.2) < 1e-9
        
        # actual=150, target=100 -> deviation = |150-100|/max(100,1) = 50/100 = 0.5
        assert abs(calculate_normalized_deviation(150.0, 100.0) - 0.5) < 1e-9
    
    def test_capping_at_max_energy(self):
        """Test that deviation is capped at MAX_ENERGY."""
        # Large deviation: actual=200, target=50 -> deviation = 150/50 = 3.0, capped at 1.0
        deviation = calculate_normalized_deviation(200.0, 50.0)
        assert deviation == MAX_ENERGY
        
        # Very large deviation
        deviation = calculate_normalized_deviation(0.0, 100.0)
        assert deviation == MAX_ENERGY
    
    def test_zero_target(self):
        """Test behavior when target is 0 (uses 1 as denominator)."""
        # actual=10, target=0 -> deviation = |10-0|/max(0,1) = 10/1 = 1.0 (capped)
        deviation = calculate_normalized_deviation(10.0, 0.0)
        assert deviation == MAX_ENERGY
        
        # actual=0, target=0 -> deviation = |0-0|/max(0,1) = 0/1 = 0.0
        deviation = calculate_normalized_deviation(0.0, 0.0)
        assert deviation == MIN_ENERGY
    
    def test_symmetry(self):
        """Test that deviation is symmetric around target."""
        # |40-50| should equal |60-50|
        dev_below = calculate_normalized_deviation(40.0, 50.0)
        dev_above = calculate_normalized_deviation(60.0, 50.0)
        assert abs(dev_below - dev_above) < 1e-9
        
        # |75-100| should equal |125-100|
        dev_below = calculate_normalized_deviation(75.0, 100.0)
        dev_above = calculate_normalized_deviation(125.0, 100.0)
        assert abs(dev_below - dev_above) < 1e-9
    
    def test_negative_values(self):
        """Test with negative values."""
        # actual=-10, target=-20 -> deviation = |-10-(-20)|/max(-20,1) = 10/1 = 1.0 (capped)
        # Note: max(target, 1) means negative targets use 1 as denominator
        deviation = calculate_normalized_deviation(-10.0, -20.0)
        assert deviation <= MAX_ENERGY
        
        # actual=-30, target=-10 -> deviation = |-30-(-10)|/max(-10,1) = 20/1 = 1.0 (capped)
        deviation = calculate_normalized_deviation(-30.0, -10.0)
        assert deviation <= MAX_ENERGY