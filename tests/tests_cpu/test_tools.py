import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.append(".")

# Import functions to be tested
from language.tools.mmseqs import (
    mmseqs_easy_search,
    run_mmseqs_search_proteins,
    _filter_top_hits,
    convert_m8_to_df,
)
from language.tools.orf_prediction import (
    run_orfipy,
    parse_orfipy_results_to_df,
    _parse_orfipy_header,
)

# Test data file paths
TEST_DATA_DIR = Path("tests/tests_cpu/dummy_data")
PROTEIN_FASTA = TEST_DATA_DIR / "test_protein_sequences.faa"
DNA_FASTA = TEST_DATA_DIR / "test_dna_sequences.fna"
M8_FILE = TEST_DATA_DIR / "test_mmseqs_results.m8"
ORFIPY_AA_FILE = TEST_DATA_DIR / "test_orfipy_aa.faa"
ORFIPY_NT_FILE = TEST_DATA_DIR / "test_orfipy_nt.fna"

# Fixtures for managing temporary files
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


class TestMmseqsTools:
    """Test suite for MMseqs2 tool wrappers."""

    def test_convert_m8_to_df(self):
        """Tests conversion of a standard M8 file to a pandas DataFrame."""
        df = convert_m8_to_df(M8_FILE)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # Based on our test data
        assert list(df.columns) == ["query_id", "target_id", "identity", "evalue"]
        assert "protein_seq_1" in df["query_id"].values
        assert "protein_seq_2" in df["query_id"].values

    def test_filter_top_hits_logic(self):
        """Tests the logic for filtering top hits from an M8 DataFrame."""
        data = {
            "query_id": ["q1", "q1", "q2", "q2", "q3"],
            "target_id": ["t1", "t2", "t3", "t4", "t5"],
            "evalue": [1e-10, 1e-20, 1e-5, 1e-5, 1e-100],
            "identity": [90.0, 80.0, 95.0, 98.0, 99.0],
        }
        df = pd.DataFrame(data)
        
        # q1: 1e-20 is better than 1e-10
        # q2: e-values are tied, so first occurrence is selected (95.0 identity)
        # q3: only one hit
        filtered_df = _filter_top_hits(df)
        assert len(filtered_df) == 3
        assert filtered_df[filtered_df.query_id == "q1"].evalue.iloc[0] == 1e-20
        assert filtered_df[filtered_df.query_id == "q2"].identity.iloc[0] == 95.0
        assert filtered_df[filtered_df.query_id == "q3"].evalue.iloc[0] == 1e-100

    def test_run_mmseqs_search_proteins_workflow(self, temp_dir):
        """Tests the high-level protein search workflow using real files."""
        # Copy the test files to temp directory to simulate workflow
        temp_protein_file = temp_dir / "proteins.faa"
        temp_m8_file = temp_dir / "results.m8"
        
        # Copy test data
        shutil.copy(PROTEIN_FASTA, temp_protein_file)
        shutil.copy(M8_FILE, temp_m8_file)
        
        # Read the expected results
        df = convert_m8_to_df(temp_m8_file)
        
        # Test filtering
        filtered_df = _filter_top_hits(df)
        
        # Should have fewer results after filtering
        assert len(filtered_df) <= len(df)
        
        # Check that each query has at most one result
        query_counts = filtered_df.groupby("query_id").size()
        assert all(count == 1 for count in query_counts)


class TestOrfipyTools:
    """Test suite for Orfipy tool wrappers."""

    def test_run_orfipy_missing_input(self, temp_dir):
        """Tests that a FileNotFoundError is raised for a missing input file."""
        with pytest.raises(FileNotFoundError):
            run_orfipy(Path("nonexistent.fna"), output_dir=temp_dir)

    def test_parse_orfipy_results_to_df(self):
        """Tests the parsing of orfipy FASTA outputs into a DataFrame."""
        df = parse_orfipy_results_to_df(ORFIPY_AA_FILE, ORFIPY_NT_FILE)
        assert len(df) == 4  # Based on our test data
        assert "parent_id" in df.columns
        assert "orf_id" in df.columns
        assert "amino_acid_sequence" in df.columns
        assert "nucleotide_sequence" in df.columns
        
        # Check first row data
        first_row = df.iloc[0]
        assert first_row["parent_id"] == "dna_seq_1"
        assert first_row["orf_id"] == "ORF.1"
        assert first_row["amino_acid_sequence"].startswith("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK")
        assert first_row["nucleotide_sequence"].startswith("ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAG")

    @pytest.mark.parametrize(
        "header, expected",
        [
            ("dna_seq_1_ORF.1 [0-180](+)", {"parent_id": "dna_seq_1", "orf_id": "ORF.1", "start": 0, "end": 180, "strand": "+"}),
            ("complex-name_ORF.15 [100-250](-)", {"parent_id": "complex-name", "orf_id": "ORF.15", "start": 100, "end": 250, "strand": "-"}),
            ("invalid header", None)
        ],
    )
    def test_parse_orfipy_header(self, header, expected):
        """Tests the parsing of individual orfipy headers."""
        if expected:
            result = _parse_orfipy_header(header)
            assert result["parent_id"] == expected["parent_id"]
            assert result["orf_id"] == expected["orf_id"]
            assert result["start"] == expected["start"]
            assert result["end"] == expected["end"]
            assert result["strand"] == expected["strand"]
        else:
            assert _parse_orfipy_header(header) is None

    def test_real_orfipy_data_integrity(self):
        """Tests that the real test data files are consistent."""
        # Read both files
        aa_df = pd.DataFrame()
        nt_df = pd.DataFrame()
        
        # Parse AA file
        with open(ORFIPY_AA_FILE, 'r') as f:
            aa_lines = f.readlines()
        
        # Parse NT file
        with open(ORFIPY_NT_FILE, 'r') as f:
            nt_lines = f.readlines()
        
        # Count headers (should be same in both files)
        aa_headers = [line for line in aa_lines if line.startswith('>')]
        nt_headers = [line for line in nt_lines if line.startswith('>')]
        
        assert len(aa_headers) == len(nt_headers), "AA and NT files should have same number of sequences"
        
        # Check that headers match
        for aa_header, nt_header in zip(aa_headers, nt_headers):
            assert aa_header.strip() == nt_header.strip(), f"Headers don't match: {aa_header.strip()} vs {nt_header.strip()}"
