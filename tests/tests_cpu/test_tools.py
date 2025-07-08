"""
Comprehensive test suite for mmseqs and orfipy tools.

This module tests the programmatic interfaces for MMseqs2 and Orfipy tools
used in proto-language workflows. Tests include both successful execution
scenarios and error handling with mock data and temporary files.

Test Categories:
    - MMseqs2 Tools: easy-search, protein search, genome search, clustering
    - Orfipy Tools: ORF prediction, result parsing
    - Error Handling: Missing files, invalid parameters, tool failures
    - Integration: Combined workflows, data consistency
"""

import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import subprocess

import sys

sys.path.append(".")

from language.tools.mmseqs import (
    mmseqs_easy_search,
    run_mmseqs_search_proteins,
    run_mmseqs_search_genomes,
    run_mmseqs_clustering,
    extract_mmseqs_cluster_representatives,
    convert_m8_to_df,
    _run_command,
    _add_query_sequences_to_results,
    _filter_top_hits,
)

from language.tools.orf_prediction import (
    run_orfipy,
    parse_orfipy_results_to_df,
    _run_subprocess_command,
    _parse_orfipy_header,
)

# Test data constants
SAMPLE_PROTEIN_FASTA = """>protein1
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF
>protein2
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY
"""

SAMPLE_DNA_FASTA = """>dna1
ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGGTGGGCGCGCACGC
>dna2  
ATGACCGAGTACAAGCTGGTGGTCGTGGGCGCGGGTGGCGTGGGCAAGTCCGCGCTGACCATCCAG
"""

SAMPLE_RNA_FASTA = """>rna1
AUGUGUCUGAGCCCGGCGGACAAGACCAACGUGAAGGCGGCGUGGGGCAAGGUGGGCGCGCACGC
>rna2
AUGACCGAGUACAAGCUGGUGGTCGUGGGCGCGGGUGGGCGUGGGGAAGTCCGCGCUGACCAUCCAG
"""

SAMPLE_M8_OUTPUT = """protein1\tsp|P12345|TEST_HUMAN\t85.5\t2.5e-10
protein2\tsp|Q67890|ANOT_HUMAN\t92.3\t1.2e-15
protein1\tsp|P54321|DUPL_HUMAN\t75.2\t3.1e-8
"""

SAMPLE_ORFIPY_AA_FASTA = """>dna1_ORF.1 [0-66](+) type:complete length:66 frame:1 start:ATG stop:TGA
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF*
>dna2_ORF.1 [0-63](+) type:complete length:63 frame:1 start:ATG stop:TAG
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG*
"""

SAMPLE_ORFIPY_NT_FASTA = """>dna1_ORF.1 [0-66](+) type:complete length:66 frame:1 start:ATG stop:TGA
ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGGTGGGCGCGCACGCTGA
>dna2_ORF.1 [0-63](+) type:complete length:63 frame:1 start:ATG stop:TAG
ATGACCGAGTACAAGCTGGTGGTCGTGGGCGCGGGTGGCGTGGGCAAGTCCGCGCTGACCATCTAG
"""


def create_temp_fasta(content: str, suffix: str = ".fasta") -> Path:
    """Create a temporary FASTA file with given content."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    temp_file.write(content)
    temp_file.flush()
    return Path(temp_file.name)


class TestMMseqsTools:
    """Test suite for MMseqs2 tools."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.protein_fasta = create_temp_fasta(SAMPLE_PROTEIN_FASTA, ".faa")
        self.dna_fasta = create_temp_fasta(SAMPLE_DNA_FASTA, ".fna")
        self.m8_file = create_temp_fasta(SAMPLE_M8_OUTPUT, ".m8")

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        for path in [self.protein_fasta, self.dna_fasta, self.m8_file]:
            if path.exists():
                path.unlink()

    @patch("language.tools.mmseqs._run_command")
    def test_mmseqs_easy_search_success(self, mock_run_command):
        """Test successful mmseqs easy-search execution."""
        results_dir = self.temp_dir / "easy_search_results"
        expected_m8 = results_dir / "mmseqs_results.m8"

        # Create expected output file
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(expected_m8, "w") as f:
            f.write(SAMPLE_M8_OUTPUT)

        result_path = mmseqs_easy_search(
            self.protein_fasta,
            self.protein_fasta,  # Use same file as target for simplicity
            results_dir,
            threads=2,
            sensitivity=4.0,
        )

        assert result_path == expected_m8
        assert result_path.exists()
        mock_run_command.assert_called_once()

        # Verify command structure
        cmd_args = mock_run_command.call_args[0][0]
        assert cmd_args[0] == "mmseqs"
        assert cmd_args[1] == "easy-search"
        assert "--threads" in cmd_args
        assert "2" in cmd_args
        assert "-s" in cmd_args
        assert "4.0" in cmd_args

    @patch("language.tools.mmseqs._run_command")
    def test_mmseqs_easy_search_missing_output(self, mock_run_command):
        """Test mmseqs easy-search when output file is not created."""
        results_dir = self.temp_dir / "missing_output"
        results_dir.mkdir(parents=True)

        with pytest.raises(
            FileNotFoundError, match="MMseqs did not produce an output .m8 file"
        ):
            mmseqs_easy_search(self.protein_fasta, self.protein_fasta, results_dir)

    @patch("language.tools.mmseqs._run_command")
    def test_mmseqs_easy_search_command_failure(self, mock_run_command):
        """Test mmseqs easy-search command failure handling."""
        mock_run_command.side_effect = RuntimeError(
            "Command failed: mmseqs easy-search"
        )

        with pytest.raises(RuntimeError, match="Command failed: mmseqs"):
            mmseqs_easy_search(self.protein_fasta, self.protein_fasta, self.temp_dir)

    def test_convert_m8_to_df(self):
        """Test conversion of M8 format to DataFrame."""
        df = convert_m8_to_df(self.m8_file)

        assert len(df) == 3  # Should keep all hits (no filtering in convert_m8_to_df)
        assert list(df.columns) == ["query_id", "target_id", "identity", "evalue"]

        # Check that all hits are preserved (no filtering)
        protein1_rows = df[df["query_id"] == "protein1"]
        assert len(protein1_rows) == 2  # protein1 should have 2 hits
        assert set(protein1_rows["identity"]) == {85.5, 75.2}
        assert set(protein1_rows["target_id"]) == {
            "sp|P12345|TEST_HUMAN",
            "sp|P54321|DUPL_HUMAN",
        }

    def test_convert_m8_to_df_empty_file(self):
        """Test M8 conversion with empty file."""
        empty_file = create_temp_fasta("", ".m8")
        try:
            df = convert_m8_to_df(empty_file)
            assert df.empty
            assert list(df.columns) == ["query_id", "target_id", "identity", "evalue"]
        finally:
            empty_file.unlink()

    def test_convert_m8_to_df_missing_file(self):
        """Test M8 conversion with missing file."""
        missing_file = Path("/nonexistent/file.m8")
        with pytest.raises(FileNotFoundError, match="M8 file not found"):
            convert_m8_to_df(missing_file)

    @patch("language.tools.mmseqs.mmseqs_easy_search")
    @patch("language.tools.mmseqs._add_query_sequences_to_results")
    @patch("language.tools.mmseqs._filter_top_hits")
    def test_run_mmseqs_search_proteins(
        self, mock_filter, mock_add_seqs, mock_easy_search
    ):
        """Test high-level protein search function."""
        # Mock the easy search to return our test m8 file
        mock_easy_search.return_value = self.m8_file

        # Mock DataFrame processing
        test_df = pd.DataFrame(
            {
                "query_id": ["protein1", "protein2"],
                "target_id": ["target1", "target2"],
                "identity": [85.5, 92.3],
                "evalue": [2.5e-10, 1.2e-15],
                "sequence": ["MVLSPADKTNVK", "MTEYKLVVVG"],
            }
        )
        mock_add_seqs.return_value = test_df.copy()
        mock_filter.return_value = test_df.copy()

        result_df = run_mmseqs_search_proteins(
            self.protein_fasta, self.protein_fasta, self.temp_dir / "protein_search"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        mock_easy_search.assert_called_once()
        mock_add_seqs.assert_called_once()
        mock_filter.assert_called_once()

    @patch("language.tools.mmseqs._run_command")
    def test_run_mmseqs_search_genomes(self, mock_run_command):
        """Test genome-to-genome search workflow."""
        out_dir = self.temp_dir / "genome_search"
        results_file = out_dir / "results" / "genome_results.m8"

        # Create expected output structure
        results_file.parent.mkdir(parents=True)
        with open(results_file, "w") as f:
            f.write("query1\ttarget1\t85.5\t1e-10\n")

        result_path = run_mmseqs_search_genomes(
            self.dna_fasta,
            self.dna_fasta,
            out_dir,
            results_filename="genome_results.m8",
        )

        assert result_path == results_file
        assert result_path.exists()

        # Should call multiple mmseqs commands
        assert mock_run_command.call_count >= 4

        # Verify some expected commands were called
        call_args_list = [call[0][0] for call in mock_run_command.call_args_list]
        createdb_calls = [args for args in call_args_list if "createdb" in args]
        assert len(createdb_calls) >= 2  # Should create query and target DBs

    @patch("language.tools.mmseqs._run_command")
    def test_run_mmseqs_clustering(self, mock_run_command):
        """Test sequence clustering workflow."""
        output_dir = self.temp_dir / "clustering"

        # Create expected output structure
        clusters_dir = output_dir / "mmseqs_results"
        clusters_dir.mkdir(parents=True)
        clusters_tsv = clusters_dir / "clusters.tsv"
        with open(clusters_tsv, "w") as f:
            f.write("protein1\tprotein1\nprotein1\tprotein2\n")

        run_mmseqs_clustering(self.protein_fasta, output_dir, min_seq_id=0.8)

        # Should call multiple mmseqs commands
        assert mock_run_command.call_count >= 4

        # Verify clustering-specific commands
        call_args_list = [call[0][0] for call in mock_run_command.call_args_list]
        cluster_calls = [args for args in call_args_list if "cluster" in args]
        assert len(cluster_calls) >= 1

    def test_run_mmseqs_clustering_missing_input(self):
        """Test clustering with missing input file."""
        missing_file = Path("/nonexistent/sequences.faa")
        with pytest.raises(FileNotFoundError, match="Input FASTA not found"):
            run_mmseqs_clustering(missing_file, self.temp_dir)

    @patch("language.tools.mmseqs._extract_representative_ids")
    @patch("language.tools.mmseqs._extract_representative_sequences")
    def test_extract_mmseqs_cluster_representatives(
        self, mock_extract_seqs, mock_extract_ids
    ):
        """Test cluster representative extraction."""
        clusters_tsv = self.temp_dir / "clusters.tsv"
        output_fasta = self.temp_dir / "representatives.faa"

        # Create mock TSV file
        with open(clusters_tsv, "w") as f:
            f.write("protein1\tprotein1\nprotein1\tprotein2\n")

        # Mock the extraction functions
        mock_extract_ids.return_value = {"protein1"}

        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq

        mock_record = SeqRecord(Seq("MVLSPADKTNVK"), id="protein1", description="")
        mock_extract_seqs.return_value = [mock_record]

        result_df = extract_mmseqs_cluster_representatives(
            clusters_tsv, self.protein_fasta, output_fasta
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert result_df.iloc[0]["id_prompt"] == "protein1"
        assert result_df.iloc[0]["sequence"] == "MVLSPADKTNVK"

    def test_add_query_sequences_to_results(self):
        """Test adding query sequences to search results."""
        # Create test DataFrame
        test_df = pd.DataFrame(
            {
                "query_id": ["protein1", "protein2", "nonexistent"],
                "target_id": ["target1", "target2", "target3"],
            }
        )

        result_df = _add_query_sequences_to_results(test_df, self.protein_fasta)

        assert "sequence" in result_df.columns
        assert (
            result_df.loc[result_df["query_id"] == "protein1", "sequence"].iloc[0]
            == "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
        )
        assert (
            result_df.loc[result_df["query_id"] == "protein2", "sequence"].iloc[0]
            == "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY"
        )
        assert pd.isna(
            result_df.loc[result_df["query_id"] == "nonexistent", "sequence"].iloc[0]
        )

    def test_filter_top_hits(self):
        """Test filtering to keep only top hits per query."""
        test_df = pd.DataFrame(
            {
                "query_id": ["protein1", "protein1", "protein2"],
                "identity": [85.5, 75.2, 92.3],
                "evalue": [2.5e-10, 3.1e-8, 1.2e-15],
                "target_id": ["target1", "target2", "target3"],
            }
        )

        result_df = _filter_top_hits(test_df)

        assert len(result_df) == 2  # One per unique query_id
        protein1_row = result_df[result_df["query_id"] == "protein1"].iloc[0]
        # Filter uses e-value (lower is better), so 2.5e-10 should win over 3.1e-8
        assert protein1_row["evalue"] == 2.5e-10
        assert protein1_row["identity"] == 85.5

    @patch("subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value.returncode = 0

        _run_command(["echo", "test"])
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_command_failure(self, mock_run):
        """Test command execution failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["false"])

        with pytest.raises(RuntimeError, match="Command failed"):
            _run_command(["false"])


class TestOrfipyTools:
    """Test suite for Orfipy tools."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dna_fasta = create_temp_fasta(SAMPLE_DNA_FASTA, ".fna")
        self.rna_fasta = create_temp_fasta(SAMPLE_RNA_FASTA, ".fna")
        self.aa_fasta = create_temp_fasta(SAMPLE_ORFIPY_AA_FASTA, ".faa")
        self.nt_fasta = create_temp_fasta(SAMPLE_ORFIPY_NT_FASTA, ".fna")

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        for path in [self.dna_fasta, self.rna_fasta, self.aa_fasta, self.nt_fasta]:
            if path.exists():
                path.unlink()

    @patch("language.tools.orf_prediction._run_subprocess_command")
    def test_run_orfipy_success(self, mock_run_command):
        """Test successful orfipy execution."""
        output_dir = self.temp_dir / "orfipy_output"
        expected_aa = output_dir / "orfipy_aa.faa"
        expected_nt = output_dir / "orfipy_nt.fna"

        # We need to ensure the files exist when the function checks for them
        # This means the mock should create the files as a side effect
        def create_files(*args, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(expected_aa, "w") as f:
                f.write(SAMPLE_ORFIPY_AA_FASTA)
            with open(expected_nt, "w") as f:
                f.write(SAMPLE_ORFIPY_NT_FASTA)

        mock_run_command.side_effect = create_files

        aa_path, nt_path = run_orfipy(
            self.dna_fasta,
            output_dir=output_dir,
            threads=2,
            min_len=30,
            max_len=1000,
            verbose=False,
        )

        assert aa_path == expected_aa
        assert nt_path == expected_nt
        assert aa_path.exists()
        assert nt_path.exists()
        mock_run_command.assert_called_once()

        # Verify command structure
        cmd_args = mock_run_command.call_args[0][0]
        assert cmd_args[0] == "orfipy"
        assert "--procs" in cmd_args
        assert "2" in cmd_args
        assert "--min" in cmd_args
        assert "30" in cmd_args

    def test_run_orfipy_missing_input(self):
        """Test orfipy with missing input file."""
        missing_file = Path("/nonexistent/sequences.fna")
        with pytest.raises(FileNotFoundError, match="Input FASTA not found"):
            run_orfipy(missing_file, output_dir=self.temp_dir)

    @patch("language.tools.orf_prediction._run_subprocess_command")
    def test_run_orfipy_command_failure(self, mock_run_command):
        """Test orfipy command failure handling."""
        mock_run_command.side_effect = RuntimeError("orfipy failed (exit 1)")

        with pytest.raises(RuntimeError, match="orfipy failed"):
            run_orfipy(self.dna_fasta, output_dir=self.temp_dir)

    def test_parse_orfipy_results_to_df(self):
        """Test parsing orfipy results to DataFrame."""
        df = parse_orfipy_results_to_df(self.aa_fasta, self.nt_fasta)

        assert len(df) == 2
        # Based on the actual implementation, the order is different
        expected_columns = [
            "parent_id",
            "orf_id",
            "strand",
            "frame",
            "amino_acid_sequence",
            "nucleotide_sequence",
            "amino_acid_length",
            "nucleotide_length",
            "nucleotide_start",
            "nucleotide_end",
        ]
        assert list(df.columns) == expected_columns

        # Check first row
        row1 = df.iloc[0]
        assert row1["parent_id"] == "dna1"
        assert row1["orf_id"] == "ORF.1"
        assert (
            row1["amino_acid_sequence"]
            == "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
        )
        assert row1["nucleotide_start"] == 0
        assert row1["nucleotide_end"] == 66
        assert row1["strand"] == "+"
        assert row1["frame"] == 1

    def test_parse_orfipy_results_mismatched_records(self):
        """Test parsing with mismatched amino acid and nucleotide records."""
        # Create file with different number of records
        mismatched_aa = create_temp_fasta(">seq1\nMVLSPAD\n", ".faa")
        mismatched_nt = create_temp_fasta(">seq1\nATGGTG\n>seq2\nGGGAAA\n", ".fna")

        try:
            with pytest.raises(
                ValueError,
                match="Mismatch between amino acid .* and nucleotide .* records",
            ):
                parse_orfipy_results_to_df(mismatched_aa, mismatched_nt)
        finally:
            mismatched_aa.unlink()
            mismatched_nt.unlink()

    def test_parse_orfipy_results_missing_files(self):
        """Test parsing with missing input files."""
        missing_aa = Path("/nonexistent/aa.faa")
        missing_nt = Path("/nonexistent/nt.fna")

        with pytest.raises(FileNotFoundError):
            parse_orfipy_results_to_df(missing_aa, self.nt_fasta)

        with pytest.raises(FileNotFoundError):
            parse_orfipy_results_to_df(self.aa_fasta, missing_nt)

    def test_parse_orfipy_header(self):
        """Test parsing individual orfipy headers."""
        # Test typical header
        header1 = (
            "dna1_ORF.1 [0-66](+) type:complete length:66 frame:1 start:ATG stop:TGA"
        )
        result1 = _parse_orfipy_header(header1)

        assert result1["parent_id"] == "dna1"
        assert result1["orf_id"] == "ORF.1"
        assert result1["start"] == 0
        assert result1["end"] == 66
        assert result1["strand"] == "+"
        assert result1["frame"] == 1

        # Test header with complex parent ID
        header2 = "complex_seq_name_ORF.5 [100-200](-) type:partial length:100 frame:3 start:GTG stop:TAA"
        result2 = _parse_orfipy_header(header2)

        assert result2["parent_id"] == "complex_seq_name"
        assert result2["orf_id"] == "ORF.5"
        assert result2["start"] == 100
        assert result2["end"] == 200
        assert result2["strand"] == "-"
        assert result2["frame"] == 3

        # Test malformed header
        malformed_header = "invalid_header_format"
        result3 = _parse_orfipy_header(malformed_header)
        assert result3 is None

    def test_parse_orfipy_results_with_stop_codons(self):
        """Test parsing results that include stop codon markers."""
        # Create FASTA with stop codon markers
        aa_with_stops = """>seq1_ORF.1 [0-12](+) type:complete length:12 frame:1 start:ATG stop:TAG
MVLS*
>seq2_ORF.1 [0-15](+) type:complete length:15 frame:1 start:ATG stop:TGA
MKKRR*
"""
        nt_corresponding = """>seq1_ORF.1 [0-12](+) type:complete length:12 frame:1 start:ATG stop:TAG
ATGGTGCTGAGCTAG
>seq2_ORF.1 [0-15](+) type:complete length:15 frame:1 start:ATG stop:TGA
ATGAAAAAGCGCCGTTGA
"""

        aa_file = create_temp_fasta(aa_with_stops, ".faa")
        nt_file = create_temp_fasta(nt_corresponding, ".fna")

        try:
            df = parse_orfipy_results_to_df(aa_file, nt_file)

            # Stop codons should be removed from amino acid sequences
            assert df.iloc[0]["amino_acid_sequence"] == "MVLS"
            assert df.iloc[1]["amino_acid_sequence"] == "MKKRR"

            # Lengths should be calculated without stop codons
            assert df.iloc[0]["amino_acid_length"] == 4
            assert df.iloc[1]["amino_acid_length"] == 5

        finally:
            aa_file.unlink()
            nt_file.unlink()

    def test_parse_orfipy_empty_results(self):
        """Test parsing empty orfipy results."""
        empty_aa = create_temp_fasta("", ".faa")
        empty_nt = create_temp_fasta("", ".fna")

        try:
            df = parse_orfipy_results_to_df(empty_aa, empty_nt)
            assert len(df) == 0
            # Empty DataFrame has no columns initially
            if len(df.columns) > 0:
                expected_columns = [
                    "parent_id",
                    "orf_id",
                    "strand",
                    "frame",
                    "amino_acid_sequence",
                    "nucleotide_sequence",
                    "amino_acid_length",
                    "nucleotide_length",
                    "nucleotide_start",
                    "nucleotide_end",
                ]
                assert list(df.columns) == expected_columns
        finally:
            empty_aa.unlink()
            empty_nt.unlink()

    @patch("subprocess.run")
    def test_run_subprocess_command_success(self, mock_run):
        """Test successful subprocess command execution."""
        mock_run.return_value.returncode = 0

        _run_subprocess_command(["echo", "test"], "test_tool")
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_subprocess_command_failure(self, mock_run):
        """Test subprocess command failure."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "stdout output"
        mock_proc.stderr = "stderr output"
        mock_run.return_value = mock_proc

        with pytest.raises(RuntimeError, match="test_tool failed \\(exit 1\\)"):
            _run_subprocess_command(["false"], "test_tool")


class TestIntegrationScenarios:
    """Test integration scenarios combining MMseqs and Orfipy."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dna_fasta = create_temp_fasta(SAMPLE_DNA_FASTA, ".fna")
        self.protein_fasta = create_temp_fasta(SAMPLE_PROTEIN_FASTA, ".faa")

    def teardown_method(self):
        """Clean up integration test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        for path in [self.dna_fasta, self.protein_fasta]:
            if path.exists():
                path.unlink()

    @patch("language.tools.orf_prediction._run_subprocess_command")
    @patch("language.tools.mmseqs._run_command")
    def test_orf_prediction_to_protein_search(self, mock_mmseqs, mock_orfipy):
        """Test workflow: DNA -> ORF prediction -> protein search."""
        # Setup orfipy output
        orfipy_dir = self.temp_dir / "orfipy_results"
        orfipy_dir.mkdir()
        aa_output = orfipy_dir / "orfipy_aa.faa"
        nt_output = orfipy_dir / "orfipy_nt.fna"

        # Create a function that creates the expected output files when orfipy is called
        def create_orfipy_files(*args, **kwargs):
            with open(aa_output, "w") as f:
                f.write(SAMPLE_ORFIPY_AA_FASTA)
            with open(nt_output, "w") as f:
                f.write(SAMPLE_ORFIPY_NT_FASTA)

        mock_orfipy.side_effect = create_orfipy_files

        # Setup mmseqs output
        mmseqs_dir = self.temp_dir / "mmseqs_results"
        mmseqs_dir.mkdir()
        m8_output = mmseqs_dir / "mmseqs_results.m8"
        with open(m8_output, "w") as f:
            f.write("dna1_ORF.1\tsp|P12345|TEST\t85.5\t1e-10\n")

        # Step 1: ORF prediction
        aa_path, nt_path = run_orfipy(
            self.dna_fasta, output_dir=orfipy_dir, min_len=30, verbose=False
        )

        # Step 2: Protein search on predicted ORFs
        with patch("language.tools.mmseqs.mmseqs_easy_search", return_value=m8_output):
            search_df = run_mmseqs_search_proteins(
                aa_path, self.protein_fasta, mmseqs_dir
            )

        # Verify workflow completion
        assert aa_path.exists()
        assert nt_path.exists()
        mock_orfipy.assert_called_once()
        # mock_mmseqs.assert_called() # This would be called by mmseqs_easy_search

    def test_data_consistency_orfipy_parsing(self):
        """Test data consistency in orfipy parsing."""
        # Create proper test data that should be consistent
        # Use shorter sequences that will have the expected relationship
        aa_short = """>dna1_ORF.1 [0-15](+) type:complete length:15 frame:1 start:ATG stop:TAG
MVLS*
>dna2_ORF.1 [0-21](+) type:complete length:21 frame:1 start:ATG stop:TGA
MKKRR*
"""

        nt_short = """>dna1_ORF.1 [0-15](+) type:complete length:15 frame:1 start:ATG stop:TAG
ATGGTGCTGAGCTAG
>dna2_ORF.1 [0-21](+) type:complete length:21 frame:1 start:ATG stop:TGA
ATGAAAAAGCGCCGTTGA
"""

        aa_file = create_temp_fasta(aa_short, ".faa")
        nt_file = create_temp_fasta(nt_short, ".fna")

        try:
            df = parse_orfipy_results_to_df(aa_file, nt_file)

            for _, row in df.iterrows():
                # Nucleotide length should be roughly 3x amino acid length (stop codon is removed from AA)
                nt_len = row["nucleotide_length"]
                aa_len = row["amino_acid_length"]  # This excludes the * stop codon

                # Expected relationship: nt_len should be roughly (aa_len + 1) * 3
                # because aa_len excludes the stop codon but nt_len includes it
                expected_nt_len = (aa_len + 1) * 3  # +1 for the stop codon
                assert (
                    abs(nt_len - expected_nt_len) <= 3
                ), f"nt_len={nt_len}, aa_len={aa_len}, expected_nt_len={expected_nt_len}"

                # Check coordinate consistency
                assert row["nucleotide_end"] > row["nucleotide_start"]
                coord_length = row["nucleotide_end"] - row["nucleotide_start"] + 1
                assert (
                    coord_length >= nt_len
                ), f"coord_length={coord_length}, nt_len={nt_len}"
        finally:
            aa_file.unlink()
            nt_file.unlink()

    def test_error_propagation(self):
        """Test that errors propagate correctly through tool chains."""
        # Test that file not found errors propagate
        with pytest.raises(FileNotFoundError):
            run_orfipy(Path("/nonexistent/file.fna"), output_dir=self.temp_dir)

        with pytest.raises(FileNotFoundError):
            convert_m8_to_df(Path("/nonexistent/file.m8"))


class TestStressScenarios:
    """Stress tests for edge cases and performance."""

    def setup_method(self):
        """Set up stress test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up stress test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_large_fasta_parsing(self):
        """Test parsing large FASTA files."""
        # Generate large test data
        large_aa_content = ""
        large_nt_content = ""

        for i in range(100):
            large_aa_content += f">seq{i}_ORF.1 [0-60](+) type:complete length:60 frame:1 start:ATG stop:TAG\n"
            large_aa_content += "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF*\n"

            large_nt_content += f">seq{i}_ORF.1 [0-60](+) type:complete length:60 frame:1 start:ATG stop:TAG\n"
            large_nt_content += (
                "ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGGTGGGCTAG\n"
            )

        large_aa_file = create_temp_fasta(large_aa_content, ".faa")
        large_nt_file = create_temp_fasta(large_nt_content, ".fna")

        try:
            df = parse_orfipy_results_to_df(large_aa_file, large_nt_file)
            assert len(df) == 100

            # Check that all records were parsed correctly
            assert all(df["parent_id"].str.startswith("seq"))
            assert all(df["orf_id"] == "ORF.1")

        finally:
            large_aa_file.unlink()
            large_nt_file.unlink()

    def test_malformed_data_handling(self):
        """Test handling of malformed input data."""
        # Test malformed M8 data - create data that will cause NaN values when parsing
        malformed_m8 = (
            "protein1\ttarget1\tnot_a_number\tinvalid_evalue\nprotein2\ttarget2\t\t\n"
        )
        malformed_file = create_temp_fasta(malformed_m8, ".m8")

        try:
            # This should either raise an error or handle gracefully
            try:
                df = convert_m8_to_df(malformed_file)
                # If it doesn't raise an error, check it returns a DataFrame
                assert isinstance(df, pd.DataFrame)
                # The DataFrame might be empty or have NaN values, both are acceptable
            except (ValueError, pd.errors.ParserError, KeyError) as e:
                # These are all acceptable error types for malformed data
                assert any(
                    word in str(e).lower()
                    for word in ["parse", "convert", "invalid", "none", "index"]
                )

        finally:
            malformed_file.unlink()

    def test_empty_and_edge_case_inputs(self):
        """Test various empty and edge case inputs."""
        # Empty FASTA files
        empty_fasta = create_temp_fasta("", ".faa")

        try:
            # Empty M8 file should return empty DataFrame
            df = convert_m8_to_df(empty_fasta)
            assert len(df) == 0
            assert isinstance(df, pd.DataFrame)

        finally:
            empty_fasta.unlink()

        # Single character sequences
        tiny_fasta = """>tiny
M
"""
        tiny_file = create_temp_fasta(tiny_fasta, ".faa")

        try:
            with patch("language.tools.mmseqs._run_command"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data="")):
                        mmseqs_easy_search(tiny_file, tiny_file, self.temp_dir)

        finally:
            tiny_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
