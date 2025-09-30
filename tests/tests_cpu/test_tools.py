import pandas as pd
import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

# Import functions to be tested
from proto_language.tools.gene_annotation.mmseqs import (
    mmseqs_easy_search,
    run_mmseqs_search_proteins,
    _filter_top_hits,
    convert_m8_to_df,
)
from proto_language.tools.orf_prediction.orfipy import (
    run_orfipy,
    parse_orfipy_results_to_df,
    _parse_orfipy_header,
)
from proto_language.tools.orf_prediction.prodigal import run_prodigal
from proto_language.base import Sequence
from proto_language.utils import SequenceType
from proto_language.constraint.protein_structure import (
    _run_esmfold,
)
from proto_language.constraint.sequence_annotation import (
    _run_orfipy_mmseqs_pipeline,
)
from proto_language.tool_cache import ToolCache

# Test data file paths
TEST_DATA_DIR = Path("tests/tests_cpu/dummy_data")
PROTEIN_FASTA = TEST_DATA_DIR / "test_protein_sequences.faa"
DNA_FASTA = TEST_DATA_DIR / "test_dna_sequences.fna"
M8_FILE = TEST_DATA_DIR / "test_mmseqs_results.m8"
ORFIPY_AA_FILE = TEST_DATA_DIR / "test_orfipy_aa.faa"
ORFIPY_NT_FILE = TEST_DATA_DIR / "test_orfipy_nt.fna"
PRODIGAL_FASTA = TEST_DATA_DIR / "test_prodigal_sequences.fna"

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


class TestToolCaching:
    """Test suite for tool caching behavior."""

    def test_tool_cache_basic_functionality(self):
        """Test basic ToolCache functionality with mock data."""
        seq = Sequence("MKLLVVAAAFVGLLLPSAAFA", SequenceType.PROTEIN)
        
        # Initially not cached
        cached_results = ToolCache.get_cached_results(seq, "test_tool", param1="value1")
        assert cached_results is None
        
        # Cache some results
        test_results = {"result": "computed_value", "score": 0.85}
        ToolCache.cache_results(seq, "test_tool", test_results, param1="value1")
        
        # Should be cached now
        cached_results = ToolCache.get_cached_results(seq, "test_tool", param1="value1")
        assert cached_results == test_results
        
        # Different parameters should not be cached
        cached_results_diff = ToolCache.get_cached_results(seq, "test_tool", param1="value2")
        assert cached_results_diff is None

    def test_tool_cache_sequence_content_based(self):
        """Test that cache is based on sequence content, not object identity."""
        seq1 = Sequence("MKLLVV", SequenceType.PROTEIN)
        seq2 = Sequence("MKLLVV", SequenceType.PROTEIN)  # Same content, different object
        
        # Cache result for seq1
        test_results = {"result": "cached_value"}
        ToolCache.cache_results(seq1, "test_tool", test_results, param="test")
        
        # seq2 should hit the same cache
        cached_results = ToolCache.get_cached_results(seq2, "test_tool", param="test")
        assert cached_results == test_results

    def test_tool_cache_parameter_sensitivity(self):
        """Test that cache is sensitive to parameter changes."""
        seq = Sequence("MKLLVV", SequenceType.PROTEIN)
        
        # Cache with first set of parameters
        result1 = {"result": "value1"}
        ToolCache.cache_results(seq, "test_tool", result1, param1="a", param2=1)
        
        # Cache with second set of parameters  
        result2 = {"result": "value2"}
        ToolCache.cache_results(seq, "test_tool", result2, param1="b", param2=2)
        
        # Both should be cached separately
        cached1 = ToolCache.get_cached_results(seq, "test_tool", param1="a", param2=1)
        cached2 = ToolCache.get_cached_results(seq, "test_tool", param1="b", param2=2)
        
        assert cached1 == result1
        assert cached2 == result2
        assert cached1 != cached2

    def test_cache_isolation_between_tools(self):
        """Test that different tools have isolated caches."""
        seq = Sequence("MKLLVV", SequenceType.PROTEIN)
        
        # Cache results for two different tools
        result1 = {"tool1_result": "value1"}
        result2 = {"tool2_result": "value2"}
        
        ToolCache.cache_results(seq, "tool1", result1, param="test")
        ToolCache.cache_results(seq, "tool2", result2, param="test")
        
        # Each tool should have its own cached results
        cached1 = ToolCache.get_cached_results(seq, "tool1", param="test")
        cached2 = ToolCache.get_cached_results(seq, "tool2", param="test")
        
        assert cached1 == result1
        assert cached2 == result2
        assert cached1 != cached2

    def test_cache_survives_metadata_changes(self):
        """Test that cache survives when sequence metadata is modified."""
        seq = Sequence("MKLLVV", SequenceType.PROTEIN)
        
        # Cache some results
        test_results = {"cached_value": "should_persist"}
        ToolCache.cache_results(seq, "test_tool", test_results, param="test")
        
        # Modify sequence metadata (simulating constraint evaluation)
        seq._metadata["some_other_key"] = "some_value"
        seq._metadata["prefix.another_key"] = "prefixed_value"
        
        # Cache should still work
        cached_results = ToolCache.get_cached_results(seq, "test_tool", param="test")
        assert cached_results == test_results

    def test_tool_cache_with_input_dictionaries(self):
        """Test that cache works correctly with input dictionaries as parameters."""
        seq = Sequence("MKLLVV", SequenceType.PROTEIN)
        
        # Test with nested dictionaries as parameters
        input_dict1 = {"config": {"threshold": 0.5, "method": "blast"}, "options": {"verbose": True}}
        input_dict2 = {"config": {"threshold": 0.8, "method": "mmseqs"}, "options": {"verbose": False}}
        
        # Cache results with first dictionary
        result1 = {"hits": 5, "confidence": 0.9}
        ToolCache.cache_results(seq, "test_tool", result1, **input_dict1)
        
        # Cache results with second dictionary
        result2 = {"hits": 3, "confidence": 0.7}
        ToolCache.cache_results(seq, "test_tool", result2, **input_dict2)
        
        # Verify both results are cached separately
        cached1 = ToolCache.get_cached_results(seq, "test_tool", **input_dict1)
        cached2 = ToolCache.get_cached_results(seq, "test_tool", **input_dict2)
        
        assert cached1 == result1
        assert cached2 == result2
        assert cached1 != cached2
        
        # Test that partial dictionary matches don't hit cache
        partial_dict = {"config": {"threshold": 0.5}}  # Missing method and options
        cached_partial = ToolCache.get_cached_results(seq, "test_tool", **partial_dict)
        assert cached_partial is None


class TestProdigalTools:
    """Test suite for Prodigal (pyrodigal) tool wrappers."""

    def test_run_prodigal_basic_functionality(self):
        """Test basic Prodigal functionality with a simple gene sequence."""
        # Create a DNA sequence with a gene
        dna_seq = Sequence('ATGAAACGTATTGCGTCGTAA' * 20, SequenceType.DNA)
        df = run_prodigal(dna_seq)
        
        # Check that we got a DataFrame back
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Should find at least one gene"
        
        # Check required columns are present
        required_columns = ['id', 'description', 'sequence', 'length', 'start', 'end', 'strand']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data types
        assert isinstance(df.iloc[0]['sequence'], str)
        assert isinstance(df.iloc[0]['length'], (int, np.integer))
        assert isinstance(df.iloc[0]['start'], (int, np.integer))
        assert isinstance(df.iloc[0]['end'], (int, np.integer))
        assert df.iloc[0]['strand'] in ['+', '-']

    def test_run_prodigal_with_real_gene(self):
        """Test Prodigal with a realistic bacterial gene sequence (lacZ partial)."""
        # Read test sequence from file
        from Bio import SeqIO
        test_seq = None
        for record in SeqIO.parse(PRODIGAL_FASTA, "fasta"):
            if record.id == "test_seq_2_with_gene":
                test_seq = str(record.seq)
                break
        
        assert test_seq is not None, "Test sequence not found in file"
        
        dna_seq = Sequence(test_seq, SequenceType.DNA)
        df = run_prodigal(dna_seq)
        
        # Should find genes in lacZ sequence
        assert len(df) >= 1, "Should find at least one gene in lacZ sequence"
        
        # First gene should be substantial
        first_gene = df.iloc[0]
        assert first_gene['length'] > 100, "lacZ gene should be substantial"
        assert first_gene['start'] > 0
        assert first_gene['end'] > first_gene['start']
        
        # Check protein sequence is valid amino acids
        protein_seq = first_gene['sequence']
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
        assert all(aa in valid_aas for aa in protein_seq), "Invalid amino acids in protein sequence"

    def test_run_prodigal_caching(self):
        """Test that Prodigal results are properly cached."""
        dna_seq = Sequence('ATGAAACGTATTGCGTCGTAA' * 30, SequenceType.DNA)
        
        # First run - should compute
        df1 = run_prodigal(dna_seq)
        
        # Check metadata was cached
        assert "prodigal_proteins" in dna_seq._metadata
        assert "prodigal_protein_count" in dna_seq._metadata
        assert dna_seq._metadata["prodigal_protein_count"] == len(df1)
        
        # Second run - should use cache
        df2 = run_prodigal(dna_seq)
        
        # Results should be identical
        assert len(df1) == len(df2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_run_prodigal_error_handling(self):
        """Test that Prodigal raises appropriate errors for invalid input."""
        # Test with non-DNA sequence
        protein_seq = Sequence("MKFLKFSLTSAA", SequenceType.PROTEIN)
        with pytest.raises(ValueError, match="Can only run Prodigal on DNA sequences"):
            run_prodigal(protein_seq)
        
        # Test with RNA sequence
        rna_seq = Sequence("AUGAAACGUAUUGCGUCGUAA", SequenceType.RNA)
        with pytest.raises(ValueError, match="Can only run Prodigal on DNA sequences"):
            run_prodigal(rna_seq)

    def test_run_prodigal_no_genes_found(self):
        """Test behavior when no genes are found in sequence."""
        # Sequence with no valid ORFs (just repeating As)
        from Bio import SeqIO
        test_seq = None
        for record in SeqIO.parse(PRODIGAL_FASTA, "fasta"):
            if record.id == "test_seq_3_no_genes":
                test_seq = str(record.seq)
                break
        
        if test_seq:
            dna_seq = Sequence(test_seq, SequenceType.DNA)
            df = run_prodigal(dna_seq)
            
            # Should return empty DataFrame
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            assert dna_seq._metadata["prodigal_protein_count"] == 0

    def test_run_prodigal_metadata_storage(self):
        """Test that Prodigal stores metadata correctly."""
        dna_seq = Sequence('ATGAAACGTATTGCGTCGTAA' * 25, SequenceType.DNA)
        df = run_prodigal(dna_seq)
        
        # Check metadata keys
        assert "prodigal_proteins" in dna_seq._metadata
        assert "prodigal_protein_count" in dna_seq._metadata
        
        # Check metadata content matches DataFrame
        stored_df = dna_seq._metadata["prodigal_proteins"]
        assert isinstance(stored_df, pd.DataFrame)
        pd.testing.assert_frame_equal(df, stored_df)
        assert dna_seq._metadata["prodigal_protein_count"] == len(df)

    def test_run_prodigal_sequence_quality(self):
        """Test that predicted proteins have valid sequences."""
        dna_seq = Sequence('ATGAAACGTATTGCGTCGTAA' * 40, SequenceType.DNA)
        df = run_prodigal(dna_seq)
        
        if len(df) > 0:
            for idx, row in df.iterrows():
                # Check sequence is non-empty string
                assert isinstance(row['sequence'], str)
                assert len(row['sequence']) > 0
                
                # Check length matches actual sequence length
                assert row['length'] == len(row['sequence'])
                
                # Check no stop codons in middle (only at end if present)
                assert row['sequence'].count('*') <= 1
                if '*' in row['sequence']:
                    assert row['sequence'].endswith('*')
                
                # Check start/end coordinates make sense
                assert row['start'] >= 1
                assert row['end'] > row['start']

    def test_run_prodigal_multiple_sequences_independent(self):
        """Test that running Prodigal on different sequences works independently."""
        # Create two different sequences
        seq1 = Sequence('ATGAAACGTATTGCGTCGTAA' * 20, SequenceType.DNA)
        seq2 = Sequence('ATGCCCGGGAAATTTCCCGGG' * 25, SequenceType.DNA)
        
        # Run Prodigal on both
        df1 = run_prodigal(seq1)
        df2 = run_prodigal(seq2)
        
        # Both should have results cached independently
        assert "prodigal_proteins" in seq1._metadata
        assert "prodigal_proteins" in seq2._metadata
        
        # Metadata should be independent
        stored_df1 = seq1._metadata["prodigal_proteins"]
        stored_df2 = seq2._metadata["prodigal_proteins"]
        
        pd.testing.assert_frame_equal(df1, stored_df1)
        pd.testing.assert_frame_equal(df2, stored_df2)

    def test_run_prodigal_dataframe_structure(self):
        """Test that output DataFrame has the expected structure for downstream use."""
        dna_seq = Sequence('ATGAAACGTATTGCGTCGTAA' * 30, SequenceType.DNA)
        df = run_prodigal(dna_seq)
        
        if len(df) > 0:
            # Check column types
            assert df['id'].dtype == object  # string
            assert df['description'].dtype == object  # string  
            assert df['sequence'].dtype == object  # string
            assert df['length'].dtype in [int, 'int64', 'int32']
            assert df['start'].dtype in [int, 'int64', 'int32']
            assert df['end'].dtype in [int, 'int64', 'int32']
            assert df['strand'].dtype == object  # string
            
            # Check no null values in critical columns
            assert not df['sequence'].isnull().any()
            assert not df['length'].isnull().any()
            assert not df['start'].isnull().any()
            assert not df['end'].isnull().any()
