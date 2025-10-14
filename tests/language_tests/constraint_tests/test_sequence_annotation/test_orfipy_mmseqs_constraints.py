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
from proto_language.language.constraint import (
    orfipy_mmseqs_gene_hit_count_constraint,
    orfipy_mmseqs_gene_homology_constraint,
)
from proto_language.language.constraint.sequence_annotation.orfipy_mmseqs_gene_hit_count_constraint import ORFipyMMseqsGeneHitCountConfig
from proto_language.language.constraint.sequence_annotation.orfipy_mmseqs_gene_homology_constraint import ORFipyMMseqsGeneHomologyConfig
from proto_language.tools.orf_prediction.orfipy import OrfipyConfig
from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
    get_test_sequences_with_real_hits,
    dummy_db_path,
    temp_dir,
    setup_test_files,
    ORFIPY_AVAILABLE,
)


@pytest.mark.skipif(not pd, reason="Pandas not installed, skipping ORF/MMseqs tests")
@pytest.mark.skipif(
    not ORFIPY_AVAILABLE, reason="orfipy not installed, skipping ORF tests"
)
class TestOrfipyMmseqsConstraints:
    @pytest.fixture
    def hit_count_config(self, dummy_db_path):
        return ORFipyMMseqsGeneHitCountConfig(
            min_hits=1,
            max_hits=3,
            mmseqs_config=MmseqsSearchProteinsConfig(
                query_fasta="",  # Filled in by pipeline
                mmseqs_db=dummy_db_path,
                results_dir="",  # Filled in by pipeline
                threads=1,
                sensitivity=1.0
            ),
            orfipy_config=OrfipyConfig(
                input_fasta="",  # Filled in by pipeline
                output_dir="",  # Filled in by pipeline
                threads=1,
                min_len=30
            ),
        )

    @pytest.fixture
    def homology_config(self, dummy_db_path):
        return ORFipyMMseqsGeneHomologyConfig(
            min_homology=80.0,
            max_homology=100.0,
            mmseqs_config=MmseqsSearchProteinsConfig(
                query_fasta="",  # Filled in by pipeline
                mmseqs_db=dummy_db_path,
                results_dir="",  # Filled in by pipeline
                threads=1,
                sensitivity=1.0
            ),
            orfipy_config=OrfipyConfig(
                input_fasta="",  # Filled in by pipeline
                output_dir="",  # Filled in by pipeline
                threads=1,
                min_len=30
            ),
        )

    def test_hit_count_constraint(self, hit_count_config, temp_dir):
        """Test hit count constraint using real test files."""
        sequences = get_test_sequences_with_real_hits()
        segment = create_segment(sequences[0])

        # Set up test files
        setup_test_files(temp_dir, sequences[0])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=hit_count_config,
        )

        # Since we're using real files, we expect the constraint to work with actual data
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0  # Score should be non-negative

        metadata = segment[0]._metadata
        assert (
            "segment_0.orfipy_mmseqs_gene_hit_count_constraint.orfipy_orfs" in metadata
        )
        assert (
            "segment_0.orfipy_mmseqs_gene_hit_count_constraint.mmseqs_results"
            in metadata
        )
        assert (
            "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            in metadata
        )
        assert isinstance(
            metadata[
                "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            ],
            int,
        )
        assert (
            metadata[
                "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            ]
            >= 0
        )

    def test_homology_constraint(self, homology_config, temp_dir):
        """Test homology constraint using real test files."""
        sequences = get_test_sequences_with_real_hits()
        segment = create_segment(sequences[0])

        # Set up test files
        setup_test_files(temp_dir, sequences[0])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_homology_constraint,
            scoring_function_config=homology_config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0

        metadata = segment[0]._metadata
        assert (
            "segment_0.orfipy_mmseqs_gene_homology_constraint.orfs_with_acceptable_homology"
            in metadata
        )
        assert (
            metadata[
                "segment_0.orfipy_mmseqs_gene_homology_constraint.orfs_with_acceptable_homology"
            ]
            >= 0
        )
        assert (
            "segment_0.orfipy_mmseqs_gene_homology_constraint.homology_compliance_rate"
            in metadata
        )
        assert (
            0.0
            <= metadata[
                "segment_0.orfipy_mmseqs_gene_homology_constraint.homology_compliance_rate"
            ]
            <= 1.0
        )

    def test_no_hits_scenario(self, hit_count_config, temp_dir):
        """Test constraint behavior when no hits are found."""
        # Use a sequence with no meaningful ORFs
        segment = create_segment("A" * 100)

        # Set up test files with empty ORF results
        dna_file = temp_dir / "input.fna"
        dna_file.write_text(">test_seq\n" + "A" * 100 + "\n")

        orfipy_dir = temp_dir / "orfipy_output"
        orfipy_dir.mkdir()

        # Create empty ORF files
        (orfipy_dir / "orfipy_aa.faa").write_text("")
        (orfipy_dir / "orfipy_nt.fna").write_text("")

        # Create empty mmseqs results
        mmseqs_file = temp_dir / "mmseqs_results.m8"
        mmseqs_file.write_text("")

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=hit_count_config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0  # Should have a penalty for not meeting min_hits
        assert (
            segment[0]._metadata[
                "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"
            ]
            == 0
        )


    def test_caching(self, hit_count_config, temp_dir):
        """Test that caching works correctly with real files."""
        from proto_language.language.constraint.sequence_annotation import (
            run_orfipy_mmseqs_pipeline,
        )
        from proto_language.tools.tool_cache import ToolCache

        seq = Sequence(
            "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA",
            SequenceType.DNA,
        )

        # Set up test files
        setup_test_files(temp_dir, seq.sequence)

        # First call, should compute
        run_orfipy_mmseqs_pipeline(
            seq,
            orfipy_config=hit_count_config.orfipy_config,
            mmseqs_config=hit_count_config.mmseqs_config,
        )
        # Check that results are in metadata
        assert "orfipy_orfs" in seq._metadata
        assert "mmseqs_results" in seq._metadata
        assert "unique_orfs_with_hits" in seq._metadata

        # Second call, should use cache
        seq._metadata["test_marker"] = "should_remain"
        run_orfipy_mmseqs_pipeline(
            seq,
            orfipy_config=hit_count_config.orfipy_config,
            mmseqs_config=hit_count_config.mmseqs_config,
        )
        assert seq._metadata["test_marker"] == "should_remain"

        # Verify cache is working by checking ToolCache directly with model parameters
        orfipy_config = hit_count_config.orfipy_config.model_dump()
        mmseqs_config = hit_count_config.mmseqs_config.model_dump()

        cached_results = ToolCache.get_cached_results(
            seq,
            "orfipy_mmseqs",
            orfipy_config=orfipy_config,
            mmseqs_config=mmseqs_config,
        )
        assert cached_results is not None
        assert "orfipy_orfs" in cached_results
        assert "mmseqs_results" in cached_results

        # Different config should recompute when pipeline parameters change
        new_mmseqs_config = MmseqsSearchProteinsConfig(
            query_fasta="",
            mmseqs_db=hit_count_config.mmseqs_config.mmseqs_db,
            results_dir="",
            threads=1,
            sensitivity=2.0,
        )  # Change pipeline parameter
        mmseqs_config_new = new_mmseqs_config.model_dump()
        cached_results_new = ToolCache.get_cached_results(
            seq,
            "orfipy_mmseqs",
            orfipy_config=orfipy_config,
            mmseqs_config=mmseqs_config_new,
        )
        assert cached_results_new is None  # Should not be cached with different params

    def test_parameter_validation(self, dummy_db_path):
        """Tests that missing required parameters raise validation errors (constraint-specific validation)."""
        segment = create_segment("ATGAAATAG")

        # Test hit count constraint - missing required fields should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
                scoring_function_config=ORFipyMMseqsGeneHitCountConfig(
                    min_hits=1,
                    # Missing max_hits
                ),
            ).evaluate()

        # Test homology constraint - invalid types should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_homology_constraint,
                scoring_function_config=ORFipyMMseqsGeneHomologyConfig(
                    min_homology="invalid",  # Should be float
                    max_homology=100.0
                ),
            ).evaluate()