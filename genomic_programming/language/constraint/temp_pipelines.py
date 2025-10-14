"""
Constraint-specific shared pipelines for ORFipy + MMseqs.

TODO: Remove this file, move this logic in the actual constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from ..core import Sequence, DNA_NUCLEOTIDES
from ...tools.tool_cache import ToolCache
from ...utils import resolve_paths
from ...tools.orf_prediction import run_orfipy_prediction, OrfipyConfig
from ...tools.gene_annotation.mmseqs import mmseqs_search_proteins, MmseqsSearchProteinsConfig


def run_orfipy_mmseqs_pipeline(
    input_sequence: Sequence,
    orfipy_config: Optional[OrfipyConfig] = None,
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = None,
) -> None:
    """
    Run the ORFipy + MMseqs pipeline for sequence analysis.

    Args:
        input_sequence: The sequence to evaluate.
        orfipy_config: ORFipy configuration arguments.
        mmseqs_config: MMseqs configuration arguments.

    Note:
        Results are cached based on sequence and parameters to avoid redundant analysis.
        Updates metadata with 'orfipy_orfs', 'mmseqs_results', and 'unique_orfs_with_hits'.
    """
    # Use defaults if not provided
    if orfipy_config is None:
        orfipy_config = OrfipyConfig(input_fasta="", output_dir="")
    if mmseqs_config is None:
        raise ValueError("MMseqs configuration with database path is required")

    # Check if analysis already cached (use model_dump for cache key)
    cached_results = ToolCache.get_cached_results(
        input_sequence,
        "orfipy_mmseqs",
        orfipy_config=orfipy_config.model_dump(),
        mmseqs_config=mmseqs_config.model_dump(),
    )
    if cached_results:
        input_sequence._metadata.update(cached_results)
        return

    # Preprocess sequence by removing all characters that are not ACGT
    sequence_to_analyze = "".join(
        char for char in input_sequence.sequence.upper() if char in DNA_NUCLEOTIDES
    )

    # Run the expensive analysis
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write sequence to temporary FASTA file
        input_fasta = temp_path / "input.fasta"
        with open(input_fasta, "w") as f:
            f.write(f">input_sequence\n{sequence_to_analyze}\n")

        # Run ORFipy - create new config with updated paths (no GCS paths to resolve here)
        orfipy_output = temp_path / "orfipy_output"
        orfipy_run_config = orfipy_config.model_copy(update={
            "input_fasta": str(input_fasta),
            "output_dir": str(orfipy_output)
        })
        result = run_orfipy_prediction(orfipy_run_config)
        
        # Get parsed ORFs from result
        orfs_df = result.results_df if result.results_df is not None else pd.DataFrame()
        aa_fasta = result.aa_fasta_path
        nt_fasta = result.nt_fasta_path

        if orfs_df.empty:
            # No ORFs found (store as empty lists for JSON serialization)
            results = {
                "orfipy_orfs": [],
                "mmseqs_results": [],
                "unique_orfs_with_hits": 0,
            }
        else:
            # Run MMseqs search for each ORF - create new config with updated paths
            # Resolve GCS paths (e.g., gcs://bucket/database) to local paths
            mmseqs_output = temp_path / "mmseqs_output"
            resolved_db = resolve_paths(mmseqs_config.mmseqs_db)
            mmseqs_run_config = mmseqs_config.model_copy(update={
                "query_fasta": str(aa_fasta),
                "mmseqs_db": resolved_db,
                "results_dir": str(mmseqs_output)
            })
            result = mmseqs_search_proteins(mmseqs_run_config)
            
            # Extract DataFrame from result
            mmseqs_results = result.results_df if result.results_df is not None else pd.DataFrame()

            # Count unique ORFs with hits
            unique_orfs_with_hits = (
                len(mmseqs_results) if not mmseqs_results.empty else 0
            )

            # Store results (convert DataFrames to dicts for JSON serialization)
            results = {
                "orfipy_orfs": orfs_df.to_dict("records") if not orfs_df.empty else [],
                "mmseqs_results": (
                    mmseqs_results.to_dict("records")
                    if not mmseqs_results.empty
                    else []
                ),
                "unique_orfs_with_hits": unique_orfs_with_hits,
            }

    # Cache results and update metadata
    ToolCache.cache_results(
        input_sequence,
        "orfipy_mmseqs",
        results,
        orfipy_config=orfipy_config.model_dump(),
        mmseqs_config=mmseqs_config.model_dump(),
    )
    input_sequence._metadata.update(results)
