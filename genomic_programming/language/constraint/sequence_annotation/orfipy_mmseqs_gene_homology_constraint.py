"""
ORFipy + MMseqs gene homology constraint for evaluating homology (percent identity) of ORF hits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import Field

from ...core import Sequence
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.orf_prediction.orfipy import OrfipyConfig
from ....tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
from ....utils import MIN_ENERGY, MAX_ENERGY, calculate_percentage_range_deviation
from ..utils import run_orfipy_mmseqs_pipeline


class ORFipyMMseqsGeneHomologyConfig(BaseConfig):
    """Configuration for ORFipy + MMseqs gene homology constraint."""
    min_homology: float = Field(ge=0.0, le=100.0, description="Minimum acceptable percent identity (0-100) for each ORF. Lower values are more permissive.")
    max_homology: float = Field(ge=0.0, le=100.0, description="Maximum acceptable percent identity (0-100) for each ORF. Higher values allow more similar hits.")
    orfipy_config: Optional[OrfipyConfig] = Field(default=None, description="ORFipy configuration for ORF prediction")
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = Field(default=None, description="MMseqs configuration for homology search (mmseqs_db path REQUIRED)")


@ConstraintRegistry.register(
    key="orfipy-mmseqs-gene-homology",
    config=ORFipyMMseqsGeneHomologyConfig,
    description="Evaluate the homology (percent identity) of each individual ORF hit",
    vectorized=False,
    concatenate=True
)
def orfipy_mmseqs_gene_homology_constraint(
    input_sequence: Sequence,
    config: ORFipyMMseqsGeneHomologyConfig
) -> float:
    """
    Evaluate the homology (percent identity) of each individual ORF hit.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration containing min_homology, max_homology, orfipy_config, and mmseqs_config parameters.

    Returns:
        Constraint score where 0.0 indicates all ORF homologies are within acceptable range
        and higher values indicate more ORFs with homology outside the acceptable range.

    Examples:
        Evaluating ORF homology constraint:

        >>> from proto_language.tools.orf_prediction.orfipy import OrfipyConfig
        >>> from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> cfg = ORFipyMMseqsGeneHomologyConfig(
        ...     min_homology=50.0,
        ...     max_homology=90.0,
        ...     orfipy_config=OrfipyConfig(input_fasta="", output_dir="", threads=48),
        ...     mmseqs_config=MmseqsSearchProteinsConfig(query_fasta="", mmseqs_db="/path/to/protein_db", results_dir="")
        ... )
        >>> score = orfipy_mmseqs_gene_homology_constraint(seq, config=cfg)
    """
    # Run the pipeline
    run_orfipy_mmseqs_pipeline(input_sequence, config.orfipy_config, config.mmseqs_config)

    # Get the MMseqs results (convert from dict records if needed)
    mmseqs_results_data = input_sequence._metadata.get("mmseqs_results", [])
    if isinstance(mmseqs_results_data, list):
        mmseqs_results = (
            pd.DataFrame(mmseqs_results_data) if mmseqs_results_data else pd.DataFrame()
        )
    else:
        mmseqs_results = mmseqs_results_data
    total_orfs_with_hits = input_sequence._metadata.get("unique_orfs_with_hits", 0)

    if mmseqs_results.empty:
        # No hits found - return max penalty
        input_sequence._metadata["orfs_with_acceptable_homology"] = 0
        input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
        input_sequence._metadata["homology_compliance_rate"] = 0.0
        return MAX_ENERGY

    # Use standardized identity column
    if "identity" not in mmseqs_results.columns:
        input_sequence._metadata["orfs_with_acceptable_homology"] = 0
        input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
        input_sequence._metadata["homology_compliance_rate"] = 0.0
        return MAX_ENERGY

    # Check each ORF's homology individually
    acceptable_homology_count = 0
    homology_violations = []

    for _, row in mmseqs_results.iterrows():
        homology = row["identity"]
        if config.min_homology <= homology <= config.max_homology:
            acceptable_homology_count += 1
        else:
            # Calculate how far this ORF's homology deviates from acceptable range
            deviation = calculate_percentage_range_deviation(
                homology, config.min_homology, config.max_homology
            )
            homology_violations.append(deviation)

    # Store metadata for inspection
    input_sequence._metadata["orfs_with_acceptable_homology"] = (
        acceptable_homology_count
    )
    input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
    input_sequence._metadata["homology_compliance_rate"] = (
        acceptable_homology_count / total_orfs_with_hits
    )

    # If all ORFs have acceptable homology, return 0
    if not homology_violations:
        return MIN_ENERGY

    # Return the average deviation of ORFs that violate the homology constraint
    return min(MAX_ENERGY, np.mean(homology_violations))
