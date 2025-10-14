"""
ORFipy + MMseqs gene hit count constraint for evaluating number of unique ORFs with hits.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ...base import Sequence
from proto_language.base_config import BaseConfig
from ..registry import ConstraintRegistry
from ....tools.orf_prediction.orfipy import OrfipyConfig
from ....tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
from ..utils import calculate_range_deviation, run_orfipy_mmseqs_pipeline


class ORFipyMMseqsGeneHitCountConfig(BaseConfig):
    """Configuration for ORFipy + MMseqs gene hit count constraint."""
    min_hits: int = Field(ge=0, description="Minimum acceptable number of unique ORFs with database hits (must be non-negative)")
    max_hits: int = Field(ge=0, description="Maximum acceptable number of unique ORFs with database hits (must be non-negative)")
    orfipy_config: Optional[OrfipyConfig] = Field(default=None, description="ORFipy configuration for ORF prediction (threads, start/stop codons, strand, min/max length, etc.)")
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = Field(default=None, description="MMseqs configuration for homology search (mmseqs_db path REQUIRED, plus threads, sensitivity, etc.)")


@ConstraintRegistry.register(
    key="orfipy-mmseqs-gene-hit-count",
    config=ORFipyMMseqsGeneHitCountConfig,
    description="Evaluate whether the number of unique ORFs with hits falls within a target range",
    vectorized=False,
    concatenate=True
)
def orfipy_mmseqs_gene_hit_count_constraint(
    input_sequence: Sequence,
    config: ORFipyMMseqsGeneHitCountConfig
) -> float:
    """
    Evaluate whether the number of unique ORFs with hits falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration containing min_hits, max_hits, orfipy_config, and mmseqs_config parameters.

    Returns:
        Constraint score where 0.0 indicates the hit count is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Examples:
        Evaluating ORF hit count constraint:

        >>> from proto_language.tools.orf_prediction.orfipy import OrfipyConfig
        >>> from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> cfg = ORFipyMMseqsGeneHitCountConfig(
        ...     min_hits=1,
        ...     max_hits=5,
        ...     orfipy_config=OrfipyConfig(input_fasta="", output_dir="", threads=48),
        ...     mmseqs_config=MmseqsSearchProteinsConfig(query_fasta="", mmseqs_db="/path/to/protein_db", results_dir="")
        ... )
        >>> score = orfipy_mmseqs_gene_hit_count_constraint(seq, config=cfg)
    """
    # Run the pipeline
    run_orfipy_mmseqs_pipeline(input_sequence, config.orfipy_config, config.mmseqs_config)

    # Get the count of unique ORFs with hits (directly from metadata)
    unique_orfs_with_hits = input_sequence._metadata.get("unique_orfs_with_hits", 0)

    # Calculate range deviation
    return calculate_range_deviation(unique_orfs_with_hits, config.min_hits, config.max_hits)
