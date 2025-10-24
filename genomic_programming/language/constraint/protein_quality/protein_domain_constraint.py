"""
Protein domain constraint function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

from ...core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from ....tools.gene_annotation.pyhmmer import pyhmmer_hmmscan, PyHmmscanInput, PyHmmerConfig
from ....utils import MIN_ENERGY, MAX_ENERGY


class ProteinDomainConfig(BaseConfig):
    """Configuration for protein domain constraint."""
    hmm_db: str = Field(
        description="Path to HMM database file for hmmscan (e.g., Pfam-A.hmm). Must be pressed with hmmpress."
    )
    keywords: List[str] = Field(
        description="Keywords to search for in domain descriptions (case-insensitive). Matches if any keyword found in hit description, unless match_all_keywords=True."
    )
    evalue_threshold: float = Field(
        default=0.005,
        description="Maximum E-value threshold for significant hits. Lower values are more stringent. Typical: 0.001-0.01."
    )
    query_coverage: Optional[float] = Field(
        default=None,
        description="Minimum query coverage percentage (0-100). If specified, filters hits by alignment coverage. None = no filter."
    )
    match_all_keywords: bool = Field(
        default=False,
        description="If True, require ALL keywords to be found. If False, require ANY keyword (default)."
    )
    hmmscan_config: Optional[PyHmmerConfig] = Field(
        default=None,
        description="Optional configuration for PyHMMER hmmscan. If None, uses default configuration. Sequences field will be set programmatically from the input sequence. Hmm_db field will be set to the provided hmm_db.",
    )


@ConstraintRegistry.register(
    key="protein-domain",
    label="Protein Domain Match",
    config=ProteinDomainConfig,
    description="Evaluate whether a sequence contains protein domains matching specified keywords",
    vectorized=False,
    concatenate=True
)
def protein_domain_constraint(
    input_sequence: Sequence,
    config: ProteinDomainConfig
) -> float:
    """
    Evaluate whether a sequence contains protein domains matching specified keywords.

    For DNA sequences, runs Prodigal first to predict proteins, then checks all predicted
    proteins. For protein sequences, checks the sequence directly.

    Args:
        input_sequence: The DNA or protein sequence to evaluate.
        config: Configuration containing hmm_db, keywords, evalue_threshold, query_coverage, match_all_keywords, and hmmer_kwargs.

    Returns:
        Constraint score where 0.0 indicates domain criteria are satisfied
        and 1.0 indicates no matching domains found.

    Raises:
        ValueError: If hmm_db doesn't exist or keywords list is empty.
        RuntimeError: If HMMER or Prodigal execution fails.

    Examples:
        Evaluating domain presence in protein:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> cfg = ProteinDomainConfig(
        ...     hmm_db="pfam.hmm",
        ...     keywords=["kinase", "ATP-binding"],
        ...     evalue_threshold=0.001
        ... )
        >>> score = protein_domain_constraint(seq, config=cfg)

        Evaluating domain presence in DNA (via Prodigal):

        >>> seq = Sequence("ATGGTACTGAGCCCAGCG...", SequenceType.DNA)
        >>> cfg = ProteinDomainConfig(
        ...     hmm_db="pfam.hmm",
        ...     keywords=["helicase"],
        ...     match_all_keywords=False
        ... )
        >>> score = protein_domain_constraint(seq, config=cfg)
    """
    # Note: hmmscan_config will be created in _check_protein_domains
    # with the appropriate sequences and hmm_db values
    hmmscan_config = config.hmmscan_config

    hmm_db = Path(config.hmm_db)
    if not hmm_db.exists():
        raise ValueError(f"HMM database not found: {hmm_db}")

    if not config.keywords or not isinstance(config.keywords, list):
        raise ValueError("Keywords must be a non-empty list")

    keywords_lower = [kw.lower() for kw in config.keywords]

    # Handle DNA vs protein sequences
    if input_sequence.sequence_type == SequenceType.DNA:
        # Run Prodigal to get predicted proteins
        try:
            prodigal_inputs = ProdigalInput(input_sequence=input_sequence.sequence)
            prodigal_config = ProdigalConfig()
            result = run_prodigal_prediction(prodigal_inputs, prodigal_config)
            proteins_df = result.results_df

            input_sequence._metadata["prodigal_proteins"] = proteins_df
            input_sequence._metadata["prodigal_protein_count"] = result.num_genes
        except Exception as e:
            raise RuntimeError(f"Prodigal execution failed: {e}")

        if len(proteins_df) == 0:
            # No proteins predicted
            input_sequence._metadata["domain_search_results"] = []
            input_sequence._metadata["domain_keywords_found"] = []
            input_sequence._metadata["domain_matching_proteins"] = []
            return MAX_ENERGY

        # Check each predicted protein
        all_results = []
        matching_proteins = []
        all_keywords_found = set()

        for idx, protein_row in proteins_df.iterrows():
            protein_seq = Sequence(protein_row["sequence"], SequenceType.PROTEIN)
            result = _check_protein_domains(
                protein_seq,
                str(hmm_db),
                keywords_lower,
                config.evalue_threshold,
                hmmscan_config,
                config.query_coverage,
            )

            result["protein_id"] = protein_row["id"]
            result["protein_description"] = protein_row["description"]
            all_results.append(result)

            if result["keywords_found"]:
                matching_proteins.append(protein_row["id"])
                all_keywords_found.update(result["keywords_found"])

        # Store metadata
        input_sequence._metadata["domain_search_results"] = all_results
        input_sequence._metadata["domain_keywords_found"] = list(all_keywords_found)
        input_sequence._metadata["domain_matching_proteins"] = matching_proteins

        # Determine constraint matching
        if config.match_all_keywords:
            return (
                MIN_ENERGY if len(all_keywords_found) == len(keywords_lower) else MAX_ENERGY
            )
        else:
            return MIN_ENERGY if all_keywords_found else MAX_ENERGY

    elif input_sequence.sequence_type == SequenceType.PROTEIN:
        # Check protein sequence directly
        try:
            result = _check_protein_domains(
                input_sequence,
                str(hmm_db),
                keywords_lower,
                config.evalue_threshold,
                hmmscan_config,
                config.query_coverage,
            )
        except Exception as e:
            raise RuntimeError(f"HMMER execution failed: {e}")

        # Store metadata
        input_sequence._metadata["domain_search_results"] = [result]
        input_sequence._metadata["domain_keywords_found"] = result["keywords_found"]
        input_sequence._metadata["domain_matching_hits"] = result["matching_hits"]
        input_sequence._metadata["hmmscan_all_hits"] = result["all_hits"]

        # Determine constraint matching
        keywords_found = set(result["keywords_found"])
        if config.match_all_keywords:
            return MIN_ENERGY if len(keywords_found) == len(keywords_lower) else MAX_ENERGY
        else:
            return MIN_ENERGY if keywords_found else MAX_ENERGY

    else:
        raise ValueError(f"Unsupported sequence type: {input_sequence.sequence_type}")


def _check_protein_domains(
    protein_sequence: Sequence,
    hmm_db: str,
    keywords_lower: List[str],
    evalue_threshold: float,
    hmmscan_config: PyHmmerConfig,
    query_coverage: float = None,
) -> Dict[str, Any]:
    """
    Helper function to check a single protein sequence for domain matches.

    Args:
        protein_sequence: Protein sequence to analyze.
        hmm_db: Path to HMM database.
        keywords_lower: Lowercase keywords to search for.
        evalue_threshold: E-value threshold for significance.
        query_coverage: Minimum query coverage (optional).
        hmmscan_config: Configuration for PyHMMER hmmscan.

    Returns:
        Dictionary with analysis results including hits and keywords found.
    """

    # Create PyHMMER config with direct sequence input (no temporary files needed)
    # Create input and config for PyHMMER hmmscan
    hmmscan_input = PyHmmscanInput(
        sequences=protein_sequence.sequence,
        hmm_db=hmm_db
    )

    # Use provided config or default
    final_config = hmmscan_config if hmmscan_config is not None else PyHmmerConfig()

    # Run PyHMMER hmmscan
    result = pyhmmer_hmmscan(inputs=hmmscan_input, config=final_config)

    if not result.success:
        raise RuntimeError(f"PyHMMER execution failed: {result.errors}")

    if result.num_domain_hits == 0:
        return {
            "all_hits": None,
            "significant_hits": None,
            "matching_hits": None,
            "keywords_found": [],
        }

    # Filter by E-value threshold
    significant_hits = result.domain_hits_df[
        result.domain_hits_df["i_evalue"] <= evalue_threshold
    ].copy()

    # Apply query coverage filter if specified
    if query_coverage is not None:
        query_len = len(
            protein_sequence.sequence
        )  # Note: .sequence to get the string
        if query_len > 0:
            coverage_pct = (
                (significant_hits["target_to"] - significant_hits["target_from"] + 1)
                / query_len
                * 100
            )
            significant_hits = significant_hits[coverage_pct >= query_coverage]

    # Find hits matching keywords
    if len(significant_hits) > 0:
        keyword_pattern = "|".join(keywords_lower)
        matching_mask = (
            significant_hits["target_description"]
            .str.lower()
            .str.contains(keyword_pattern, na=False, regex=True)
        )
        matching_hits = significant_hits[matching_mask]
    else:
        matching_hits = significant_hits

    # Extract found keywords
    found_keywords = []
    if len(matching_hits) > 0:
        for _, hit in matching_hits.iterrows():
            description_lower = str(hit["target_description"]).lower()
            for keyword in keywords_lower:
                if keyword in description_lower and keyword not in found_keywords:
                    found_keywords.append(keyword)

    return {
        "all_hits": result.domain_hits_df,
        "significant_hits": significant_hits,
        "matching_hits": matching_hits,
        "keywords_found": found_keywords,
    }
