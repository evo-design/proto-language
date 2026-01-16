"""
Protein domain constraint function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from proto_language.tools.gene_annotation.pyhmmer import pyhmmer_hmmscan, PyHmmscanInput, PyHmmerConfig
from proto_language.utils import MIN_ENERGY, MAX_ENERGY


class ProteinDomainConfig(BaseConfig):
    """Configuration for protein domain constraint.
    
    This class defines configuration parameters for evaluating whether protein
    sequences contain specific functional domains identified by keyword searches
    against HMM (Hidden Markov Model) profile databases. The constraint uses
    HMMER's hmmscan tool to identify protein domains and matches them against
    user-specified keywords, enabling targeted selection for proteins with
    desired functional characteristics.
    
    Attributes:
        hmm_db (str): Path to HMM database file for hmmscan (e.g., Pfam-A.hmm,
            TIGRFAM.hmm). The database must be preprocessed with hmmpress before
            use. Download Pfam from: https://www.ebi.ac.uk/interpro/download/pfam/

        keywords (List[str]): Keywords to search for in domain descriptions
            (case-insensitive). For example, ["kinase", "ATP-binding"] will match
            any domain description containing either term. Matches if ANY keyword
            is found in hit description, unless ``match_all_keywords=True``.

        evalue_threshold (float): Maximum E-value threshold for significant hits.
            Lower values are more stringent. E-values indicate the number of hits
            expected by chance. Typical values range from 0.0001 (strict) to
            0.01 (permissive). Default: 0.005.

        query_coverage (Optional[float]): Minimum query coverage percentage (0-100)
            for significant hits. If specified, filters hits by alignment coverage
            of the query sequence. For example, 50.0 requires at least 50% of the
            query to align to the domain. None means no coverage filter applied.
            Default: None.

        match_all_keywords (bool): If True, require ALL keywords to be found in
            domain descriptions. If False, require ANY keyword (default). Use True
            for strict multi-domain requirements (e.g., ["kinase", "ATP-binding"]
            both must be present). Default: False.

        hmmscan_config (Optional[PyHmmerConfig]): Optional advanced configuration
            for PyHMMER hmmscan (threading, bit score thresholds, etc.). If None,
            uses default configuration. The ``sequences`` and ``hmm_db`` fields
            are set programmatically and should not be specified here. Default: None.
            Example usage:
            ``PyHmmerConfig(cpus=4, Z=1000, domZ=1000)`` to use 4 CPU cores and
            set database size parameters for E-value calculation. Default: None.
    
    Note:
        For DNA sequences, Prodigal is used to predict ORFs first, then each
        predicted protein is searched for domains. For protein sequences, the
        search is performed directly.
    """
    # Required parameters
    hmm_db: str = ConfigField(
        title="HMM Database",
        description="Path to HMM database file for hmmscan (e.g., Pfam-A.hmm). Must be pressed with hmmpress.",
    )
    keywords: List[str] = ConfigField(
        title="Keywords to Search",
        description="Keywords to search for in domain descriptions (case-insensitive).",
    )

    # Advanced parameters
    evalue_threshold: float = ConfigField(
        title="Max E-value Threshold",
        default=0.005,
        description="Max E-value threshold for significant hits. Lower values are more stringent. Typical: 0.0001-0.01",
        advanced=True,
        examples=[0.0001, 0.01],
    )
    query_coverage: Optional[float] = ConfigField(
        title="Min Query Coverage",
        default=None,
        description="Min query coverage percentage for significant hits (0-100).",
        advanced=True,
    )
    match_all_keywords: bool = ConfigField(
        title="Match All Keywords",
        default=False,
        description="If True, require ALL keywords to be found. If False, require ANY keyword (default).",
        advanced=True,
    )
    hmmscan_config: Optional[PyHmmerConfig] = ConfigField(
        title="PyHMMER Config",
        default=None,
        description="Optional configuration for PyHMMER hmmscan. If None, uses default configuration.",
        advanced=True,
    )


@ConstraintRegistry.register(
    key="protein-domain",
    label="Protein Domain Match",
    config=ProteinDomainConfig,
    description="Evaluate whether sequences contains protein domains matching specified keywords",
    batched=True,
    concatenate=True,
    tools_called=["pyhmmer", "prodigal"],
    category="protein quality",
    supported_sequence_types=["dna", "protein"],
)
def protein_domain_constraint(sequences: List[Sequence], config: ProteinDomainConfig) -> List[float]:
    """Evaluate whether sequences contain protein domains matching specified keywords.
    
    This constraint function searches for functional protein domains using HMMER's
    hmmscan tool against HMM profile databases. It identifies domains in protein 
    sequences and matches them against user-specified keywords, enabling selection
    of proteins with desired functional domains.
    
    For DNA sequences, the function first runs Prodigal to predict protein-coding
    regions (ORFs), then searches each predicted protein for matching domains. For
    protein sequences, the domain search is performed directly. The constraint is
    satisfied when the specified keyword criteria are met (any or all keywords,
    depending on configuration).

    Args:
        sequences (List[Sequence]): List of DNA or protein sequences to evaluate.
            DNA sequences are first processed through ORF prediction. All sequences
            in the list must be the same type (all DNA or all PROTEIN).
            
        config (ProteinDomainConfig): Configuration object containing ``hmm_db``
            (path to HMM database), ``keywords`` (list of domain keywords to search),
            ``evalue_threshold`` (default: 0.005), ``query_coverage`` (default: None),
            ``match_all_keywords`` (default: False), and ``hmmscan_config``
            (default: None).

    Returns:
        List[float]: Constraint scores for each sequence, where 0.0 indicates domain
            criteria are satisfied (matching domains found) and 1.0 indicates no
            matching domains found or failure to meet keyword requirements.

    Raises:
        ValueError: If ``hmm_db`` path doesn't exist, ``keywords`` list is empty,
            input sequence list is empty, or sequences are of mixed types (DNA and
            protein mixed).
        RuntimeError: If HMMER hmmscan execution fails or Prodigal ORF prediction
            fails for DNA sequences.
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Metadata keys vary by
        sequence type:
        
        **For DNA sequences:**
        - ``prodigal_proteins``: DataFrame of predicted proteins from Prodigal
        - ``prodigal_protein_count``: Integer count of predicted ORFs
        - ``domain_search_results``: List of domain search results for each
          predicted protein
        - ``domain_keywords_found``: List of unique keywords found across all
          predicted proteins
        - ``domain_matching_proteins``: List of protein IDs that matched keywords
        
        **For protein sequences:**
        - ``domain_search_results``: List containing domain search results
        - ``domain_keywords_found``: List of keywords found in domain descriptions
        - ``domain_matching_hits``: DataFrame of domain hits matching keywords
        - ``hmmscan_all_hits``: DataFrame of all significant hmmscan hits
    
    Examples:
        Evaluating domain presence in protein with single keyword:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> cfg = ProteinDomainConfig(
        ...     hmm_db="Pfam-A.hmm",
        ...     keywords=["kinase"],
        ...     evalue_threshold=0.001
        ... )
        >>> scores = protein_domain_constraint([seq], config=cfg)
        >>> print(scores[0])  # 0.0 if kinase domain found, 1.0 if not
        >>> print(seq._metadata["domain_keywords_found"])  # ['kinase'] if found
        
        Evaluating DNA sequence (with automatic ORF prediction):
        
        >>> dna_seq = Sequence("ATGGTACTGAGCCCAGCG...", "dna")
        >>> cfg = ProteinDomainConfig(
        ...     hmm_db="Pfam-A.hmm",
        ...     keywords=["helicase"]
        ... )
        >>> scores = protein_domain_constraint([dna_seq], config=cfg)
        >>> print(dna_seq._metadata["prodigal_protein_count"])  # Number of predicted ORFs
        >>> print(dna_seq._metadata["domain_matching_proteins"])  # IDs of proteins with helicase domain
    """

    hmm_db = Path(config.hmm_db)
    if not hmm_db.exists():
        raise ValueError(f"HMM database not found: {hmm_db}")

    if not config.keywords or not isinstance(config.keywords, list):
        raise ValueError("Keywords must be a non-empty list")

    if not sequences:
        raise ValueError("Input sequence list must not be empty")
    
    dna_sequences = []
    protein_sequences = []
    sequence_type_map = []

    for idx, seq in enumerate(sequences):
        if seq.sequence_type == "dna":
            dna_sequences.append((idx, seq))
            sequence_type_map.append(('dna', len(dna_sequences) - 1))
        else:  # protein (validated by Constraint._validate_sequence_types)
            protein_sequences.append((idx, seq))
            sequence_type_map.append(('protein', len(protein_sequences) - 1))

    dna_scores = {}
    protein_scores = {}
    keywords_lower = [kw.lower() for kw in config.keywords]

    # Handle DNA vs protein sequences
    if dna_sequences:
        dna_indices, dna_seqs = zip(*dna_sequences)
        dna_results = _process_dna_sequences(
            list(dna_seqs), hmm_db, keywords_lower, config
        )
        dna_scores = dict(zip(dna_indices, dna_results))
    if protein_sequences:
        protein_indices, protein_seqs = zip(*protein_sequences)
        protein_results = _process_protein_sequences(
            list(protein_seqs), hmm_db, keywords_lower, config
        )
        protein_scores = dict(zip(protein_indices, protein_results))
    
    scores = []
    for idx, (seq_type, type_idx) in enumerate(sequence_type_map):
        if seq_type == 'dna':
            scores.append(dna_scores[idx])
        else:
            scores.append(protein_scores[idx])
    
    return scores

def _process_dna_sequences(
    input_sequences: List[Sequence],
    hmm_db: Path,
    keywords_lower: List[str],
    config: ProteinDomainConfig
) -> List[float]:
    """
    Process DNA sequences: Run Prodigal in batch, then check domains. Returns list of constraint scores
    """
    # Run Prodigal to get predicted proteins
    try:
        dna_sequences = [seq.sequence for seq in input_sequences]
        prodigal_inputs = ProdigalInput(input_sequences=dna_sequences)
        prodigal_config = ProdigalConfig()
        result = run_prodigal_prediction(prodigal_inputs, prodigal_config)
        all_proteins_per_seq = result.predicted_orfs
        gene_counts = result.num_orfs_per_sequence

    except Exception as e:
        raise RuntimeError(f"Prodigal execution failed: {e}")

    scores = []
    for input_sequence, proteins_list, gene_count in zip(input_sequences, all_proteins_per_seq, gene_counts):
        # Store Prodigal results in metadata
        input_sequence._metadata["prodigal_proteins"] = [orf.model_dump() for orf in proteins_list]
        input_sequence._metadata["prodigal_protein_count"] = gene_count

        if len(proteins_list) == 0:
            # No proteins predicted
            input_sequence._metadata["domain_search_results"] = []
            input_sequence._metadata["domain_keywords_found"] = []
            input_sequence._metadata["domain_matching_proteins"] = []
            scores.append(MAX_ENERGY)
            continue
        
        protein_sequences = [orf.amino_acid_sequence for orf in proteins_list]
        batch_results = _check_protein_domains_batch(
            protein_sequences,
            str(hmm_db),
            keywords_lower,
            config.evalue_threshold,
            config.hmmscan_config,
            config.query_coverage,
        )

        # Check each predicted protein
        all_results = []
        matching_proteins = []
        all_keywords_found = set()

        for orf, result in zip(proteins_list, batch_results):
            result["protein_id"] = orf.id
            result["protein_description"] = orf.description if hasattr(orf, 'description') and orf.description else ""
            all_results.append(result)

            if result["keywords_found"]:
                matching_proteins.append(orf.id)
                all_keywords_found.update(result["keywords_found"])

        # Store metadata
        input_sequence._metadata["domain_search_results"] = all_results
        input_sequence._metadata["domain_keywords_found"] = list(all_keywords_found)
        input_sequence._metadata["domain_matching_proteins"] = matching_proteins

        # Determine constraint score
        if config.match_all_keywords:
            score = (
                MIN_ENERGY if len(all_keywords_found) == len(keywords_lower) else MAX_ENERGY
            )
        else:
            score = MIN_ENERGY if all_keywords_found else MAX_ENERGY
        
        scores.append(score)

    return scores

def _process_protein_sequences(
    input_sequences: List[Sequence],
    hmm_db: Path,
    keywords_lower: List[str],
    config: ProteinDomainConfig
) -> List[float]:
    """
    Process protein sequences: Check domains in batch.
    
    Args:
        input_sequences: List of protein sequences
        hmm_db: Path to HMM database
        keywords_lower: Lowercase keywords to search for
        config: Domain constraint configuration
        
    Returns:
        List of constraint scores
    """
    # Extract protein sequence strings
    protein_sequences = [seq.sequence for seq in input_sequences]
    
    # Batch check all protein sequences
    try:
        batch_results = _check_protein_domains_batch(
            protein_sequences,
            str(hmm_db),
            keywords_lower,
            config.evalue_threshold,
            config.hmmscan_config,
            config.query_coverage,
        )
    except Exception as e:
        raise RuntimeError(f"HMMER execution failed: {e}")

    # Process results for each sequence
    scores = []
    for input_sequence, result in zip(input_sequences, batch_results):
        # Store metadata
        input_sequence._metadata["domain_search_results"] = [result]
        input_sequence._metadata["domain_keywords_found"] = result["keywords_found"]
        input_sequence._metadata["domain_matching_hits"] = result["matching_hits"]
        input_sequence._metadata["hmmscan_all_hits"] = result["all_hits"]

        # Determine constraint score
        keywords_found = set(result["keywords_found"])
        if config.match_all_keywords:
            score = MIN_ENERGY if len(keywords_found) == len(keywords_lower) else MAX_ENERGY
        else:
            score = MIN_ENERGY if keywords_found else MAX_ENERGY
        
        scores.append(score)

    return scores

def _check_protein_domains_batch(
    protein_sequences: List[Sequence],
    hmm_db: str,
    keywords_lower: List[str],
    evalue_threshold: float,
    hmmscan_config: PyHmmerConfig,
    query_coverage: float = None,
) -> List[Dict[str, Any]]:
    """
    Helper function to check a batch of protein sequences for domain matches.

    Args:
        protein_sequences: Protein sequence to analyze.
        hmm_db: Path to HMM database.
        keywords_lower: Lowercase keywords to search for.
        evalue_threshold: E-value threshold for significance.
        query_coverage: Minimum query coverage (optional).
        hmmscan_config: Configuration for PyHMMER hmmscan.

    Returns:
        List of dictionaries with analysis results including hits and keywords found.
    """

    # Create PyHMMER config with direct sequence input (no temporary files needed)
    # Create input and config for PyHMMER hmmscan
    hmmscan_input = PyHmmscanInput(
        sequences=protein_sequences,
        hmm_db=hmm_db
    )

    # Use provided config or default
    final_config = hmmscan_config if hmmscan_config is not None else PyHmmerConfig()

    # Run PyHMMER hmmscan
    result = pyhmmer_hmmscan(inputs=hmmscan_input, config=final_config)

    if not result.success:
        raise RuntimeError(f"PyHMMER execution failed: {result.errors}")

    batch_results = []

    for seq_idx, protein_seq in enumerate(protein_sequences):
        if result.num_domain_hits == 0:
            batch_results.append({
                "all_hits": None,
                "significant_hits": None,
                "matching_hits": None,
                "keywords_found": [],
            })
            continue

        seq_domain_hits = result.domain_hits_df[
            result.domain_hits_df["query_idx"] == seq_idx
        ].copy() if result.domain_hits_df is not None else None

        if seq_domain_hits is None or len(seq_domain_hits) == 0:
            batch_results.append({
                "all_hits": None,
                "significant_hits": None,
                "matching_hits": None,
                "keywords_found": [],
            })
            continue

        # Filter by E-value threshold
        significant_hits = seq_domain_hits[
            seq_domain_hits["i_evalue"] <= evalue_threshold
        ].copy()

        # Apply query coverage filter if specified
        if query_coverage is not None:
            query_len = len(
                protein_seq
            )
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

        batch_results.append({
            "all_hits": seq_domain_hits,
            "significant_hits": significant_hits,
            "matching_hits": matching_hits,
            "keywords_found": found_keywords,
        })

    return batch_results
