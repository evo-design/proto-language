"""Protein domain constraint function."""

from pathlib import Path
from typing import Any

from proto_tools import (
    ProdigalConfig,
    ProdigalInput,
    PyHmmerConfig,
    PyHmmscanInput,
    run_prodigal_prediction,
    run_pyhmmer_hmmscan,
)

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField


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

        keywords (list[str]): Keywords to search for in domain descriptions
            (case-insensitive). For example, ["kinase", "ATP-binding"] will match
            any domain description containing either term. Matches if ANY keyword
            is found in hit description, unless ``match_all_keywords=True``.

        evalue_threshold (float): Maximum E-value threshold for significant hits.
            Lower values are more stringent. E-values indicate the number of hits
            expected by chance. Typical values range from 0.0001 (strict) to
            0.01 (permissive). Default: 0.005.

        query_coverage (float | None): Minimum query coverage percentage (0-100)
            for significant hits. If specified, filters hits by alignment coverage
            of the query sequence. For example, 50.0 requires at least 50% of the
            query to align to the domain. None means no coverage filter applied.
            Default: None.

        match_all_keywords (bool): If True, require ALL keywords to be found in
            domain descriptions. If False, require ANY keyword (default). Use True
            for strict multi-domain requirements (e.g., ["kinase", "ATP-binding"]
            both must be present). Default: False.

        hmmscan_config (PyHmmerConfig): Advanced configuration for PyHMMER hmmscan
            (threading, bit score thresholds, etc.). The ``sequences`` and ``hmm_db``
            fields are set programmatically and should not be specified here.
            Example: ``PyHmmerConfig(cpus=4, Z=1000, domZ=1000)`` to use 4 CPU cores
            and set database size parameters for E-value calculation.
            Default: PyHmmerConfig().

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
    keywords: list[str] = ConfigField(
        title="Keywords to Search",
        description="Keywords to search for in domain descriptions (case-insensitive).",
    )

    # Advanced parameters
    evalue_threshold: float = ConfigField(
        title="Max E-value",
        default=0.005,
        description="Maximum E-value for significant HMM hits; lower is more stringent (typical range 0.0001 to 0.01).",
        examples=[0.0001, 0.01],
    )
    query_coverage: float | None = ConfigField(
        title="Min Query Coverage",
        default=None,
        description="Min query coverage percentage for significant hits (0-100).",
    )
    match_all_keywords: bool = ConfigField(
        title="Match All Keywords",
        default=False,
        description="If True, require ALL keywords to be found. If False, require ANY keyword (default).",
    )
    hmmscan_config: PyHmmerConfig = ConfigField(
        title="PyHMMER Config",
        default_factory=PyHmmerConfig,
        description="Configuration for PyHMMER hmmscan.",
    )


@constraint(
    key="protein-domain",
    label="Protein Domain Match",
    config=ProteinDomainConfig,
    description="Evaluate whether sequences contains protein domains matching specified keywords",
    tools_called=["pyhmmer-hmmscan", "prodigal-prediction"],
    category="protein_quality",
    supported_sequence_types=["dna", "protein"],
)
def protein_domain_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinDomainConfig
) -> list[ConstraintOutput]:
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
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one DNA or protein sequence. DNA sequences are first
            processed through ORF prediction.

        config (ProteinDomainConfig): Configuration object containing ``hmm_db``
            (path to HMM database), ``keywords`` (list of domain keywords to search),
            ``evalue_threshold`` (default: 0.005), ``query_coverage`` (default: None),
            ``match_all_keywords`` (default: False), and ``hmmscan_config``
            (default: PyHmmerConfig()).

    Returns:
        list[ConstraintOutput]: One result per sequence. A score of 0.0 indicates
            domain criteria are satisfied (matching domains found) and 1.0 indicates
            no matching domains found or failure to meet keyword requirements.
            ``metadata`` carries:

            **For DNA sequences:**

            - ``prodigal_proteins``: List of dicts of predicted proteins from
              Prodigal (or ``None`` if no ORFs were predicted)
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

    Raises:
        ValueError: If ``hmm_db`` path doesn't exist or ``keywords`` list is empty.
        RuntimeError: If HMMER hmmscan execution fails or Prodigal ORF prediction
            fails for DNA sequences.

    Examples:
        Evaluating domain presence in protein with single keyword:

        >>> from proto_language.core import Sequence, SequenceType
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> cfg = ProteinDomainConfig(hmm_db="Pfam-A.hmm", keywords=["kinase"], evalue_threshold=0.001)
        >>> results = protein_domain_constraint([(seq,)], config=cfg)
        >>> print(results[0].score)  # 0.0 if kinase domain found, 1.0 if not
        >>> print(results[0].metadata["domain_keywords_found"])  # ['kinase'] if found

        Evaluating DNA sequence (with automatic ORF prediction):

        >>> dna_seq = Sequence("ATGGTACTGAGCCCAGCG...", "dna")
        >>> cfg = ProteinDomainConfig(hmm_db="Pfam-A.hmm", keywords=["helicase"])
        >>> results = protein_domain_constraint([(dna_seq,)], config=cfg)
        >>> print(results[0].metadata["prodigal_protein_count"])  # Number of predicted ORFs
        >>> print(results[0].metadata["domain_matching_proteins"])  # IDs of proteins with helicase domain
    """
    sequences = [seq for (seq,) in input_sequences]

    hmm_db = Path(config.hmm_db)
    if not hmm_db.exists():
        raise ValueError(f"HMM database not found: {hmm_db}")

    if not config.keywords or not isinstance(config.keywords, list):
        raise ValueError("Keywords must be a non-empty list")

    dna_sequences: list[tuple[int, Sequence]] = []
    protein_sequences: list[tuple[int, Sequence]] = []

    for idx, seq in enumerate(sequences):
        if seq.sequence_type == "dna":
            dna_sequences.append((idx, seq))
        else:  # protein (validated by Constraint._validate_sequence_types)
            protein_sequences.append((idx, seq))

    keywords_lower = [kw.lower() for kw in config.keywords]

    dna_results: dict[int, ConstraintOutput] = {}
    protein_results: dict[int, ConstraintOutput] = {}

    if dna_sequences:
        dna_indices, dna_seqs = zip(*dna_sequences, strict=False)
        dna_output = _process_dna_sequences(list(dna_seqs), hmm_db, keywords_lower, config)
        dna_results = dict(zip(dna_indices, dna_output, strict=False))
    if protein_sequences:
        protein_indices, protein_seqs = zip(*protein_sequences, strict=False)
        protein_output = _process_protein_sequences(list(protein_seqs), hmm_db, keywords_lower, config)
        protein_results = dict(zip(protein_indices, protein_output, strict=False))

    return [dna_results[idx] if idx in dna_results else protein_results[idx] for idx in range(len(sequences))]


def _process_dna_sequences(
    input_sequences: list[Sequence], hmm_db: Path, keywords_lower: list[str], config: ProteinDomainConfig
) -> list[ConstraintOutput]:
    """Process DNA sequences: Run Prodigal in batch, then check domains. Returns one result per sequence."""
    try:
        dna_sequences = [seq.sequence for seq in input_sequences]
        prodigal_inputs = ProdigalInput(input_sequences=dna_sequences)
        prodigal_config = ProdigalConfig()
        result = run_prodigal_prediction(prodigal_inputs, prodigal_config)
        all_proteins_per_seq = result.predicted_orfs
        gene_counts = result.num_orfs_per_sequence

    except Exception as e:
        raise RuntimeError(f"Prodigal execution failed: {e}") from e

    results = []
    for proteins_list, gene_count in zip(all_proteins_per_seq, gene_counts, strict=False):
        orf_dicts = [orf.model_dump() for orf in proteins_list]
        metadata: dict[str, Any] = {
            "prodigal_proteins": orf_dicts or None,
            "prodigal_protein_count": gene_count,
        }

        if len(proteins_list) == 0:
            metadata["domain_search_results"] = None
            metadata["domain_keywords_found"] = []
            metadata["domain_matching_proteins"] = []
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata=metadata))
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

        serializable_results = []
        matching_proteins = []
        all_keywords_found: set[str] = set()

        for orf, batch_result in zip(proteins_list, batch_results, strict=False):
            serializable_result = {
                k: v for k, v in batch_result.items() if k not in ("matching_hits", "all_hits", "significant_hits")
            }
            serializable_result["protein_id"] = orf.id
            serializable_result["protein_description"] = (
                orf.description if hasattr(orf, "description") and orf.description else ""
            )
            serializable_results.append(serializable_result)

            if batch_result["keywords_found"]:
                matching_proteins.append(orf.id)
                all_keywords_found.update(batch_result["keywords_found"])

        metadata["domain_search_results"] = serializable_results
        metadata["domain_keywords_found"] = sorted(all_keywords_found)
        metadata["domain_matching_proteins"] = matching_proteins

        if config.match_all_keywords:
            score = MIN_ENERGY if len(all_keywords_found) == len(keywords_lower) else MAX_ENERGY
        else:
            score = MIN_ENERGY if all_keywords_found else MAX_ENERGY

        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results


def _process_protein_sequences(
    input_sequences: list[Sequence], hmm_db: Path, keywords_lower: list[str], config: ProteinDomainConfig
) -> list[ConstraintOutput]:
    """Process protein sequences: Check domains in batch.

    Args:
        input_sequences (list[Sequence]): List of protein sequences.
        hmm_db (Path): Path to HMM database.
        keywords_lower (list[str]): Lowercase keywords to search for.
        config (ProteinDomainConfig): Domain constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per sequence.
    """
    protein_sequences = [seq.sequence for seq in input_sequences]

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
        raise RuntimeError(f"HMMER execution failed: {e}") from e

    results = []
    for batch_result in batch_results:
        matching_hits = batch_result["matching_hits"]
        all_hits = batch_result["all_hits"]
        serializable_result = {
            k: v for k, v in batch_result.items() if k not in ("matching_hits", "all_hits", "significant_hits")
        }

        metadata: dict[str, Any] = {
            "domain_search_results": [serializable_result],
            "domain_keywords_found": batch_result["keywords_found"],
            "domain_matching_hits": [hit.model_dump() for hit in matching_hits] or None,
            "hmmscan_all_hits": [hit.model_dump() for hit in all_hits] or None,
        }

        keywords_found = set(batch_result["keywords_found"])
        if config.match_all_keywords:
            score = MIN_ENERGY if len(keywords_found) == len(keywords_lower) else MAX_ENERGY
        else:
            score = MIN_ENERGY if keywords_found else MAX_ENERGY

        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results


def _check_protein_domains_batch(
    protein_sequences: list[str],
    hmm_db: str,
    keywords_lower: list[str],
    evalue_threshold: float,
    hmmscan_config: PyHmmerConfig,
    query_coverage: float | None = None,
) -> list[dict[str, Any]]:
    """Helper function to check a batch of protein sequences for domain matches.

    Args:
        protein_sequences (list[str]): Protein sequence strings to analyze.
        hmm_db (str): Path to HMM database.
        keywords_lower (list[str]): Lowercase keywords to search for.
        evalue_threshold (float): E-value threshold for significance.
        query_coverage (float | None): Minimum query coverage (optional).
        hmmscan_config (PyHmmerConfig): Configuration for PyHMMER hmmscan.

    Returns:
        list[dict[str, Any]]: List of dictionaries with analysis results including hits and keywords found.
    """
    # Build PyHMMER hmmscan input directly from sequences (no temp files).
    hmmscan_input = PyHmmscanInput(sequences=protein_sequences, hmm_db=hmm_db)

    # Use provided config or default
    final_config = hmmscan_config

    # Run PyHMMER hmmscan
    result = run_pyhmmer_hmmscan(inputs=hmmscan_input, config=final_config)

    batch_results: list[dict[str, Any]] = []

    # Pre-group domain hits by query index for O(H + S) lookup
    hits_by_query: dict[int, list[Any]] = {}
    for hit in result.domain_hits:
        hits_by_query.setdefault(hit.query_idx, []).append(hit)

    # Early exit if no domain hits at all
    if result.num_domain_hits == 0:
        return [
            {"all_hits": [], "significant_hits": [], "matching_hits": [], "keywords_found": []}
            for _ in protein_sequences
        ]

    for seq_idx, protein_seq in enumerate(protein_sequences):
        # Look up domain hits for this sequence
        seq_domain_hits = hits_by_query.get(seq_idx, [])

        if not seq_domain_hits:
            batch_results.append(
                {
                    "all_hits": [],
                    "significant_hits": [],
                    "matching_hits": [],
                    "keywords_found": [],
                }
            )
            continue

        # Filter by E-value threshold
        significant_hits = [hit for hit in seq_domain_hits if hit.i_evalue <= evalue_threshold]

        # Apply query coverage filter if specified
        if query_coverage is not None:
            query_len = len(protein_seq)
            if query_len > 0:
                significant_hits = [
                    hit
                    for hit in significant_hits
                    if ((hit.target_to - hit.target_from + 1) / query_len * 100) >= query_coverage
                ]

        # Find hits matching keywords
        if significant_hits:
            matching_hits = [
                hit
                for hit in significant_hits
                if any(kw in (hit.target_description or "").lower() for kw in keywords_lower)
            ]
        else:
            matching_hits = []

        # Extract found keywords
        found_keywords: list[str] = []
        for hit in matching_hits:
            description_lower = (hit.target_description or "").lower()
            for keyword in keywords_lower:
                if keyword in description_lower and keyword not in found_keywords:
                    found_keywords.append(keyword)

        batch_results.append(
            {
                "all_hits": seq_domain_hits,
                "significant_hits": significant_hits,
                "matching_hits": matching_hits,
                "keywords_found": found_keywords,
            }
        )

    return batch_results
