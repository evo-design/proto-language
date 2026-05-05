"""Supports DNA (with ORF prediction) and Protein sequences (direct search)."""

import json
from typing import Any, Literal

import numpy as np
from proto_tools import (
    Mmseqs2SearchProteinsConfig,
    Mmseqs2SearchProteinsInput,
    OrfipyConfig,
    OrfipyInput,
    ProdigalConfig,
    ProdigalInput,
    run_mmseqs2_search_proteins,
    run_orfipy_prediction,
    run_prodigal_prediction,
)
from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import DNA_NUCLEOTIDES, ConstraintOutput, Sequence
from proto_language.storage import FileType, resolve_paths, store_file
from proto_language.utils import (
    MAX_ENERGY,
    MIN_ENERGY,
    calculate_percentage_range_deviation,
)


class MMseqsSimilarityConfig(BaseConfig):
    """Configuration for MMseqs gene similarity constraint.

    This class defines configuration parameters for evaluating sequence similarity
    (percent identity) to known proteins using MMseqs2, an ultra-fast sequence
    search tool. For DNA sequences, the constraint first predicts open reading
    frames (ORFs) using either Prodigal or ORFipy, then searches the translated
    proteins against a reference database. For protein sequences, the search is
    performed directly.

    Attributes:
        min_similarity (float): Minimum acceptable percent identity (0-100). Hits
            below this threshold are penalized. Lower values are more permissive.
            For example, 30 means sequences must have at least 30% identity to
            database hits. Typical values: 20-30 for remote similarity, 40-80 for
            moderate similarity, 80-95 for close similarity.

        max_similarity (float): Maximum acceptable percent identity (0-100). Hits
            above this threshold are penalized. Higher values allow more similar
            matches. For example, 70 means sequences must have at most 70% identity
            to avoid being too similar. Use this to filter out sequences that are
            too similar to existing proteins (e.g., novelty filter).

        mmseqs_db (str): Path to MMseqs2 protein database for similarity searching.
            Must be a preprocessed MMseqs2 database (created with `mmseqs createdb`).
            Example: "/data/databases/uniref50" or "~/databases/ncbi_nr_mmseqs".

        mmseqs_config (Mmseqs2SearchProteinsConfig): MMseqs2 configuration including
            sensitivity, threads, E-value threshold, and other search parameters.
            Example: ``Mmseqs2SearchProteinsConfig(threads=16, sensitivity=7.5)``
            to parallelize across 16 cores at maximum sensitivity (valid range
            is 1.0-7.5). Default: Mmseqs2SearchProteinsConfig().

        orf_predictor (Literal['orfipy', 'prodigal']): ORF prediction tool for
            DNA sequences (ignored for protein inputs). Options:
            - "prodigal": Prokaryotic gene finder, faster and more accurate for
              bacterial/archaeal genomes (default)
            - "orfipy": Viral ORF finder
            Choose based on your organism: prodigal for bacteria, orfipy for
            viral sequences. Default: "prodigal".

        orfipy_config (OrfipyConfig): ORFipy configuration (DNA only, used when
            ``orf_predictor="orfipy"``). Controls minimum ORF length, start codons,
            and other ORF parameters. Default: OrfipyConfig().

        prodigal_config (ProdigalConfig): Prodigal configuration (DNA only, used
            when ``orf_predictor="prodigal"``). Controls gene finding mode and
            translation table. Default: ProdigalConfig().

    Note:
        For examples with tool configuration, see:
        >>> from proto_tools import Mmseqs2SearchProteinsConfig
        The similarity range [min_similarity, max_similarity] defines acceptable percent
        identity. Sequences with hits outside this range are penalized. For example:
        - [40, 70]: Moderate similarity, useful for inferring functional similarity while
          avoiding identical sequences
        - [0, 40]: Low similarity filter, for novelty/uniqueness constraints
        - [80, 100]: High similarity filter, for functional conservation requirements
    """

    # Required parameters
    min_similarity: float = ConfigField(
        title="Min Acceptable Percent Identity",
        ge=0.0,
        le=100.0,
        description="Minimum acceptable percent identity (0-100). Lower values are more permissive.",
        examples=[20, 50],
    )
    max_similarity: float = ConfigField(
        title="Max Acceptable Percent Identity",
        ge=0.0,
        le=100.0,
        description="Maximum acceptable percent identity (0-100). Higher values allow more similar hits.",
        examples=[51, 95],
    )
    mmseqs_db: str = ConfigField(
        title="MMseqs2 Protein Database",
        description="Path to MMseqs2 protein database for similarity search",
    )

    # Advanced parameters
    mmseqs_config: Mmseqs2SearchProteinsConfig = ConfigField(
        title="MMseqs Configuration",
        default_factory=Mmseqs2SearchProteinsConfig,
        description="MMseqs configuration (threads, sensitivity, etc.).",
        advanced=True,
    )
    orf_predictor: Literal["orfipy", "prodigal"] = ConfigField(
        title="ORF Predictor",
        default="prodigal",
        description="ORF prediction tool (DNA only): 'orfipy' (viral) or 'prodigal' (prokaryotic).",  #  Ignored for protein sequences.
        advanced=True,
    )
    orfipy_config: OrfipyConfig = ConfigField(
        title="ORFipy Configuration",
        default_factory=OrfipyConfig,
        description="ORFipy configuration (DNA only, used if orf_predictor='orfipy').",
        advanced=True,
        depends_on={"field": "orf_predictor", "value": "orfipy"},
    )
    prodigal_config: ProdigalConfig = ConfigField(
        title="Prodigal Configuration",
        default_factory=ProdigalConfig,
        description="Prodigal configuration (DNA only, used if orf_predictor='prodigal').",
        advanced=True,
        depends_on={"field": "orf_predictor", "value": "prodigal"},
    )

    @model_validator(mode="after")
    def validate_similarity_range(self) -> "MMseqsSimilarityConfig":
        """Ensure min_similarity does not exceed max_similarity."""
        if self.min_similarity > self.max_similarity:
            raise ValueError(
                f"min_similarity ({self.min_similarity}) must be <= max_similarity ({self.max_similarity})."
            )
        return self


@constraint(
    key="mmseqs-gene-similarity",
    label="Gene/Protein Similarity",
    config=MMseqsSimilarityConfig,
    description="Evaluate similarity (percent identity) using MMseqs. For DNA: predicts ORFs first. For proteins: searches directly.",
    tools_called=["mmseqs2-search-proteins", "prodigal-prediction", "orfipy-prediction"],
    category="sequence annotation",
    supported_sequence_types=["dna", "protein"],
)
def mmseqs_similarity_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: MMseqsSimilarityConfig
) -> list[ConstraintOutput]:
    """Evaluate sequence similarity using MMseqs2 protein database search.

    This constraint function evaluates whether protein sequences (or proteins
    predicted from DNA sequences) have percent identity to known proteins within
    an acceptable range. It uses MMseqs2, an ultra-fast sequence search tool,
    to search against a reference protein database and calculates similarity scores.

    For DNA sequences, the function first predicts open reading frames (ORFs)
    using either Prodigal (for prokaryotes) or ORFipy (viral), then
    searches the translated proteins. For protein sequences, the search is
    performed directly. The constraint is satisfied when all database hits have
    percent identity within the specified [min_similarity, max_similarity] range.

    Args:
        sequences: List of DNA or protein sequences to evaluate.
            All sequences in the list must be the same type (all DNA or all PROTEIN).
            For DNA sequences, ORF prediction is performed automatically based on
            the configured predictor.

        config (MMseqsSimilarityConfig): Configuration object containing ``min_similarity``
            (minimum percent identity, default: 0.0), ``max_similarity`` (maximum
            percent identity, default: 100.0), ``mmseqs_db`` (database path),
            ``orf_predictor`` (default: "prodigal"), and optional advanced configs
            for MMseqs2, ORFipy, and Prodigal.
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.

    Returns:
        list[ConstraintOutput]: One result per sequence. Score 0.0 means all hits
            fall within [min_similarity, max_similarity]; higher scores indicate
            greater deviation. Score 1.0 (MAX_ENERGY) is returned if no ORFs are
            found or MMseqs2 search fails. ``metadata`` carries:

            **For DNA sequences (with Prodigal):**

            - ``prodigal_orfs``: List of dictionaries containing predicted ORF information
              (id, start, end, strand, protein_sequence, etc.)
            - ``mmseqs_results``: List of dictionaries with MMseqs2 hit information
              (target_id, pident, evalue)
            - ``unique_orfs_with_hits``: Integer count of ORFs with database matches
            - ``orfs_with_acceptable_similarity``: Integer count of ORFs with hits in
              acceptable range
            - ``total_orfs_with_hits``: Integer total number of ORF-hit pairs
            - ``similarity_compliance_rate``: Float fraction of hits within acceptable
              range (0.0-1.0)

            **For DNA sequences (with ORFipy):**

            - ``orfipy_orfs``: List of dictionaries with ORFipy ORF predictions
            - Other fields same as Prodigal above

            **For protein sequences:**

            - ``direct_protein``: Dictionary with protein information (id, sequence, length)
            - ``mmseqs_results``: List of MMseqs2 hit dictionaries
            - ``unique_orfs_with_hits``: Count of hits (always 1 or 0 for single proteins)
            - ``orfs_with_acceptable_similarity``: Count of acceptable hits
            - ``total_orfs_with_hits``: Total hit count
            - ``similarity_compliance_rate``: Fraction of hits in range

            **Error metadata (when MMseqs2 fails):**

            - ``mmseqs_error``: Boolean True
            - ``mmseqs_error_messages``: List of error message strings

    Raises:
        ValueError: If sequences are of mixed types (some DNA, some protein).

    Examples:
        Filtering for sequences with low similarity to existing proteins:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> protein_seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> config = MMseqsSimilarityConfig(
        ...     min_similarity=10.0, max_similarity=30.0, mmseqs_db="/data/databases/uniref90"
        ... )
        >>> results = mmseqs_similarity_constraint([(protein_seq,)], config)
        >>> print(results[0].score)  # 0.0 means no high-similarity hits
        >>> print(results[0].metadata["similarity_compliance_rate"])
    """
    # Extract sequences from tuples
    sequences = [seq for (seq,) in input_sequences]
    sequence_type = sequences[0].sequence_type

    # Validate all same type
    if not all(seq.sequence_type == sequence_type for seq in sequences):
        types_seen = sorted({seq.sequence_type for seq in sequences})
        raise ValueError(f"mmseqs-similarity: all sequences must share one type; got mixed {types_seen}")

    # Per-sequence metadata accumulator (one dict per input sequence)
    metadata_by_seq: list[dict[str, Any]] = [{} for _ in sequences]

    # Build protein list with mapping back to input sequences
    # protein_data: List of (seq_idx, protein_sequence) tuples
    protein_data: list[tuple[int, str]] = []

    # Get proteins (ORF prediction for DNA, direct for protein)
    if sequence_type == "dna":
        sequences_clean = [
            "".join(c for c in seq.sequence.upper() if c in DNA_NUCLEOTIDES) for (seq,) in input_sequences
        ]

        if config.orf_predictor == "prodigal":
            prodigal_config = config.prodigal_config
            prodigal_input = ProdigalInput(input_sequences=sequences_clean)
            result = run_prodigal_prediction(inputs=prodigal_input, config=prodigal_config)

            for seq_idx, orfs_list in enumerate(result.predicted_orfs):
                orf_dicts = [orf.model_dump() for orf in orfs_list]
                metadata_by_seq[seq_idx]["prodigal_orfs"] = (
                    store_file(json.dumps(orf_dicts), FileType.JSON) if orf_dicts else None
                )
                protein_data.extend((seq_idx, orf.amino_acid_sequence) for orf in orfs_list)

        else:  # orfipy
            orfipy_input = OrfipyInput(sequences=sequences_clean)
            orfipy_config = config.orfipy_config
            result = run_orfipy_prediction(inputs=orfipy_input, config=orfipy_config)

            for seq_idx, orfs_list in enumerate(result.predicted_orfs):
                orf_dicts = [orf.model_dump() for orf in orfs_list]
                metadata_by_seq[seq_idx]["orfipy_orfs"] = (
                    store_file(json.dumps(orf_dicts), FileType.JSON) if orf_dicts else None
                )
                protein_data.extend((seq_idx, orf.amino_acid_sequence) for orf in orfs_list)

    else:  # PROTEIN sequences - use directly
        for seq_idx, seq in enumerate(sequences):
            protein_data.append((seq_idx, seq.sequence))
            metadata_by_seq[seq_idx]["direct_protein"] = {
                "id": f"protein_{seq_idx}",
                "sequence": seq.sequence,
                "length": len(seq.sequence),
            }

    # Handle case where no proteins were found
    if not protein_data:
        for meta in metadata_by_seq:
            meta.update(
                {
                    "mmseqs_results": None,
                    "unique_orfs_with_hits": 0,
                    "orfs_with_acceptable_similarity": 0,
                    "total_orfs_with_hits": 0,
                    "similarity_compliance_rate": 0.0,
                }
            )
        return [ConstraintOutput(score=MAX_ENERGY, metadata=meta) for meta in metadata_by_seq]

    # Extract protein sequences for MMseqs search
    protein_sequences = [prot_seq for _, prot_seq in protein_data]
    protein_to_seq_idx = [seq_idx for seq_idx, _ in protein_data]

    # Run MMseqs search
    resolved_db = resolve_paths(config.mmseqs_db)
    mmseqs_config = config.mmseqs_config

    mmseqs_input = Mmseqs2SearchProteinsInput(
        query_sequences=protein_sequences,
        mmseqs_db=resolved_db,
    )
    mmseqs_result = run_mmseqs2_search_proteins(mmseqs_input, mmseqs_config)

    if not mmseqs_result.success:
        for meta in metadata_by_seq:
            meta.update(
                {
                    "mmseqs_error": True,
                    "mmseqs_error_messages": mmseqs_result.errors,
                    "mmseqs_results": None,
                    "unique_orfs_with_hits": 0,
                    "orfs_with_acceptable_similarity": 0,
                    "total_orfs_with_hits": 0,
                    "similarity_compliance_rate": 0.0,
                }
            )
        return [ConstraintOutput(score=MAX_ENERGY, metadata=meta) for meta in metadata_by_seq]

    # Aggregate hits by input sequence
    # seq_hits[seq_idx] = list of all hits for that input sequence
    seq_hits: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(sequences))}

    for prot_idx, result in enumerate(mmseqs_result.results):
        seq_idx = protein_to_seq_idx[prot_idx]
        for hit in result.hits:
            seq_hits[seq_idx].append(
                {
                    "target_id": hit.target_id,
                    "pident": hit.pident,
                    "evalue": hit.evalue,
                }
            )

    # Score each sequence
    results: list[ConstraintOutput] = []
    for seq_idx in range(len(sequences)):
        hits = seq_hits[seq_idx]
        num_hits = len(hits)
        meta = metadata_by_seq[seq_idx]

        meta.update(
            {
                "mmseqs_results": store_file(json.dumps(hits), FileType.JSON) if hits else None,
                "unique_orfs_with_hits": num_hits,
            }
        )

        if num_hits == 0:
            meta.update(
                {
                    "orfs_with_acceptable_similarity": 0,
                    "total_orfs_with_hits": 0,
                    "similarity_compliance_rate": 0.0,
                }
            )
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata=meta))
            continue

        # Score each hit
        acceptable = 0
        violations = []

        for hit in hits:
            pident = hit["pident"]
            if config.min_similarity <= pident <= config.max_similarity:
                acceptable += 1
            else:
                violations.append(
                    calculate_percentage_range_deviation(pident, config.min_similarity, config.max_similarity)
                )

        meta.update(
            {
                "orfs_with_acceptable_similarity": acceptable,
                "total_orfs_with_hits": num_hits,
                "similarity_compliance_rate": acceptable / num_hits,
            }
        )

        score = MIN_ENERGY if not violations else min(MAX_ENERGY, float(np.mean(violations)))
        results.append(ConstraintOutput(score=score, metadata=meta))

    return results
