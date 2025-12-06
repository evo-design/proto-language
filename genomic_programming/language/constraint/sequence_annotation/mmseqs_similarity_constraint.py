"""
ORF prediction + MMseqs gene similarity constraint for evaluating similarity (percent identity).
Supports DNA (with ORF prediction) and Protein sequences (direct search).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
import pandas as pd
from pydantic import model_validator

from proto_language.language.core import Sequence, SequenceType, DNA_NUCLEOTIDES
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.orf_prediction.orfipy import OrfipyConfig, OrfipyInput, run_orfipy_prediction
from proto_language.tools.orf_prediction.prodigal import ProdigalConfig, ProdigalInput, run_prodigal_prediction
from proto_language.tools.gene_annotation.mmseqs import (
    MmseqsSearchProteinsConfig,
    MmseqsSearchProteinsInput,
    mmseqs_search_proteins,
)
from proto_language.utils import MIN_ENERGY, MAX_ENERGY, calculate_percentage_range_deviation, resolve_paths


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

        mmseqs_config (Optional[MmseqsSearchProteinsConfig]): Optional MMseqs2
            configuration including sensitivity, threads, E-value threshold, and
            other search parameters. If None, uses default configuration (sensitivity=7.5,
            threads=4). Example: ``MmseqsSearchProteinsConfig(threads=16, sensitivity=8.0)``
            for faster, more sensitive searches. Default: None.

        orf_predictor (Literal["orfipy", "prodigal"]): ORF prediction tool for
            DNA sequences (ignored for protein inputs). Options:
            - "prodigal": Prokaryotic gene finder, faster and more accurate for
              bacterial/archaeal genomes (default)
            - "orfipy": Viral ORF finder
            Choose based on your organism: prodigal for bacteria, orfipy for
            viral sequences. Default: "prodigal".

        orfipy_config (Optional[OrfipyConfig]): Optional ORFipy configuration
            (DNA only, used when ``orf_predictor="orfipy"``). Controls minimum
            ORF length, start codons, and other ORF parameters. If None, uses
            default configuration (min_length=75, all standard start codons).
            Default: None.

        prodigal_config (Optional[ProdigalConfig]): Optional Prodigal configuration
            (DNA only, used when ``orf_predictor="prodigal"``). Controls gene
            finding mode and translation table. If None, uses default configuration
            (single mode, translation_table=11). Default: None.
    
    Note:
        The similarity range [min_similarity, max_similarity] defines acceptable percent
        identity. Sequences with hits outside this range are penalized. For example:
        - [40, 70]: Moderate similarity, useful for inferring functional similarity while
          avoiding identical seuqences
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
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = ConfigField(
        title="MMseqs Configuration",
        default=None,
        description="MMseqs configuration (threads, sensitivity, etc.). If None, uses default configuration.",
        advanced=True,
    )
    orf_predictor: Literal["orfipy", "prodigal"] = ConfigField(
        title="ORF Predictor",
        default="prodigal",
        description="ORF prediction tool (DNA only): 'orfipy' (viral) or 'prodigal' (prokaryotic).",  #  Ignored for protein sequences.
        advanced=True,
    )
    # TODO: These should be the same parameter
    orfipy_config: Optional[OrfipyConfig] = ConfigField(
        title="ORFipy Configuration",
        default=None,
        description="ORFipy configuration (DNA only, used if orf_predictor='orfipy'). If None, uses default.",
        advanced=True,
    )
    prodigal_config: Optional[ProdigalConfig] = ConfigField(
        title="Prodigal Configuration",
        default=None,
        description="Prodigal configuration (DNA only, used if orf_predictor='prodigal'). If None, uses default.",
        advanced=True,
    )


@ConstraintRegistry.register(
    key="mmseqs-gene-similarity",
    label="Gene/Protein Similarity",
    config=MMseqsSimilarityConfig,
    description="Evaluate similarity (percent identity) using MMseqs. For DNA: predicts ORFs first. For proteins: searches directly.",
    mode="score",
    batched=True,
    concatenate=True,
    tools_called=["mmseqs", "prodigal", "orfipy"],
    category="sequence annotation",
)
def mmseqs_similarity_constraint(sequences: List[Sequence], config: MMseqsSimilarityConfig) -> List[float]:
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
        sequences (List[Sequence]): List of DNA or protein sequences to evaluate.
            All sequences in the list must be the same type (all DNA or all PROTEIN).
            For DNA sequences, ORF prediction is performed automatically based on
            the configured predictor.
            
        config (MMseqsSimilarityConfig): Configuration object containing ``min_similarity``
            (minimum percent identity, default: 0.0), ``max_similarity`` (maximum
            percent identity, default: 100.0), ``mmseqs_db`` (database path),
            ``orf_predictor`` (default: "prodigal"), and optional advanced configs
            for MMseqs2, ORFipy, and Prodigal.

    Returns:
        List[float]: Constraint scores for each sequence. A score of 0.0 indicates
            all database hits have percent identity within the acceptable range
            [min_similarity, max_similarity]. Higher scores indicate violations, with
            the score proportional to the average deviation from the acceptable
            range. Maximum penalty (1.0) is returned if no ORFs are found or if
            MMseqs2 search fails.

    Raises:
        ValueError: If sequences are of mixed types (some DNA, some protein).
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Metadata varies by
        sequence type and ORF predictor:
        
        **For DNA sequences (with Prodigal):**
        - ``prodigal_orfs``: List of dictionaries containing predicted ORF information
          (id, start, end, strand, protein_sequence, etc.)
        - ``mmseqs_results``: List of dictionaries with MMseqs2 hit information
          (query, target, pident, evalue, bitscore, etc.)
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
    
    Examples:
        Filtering for sequences with low similarity to existing proteins:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> protein_seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", SequenceType.PROTEIN)
        >>> config = MMseqsSimilarityConfig(
        ...     min_similarity=10.0,
        ...     max_similarity=30.0,
        ...     mmseqs_db="/data/databases/uniref90"
        ... )
        >>> scores = mmseqs_similarity_constraint([protein_seq], config)
        >>> if scores[0] == 0.0:
        ...     print("Novel protein (no high-similarity hits)")
        >>> print(protein_seq._metadata["similarity_compliance_rate"])
        
        Custom MMseqs2 configuration for sensitive search:
        
        >>> from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
        >>> mmseqs_cfg = MmseqsSearchProteinsConfig(
        ...     threads=32,         # Use 32 CPU cores
        ...     max_seqs=1000       # Return up to 1000 hits per query
        ... )
        >>> config = MMseqsSimilarityConfig(
        ...     min_similarity=20.0,
        ...     max_similarity=60.0,
        ...     mmseqs_db="/data/databases/trembl",
        ...     mmseqs_config=mmseqs_cfg
        ... )
        >>> scores = mmseqs_similarity_constraint([protein_seq], config)
    """
    if not sequences:
        return []

    sequence_type = sequences[0].sequence_type

    # Validate all same type
    if not all(seq.sequence_type == sequence_type for seq in sequences):
        raise ValueError("All sequences must be same type (all DNA or all PROTEIN)")

    scores = []
    all_proteins_data = [] 

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Get proteins (ORF prediction for DNA, direct for protein)
        if sequence_type == SequenceType.DNA:
            if config.orf_predictor == "prodigal":
                prodigal_config = config.prodigal_config or ProdigalConfig()
                sequences_clean = [
                    "".join(c for c in seq.sequence.upper() if c in DNA_NUCLEOTIDES)
                    for seq in sequences
                ]

                prodigal_input = ProdigalInput(input_sequences=sequences_clean)
                result = run_prodigal_prediction(inputs=prodigal_input, config=prodigal_config)

                for seq_idx, orfs_df in enumerate(result.results_per_sequence):
                    sequences[seq_idx]._metadata["prodigal_orfs"] = (
                        orfs_df.to_dict("records") if not orfs_df.empty else []
                    )

                    if not orfs_df.empty:
                        for _, row in orfs_df.iterrows():
                            all_proteins_data.append((
                                seq_idx, row['id'], row['protein_sequence']
                            ))

            else:  # orfipy
                orfipy_input = OrfipyInput(sequences=sequences)
                orfipy_config = config.orfipy_config or OrfipyConfig(output_dir="")

                orfipy_run_config = orfipy_config.model_copy(
                    update={"output_dir": str(temp_path / "orfipy_out")}
                )
                result = run_orfipy_prediction(inputs=orfipy_input, config=orfipy_run_config)
                full_orfs = result.results_df if result.results_df is not None else pd.DataFrame()

                for seq_idx in range(len(sequences)):
                    if not full_orfs.empty:
                        seq_orfs = full_orfs[full_orfs['parent_id'] == f'seq_{seq_idx}'].copy()
                    else:
                        seq_orfs = pd.DataFrame()

                    sequences[seq_idx]._metadata["orfipy_orfs"] = (
                        seq_orfs.to_dict("records") if not seq_orfs.empty else []
                    )

                    if not seq_orfs.empty:
                        for _, row in seq_orfs.iterrows():
                            all_proteins_data.append((
                                seq_idx, row['orf_id'], row['amino_acid_sequence']
                            ))

        else:  # PROTEIN sequences - use directly
            for seq_idx, seq in enumerate(sequences):
                protein_id = f"protein_{seq_idx}"
                all_proteins_data.append((seq_idx, protein_id, seq.sequence))
                sequences[seq_idx]._metadata["direct_protein"] = {
                    "id": protein_id,
                    "sequence": seq.sequence,
                    "length": len(seq.sequence)
                }

        # Write all proteins to combined FASTA
        if not all_proteins_data:
            # No proteins found
            for seq in sequences:
                seq._metadata.update({
                    "mmseqs_results": [],
                    "unique_orfs_with_hits": 0,
                    "orfs_with_acceptable_similarity": 0,
                    "total_orfs_with_hits": 0,
                    "similarity_compliance_rate": 0.0,
                })
            return [MAX_ENERGY] * len(sequences)

        combined_fasta = temp_path / "all_proteins.faa"
        protein_to_seq = {}

        with open(combined_fasta, "w") as f:
            for seq_idx, prot_id, prot_seq in all_proteins_data:
                f.write(f">{prot_id}\n{prot_seq}\n")
                protein_to_seq[prot_id] = seq_idx

        # MMseqs call
        resolved_db = resolve_paths(config.mmseqs_db)
        mmseqs_config = config.mmseqs_config or MmseqsSearchProteinsConfig(results_dir="")

        mmseqs_input = MmseqsSearchProteinsInput(
            query_sequences=str(combined_fasta), mmseqs_db=resolved_db
        )
        mmseqs_run_config = mmseqs_config.model_copy(
            update={"results_dir": str(temp_path / "mmseqs_out")}
        )
        mmseqs_result = mmseqs_search_proteins(mmseqs_input, mmseqs_run_config)

        if not mmseqs_result.success:
            for seq in sequences:
                seq._metadata.update({
                    "mmseqs_error": True,
                    "mmseqs_error_messages": mmseqs_result.errors,
                    "mmseqs_results": [],
                    "unique_orfs_with_hits": 0,
                    "orfs_with_acceptable_similarity": 0,
                    "total_orfs_with_hits": 0,
                    "similarity_compliance_rate": 0.0,
                })
            return [MAX_ENERGY] * len(sequences)

        all_results = mmseqs_result.results_df if mmseqs_result.results_df is not None else pd.DataFrame()

    # Split results back to sequences and score
    for seq_idx, seq in enumerate(sequences):
        seq_prot_ids = [pid for pid, sid in protein_to_seq.items() if sid == seq_idx]

        if not all_results.empty and seq_prot_ids:
            seq_results = all_results[all_results['query'].isin(seq_prot_ids)].copy()
        else:
            seq_results = pd.DataFrame()

        num_hits = len(seq_results) if not seq_results.empty else 0

        seq._metadata.update({
            "mmseqs_results": seq_results.to_dict("records") if not seq_results.empty else [],
            "unique_orfs_with_hits": num_hits,
        })

        if seq_results.empty or "pident" not in seq_results.columns:
            seq._metadata.update({
                "orfs_with_acceptable_similarity": 0,
                "total_orfs_with_hits": 0,
                "similarity_compliance_rate": 0.0,
            })
            scores.append(MAX_ENERGY)
            continue

        # Score each hit
        acceptable = 0
        violations = []

        for _, row in seq_results.iterrows():
            pident = row["pident"]
            if config.min_similarity <= pident <= config.max_similarity:
                acceptable += 1
            else:
                violations.append(calculate_percentage_range_deviation(
                    pident, config.min_similarity, config.max_similarity
                ))

        seq._metadata.update({
            "orfs_with_acceptable_similarity": acceptable,
            "total_orfs_with_hits": num_hits,
            "similarity_compliance_rate": acceptable / num_hits,
        })

        scores.append(MIN_ENERGY if not violations else min(MAX_ENERGY, np.mean(violations)))

    return scores
