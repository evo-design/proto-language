"""
Protein quality and functional analysis constraints.
"""

from __future__ import annotations
import tempfile
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
from collections import Counter
from ..base import *
from ..tools.orf_prediction.prodigal import run_prodigal
from ..tools.gene_annotation.blast import calculate_segmasker_score
from ..tools.gene_annotation.hmmer import run_hmmscan
from .utils import (
    MIN_ENERGY,
    MAX_ENERGY,
    _validate_required_config,
    _calculate_range_deviation,
    _calculate_percentage_range_deviation,
)


def _calculate_repetitiveness_score(seq: str, min_repeat_length: int = 3) -> float:
    """
    Calculate repetitiveness score based on k-mer frequency analysis

    Args:
        seq: Protein sequence to analyze
        min_repeat_length: Minimum length of repeats to consider

    Returns:
        Maximum fraction of sequence covered by repeated k-mers (0.0 to 1.0)

    Raises:
        ValueError: If length of sequence is shorter than the minimum repeat length
    """
    if len(seq) < min_repeat_length:
        raise ValueError("Sequence must be longer that the minimum repeat length")

    seq_len = len(seq)
    seq_array = np.array(list(seq))
    max_repetitive_fraction = 0.0

    for k in range(min_repeat_length, min(min_repeat_length + 7, seq_len + 1)):
        kmers = np.lib.stride_tricks.sliding_window_view(seq_array, k)
        kmer_strings = ["".join(kmer) for kmer in kmers]
        if kmer_strings:
            max_count = max(Counter(kmer_strings).values())
            repetitive_fraction = (max_count * k) / seq_len
            max_repetitive_fraction = max(max_repetitive_fraction, repetitive_fraction)

    return max_repetitive_fraction


def _check_protein_domains(
    protein_sequence: Sequence,
    hmm_db: str,
    keywords_lower: List[str],
    evalue_threshold: float,
    query_coverage: float = None,
    hmmer_kwargs: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Helper function to check a single protein sequence for domain matches.

    Args:
        protein_sequence: Protein sequence to analyze.
        hmm_db: Path to HMM database.
        keywords_lower: Lowercase keywords to search for.
        evalue_threshold: E-value threshold for significance.
        query_coverage: Minimum query coverage (optional).
        hmmer_kwargs: Additional HMMER parameters.

    Returns:
        Dictionary with analysis results including hits and keywords found.
    """
    hmmer_kwargs = hmmer_kwargs or {}

    # Write sequence to temporary file for HMMER
    with tempfile.NamedTemporaryFile(mode="w", suffix=".faa", delete=False) as temp_seq:
        temp_seq.write(f">query\n{protein_sequence.sequence}\n")
        temp_seq_path = temp_seq.name

    with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as temp_out:
        temp_out_path = temp_out.name

    try:
        # Run hmmscan
        results = run_hmmscan(
            hmm_db=hmm_db,
            query=temp_seq_path,
            output_path=temp_out_path,
            **hmmer_kwargs,
        )
        if isinstance(results, dict) and "domain" in results:
            hits_df = results["domain"]
        else:
            hits_df = results

        if len(hits_df) == 0:
            return {
                "all_hits": hits_df,
                "significant_hits": hits_df,
                "matching_hits": hits_df,
                "keywords_found": [],
            }

        # Filter by E-value threshold
        significant_hits = hits_df[hits_df["evalue"] <= evalue_threshold].copy()

        # Apply query coverage filter if specified
        if query_coverage is not None:
            query_len = len(
                protein_sequence.sequence
            )  # Note: .sequence to get the string
            if query_len > 0:
                coverage_pct = (
                    (significant_hits["ali_to"] - significant_hits["ali_from"] + 1)
                    / query_len
                    * 100
                )
                significant_hits = significant_hits[coverage_pct >= query_coverage]

        # Find hits matching keywords
        if len(significant_hits) > 0:
            keyword_pattern = "|".join(keywords_lower)
            matching_mask = (
                significant_hits["description"]
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
                description_lower = str(hit["description"]).lower()
                for keyword in keywords_lower:
                    if keyword in description_lower and keyword not in found_keywords:
                        found_keywords.append(keyword)

        return {
            "all_hits": hits_df,
            "significant_hits": significant_hits,
            "matching_hits": matching_hits,
            "keywords_found": found_keywords,
        }

    finally:
        # Clean up temporary files
        for path in [temp_seq_path, temp_out_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


def protein_length_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate whether a protein sequence length falls within acceptable range.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - min_length (int): Minimum acceptable protein length.
            - max_length (int): Maximum acceptable protein length.

    Returns:
        Constraint score where 0.0 indicates length is within acceptable range
        and higher values indicate greater deviation from acceptable range.
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"
    _validate_required_config(config, ["min_length", "max_length"])

    min_length = config["min_length"]
    max_length = config["max_length"]
    actual_length = len(input_sequence)

    input_sequence._metadata["protein_length"] = actual_length

    return _calculate_range_deviation(actual_length, min_length, max_length)


def protein_complexity_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate protein sequence complexity using segmasker to detect low-complexity regions.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - max_low_complexity (float): Maximum acceptable fraction of low-complexity regions (0.0-1.0).
            - segmasker_path (str, optional): Path to segmasker executable (default: 'segmasker').

    Returns:
        Constraint score where 0.0 indicates acceptable complexity
        and higher values indicate excessive low-complexity regions.
        Returns MAX_ENERGY if segmasker fails.

    Raises:
        ValueError: If segmasker execution fails.
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"
    _validate_required_config(config, ["max_low_complexity"])

    max_low_complexity = config["max_low_complexity"]
    segmasker_path = config.get("segmasker_path", "segmasker")

    try:
        low_complexity_fraction = calculate_segmasker_score(
            input_sequence.sequence, segmasker_path
        )

        input_sequence._metadata["low_complexity_fraction"] = low_complexity_fraction
        input_sequence._metadata["segmasker_X_count"] = int(
            low_complexity_fraction * len(input_sequence)
        )
        input_sequence._metadata["segmasker_error"] = False

        if low_complexity_fraction <= max_low_complexity:
            return MIN_ENERGY

        excess = low_complexity_fraction - max_low_complexity
        return min(MAX_ENERGY, excess / (1.0 - max_low_complexity))

    except ValueError as e:
        # Store error information in metadata
        input_sequence._metadata["low_complexity_fraction"] = float("nan")
        input_sequence._metadata["segmasker_X_count"] = float("nan")
        input_sequence._metadata["segmasker_error"] = True
        input_sequence._metadata["segmasker_error_message"] = str(e)

        # Re-raise the exception to propagate the error
        raise ValueError(f"Segmasker analysis failed: {str(e)}")


def protein_repetitiveness_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate protein sequence repetitiveness based on k-mer analysis.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - max_repetitiveness (float): Maximum acceptable repetitiveness fraction (0.0-1.0).
            - min_repeat_length (int, optional): Minimum repeat length to consider (default: 3).

    Returns:
        Constraint score where 0.0 indicates acceptable repetitiveness
        and higher values indicate excessive repetitive content.
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"
    _validate_required_config(config, ["max_repetitiveness"])

    max_repetitiveness = config["max_repetitiveness"]
    min_repeat_length = config.get("min_repeat_length", 3)

    repetitiveness_score = _calculate_repetitiveness_score(
        input_sequence.sequence, min_repeat_length
    )
    input_sequence._metadata["repetitiveness_score"] = repetitiveness_score
    input_sequence._metadata["max_repetitive_fraction"] = repetitiveness_score

    if repetitiveness_score <= max_repetitiveness:
        return MIN_ENERGY

    excess = repetitiveness_score - max_repetitiveness
    return min(MAX_ENERGY, excess / (1.0 - max_repetitiveness))


def protein_diversity_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate amino acid diversity in a protein sequence.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - min_diversity (float): Minimum acceptable amino acid diversity (0.0-1.0).

    Returns:
        Constraint score where 0.0 indicates sufficient diversity
        and higher values indicate insufficient amino acid diversity.

    Raises:
        ValueError: If sequence has length 0
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"
    _validate_required_config(config, ["min_diversity"])

    min_diversity = config["min_diversity"]
    seq = input_sequence.sequence

    # Calculate amino acid diversity score
    if len(seq) == 0:
        raise ValueError("Sequence is non-existent.")

    unique_aas = len(set(seq))
    diversity_score = unique_aas / 20.0  # 20 standard amino acids

    # Store metadata
    input_sequence._metadata["aa_diversity_score"] = diversity_score
    input_sequence._metadata["unique_amino_acid_count"] = unique_aas
    input_sequence._metadata["unique_amino_acids"] = sorted(list(set(seq)))

    # Return constraint score
    if diversity_score >= min_diversity:
        return MIN_ENERGY

    deficit = min_diversity - diversity_score
    return min(MAX_ENERGY, deficit / min_diversity)


def balanced_aa_constraint(input_sequence: Sequence, config: Dict[str, Any]) -> float:
    """
    Evaluate the presence of underrepresented amino acids in a protein sequence.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - min_aa_frequency (float): Minimum acceptable relative frequency for amino acids (0.0-1.0).
            - max_underrepresented_count (int): Maximum acceptable number of underrepresented amino acid types (0-20).

    Returns:
        Constraint score from 0.0 (best, acceptable number of underrepresented amino acids) to 1.0 (worst).
        Score is scaled based on how many excess underrepresented amino acids there are and their severity.
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"

    min_aa_frequency = config.get("min_aa_frequency", 0.02)
    max_underrepresented_count = config.get("max_underrepresented_count", 3)
    seq = input_sequence.sequence

    if len(seq) == 0:
        underrepresented_score = 1.0
        aa_counts = Counter()
        underrepresented_aas = []
        penalty_score = 1.0
    else:
        aa_counts = Counter(seq)
        if len(aa_counts) == 0:
            underrepresented_score = 1.0
            underrepresented_aas = []
            penalty_score = 1.0
        else:
            # Identify underrepresented amino acids (below minimum frequency threshold)
            frequency_threshold = min_aa_frequency * len(seq)
            underrepresented_aas = [
                aa for aa, count in aa_counts.items() if count < frequency_threshold
            ]

            # Calculate fraction of sequence that consists of underrepresented amino acids
            underrepresented_total = sum(aa_counts[aa] for aa in underrepresented_aas)
            underrepresented_score = underrepresented_total / len(seq)

            # Calculate penalty score based on count of underrepresented amino acids
            underrepresented_aa_count = len(underrepresented_aas)

            if underrepresented_aa_count <= max_underrepresented_count:
                penalty_score = 0.0
            else:
                # Scale penalty based on both excess count and how far amino acids are from threshold
                excess_count = underrepresented_aa_count - max_underrepresented_count
                max_possible_excess = (
                    20 - max_underrepresented_count
                )  # 20 standard amino acids

                # Calculate average "deficit" - how far underrepresented AAs are from threshold
                total_deficit = 0.0
                for aa in underrepresented_aas:
                    current_freq = aa_counts[aa] / len(seq)
                    deficit = min_aa_frequency - current_freq
                    total_deficit += deficit * aa_counts[aa]  # Weight by actual count

                avg_deficit = (
                    total_deficit / underrepresented_total
                    if underrepresented_total > 0
                    else 0.0
                )

                # Combine excess count with severity of underrepresentation
                count_penalty = (
                    excess_count / max_possible_excess
                    if max_possible_excess > 0
                    else 1.0
                )
                severity_penalty = (
                    avg_deficit / min_aa_frequency if min_aa_frequency > 0 else 0.0
                )
                penalty_score = min(1.0, count_penalty * (1.0 + severity_penalty))

    # Store metadata
    input_sequence._metadata["underrepresented_aa_score"] = underrepresented_score
    input_sequence._metadata["amino_acid_counts"] = dict(aa_counts)
    input_sequence._metadata["underrepresented_amino_acids"] = underrepresented_aas
    input_sequence._metadata["underrepresented_aa_count"] = len(underrepresented_aas)
    input_sequence._metadata["min_aa_frequency_threshold"] = min_aa_frequency

    # Return penalty score
    return penalty_score


def overall_protein_quality_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate protein quality either from predicted proteins (DNA input) or directly (protein input).

    For DNA sequences, runs Prodigal first to predict proteins, then checks all predicted
    proteins. For protein sequences, checks the sequence directly.

    Args:
        input_sequence: The DNA or protein sequence to analyze.
        config: Configuration dictionary containing:
            For DNA input:
                - min_high_quality_fraction (float): Minimum fraction of predicted proteins that must be high quality.
                - protein_quality_config (dict): Configuration dictionary with the following structure:
                {
                    "min_high_quality_fraction": 0.8,  # Minimum fraction of predicted proteins that must be high quality (0.0-1.0)
                    "protein_quality_config": {
                        "quality_threshold": 0.1,  # Maximum acceptable constraint score for a protein to be considered "high quality"

                        # Individual protein quality constraints (all optional):
                        "length": {
                            "min_length": 50,     # Minimum acceptable protein length (amino acids)
                            "max_length": 2000    # Maximum acceptable protein length (amino acids)
                        },
                        "complexity": {
                            "max_low_complexity": 0.3,              # Maximum fraction of low-complexity regions (0.0-1.0)
                            "segmasker_path": "segmasker"           # Path to segmasker executable (optional)
                        },
                        "repetitiveness": {
                            "max_repetitiveness": 0.4,              # Maximum acceptable repetitiveness fraction (0.0-1.0)
                            "min_repeat_length": 3                  # Minimum repeat length to consider (optional, default: 3)
                        },
                        "diversity": {
                            "min_diversity": 0.3                    # Minimum acceptable amino acid diversity (0.0-1.0, where 1.0 = all 20 amino acids)
                        },
                        "balanced_aas": {
                            "max_underrepresented": 0.2             # Maximum acceptable fraction of underrepresented amino acids (0.0-1.0)
                        }
                    }
                }

            For protein input:
                - protein_quality_config (dict): Configuration for protein quality checks with the following structure:
                {
                    "protein_quality_config": {
                        "quality_threshold": 0.1,  # Maximum acceptable constraint score for overall quality assessment

                        # Same individual constraints as above (all optional)
                        "length": { ... },
                        "complexity": { ... },
                        "repetitiveness": { ... },
                        "diversity": { ... },
                        "balanced_aas": { ... }
                    }
                }

    Returns:
        Constraint score between 0.0 and 1.0 where:
        - 0.0 indicates perfect/optimal protein quality (all constraints satisfied)
        - Values closer to 0.0 indicate better constraint satisfaction
        - 1.0 indicates worst possible protein quality (maximum constraint violation)

    Example:
        >>> dna_seq = Sequence("ATGAAACGTATTGCGTCG...", SequenceType.DNA)
        >>> minimal_config = {
        ...     "min_high_quality_fraction": 0.5,
        ...     "protein_quality_config": {
        ...         "quality_threshold": 0.2,
        ...         "length": {
        ...             "min_length": 100,
        ...             "max_length": 800
        ...         }
        ...     }
        ... }
        >>> score = overall_protein_quality_constraint(dna_seq, minimal_config)
    """
    _validate_required_config(config, ["protein_quality_config"])
    protein_config = config["protein_quality_config"]

    if input_sequence.sequence_type == SequenceType.DNA:
        # For DNA sequences: predict proteins first
        _validate_required_config(config, ["min_high_quality_fraction"])
        min_high_quality_fraction = config["min_high_quality_fraction"]

        # Get predicted proteins, this will load cached proteins if they already exist
        proteins_df = run_prodigal(input_sequence)

        if len(proteins_df) == 0:
            input_sequence._metadata["predicted_protein_count"] = 0
            input_sequence._metadata["high_quality_protein_count"] = 0
            input_sequence._metadata["high_quality_protein_fraction"] = 0.0
            input_sequence._metadata["protein_quality_details"] = []
            return 1.0  # Maximum penalty for no proteins found

        # Evaluate each predicted protein
        high_quality_count = 0
        protein_quality_details = []
        all_protein_avg_scores = []

        for idx, protein_row in proteins_df.iterrows():
            protein_seq = Sequence(protein_row["sequence"], SequenceType.PROTEIN)

            # Apply all protein quality constraints
            quality_scores = {}
            overall_scores = []

            if "length" in protein_config:
                score = protein_length_constraint(protein_seq, protein_config["length"])
                quality_scores["length"] = score
                overall_scores.append(quality_scores["length"])

            if "complexity" in protein_config:
                score = protein_complexity_constraint(
                    protein_seq, protein_config["complexity"]
                )
                quality_scores["complexity"] = score
                overall_scores.append(quality_scores["complexity"])

            if "repetitiveness" in protein_config:
                score = protein_repetitiveness_constraint(
                    protein_seq, protein_config["repetitiveness"]
                )
                quality_scores["repetitiveness"] = score
                overall_scores.append(quality_scores["repetitiveness"])

            if "diversity" in protein_config:
                score = protein_diversity_constraint(
                    protein_seq, protein_config["diversity"]
                )
                quality_scores["diversity"] = score
                overall_scores.append(quality_scores["diversity"])

            if "balanced_aas" in protein_config:
                score = balanced_aa_constraint(
                    protein_seq, protein_config["balanced_aas"]
                )
                quality_scores["balanced_aas"] = score
                overall_scores.append(quality_scores["balanced_aas"])

            # Calculate average score for this protein
            avg_score = (
                sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            )
            all_protein_avg_scores.append(avg_score)

            # Consider protein high quality if average score is below threshold
            threshold = protein_config.get("quality_threshold", 0.1)
            is_high_quality = avg_score <= threshold

            if is_high_quality:
                high_quality_count += 1

            protein_quality_details.append(
                {
                    "protein_id": protein_row["id"],
                    "length": len(protein_row["sequence"]),
                    "is_high_quality": is_high_quality,
                    "avg_constraint_score": avg_score,
                    "quality_scores": quality_scores,
                    "metadata": protein_seq._metadata.copy(),
                }
            )

        high_quality_fraction = high_quality_count / len(proteins_df)

        # Store comprehensive metadata
        input_sequence._metadata["predicted_protein_count"] = len(proteins_df)
        input_sequence._metadata["high_quality_protein_count"] = high_quality_count
        input_sequence._metadata["high_quality_protein_fraction"] = (
            high_quality_fraction
        )
        input_sequence._metadata["protein_quality_details"] = protein_quality_details
        input_sequence._metadata["protein_quality_threshold"] = protein_config.get(
            "quality_threshold", 0.1
        )

        # If we high quality fraction requirement is met, return 0
        if high_quality_fraction >= min_high_quality_fraction:
            return 0.0

        # Otherwise, return a score based on how far we are from meeting the requirement
        overall_avg_protein_score = sum(all_protein_avg_scores) / len(
            all_protein_avg_scores
        )
        fraction_deficit = (
            min_high_quality_fraction - high_quality_fraction
        ) / min_high_quality_fraction

        # Combine the average protein quality with the fraction deficit
        combined_score = (overall_avg_protein_score + fraction_deficit) / 2.0
        return min(1.0, max(0.0, combined_score))

    elif input_sequence.sequence_type == SequenceType.PROTEIN:
        # For protein sequences: evaluate quality directly on input sequence
        quality_scores = {}
        overall_scores = []

        if "length" in protein_config:
            score = protein_length_constraint(input_sequence, protein_config["length"])
            quality_scores["length"] = score
            overall_scores.append(quality_scores["length"])

        if "complexity" in protein_config:
            score = protein_complexity_constraint(
                input_sequence, protein_config["complexity"]
            )
            quality_scores["complexity"] = score
            overall_scores.append(quality_scores["complexity"])

        if "repetitiveness" in protein_config:
            score = protein_repetitiveness_constraint(
                input_sequence, protein_config["repetitiveness"]
            )
            quality_scores["repetitiveness"] = score
            overall_scores.append(quality_scores["repetitiveness"])

        if "diversity" in protein_config:
            score = protein_diversity_constraint(
                input_sequence, protein_config["diversity"]
            )
            quality_scores["diversity"] = score
            overall_scores.append(quality_scores["diversity"])

        if "balanced_aas" in protein_config:
            score = balanced_aa_constraint(
                input_sequence, protein_config["balanced_aas"]
            )
            quality_scores["balanced_aas"] = score
            overall_scores.append(quality_scores["balanced_aas"])

        # Calculate overall quality score as average of individual constraint scores
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        threshold = protein_config.get("quality_threshold", 0.0)
        is_high_quality = avg_score <= threshold

        # Store metadata for protein input
        input_sequence._metadata["protein_quality_scores"] = quality_scores
        input_sequence._metadata["avg_constraint_score"] = avg_score
        input_sequence._metadata["is_high_quality"] = is_high_quality
        input_sequence._metadata["protein_quality_threshold"] = threshold

        # If protein meets quality threshold, return 0, otherwise return the average score
        if is_high_quality:
            return 0.0
        else:
            return min(1.0, max(0.0, avg_score))

    else:
        raise ValueError("Input sequence must be either DNA or PROTEIN type")


def protein_domain_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate whether a sequence contains protein domains matching specified keywords.

    For DNA sequences, runs Prodigal first to predict proteins, then checks all predicted
    proteins. For protein sequences, checks the sequence directly.

    Args:
        input_sequence: The DNA or protein sequence to evaluate.
        config: Configuration dictionary containing:
            - hmm_db (str): Path to HMM database for hmmscan.
            - keywords (List[str]): Keywords to search for in domain descriptions.
            - evalue_threshold (float, optional): Maximum E-value for hits (default: 0.005).
            - query_coverage (float, optional): Minimum query coverage percentage (0-100).
            - match_all_keywords (bool, optional): Require all keywords vs any (default: False).
            - hmmer_kwargs (dict, optional): Additional HMMER parameters.

    Returns:
        Constraint score where 0.0 indicates domain criteria are satisfied
        and 1.0 indicates no matching domains found.

    Raises:
        ValueError: If hmm_db doesn't exist or keywords list is empty.
        RuntimeError: If HMMER or Prodigal execution fails.

    Examples:
        Evaluating domain presence in protein:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> config = {
        ...     "hmm_db": "pfam.hmm",
        ...     "keywords": ["kinase", "ATP-binding"],
        ...     "evalue_threshold": 0.001
        ... }
        >>> score = protein_domain_keyword_constraint(seq, config)

        Evaluating domain presence in DNA (via Prodigal):

        >>> seq = Sequence("ATGGTACTGAGCCCAGCG...", SequenceType.DNA)
        >>> config = {
        ...     "hmm_db": "pfam.hmm",
        ...     "keywords": ["helicase"],
        ...     "match_all_keywords": False
        ... }
        >>> score = protein_domain_keyword_constraint(seq, config)
    """
    _validate_required_config(config, ["hmm_db", "keywords"])

    hmm_db = Path(config["hmm_db"])
    if not hmm_db.exists():
        raise ValueError(f"HMM database not found: {hmm_db}")

    keywords = config["keywords"]
    if not keywords or not isinstance(keywords, list):
        raise ValueError("Keywords must be a non-empty list")

    keywords_lower = [kw.lower() for kw in keywords]
    evalue_threshold = config.get("evalue_threshold", 0.005)
    query_coverage = config.get("query_coverage")
    match_all_keywords = config.get("match_all_keywords", False)
    hmmer_kwargs = config.get("hmmer_kwargs", {})

    # Handle DNA vs protein sequences
    if input_sequence.sequence_type == SequenceType.DNA:
        # Run Prodigal to get predicted proteins
        try:
            proteins_df = run_prodigal(input_sequence)
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
                evalue_threshold,
                query_coverage,
                hmmer_kwargs,
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
        if match_all_keywords:
            return (
                MIN_ENERGY if len(all_keywords_found) == len(keywords) else MAX_ENERGY
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
                evalue_threshold,
                query_coverage,
                hmmer_kwargs,
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
        if match_all_keywords:
            return MIN_ENERGY if len(keywords_found) == len(keywords) else MAX_ENERGY
        else:
            return MIN_ENERGY if keywords_found else MAX_ENERGY

    else:
        raise ValueError(f"Unsupported sequence type: {input_sequence.sequence_type}")
