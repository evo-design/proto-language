"""
Constraint functions for sequence optimization and validation.

This module provides constraint functions for evaluating and optimizing biological
sequences. Constraints assess sequence properties like length, composition, structure,
and functional characteristics.

Constraint Categories:
    Sequence Composition: length, GC content, homopolymers, dinucleotide frequencies
    Protein Structure: ESMFold pLDDT/pTM, symmetry, globularity  

"""

from __future__ import annotations
import itertools
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from .base import *
from .tools.orf_prediction import run_orfipy, parse_orfipy_results_to_df
from .tools.mmseqs import (
    run_mmseqs_search_proteins,
)
from .tools.structure_prediction import predict_structure_esmfold

# Constants
DEFAULT_ORFIPY_PARAMS = {
    "threads": 96,
    "start_codons": "ATG",
    "stop_codons": "TAA,TAG,TGA",
    "strand": "b",
    "min_len": 0,
    "max_len": 3000,
    "include_stop": True,
}
DEFAULT_MMSEQS_PARAMS = {"threads": 96, "sensitivity": 4.0, "only_top_hits": True}

# Valid nucleotides for different sequence types
DNA_NUCLEOTIDES = "ATCG"
RNA_NUCLEOTIDES = "AUCG"

# Constraint scoring constants
MIN_ENERGY = 0.0
MAX_ENERGY = 1.0
LOG_BASE = 2

# GC content constants (0-100%)
MIN_GC_CONTENT = 0.0
MAX_GC_CONTENT = 100.0


def _validate_required_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that all required configuration keys are present.

    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required configuration keys.

    Raises:
        ValueError: If any required keys are missing from the configuration.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")


def _validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """
    Validate that a value falls within the specified range.

    Args:
        value: The value to validate.
        min_val: Minimum acceptable value (inclusive).
        max_val: Maximum acceptable value (inclusive).
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If value is outside the specified range.
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def _calculate_normalized_deviation(actual: float, target: float) -> float:
    """
    Calculate normalized deviation from target value.

    Args:
        actual: The actual measured value.
        target: The desired target value.

    Returns:
        Normalized deviation score where 0.0 indicates perfect match
        and higher values indicate greater deviation from target.
    """
    return min(MAX_ENERGY, abs(actual - target) / max(target, 1))


def _calculate_range_deviation(actual: float, min_val: float, max_val: float) -> float:
    """
    Calculate deviation from acceptable range for general constraints.

    Args:
        actual: The actual measured value.
        min_val: Minimum acceptable value.
        max_val: Maximum acceptable value.

    Returns:
        Range deviation score where 0.0 indicates the value is within range
        and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / min_val)
    else:
        return min(MAX_ENERGY, (actual - max_val) / max_val)


def _calculate_percentage_range_deviation(
    actual: float, min_val: float, max_val: float
) -> float:
    """
    Calculate deviation from acceptable range for percentage-based constraints (0-100%).

    Args:
        actual: The actual measured percentage value.
        min_val: Minimum acceptable percentage.
        max_val: Maximum acceptable percentage.

    Returns:
        Percentage range deviation score where 0.0 indicates the value is within range
        and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, 1))
    else:
        return min(MAX_ENERGY, (actual - max_val) / max(100 - max_val, 1))


def sequence_length_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate how well a sequence matches a target length.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration dictionary containing:
            - target_length (int): Desired sequence length.

    Returns:
        Constraint score where 0.0 indicates perfect length match
        and higher values indicate greater deviation from target length.

    Examples:
        Evaluating length constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> config = {"target_length": 8}
        >>> score = sequence_length_constraint(seq, config)
        >>> print(score)  # 0.0 (perfect match)
    """
    _validate_required_config(config, ["target_length"])
    target_length = config["target_length"]

    input_sequence._metadata["length"] = len(input_sequence)
    return _calculate_normalized_deviation(len(input_sequence), target_length)


def gc_content_constraint(input_sequence: Sequence, config: Dict[str, Any]) -> float:
    """
    Evaluate whether a sequence's GC content falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration dictionary containing:
            - min_gc (float): Minimum acceptable GC content percentage (0-100).
            - max_gc (float): Maximum acceptable GC content percentage (0-100).

    Returns:
        Constraint score where 0.0 indicates GC content is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Raises:
        ValueError: If min_gc or max_gc are outside the range [0, 100].
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating GC content constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> config = {"min_gc": 40.0, "max_gc": 60.0}
        >>> score = gc_content_constraint(seq, config)
        >>> print(score)  # 0.0 (50% GC content is within acceptable range)
    """
    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"
    _validate_required_config(config, ["min_gc", "max_gc"])
    min_gc, max_gc = config["min_gc"], config["max_gc"]

    _validate_range(min_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "min_gc")
    _validate_range(max_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "max_gc")

    gc_content = (
        100.0
        * sum(nt in "GC" for nt in input_sequence.sequence.upper())
        / max(len(input_sequence), 1)
    )

    input_sequence._metadata["gc_content"] = gc_content

    return _calculate_percentage_range_deviation(gc_content, min_gc, max_gc)


def max_homopolymer_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Penalize sequences containing homopolymers longer than a specified maximum.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration dictionary containing:
            - max_length (int): Maximum allowed homopolymer length.

    Returns:
        Constraint score where 0.0 indicates no homopolymers exceed the maximum length
        and higher values indicate longer homopolymers with logarithmic scaling.

    Examples:
        Evaluating homopolymer constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> config = {"max_length": 3}
        >>> score = max_homopolymer_constraint(seq, config)
        >>> print(score)  # 0.0 (no long homopolymers)

    Note:
        The constraint uses logarithmic scaling to penalize excessive homopolymer lengths
        while avoiding extreme penalty values.
    """
    _validate_required_config(config, ["max_length"])
    max_length = config["max_length"]

    if len(input_sequence) <= 1:
        longest_homopolymer = len(input_sequence)
    else:
        homopolymer_lengths = [
            len(list(group)) for _, group in itertools.groupby(input_sequence.sequence)
        ]
        longest_homopolymer = max(homopolymer_lengths)

    input_sequence._metadata["max_homopolymer_length"] = longest_homopolymer

    if longest_homopolymer <= max_length:
        return MIN_ENERGY

    excess_length = longest_homopolymer - max_length
    log_ratio = np.log(1 + excess_length / max_length) / np.log(LOG_BASE)
    return min(MAX_ENERGY, log_ratio)


def dinucleotide_frequency_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate whether dinucleotide frequencies fall within acceptable ranges.

    Args:
        input_sequence: The DNA or RNA sequence to evaluate.
        config: Configuration dictionary containing:
            - min_freq (float): Minimum acceptable frequency for each dinucleotide (0.0-1.0).
            - max_freq (float): Maximum acceptable frequency for each dinucleotide (0.0-1.0).

    Returns:
        Constraint score where 0.0 indicates all dinucleotide frequencies are within acceptable range
        and higher values indicate the maximum deviation across all dinucleotides.

    Raises:
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating dinucleotide frequency constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> config = {"min_freq": 0.0, "max_freq": 0.3}
        >>> score = dinucleotide_frequency_constraint(seq, config)
    """
    _validate_required_config(config, ["min_freq", "max_freq"])
    min_freq, max_freq = config["min_freq"], config["max_freq"]

    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    if len(input_sequence) < 2:
        input_sequence._metadata["dinucleotide_freqs"] = {}
        return MAX_ENERGY

    valid_nucleotides = (
        DNA_NUCLEOTIDES
        if input_sequence.sequence_type == SequenceType.DNA
        else RNA_NUCLEOTIDES
    )
    dinucleotides = [
        "".join(pair) for pair in itertools.product(valid_nucleotides, repeat=2)
    ]

    # Count dinucleotides
    dinucleotide_counts = {}
    total_count = 0
    for i in range(len(input_sequence) - 1):
        dinuc = str(input_sequence)[i : i + 2]
        if all(nt in valid_nucleotides for nt in dinuc):
            dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
            total_count += 1

    if total_count == 0:
        input_sequence._metadata["dinucleotide_freqs"] = {}
        return MAX_ENERGY

    max_deviation = 0.0
    dinucleotide_freqs = {}

    for dinuc in dinucleotides:
        freq = dinucleotide_counts.get(dinuc, 0) / total_count
        dinucleotide_freqs[dinuc] = freq
        max_deviation = max(
            max_deviation, _calculate_range_deviation(freq, min_freq, max_freq)
        )

    input_sequence._metadata["dinucleotide_freqs"] = dinucleotide_freqs
    return min(MAX_ENERGY, max_deviation)


def tetranucleotide_usage_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate tetranucleotide usage deviation (TUD) for a specific 4-base motif.

    Args:
        input_sequence: The DNA or RNA sequence to evaluate.
        config: Configuration dictionary containing:
            - tetranucleotide (str): The 4-base DNA sequence motif to analyze.
            - min_tud (float): Minimum acceptable tetranucleotide usage deviation.
            - max_tud (float): Maximum acceptable tetranucleotide usage deviation.

    Returns:
        Constraint score where 0.0 indicates tetranucleotide usage deviation (TUD) is within acceptable range
        and higher values indicate greater deviation from the acceptable TUD range.

    Raises:
        ValueError: If tetranucleotide is not exactly 4 bases long.
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating tetranucleotide usage constraint:

        >>> seq = Sequence("ATCGATCGATCG", SequenceType.DNA)
        >>> config = {"tetranucleotide": "ATCG", "min_tud": 0.5, "max_tud": 2.0}
        >>> score = tetranucleotide_usage_constraint(seq, config)
    """
    _validate_required_config(config, ["tetranucleotide", "min_tud", "max_tud"])
    tetranucleotide = config["tetranucleotide"].upper()
    min_tud, max_tud = config["min_tud"], config["max_tud"]

    if len(tetranucleotide) != 4:
        raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")

    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    if len(input_sequence) < 4:
        input_sequence._metadata[tetranucleotide + "_tud"] = 0.0
        return MIN_ENERGY

    nucleotide_keys = list(
        DNA_NUCLEOTIDES
        if input_sequence.sequence_type == SequenceType.DNA
        else RNA_NUCLEOTIDES
    )

    # Calculate nucleotide frequencies
    seq_length = len(input_sequence)
    nucleotide_freqs = {
        nt: str(input_sequence).count(nt) / seq_length for nt in nucleotide_keys
    }

    # Count tetranucleotide occurrences
    tetra_count = sum(
        1
        for i in range(len(input_sequence) - 3)
        if str(input_sequence)[i : i + 4] == tetranucleotide
    )

    # Calculate expected frequency using zero-order Markov model
    tetra_expected_freq = 1.0
    for nt in tetranucleotide:
        if nt in nucleotide_freqs:
            tetra_expected_freq *= nucleotide_freqs[nt]
        else:
            tetra_expected_freq = 0
            break

    expected_occurrences = tetra_expected_freq * (seq_length - 3)
    tetra_tud = tetra_count / expected_occurrences if expected_occurrences > 0 else 0
    input_sequence._metadata[tetranucleotide + "_tud"] = tetra_tud

    return _calculate_range_deviation(tetra_tud, min_tud, max_tud)


def _run_esmfold(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Execute ESMFold protein structure prediction on a sequence.

    Args:
        input_sequence: The protein sequence to fold.
        n_replications: Number of sequence replications for multimeric prediction (default: 1).
        esmfold_kwargs: Additional keyword arguments passed to ESMFold.

    Raises:
        ValueError: If input_sequence is not SequenceType.PROTEIN.

    Note:
        Results are cached in input_sequence._metadata to avoid redundant predictions.
        Updates metadata with 'avg_plddt', 'ptm', 'pdb_output', and 'esmfolded_sequence'.
    """

    if input_sequence.sequence_type != SequenceType.PROTEIN:
        raise ValueError("Can only run ESMFold on a protein sequence.")

    esmfolded_sequence = ":".join([input_sequence.sequence] * n_replications)

    # Check if prediction already cached
    if (
        "esmfolded_sequence" not in input_sequence._metadata
        or esmfolded_sequence != input_sequence._metadata["esmfolded_sequence"]
        or not all(
            key in input_sequence._metadata
            for key in ["avg_plddt", "ptm", "pdb_output"]
        )
    ):

        folding_output = predict_structure_esmfold(
            sequences=esmfolded_sequence, **esmfold_kwargs
        )
        input_sequence._metadata.update(folding_output.metrics)
        input_sequence._metadata["pdb_output"] = folding_output.structure_pdb_output
        input_sequence._metadata["esmfolded_sequence"] = esmfolded_sequence


def esmfold_plddt_constraint(input_sequence: Sequence, config: Dict[str, Any]) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted LDDT (pLDDT) score.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - n_replications (int, optional): Number of sequence replications (default: 1).
            - esmfold_kwargs (dict, optional): Additional ESMFold parameters.

    Returns:
        Constraint score where 0.0 indicates perfect structure confidence (pLDDT = 1.0)
        and higher values indicate lower structure confidence.

    Examples:
        Evaluating protein structure confidence:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> config = {"n_replications": 1}
        >>> score = esmfold_plddt_constraint(seq, config)
    """

    n_replications = config.get("n_replications", 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))
    return 1.0 - input_sequence._metadata["avg_plddt"]


def esmfold_ptm_constraint(input_sequence: Sequence, config: Dict[str, Any]) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted TM-score (pTM).

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - n_replications (int, optional): Number of sequence replications (default: 1).
            - esmfold_kwargs (dict, optional): Additional ESMFold parameters.

    Returns:
        Constraint score where 0.0 indicates perfect structure quality (pTM = 1.0)
        and higher values indicate lower structure quality.

    Examples:
        Evaluating protein structure quality:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> config = {"n_replications": 1}
        >>> score = esmfold_ptm_constraint(seq, config)
    """

    n_replications = config.get("n_replications", 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))
    return 1.0 - input_sequence._metadata["ptm"]


def protein_symmetry_ring_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Constrain a protein to form a symmetric ring-like multimeric structure.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - n_replications (int, optional): Number of protomers in the ring (default: 1).
            - all_to_all_protomer_symmetry (bool, optional): Use all pairwise distances vs adjacent (default: False).
            - esmfold_kwargs (dict, optional): Additional ESMFold parameters.

    Returns:
        Constraint score based on standard deviation of inter-protomer distances.
        Lower values indicate more symmetric ring-like arrangements.

    Examples:
        Evaluating ring symmetry:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> config = {"n_replications": 6}  # Hexameric ring
        >>> score = protein_symmetry_ring_constraint(seq, config)
    """
    from biotite.structure import get_chains
    from .utils import (
        adjacent_distances,
        get_backbone_atoms,
        get_centroid,
        pairwise_distances,
        pdb_file_to_atomarray,
    )

    n_replications = config.get("n_replications", 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))

    centroids = []
    for chain_id in get_chains(atom_array):
        chain_backbone = get_backbone_atoms(
            atom_array[atom_array.chain_id == chain_id]
        ).coord
        centroids.append(get_centroid(chain_backbone))

    assert len(centroids) == n_replications
    centroids = np.vstack(centroids)

    distance_func = (
        pairwise_distances
        if config.get("all_to_all_protomer_symmetry", False)
        else adjacent_distances
    )
    return float(np.std(distance_func(centroids)))


def protein_globularity_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Encourage compact, globular protein structures.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - n_replications (int, optional): Number of sequence replications (default: 1).
            - esmfold_kwargs (dict, optional): Additional ESMFold parameters.

    Returns:
        Constraint score based on standard deviation of distances from backbone atoms to centroid.
        Lower values indicate more compact, globular structures.

    Examples:
        Evaluating protein globularity:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> config = {"n_replications": 1}
        >>> score = protein_globularity_constraint(seq, config)
    """
    from .utils import distances_to_centroid, get_backbone_atoms, pdb_file_to_atomarray

    n_replications = config.get("n_replications", 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))
    backbone = get_backbone_atoms(atom_array).coord
    return float(np.std(distances_to_centroid(backbone)))


def _run_orfipy_mmseqs_pipeline(
    input_sequence: Sequence, config: Dict[str, Any]
) -> None:
    """
    Run the ORFipy + MMseqs pipeline for sequence analysis.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration dictionary containing:
            - orfipy_kwargs (dict, optional): Additional ORFipy parameters.
            - mmseqs_kwargs (dict, optional): Additional MMseqs parameters.

    Note:
        Results are cached in input_sequence._metadata to avoid redundant analysis.
        Updates metadata with 'orfipy_orfs', 'mmseqs_results', 'unique_orfs_with_hits', and 'analyzed_sequence'.
    """
    # Extract ORFipy and MMseqs parameters
    orfipy_kwargs = {**DEFAULT_ORFIPY_PARAMS, **config.get("orfipy_kwargs", {})}
    mmseqs_kwargs = {**DEFAULT_MMSEQS_PARAMS, **config.get("mmseqs_kwargs", {})}

    # Create cache key based on sequence and analysis parameters
    sequence_to_analyze = input_sequence.sequence

    # Create a deterministic cache key from config parameters
    cache_key_parts = [
        sequence_to_analyze,
        str(sorted(orfipy_kwargs.items())),
        str(sorted(mmseqs_kwargs.items())),
    ]
    analyzed_sequence_key = "|".join(cache_key_parts)

    # Check if analysis already cached
    if (
        "analyzed_sequence" not in input_sequence._metadata
        or analyzed_sequence_key != input_sequence._metadata["analyzed_sequence"]
        or not all(
            key in input_sequence._metadata
            for key in ["orfipy_orfs", "mmseqs_results", "unique_orfs_with_hits"]
        )
    ):

        # Run the expensive analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write sequence to temporary FASTA file
            input_fasta = temp_path / "input.fasta"
            with open(input_fasta, "w") as f:
                f.write(f">input_sequence\n{sequence_to_analyze}\n")

            # Run ORFipy
            orfipy_output = temp_path / "orfipy_output"
            aa_fasta, nt_fasta = run_orfipy(
                input_fasta, output_dir=orfipy_output, **orfipy_kwargs
            )

            # Parse ORFipy results
            orfs_df = parse_orfipy_results_to_df(aa_fasta, nt_fasta)

            if orfs_df.empty:
                # No ORFs found
                input_sequence._metadata["orfipy_orfs"] = orfs_df
                input_sequence._metadata["mmseqs_results"] = pd.DataFrame()
                input_sequence._metadata["unique_orfs_with_hits"] = 0
            else:
                # Run MMseqs search for each ORF
                mmseqs_output = temp_path / "mmseqs_output"
                mmseqs_results = run_mmseqs_search_proteins(
                    aa_fasta,
                    mmseqs_kwargs.get(
                        "database", ""
                    ),  # Database path should be provided in config
                    mmseqs_output,
                    **{k: v for k, v in mmseqs_kwargs.items() if k != "database"},
                )

                # Count unique ORFs with hits
                unique_orfs_with_hits = (
                    len(mmseqs_results) if not mmseqs_results.empty else 0
                )

                # Store results in metadata
                input_sequence._metadata["orfipy_orfs"] = orfs_df
                input_sequence._metadata["mmseqs_results"] = mmseqs_results
                input_sequence._metadata["unique_orfs_with_hits"] = (
                    unique_orfs_with_hits
                )

            # Cache the analysis key to avoid recomputation
            input_sequence._metadata["analyzed_sequence"] = analyzed_sequence_key


def orfipy_mmseqs_gene_hit_count_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate whether the number of unique ORFs with hits falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration dictionary containing:
            - min_hits (int): Minimum acceptable number of unique ORFs with hits.
            - max_hits (int): Maximum acceptable number of unique ORFs with hits.
            - orfipy_kwargs (dict, optional): Additional ORFipy parameters.
            - mmseqs_kwargs (dict, optional): Additional MMseqs parameters.

    Returns:
        Constraint score where 0.0 indicates the hit count is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Examples:
        Evaluating ORF hit count constraint:

        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> config = {
        ...     "min_hits": 1,
        ...     "max_hits": 5,
        ...     "mmseqs_kwargs": {"database": "/path/to/protein_db"}
        ... }
        >>> score = orfipy_mmseqs_gene_hit_count_constraint(seq, config)
    """
    _validate_required_config(config, ["min_hits", "max_hits"])
    min_hits, max_hits = config["min_hits"], config["max_hits"]

    # Run the pipeline
    _run_orfipy_mmseqs_pipeline(input_sequence, config)

    # Get the count of unique ORFs with hits (directly from metadata)
    unique_orfs_with_hits = input_sequence._metadata.get("unique_orfs_with_hits", 0)

    # Calculate range deviation
    return _calculate_range_deviation(unique_orfs_with_hits, min_hits, max_hits)


def orfipy_mmseqs_gene_homology_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate the homology (percent identity) of each individual ORF hit.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration dictionary containing:
            - min_homology (float): Minimum acceptable percent identity (0-100) for each ORF.
            - max_homology (float): Maximum acceptable percent identity (0-100) for each ORF.
            - orfipy_kwargs (dict, optional): Additional ORFipy parameters.
            - mmseqs_kwargs (dict, optional): Additional MMseqs parameters.

    Returns:
        Constraint score where 0.0 indicates all ORF homologies are within acceptable range
        and higher values indicate more ORFs with homology outside the acceptable range.

    Examples:
        Evaluating ORF homology constraint:

        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> config = {
        ...     "min_homology": 50.0,
        ...     "max_homology": 90.0,
        ...     "mmseqs_kwargs": {"database": "/path/to/protein_db"}
        ... }
        >>> score = orfipy_mmseqs_gene_homology_constraint(seq, config)
    """
    _validate_required_config(config, ["min_homology", "max_homology"])
    min_homology, max_homology = config["min_homology"], config["max_homology"]

    # Run the pipeline
    _run_orfipy_mmseqs_pipeline(input_sequence, config)

    # Get the MMseqs results (directly from metadata)
    mmseqs_results = input_sequence._metadata.get("mmseqs_results", pd.DataFrame())
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
        if min_homology <= homology <= max_homology:
            acceptable_homology_count += 1
        else:
            # Calculate how far this ORF's homology deviates from acceptable range
            deviation = _calculate_percentage_range_deviation(
                homology, min_homology, max_homology
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
