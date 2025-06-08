import itertools
from io import StringIO
import numpy as np
from typing import Any, Dict, List
import warnings
from .base import *


def sequence_length_constraint(
    input: ProgramSequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate how well a sequence matches a target length.

    This constraint penalizes sequences that deviate from the specified target length,
    returning a normalized deviation score. Perfect length matches return 0.0, while
    larger deviations return higher scores up to 1.0.

    Args:
        input: The sequence to evaluate for length constraint.
        config: Configuration dictionary containing:
            - target_length (int): The desired sequence length.

    Returns:
        A constraint score between 0.0 and 1.0, where 0.0 indicates perfect
        length matching and higher values indicate greater deviation.

    Raises:
        ValueError: If target_length is not specified in the config.

    Examples:
        >>> config = {"target_length": 100}
        >>> seq = ProgramSequence("ATCG" * 25, SequenceType.DNA)  # 100 nucleotides
        >>> score = sequence_length_constraint(seq, config)
        >>> print(score)  # 0.0 (perfect match)
    """
    if "target_length" not in config:
        raise ValueError("target_length must be specified in config")
    target_length = config["target_length"]

    input._metadata["length"] = len(input)

    # Calculate deviation based on total length.
    full_length = len(input)
    if full_length == target_length:
        return 0.0

    # Calculate normalized deviation from target length.
    deviation = abs(full_length - target_length) / target_length
    return min(1.0, deviation)


def gc_content_constraint(
    input: ProgramSequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate whether a sequence's GC content falls within a target range.

    This constraint ensures that the percentage of G and C nucleotides in a DNA or RNA
    sequence falls within specified bounds. GC content affects sequence properties like
    melting temperature and secondary structure stability.

    Args:
        input: The DNA or RNA sequence to evaluate.
        config: Configuration dictionary containing:
            - min_gc (float): Minimum acceptable GC content percentage.
            - max_gc (float): Maximum acceptable GC content percentage.

    Returns:
        A constraint score between 0.0 and 1.0, where 0.0 indicates GC content
        within the acceptable range and higher values indicate greater deviation.

    Raises:
        ValueError: If min_gc or max_gc are not specified in config, or if the
                   GC content range is invalid (min_gc < 0 or max_gc > 100).

    Examples:
        >>> config = {"min_gc": 40.0, "max_gc": 60.0}
        >>> seq = ProgramSequence("ATCGATCG", SequenceType.DNA)  # 50% GC
        >>> score = gc_content_constraint(seq, config)
        >>> print(score)  # 0.0 (within range)
    """
    if "min_gc" not in config:
        raise ValueError("min_gc must be specified in config")
    if "max_gc" not in config:
        raise ValueError("max_gc must be specified in config")
    min_gc = config["min_gc"]
    max_gc = config["max_gc"]

    # Validate range.
    if min_gc < 0 or max_gc > 100:
        raise ValueError("GC content range must be between 0 and 100 percent.")

    # Calculate GC content.
    gc_content = (
        100.0 * sum(nt in "GC" for nt in input.sequence.upper()) / max(len(input), 1)
    )

    # Return 0.0 if GC content is within the range.
    if min_gc <= gc_content <= max_gc:
        return 0.0
    else:
        if gc_content < min_gc:
            deviation = (min_gc - gc_content) / min_gc
        else:
            deviation = (gc_content - max_gc) / (100 - max_gc)
        return min(1.0, deviation)


def max_homopolymer_constraint(
    input: ProgramSequence, config: Dict[str, Any]
) -> float:
    """
    Penalize sequences containing homopolymers longer than a specified maximum.

    Homopolymers are runs of consecutive identical nucleotides (e.g., "AAAA" or "TTTT").
    Long homopolymers can cause problems in sequencing, synthesis, and amplification,
    so this constraint helps avoid them by penalizing sequences with excessive runs.

    Args:
        input: The sequence to evaluate for homopolymer content.
        config: Configuration dictionary containing:
            - max_length (int): Maximum allowed homopolymer length.

    Returns:
        A constraint score between 0.0 and 1.0, where 0.0 indicates all homopolymers
        are within the length limit and higher values indicate longer homopolymers.
        Uses a logarithmic scale for scoring excess length.

    Raises:
        ValueError: If max_length is not specified in the config.

    Examples:
        >>> config = {"max_length": 3}
        >>> seq = ProgramSequence("ATCCCGATCG", SequenceType.DNA)  # "CCC" = 3 bp
        >>> score = max_homopolymer_constraint(seq, config)
        >>> print(score)  # 0.0 (within limit)
        
        >>> seq2 = ProgramSequence("ATCCCCGATCG", SequenceType.DNA)  # "CCCC" = 4 bp
        >>> score2 = max_homopolymer_constraint(seq2, config)
        >>> print(score2 > 0)  # True (exceeds limit)
    """
    if "max_length" not in config:
        raise ValueError("max_length must be specified in config")
    max_length = config["max_length"]

    if len(input) <= 1:
        # Edge case.
        longest_homopolymer = len(input)
    else:
        # Find length of each homopolymer.
        homopolymer_lengths = [
            len(list(group)) for _, group in itertools.groupby(input.sequence)
        ]
        longest_homopolymer = max(homopolymer_lengths)

    input._metadata["max_homopolymer_length"] = longest_homopolymer

    # Return 0.0 if longest homopolymer is within range.
    if longest_homopolymer <= max_length:
        return 0.0
    else:
        # Use a logarithmic scale for scoring.
        excess_length = longest_homopolymer - max_length
        log_ratio = np.log(1 + excess_length / max_length) / np.log(2)
        return min(1.0, log_ratio)


def dinucleotide_frequency_constraint(
    input: ProgramSequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate whether dinucleotide frequencies fall within acceptable ranges.

    This constraint analyzes the frequency of all possible two-nucleotide combinations
    (dinucleotides) in a sequence. Balanced dinucleotide frequencies can be important
    for avoiding bias in sequencing, PCR amplification, and other molecular processes.

    Args:
        input: The DNA or RNA sequence to evaluate.
        config: Configuration dictionary containing:
            - min_freq (float): Minimum acceptable frequency for each dinucleotide.
            - max_freq (float): Maximum acceptable frequency for each dinucleotide.

    Returns:
        A constraint score between 0.0 and 1.0, where 0.0 indicates all dinucleotide
        frequencies are within acceptable ranges and higher values indicate greater
        deviation from the target frequency distribution.

    Raises:
        ValueError: If min_freq or max_freq are not specified in the config.
        AssertionError: If the input sequence is not DNA or RNA type.

    Examples:
        >>> config = {"min_freq": 0.03, "max_freq": 0.08}
        >>> seq = ProgramSequence("ATCGATCGATCG", SequenceType.DNA)
        >>> score = dinucleotide_frequency_constraint(seq, config)
        >>> # Score depends on dinucleotide balance in the sequence
    """
    if "min_freq" not in config:
        raise ValueError("min_freq must be specified in config")
    min_freq = config["min_freq"]

    if "max_freq" not in config:
        raise ValueError("max_freq must be specified in config")
    max_freq = config["max_freq"]

    assert input.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    # Edge case.
    if len(input) < 2:
        input._metadata["dinucleotide_freqs"] = {}
        return 1.0

    # Determine valid nucleotides.
    valid_nucleotides = "ATCG" if input.sequence_type == SequenceType.DNA else "AUCG"

    # Precompute dinucleotides.
    dinucleotides = [
        "".join(pair) for pair in itertools.product(valid_nucleotides, repeat=2)
    ]

    # Count dinucleotides.
    dinucleotide_counts = {}
    total_count = 0
    for i in range(len(input) - 1):
        dinuc = str(input)[i : i + 2]
        if all(nt in valid_nucleotides for nt in dinuc):
            dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
            total_count += 1

    # If no valid dinucleotides found.
    if total_count == 0:
        input._metadata["dinucleotide_freqs"] = {}
        return 1.0

    # Calculate frequencies and check if they are in range.
    max_deviation = 0.0
    dinucleotide_freqs = {}

    # Score based on deviation from target dinucleotide frequencies.
    for dinuc in dinucleotides:
        freq = dinucleotide_counts.get(dinuc, 0) / total_count
        dinucleotide_freqs[dinuc] = freq

        # Calculate deviation if outside acceptable range.
        if freq < min_freq:
            deviation = (min_freq - freq) / min_freq
            max_deviation = max(max_deviation, deviation)
        elif freq > max_freq:
            deviation = (freq - max_freq) / (1.0 - max_freq)
            max_deviation = max(max_deviation, deviation)

    input._metadata["dinucleotide_freqs"] = dinucleotide_freqs
    return min(1.0, max_deviation)


def tetranucleotide_usage_constraint(
    input: ProgramSequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate tetranucleotide usage deviation (TUD) for a specific 4-base motif.

    This constraint analyzes how often a specific tetranucleotide appears compared
    to what would be expected based on the overall nucleotide composition. The
    tetranucleotide usage deviation (TUD) compares observed vs. expected frequencies
    using a zero-order Markov model.

    Args:
        input: The DNA or RNA sequence to evaluate.
        config: Configuration dictionary containing:
            - tetranucleotide (str): The specific 4-base sequence to analyze.
            - min_tud (float): Minimum acceptable TUD value.
            - max_tud (float): Maximum acceptable TUD value.

    Returns:
        A constraint score between 0.0 and 1.0, where 0.0 indicates the TUD
        is within the acceptable range and higher values indicate greater deviation.

    Raises:
        ValueError: If required config parameters are missing or if the tetranucleotide
                   is not exactly 4 bases long.
        AssertionError: If the input sequence is not DNA or RNA type.

    Examples:
        >>> config = {
        ...     "tetranucleotide": "GATC",
        ...     "min_tud": 0.8,
        ...     "max_tud": 1.2
        ... }
        >>> seq = ProgramSequence("ATCGATCGATCGATC", SequenceType.DNA)
        >>> score = tetranucleotide_usage_constraint(seq, config)
        >>> # Score depends on GATC frequency vs. expected frequency
    """
    if "tetranucleotide" not in config:
        raise ValueError("tetranucleotide must be specified in config")
    tetranucleotide = config["tetranucleotide"].upper()

    if "min_tud" not in config:
        raise ValueError("min_tud must be specified in config")
    min_tud = config["min_tud"]

    if "max_tud" not in config:
        raise ValueError("max_tud must be specified in config")
    max_tud = config["max_tud"]

    # Validate tetranucleotide input.
    if len(tetranucleotide) != 4:
        raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")

    assert input.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    # Set appropriate nucleotide keys based on sequence type.
    nucleotide_keys = (
        ["A", "T", "C", "G"]
        if input.sequence_type == SequenceType.DNA
        else ["A", "U", "C", "G"]
    )

    # Edge case.
    if len(input) < 4:
        input._metadata[tetranucleotide + "_tud"] = 0.0
        return 0.0

    # Calculate nucleotide frequencies.
    nucleotide_freqs = {}
    seq_length = len(input)
    for nt in nucleotide_keys:
        nucleotide_freqs[nt] = str(input).count(nt) / seq_length

    # Count occurrences of tetranucleotide.
    tetra_count = 0
    for i in range(len(input) - 3):
        if str(input)[i : i + 4] == tetranucleotide:
            tetra_count += 1

    # Calculate expected frequency using zero-order Markov model.
    tetra_expected_freq = 1.0
    for nt in tetranucleotide:
        if nt in nucleotide_freqs:
            tetra_expected_freq *= nucleotide_freqs[nt]
        else:
            # If invalid nucleotide, set to 0
            tetra_expected_freq = 0
            break

    # Calculate expected occurrences and TUD.
    expected_occurrences = tetra_expected_freq * (seq_length - 3)
    tetra_tud = tetra_count / expected_occurrences if expected_occurrences > 0 else 0
    input._metadata[tetranucleotide + "_tud"] = tetra_tud

    # Score based on TUD range.
    if min_tud <= tetra_tud <= max_tud:
        return 0.0
    else:
        # Calculate normalized deviation.
        if tetra_tud < min_tud:
            deviation = (min_tud - tetra_tud) / min_tud
        else:
            deviation = (tetra_tud - max_tud) / max_tud
        return min(1.0, deviation)


def _run_esmfold(
    input_sequence: ProgramSequence,
    n_replications: int = 1,
    esmfold_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Execute ESMFold protein structure prediction on a sequence.

    This internal helper function runs ESMFold on a protein sequence with optional
    replication for symmetric multimer design. Results are stored in the sequence's
    metadata for use by constraint functions.

    Args:
        input_sequence: The protein sequence to fold. Must have sequence_type=PROTEIN.
        n_replications: Number of times to replicate the sequence (for multimers).
                       Sequences are joined with ":" separators.
        esmfold_kwargs: Additional keyword arguments to pass to ESMFold.

    Raises:
        ValueError: If the input sequence is not a protein type.

    Note:
        This function caches results in the sequence metadata to avoid redundant
        computations. Results include avg_plddt, ptm, and pdb_output fields.
    """
    from .tools.structure_prediction import esmfold_protein_sequence

    if input_sequence.sequence_type != SequenceType.PROTEIN:
        raise ValueError("Can only run ESMFold on a protein sequence.")

    esmfolded_sequence = ":".join([input_sequence.sequence] * n_replications)

    if "esmfolded_sequence" not in input_sequence._metadata or \
       (esmfolded_sequence != input_sequence._metadata["esmfolded_sequence"]) or \
       "avg_plddt" not in input_sequence._metadata or \
       "ptm" not in input_sequence._metadata or \
       "pdb_output" not in input_sequence._metadata:
        folding_output = esmfold_protein_sequence(
            esmfolded_sequence,
            **esmfold_kwargs,
        )
        input_sequence._metadata.update(folding_output)
        input_sequence._metadata["esmfolded_sequence"] = esmfolded_sequence


def esmfold_plddt_constraint(
    input: ProgramSequence,
    config: Dict[str, Any],
) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted LDDT (pLDDT) score.

    This constraint uses ESMFold to predict protein structure and evaluates the
    confidence using the pLDDT (predicted Local Distance Difference Test) score.
    Higher pLDDT values indicate more confident structure predictions.

    Args:
        input: A ProgramSequence with a protein sequence.
        config: Configuration dictionary containing:
            - esmfold_kwargs (Dict[str, Any], optional): Arguments to pass to ESMFold.
            - n_replications (int, optional): Number of times to replicate sequence
                                            for symmetric multimer design. Default: 1.

    Returns:
        Constraint score calculated as (1 - pLDDT), where lower values indicate
        better predicted structure quality. Range: [0.0, 1.0].

    Examples:
        >>> config = {"n_replications": 2, "esmfold_kwargs": {}}
        >>> seq = ProgramSequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> score = esmfold_plddt_constraint(seq, config)
        >>> # Lower score = higher confidence structure prediction
    """
    input_sequence = input
    n_replications = config.get('n_replications', 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))
    return 1.0 - input_sequence._metadata["avg_plddt"]


def esmfold_ptm_constraint(
    input: ProgramSequence,
    config: Dict[str, Any],
) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted TM-score (pTM).

    This constraint uses ESMFold to predict protein structure and evaluates the
    quality using the pTM (predicted Template Modeling) score. Higher pTM values
    indicate better predicted structure quality and domain organization.

    Args:
        input: A ProgramSequence with a protein sequence.
        config: Configuration dictionary containing:
            - esmfold_kwargs (Dict[str, Any], optional): Arguments to pass to ESMFold.
            - n_replications (int, optional): Number of times to replicate sequence
                                            for symmetric multimer design. Default: 1.

    Returns:
        Constraint score calculated as (1 - pTM), where lower values indicate
        better predicted structure quality. Range: [0.0, 1.0].

    Examples:
        >>> config = {"n_replications": 1, "esmfold_kwargs": {}}
        >>> seq = ProgramSequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> score = esmfold_ptm_constraint(seq, config)
        >>> # Lower score = better predicted fold quality
    """
    input_sequence = input
    n_replications = config.get('n_replications', 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))
    return 1.0 - input_sequence._metadata["ptm"]


def protein_symmetry_ring_constraint(
    input: ProgramSequence,
    config: Dict[str, Any],
) -> float:
    """
    Constrain a protein to form a symmetric ring-like multimeric structure.

    This constraint evaluates whether replicated protein chains arrange themselves
    in a symmetric ring configuration by analyzing the variance in distances between
    chain centroids. Lower variance indicates better ring symmetry.

    Args:
        input: A ProgramSequence with a protein sequence.
        config: Configuration dictionary containing:
            - esmfold_kwargs (Dict[str, Any], optional): Arguments to pass to ESMFold.
            - n_replications (int, optional): Number of chains in the multimer.
            - all_to_all_protomer_symmetry (bool, optional): Whether to compare
                all centroids (True) or only adjacent ones (False, default).

    Returns:
        The standard deviation of distances between chain centroids. Lower values
        indicate better ring symmetry.

    Raises:
        AssertionError: If the number of chains in the predicted structure does not
                       match the specified n_replications.

    Examples:
        >>> config = {
        ...     "n_replications": 4,
        ...     "all_to_all_protomer_symmetry": False
        ... }
        >>> seq = ProgramSequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> score = protein_symmetry_ring_constraint(seq, config)
        >>> # Lower score = more symmetric ring arrangement
    """
    from biotite.structure import get_chains
    from .utils import (
        adjacent_distances,
        get_backbone_atoms,
        get_centroid,
        pairwise_distances,
        pdb_file_to_atomarray,
    )

    input_sequence = input
    n_replications = config.get('n_replications', 1)
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

    return (
        float(np.std(pairwise_distances(centroids)))
        if config.get("all_to_all_protomer_symmetry", False) else
        float(np.std(adjacent_distances(centroids)))
    )


def protein_globularity_constraint(
    input: ProgramSequence,
    config: Dict[str, Any],
) -> float:
    """
    Encourage compact, globular protein structures.

    This constraint evaluates protein structure compactness by measuring the variance
    in distances from all backbone atoms to the protein's centroid. Lower variance
    indicates a more spherical, globular structure, while higher variance suggests
    an elongated or irregular shape.

    Args:
        input: A ProgramSequence with a protein sequence.
        config: Configuration dictionary containing:
            - esmfold_kwargs (Dict[str, Any], optional): Arguments to pass to ESMFold.
            - n_replications (int, optional): Number of times to replicate sequence
                                            for symmetric multimer design. Default: 1.

    Returns:
        The standard deviation of distances from backbone atoms to the protein centroid.
        Lower values indicate more globular (spherical) structures.

    Examples:
        >>> config = {"n_replications": 1}
        >>> seq = ProgramSequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> score = protein_globularity_constraint(seq, config)
        >>> # Lower score = more compact, globular structure
    """
    from .utils import (
        distances_to_centroid,
        get_backbone_atoms,
        pdb_file_to_atomarray,
    )

    input_sequence = input
    n_replications = config.get('n_replications', 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))

    backbone = get_backbone_atoms(atom_array).coord

    return float(np.std(distances_to_centroid(backbone)))
