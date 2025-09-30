"""
Sequence composition constraints for evaluating basic sequence properties.
"""

from __future__ import annotations
import itertools
import numpy as np
from ..base import *
from .utils import (
    DNA_NUCLEOTIDES,
    RNA_NUCLEOTIDES,
    MIN_ENERGY,
    MAX_ENERGY,
    LOG_BASE,
    MIN_GC_CONTENT,
    MAX_GC_CONTENT,
    _validate_range,
    _calculate_range_deviation,
    _calculate_percentage_range_deviation,
)


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


def sequence_length_constraint(input_sequence: Sequence, target_length: int) -> float:
    """
    Evaluate how well a sequence matches a target length.

    Args:
        input_sequence: The sequence to evaluate.
        target_length: Desired sequence length.

    Returns:
        Constraint score where 0.0 indicates perfect length match
        and higher values indicate greater deviation from target length.

    Examples:
        Evaluating length constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> score = sequence_length_constraint(seq, 8)
        >>> print(score)  # 0.0 (perfect match)
    """
    input_sequence._metadata["length"] = len(input_sequence)
    return _calculate_normalized_deviation(len(input_sequence), target_length)


def gc_content_constraint(
    input_sequence: Sequence, min_gc: float, max_gc: float
) -> float:
    """
    Evaluate whether a sequence's GC content falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        min_gc: Minimum acceptable GC content percentage (0-100).
        max_gc: Maximum acceptable GC content percentage (0-100).

    Returns:
        Constraint score where 0.0 indicates GC content is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Raises:
        ValueError: If min_gc or max_gc are outside the range [0, 100].
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating GC content constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> score = gc_content_constraint(seq, 40.0, 60.0)
        >>> print(score)  # 0.0 (50% GC content is within acceptable range)
    """
    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    _validate_range(min_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "min_gc")
    _validate_range(max_gc, MIN_GC_CONTENT, MAX_GC_CONTENT, "max_gc")

    gc_content = (
        100.0
        * sum(nt in "GC" for nt in input_sequence.sequence.upper())
        / max(len(input_sequence), 1)
    )

    input_sequence._metadata["gc_content"] = gc_content

    return _calculate_percentage_range_deviation(gc_content, min_gc, max_gc)


def max_homopolymer_constraint(input_sequence: Sequence, max_length: int) -> float:
    """
    Penalize sequences containing homopolymers longer than a specified maximum.

    Args:
        input_sequence: The sequence to evaluate.
        max_length: Maximum allowed homopolymer length.

    Returns:
        Constraint score where 0.0 indicates no homopolymers exceed the maximum length
        and higher values indicate longer homopolymers with logarithmic scaling.

    Examples:
        Evaluating homopolymer constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> score = max_homopolymer_constraint(seq, 3)
        >>> print(score)  # 0.0 (no long homopolymers)

    Note:
        The constraint uses logarithmic scaling to penalize excessive homopolymer lengths
        while avoiding extreme penalty values.
    """

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
    input_sequence: Sequence, min_freq: float, max_freq: float
) -> float:
    """
    Evaluate whether dinucleotide frequencies fall within acceptable ranges.

    Args:
        input_sequence: The DNA or RNA sequence to evaluate.
        min_freq: Minimum acceptable frequency for each dinucleotide (0.0-1.0).
        max_freq: Maximum acceptable frequency for each dinucleotide (0.0-1.0).

    Returns:
        Constraint score where 0.0 indicates all dinucleotide frequencies are within acceptable range
        and higher values indicate the maximum deviation across all dinucleotides.

    Raises:
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating dinucleotide frequency constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> score = dinucleotide_frequency_constraint(seq, 0.0, 0.3)
    """

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
    input_sequence: Sequence, tetranucleotide: str, min_tud: float, max_tud: float
) -> float:
    """
    Evaluate tetranucleotide usage deviation (TUD) for a specific 4-base motif.

    Args:
        input_sequence: The DNA or RNA sequence to evaluate.
        tetranucleotide: The 4-base DNA sequence motif to analyze.
        min_tud: Minimum acceptable tetranucleotide usage deviation.
        max_tud: Maximum acceptable tetranucleotide usage deviation.

    Returns:
        Constraint score where 0.0 indicates tetranucleotide usage deviation (TUD) is within acceptable range
        and higher values indicate greater deviation from the acceptable TUD range.

    Raises:
        ValueError: If tetranucleotide is not exactly 4 bases long.
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating tetranucleotide usage constraint:

        >>> seq = Sequence("ATCGATCGATCG", SequenceType.DNA)
        >>> score = tetranucleotide_usage_constraint(seq, "ATCG", 0.5, 2.0)
    """
    tetranucleotide = tetranucleotide.upper()

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
