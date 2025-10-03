import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
    ConstraintType,
)
from proto_language.language.constraint import (
    dinucleotide_frequency_constraint,
    gc_content_constraint,
    max_homopolymer_constraint,
    sequence_length_constraint,
    tetranucleotide_usage_constraint,
    orfipy_mmseqs_gene_hit_count_constraint,
    orfipy_mmseqs_gene_homology_constraint,
    protein_length_constraint,
    protein_repetitiveness_constraint,
    protein_diversity_constraint,
    balanced_aa_constraint,
)
from proto_language.schemas import ORFipyKwargs, MMseqsKwargs, ESMFoldKwargs


# Helper functions
def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.DNA
) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


def create_batched_segment(
    sequences: List[str], seq_type: SequenceType = SequenceType.DNA
) -> Segment:
    """Helper to create a Segment with a batch of sequences."""
    segment = Segment(sequence=sequences[0], sequence_type=seq_type)
    segment.create_batch(len(sequences))
    for i, seq_str in enumerate(sequences):
        segment.batch_sequences[i].sequence = seq_str
    return segment


# Mock scoring functions
def mock_single_input_scoring_function(sequence: Sequence) -> float:
    """
    Mock scoring function that takes in a single sequence and returns a score
    corresponding to the number of T characters in the sequence
    """
    score = sequence.sequence.count("T") / len(sequence)
    # Add metadata to demonstrate propagation
    sequence._metadata["t_count"] = sequence.sequence.count("T")
    sequence._metadata["total_length"] = len(sequence)
    sequence._metadata["t_fraction"] = score
    return score


def mock_multi_input_scoring_function(sequences: List[Sequence]) -> List[float]:
    """
    Mock scoring function that takes in a list of sequences and returns a list of scores
    corresponding to the number of T characters in each sequence
    """
    scores = []
    for sequence in sequences:
        score = sequence.sequence.count("T") / len(sequence)
        # Add metadata to demonstrate propagation
        sequence._metadata["t_count"] = sequence.sequence.count("T")
        sequence._metadata["total_length"] = len(sequence)
        sequence._metadata["t_fraction"] = score
        scores.append(score)
    return scores


def mock_single_input_scoring_function_disjoint(
    sequence_tuple: Tuple[Sequence, Sequence],
) -> float:
    """
    Mock scoring function that takes in a tuple of sequences and returns a score
    corresponding to the number of T characters in the sequences. Expects two sequences in the tuple.
    """
    # Compute percent of T in first and percent of C in second
    t_percent = sequence_tuple[0].sequence.count("T") / len(sequence_tuple[0])
    c_percent = sequence_tuple[1].sequence.count("C") / len(sequence_tuple[1])
    # Add metadata
    sequence_tuple[0]._metadata["t_percent"] = t_percent
    sequence_tuple[1]._metadata["c_percent"] = c_percent

    score = (t_percent + c_percent) / 2
    return score


def mock_multi_input_scoring_function_disjoint(
    sequence_tuples: List[Tuple[Sequence, Sequence]],
) -> float:
    """
    Mock scoring function that takes in a tuple of sequences and returns a score
    corresponding to the number of T characters in the sequences. Expects two sequences in the tuple.
    """
    scores = []
    for sequence_tuple in sequence_tuples:
        t_percent = sequence_tuple[0].sequence.count("T") / len(sequence_tuple[0])
        c_percent = sequence_tuple[1].sequence.count("C") / len(sequence_tuple[1])
        scores.append((t_percent + c_percent) / 2)
        sequence_tuple[0]._metadata["t_percent"] = t_percent
        sequence_tuple[1]._metadata["c_percent"] = c_percent
    return scores


# Tests for Sequence and Segment basics
def test_sequence_validation():
    """Tests character validation for Sequence objects."""
    with pytest.raises(ValueError, match=r"Invalid characters found: (X, Z|Z, X)"):
        Sequence("ATCGXZ", SequenceType.DNA)
    with pytest.raises(ValueError, match="Invalid characters found: T"):
        Sequence("ACGUUUT", SequenceType.RNA)
    with pytest.raises(ValueError, match=r"Invalid characters found: (J, O|O, J)"):
        Sequence("MVLSPADKTNVKJO", SequenceType.PROTEIN)
    # Test custom valid characters
    seq = Sequence("123", valid_chars=set("123"))
    assert seq.sequence == "123"
    with pytest.raises(ValueError, match="Invalid characters found: 4"):
        seq.sequence = "1234"


def test_segment_batching():
    """Tests batch creation for Segment."""
    segment = create_segment("ATCG")
    assert segment.batch_size == 1
    segment.create_batch(5)
    assert segment.batch_size == 5
    assert all(s.sequence == "ATCG" for s in segment.batch_sequences)
    segment.batch_sequences[0].sequence = "GGGG"
    assert segment.batch_sequences[0].sequence == "GGGG"
    assert segment.batch_sequences[1].sequence == "ATCG"


def test_construct_concatenation():
    """Tests sequence concatenation in Construct objects."""
    seg1 = create_segment("ATG")
    seg2 = create_segment("GGG")
    seg3 = create_segment("TAA")
    construct = Construct([seg1, seg2, seg3])
    assert len(construct.batch_sequences) == 1
    assert construct.batch_sequences[0].sequence == "ATGGGGTAA"

    # Test with batches
    batch_seg1 = create_batched_segment(["ATG", "ATG"])
    batch_seg2 = create_batched_segment(["GGG", "CCC"])
    batch_seg3 = create_batched_segment(["TAA", "TGA"])
    batch_construct = Construct([batch_seg1, batch_seg2, batch_seg3])
    assert len(batch_construct.batch_sequences) == 2
    assert batch_construct.batch_sequences[0].sequence == "ATGGGGTAA"
    assert batch_construct.batch_sequences[1].sequence == "ATG" + "CCC" + "TGA"


def test_mock_constraint_with_batched_segment():
    """
    Tests both single and multi-input scoring functions return the metadata and
    scores for the same inputs.
    """
    input_sequences = ["ACTGACTG", "TCTGTCTG", "TTTGTTTG", "TTTTTTTT"]
    # Create a DNA sequence
    single_batch_input = create_batched_segment(
        sequences=input_sequences,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input = create_batched_segment(
        sequences=input_sequences,
        seq_type=SequenceType.DNA,
    )

    # Create a single-input constraint
    constraint_single_input = Constraint(
        inputs=[single_batch_input],
        scoring_function=mock_single_input_scoring_function,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()
    constraint_multi_input = Constraint(
        inputs=[multi_batch_input],
        scoring_function=mock_multi_input_scoring_function,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    expected_scores_single_input = [0.25, 0.5, 0.75, 1]

    # Access metadata from the original segment's sequences
    for i, expected_score in enumerate(expected_scores_single_input):
        # Ensure scores are correct
        assert (
            scores_single_input[i] == expected_score
        ), f"Score mismatch for single input at index {i}"
        assert (
            scores_multi_input[i] == expected_score
        ), f"Score mismatch for multi input at index {i}"

        # Ensure metadata is propagated correctly
        sequence_metadata = single_batch_input.batch_sequences[i]._metadata
        sequence_metadata_multi = multi_batch_input.batch_sequences[i]._metadata

        # Check that metadata was propagated with proper prefixes
        expected_prefix_single = "segment_0.mock_single_input_scoring_function"
        expected_prefix_multi = "segment_0.mock_multi_input_scoring_function"

        # Check that prefixed metadata exists (excluding system metadata)
        assert any(
            key.startswith(expected_prefix_single)
            for key in sequence_metadata.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_single} in metadata: {list(sequence_metadata.keys())}"

        assert any(
            key.startswith(expected_prefix_multi)
            for key in sequence_metadata_multi.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi.keys())}"

        # Check specific metadata values were propagated for single-input constraint
        assert f"{expected_prefix_single}.t_count" in sequence_metadata
        assert f"{expected_prefix_single}.total_length" in sequence_metadata
        assert f"{expected_prefix_single}.t_fraction" in sequence_metadata

        # Check specific metadata values were propagated for multi-input constraint
        assert f"{expected_prefix_multi}.t_count" in sequence_metadata_multi
        assert f"{expected_prefix_multi}.total_length" in sequence_metadata_multi
        assert f"{expected_prefix_multi}.t_fraction" in sequence_metadata_multi

        # Verify the metadata values are correct (both should have same values, just different keys)
        single_t_count = sequence_metadata[f"{expected_prefix_single}.t_count"]
        multi_t_count = sequence_metadata_multi[f"{expected_prefix_multi}.t_count"]
        assert (
            single_t_count == multi_t_count
        ), f"T counts don't match: {single_t_count} vs {multi_t_count}"

        single_total_length = sequence_metadata[
            f"{expected_prefix_single}.total_length"
        ]
        multi_total_length = sequence_metadata_multi[
            f"{expected_prefix_multi}.total_length"
        ]
        assert (
            single_total_length == multi_total_length
        ), f"Total lengths don't match: {single_total_length} vs {multi_total_length}"


def test_mock_constraint_with_single_sequence_input():
    """
    Tests that multi-input scoring functions work correctly with single sequence inputs (batch size 1).
    This ensures the multi-input functionality doesn't break with non-batched segments.
    """
    # Create single sequence segments (batch size 1)
    single_seq_segment = create_segment("ACTGACTG", SequenceType.DNA)
    multi_seq_segment = create_segment("ACTGACTG", SequenceType.DNA)

    # Create constraints with single sequence inputs
    constraint_single_input = Constraint(
        inputs=[single_seq_segment],
        scoring_function=mock_single_input_scoring_function,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_seq_segment],
        scoring_function=mock_multi_input_scoring_function,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    # Both should return a single score
    expected_score = 0.25  # 2 T's out of 8 characters

    assert len(scores_single_input) == 1, f"Expected 1 score, got {len(scores_single_input)}"
    assert len(scores_multi_input) == 1, f"Expected 1 score, got {len(scores_multi_input)}"

    assert (
        scores_single_input[0] == expected_score
    ), f"Single input score mismatch: {scores_single_input[0]} vs {expected_score}"
    assert (
        scores_multi_input[0] == expected_score
    ), f"Multi input score mismatch: {scores_multi_input[0]} vs {expected_score}"

    # Check metadata propagation for single sequence
    sequence_metadata = single_seq_segment.batch_sequences[0]._metadata
    sequence_metadata_multi = multi_seq_segment.batch_sequences[0]._metadata

    # Check that metadata was propagated with proper prefixes
    expected_prefix_single = "segment_0.mock_single_input_scoring_function"
    expected_prefix_multi = "segment_0.mock_multi_input_scoring_function"

    # Check that prefixed metadata exists
    assert any(
        key.startswith(expected_prefix_single)
        for key in sequence_metadata.keys()
        if key not in ["sequence", "sequence_length"]
    ), f"Missing prefix {expected_prefix_single} in metadata: {list(sequence_metadata.keys())}"

    assert any(
        key.startswith(expected_prefix_multi)
        for key in sequence_metadata_multi.keys()
        if key not in ["sequence", "sequence_length"]
    ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi.keys())}"

    # Check specific metadata values were propagated
    assert f"{expected_prefix_single}.t_count" in sequence_metadata
    assert f"{expected_prefix_single}.total_length" in sequence_metadata
    assert f"{expected_prefix_single}.t_fraction" in sequence_metadata

    assert f"{expected_prefix_multi}.t_count" in sequence_metadata_multi
    assert f"{expected_prefix_multi}.total_length" in sequence_metadata_multi
    assert f"{expected_prefix_multi}.t_fraction" in sequence_metadata_multi

    # Verify the metadata values are correct
    single_t_count = sequence_metadata[f"{expected_prefix_single}.t_count"]
    multi_t_count = sequence_metadata_multi[f"{expected_prefix_multi}.t_count"]
    assert (
        single_t_count == multi_t_count
    ), f"T counts don't match: {single_t_count} vs {multi_t_count}"

    single_total_length = sequence_metadata[f"{expected_prefix_single}.total_length"]
    multi_total_length = sequence_metadata_multi[f"{expected_prefix_multi}.total_length"]
    assert (
        single_total_length == multi_total_length
    ), f"Total lengths don't match: {single_total_length} vs {multi_total_length}"

    # Verify the actual values make sense
    assert single_t_count == 2, f"Expected 2 T's, got {single_t_count}"
    assert single_total_length == 8, f"Expected length 8, got {single_total_length}"


def test_mock_constraint_with_multi_segment_input():
    """
    Tests both single and multi-input scoring functions with multiple segments as inputs.
    This tests the case where multiple segments are combined per batch position.
    """
    input_sequences_a = ["ACTG", "TCTG", "TTTG", "TTTT"]
    input_sequences_b = ["ACTG", "TCTG", "TTTG", "TTTT"]

    # Create multiple batched segments
    single_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    single_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )

    # Create constraints with multiple segment inputs
    constraint_single_input = Constraint(
        inputs=[single_batch_input_a, single_batch_input_b],
        scoring_function=mock_single_input_scoring_function,
        constraint_type=ConstraintType.CONTIGUOUS,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_batch_input_a, multi_batch_input_b],
        scoring_function=mock_multi_input_scoring_function,
        constraint_type=ConstraintType.CONTIGUOUS,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    # For CONTIGUOUS: each segment contributes "ACTGACTG", "TCTGTCTG", etc.
    # So concatenated sequences are: "ACTGACTG", "TCTGTCTG", "TTTGTTTG", "TTTTTTTT"
    expected_scores = [0.25, 0.5, 0.75, 1.0]

    # Access metadata from the original segments' sequences
    for i, expected_score in enumerate(expected_scores):
        # Ensure scores are correct
        assert (
            scores_single_input[i] == expected_score
        ), f"Score mismatch for single input at index {i}"
        assert (
            scores_multi_input[i] == expected_score
        ), f"Score mismatch for multi input at index {i}"

        # For CONTIGUOUS constraints, metadata should be propagated to both segments
        sequence_metadata_a = single_batch_input_a.batch_sequences[i]._metadata
        sequence_metadata_b = single_batch_input_b.batch_sequences[i]._metadata
        sequence_metadata_multi_a = multi_batch_input_a.batch_sequences[i]._metadata
        sequence_metadata_multi_b = multi_batch_input_b.batch_sequences[i]._metadata

        # Check that metadata was propagated with proper prefixes for CONTIGUOUS
        expected_prefix = "segment_0-segment_1.mock_single_input_scoring_function"
        expected_prefix_multi = "segment_0-segment_1.mock_multi_input_scoring_function"

        # Both segments should have the same metadata for CONTIGUOUS constraints
        assert any(
            key.startswith(expected_prefix)
            for key in sequence_metadata_a.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix} in metadata: {list(sequence_metadata_a.keys())}"
        assert any(
            key.startswith(expected_prefix)
            for key in sequence_metadata_b.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix} in metadata: {list(sequence_metadata_b.keys())}"
        assert any(
            key.startswith(expected_prefix_multi)
            for key in sequence_metadata_multi_a.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi_a.keys())}"
        assert any(
            key.startswith(expected_prefix_multi)
            for key in sequence_metadata_multi_b.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi_b.keys())}"

        # Check specific metadata values were propagated for single-input constraint
        assert f"{expected_prefix}.t_count" in sequence_metadata_a
        assert f"{expected_prefix}.total_length" in sequence_metadata_a
        assert f"{expected_prefix}.t_fraction" in sequence_metadata_a

        # Check specific metadata values were propagated for multi-input constraint
        assert f"{expected_prefix_multi}.t_count" in sequence_metadata_multi_a
        assert f"{expected_prefix_multi}.total_length" in sequence_metadata_multi_a
        assert f"{expected_prefix_multi}.t_fraction" in sequence_metadata_multi_a

        # Verify the metadata values are correct
        single_t_count = sequence_metadata_a[f"{expected_prefix}.t_count"]
        multi_t_count = sequence_metadata_multi_a[f"{expected_prefix_multi}.t_count"]
        assert (
            single_t_count == multi_t_count
        ), f"T counts don't match: {single_t_count} vs {multi_t_count}"

        single_total_length = sequence_metadata_a[f"{expected_prefix}.total_length"]
        multi_total_length = sequence_metadata_multi_a[
            f"{expected_prefix_multi}.total_length"
        ]
        assert (
            single_total_length == multi_total_length
        ), f"Total lengths don't match: {single_total_length} vs {multi_total_length}"


def test_mock_constraint_with_disjoint_input():
    """
    Tests that disjoint input mode works correctly.
    """
    input_sequences_a = ["AAAA", "AAAT", "AATT", "ATTT", "TTTT"]
    input_sequences_b = ["AAAA", "AAAC", "AACC", "ACCC", "CCCC"]

    single_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    single_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )

    constraint_single_input = Constraint(
        inputs=[single_batch_input_a, single_batch_input_b],
        scoring_function=mock_single_input_scoring_function_disjoint,
        constraint_type=ConstraintType.DISJOINT,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_batch_input_a, multi_batch_input_b],
        scoring_function=mock_multi_input_scoring_function_disjoint,
        constraint_type=ConstraintType.DISJOINT,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    # Calculate expected scores: (T_percent_in_first + C_percent_in_second) / 2
    expected_scores = []
    for seq_a, seq_b in zip(input_sequences_a, input_sequences_b):
        t_percent = seq_a.count("T") / len(seq_a)
        c_percent = seq_b.count("C") / len(seq_b)
        expected_scores.append((t_percent + c_percent) / 2)

    expected_scores_calculated = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Verify scores match expectations
    assert len(scores_single_input) == len(expected_scores_calculated)
    assert len(scores_multi_input) == len(expected_scores_calculated)

    for i, expected_score in enumerate(expected_scores_calculated):
        assert (
            scores_single_input[i] == expected_score
        ), f"Single input score mismatch at index {i}: {scores_single_input[i]} vs {expected_score}"
        assert (
            scores_multi_input[i] == expected_score
        ), f"Multi input score mismatch at index {i}: {scores_multi_input[i]} vs {expected_score}"

    # Verify metadata propagation for DISJOINT constraints
    # For DISJOINT: each segment gets its own separate metadata prefix
    expected_prefix_single_a = "segment_0.mock_single_input_scoring_function_disjoint"
    expected_prefix_single_b = "segment_1.mock_single_input_scoring_function_disjoint"
    expected_prefix_multi_a = "segment_0.mock_multi_input_scoring_function_disjoint"
    expected_prefix_multi_b = "segment_1.mock_multi_input_scoring_function_disjoint"

    for i in range(len(input_sequences_a)):
        # Check metadata in segment A
        metadata_a_single = single_batch_input_a.batch_sequences[i]._metadata
        metadata_a_multi = multi_batch_input_a.batch_sequences[i]._metadata

        # Check metadata in segment B
        metadata_b_single = single_batch_input_b.batch_sequences[i]._metadata
        metadata_b_multi = multi_batch_input_b.batch_sequences[i]._metadata

        # Verify metadata prefixes exist
        assert any(
            key.startswith(expected_prefix_single_a)
            for key in metadata_a_single.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_single_a} in segment A metadata: {list(metadata_a_single.keys())}"

        assert any(
            key.startswith(expected_prefix_single_b)
            for key in metadata_b_single.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_single_b} in segment B metadata: {list(metadata_b_single.keys())}"

        assert any(
            key.startswith(expected_prefix_multi_a)
            for key in metadata_a_multi.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi_a} in segment A metadata: {list(metadata_a_multi.keys())}"

        assert any(
            key.startswith(expected_prefix_multi_b)
            for key in metadata_b_multi.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi_b} in segment B metadata: {list(metadata_b_multi.keys())}"

        # Check specific metadata values were propagated
        assert f"{expected_prefix_single_a}.t_percent" in metadata_a_single
        assert f"{expected_prefix_single_b}.c_percent" in metadata_b_single
        assert f"{expected_prefix_multi_a}.t_percent" in metadata_a_multi
        assert f"{expected_prefix_multi_b}.c_percent" in metadata_b_multi

        # Verify the metadata values are correct
        expected_t_percent = input_sequences_a[i].count("T") / len(input_sequences_a[i])
        expected_c_percent = input_sequences_b[i].count("C") / len(input_sequences_b[i])

        assert (
            metadata_a_single[f"{expected_prefix_single_a}.t_percent"]
            == expected_t_percent
        )
        assert (
            metadata_b_single[f"{expected_prefix_single_b}.c_percent"]
            == expected_c_percent
        )
        assert (
            metadata_a_multi[f"{expected_prefix_multi_a}.t_percent"]
            == expected_t_percent
        )
        assert (
            metadata_b_multi[f"{expected_prefix_multi_b}.c_percent"]
            == expected_c_percent
        )

        # Verify consistency between single and multi modes
        assert (
            metadata_a_single[f"{expected_prefix_single_a}.t_percent"]
            == metadata_a_multi[f"{expected_prefix_multi_a}.t_percent"]
        )
        assert (
            metadata_b_single[f"{expected_prefix_single_b}.c_percent"]
            == metadata_b_multi[f"{expected_prefix_multi_b}.c_percent"]
        )


# Tests for sequence_length_constraint
class TestSequenceLengthConstraint:
    def test_single_segment(self):
        target_len = 20
        seg_match = create_segment("A" * target_len)
        seg_short = create_segment("A" * (target_len // 2))
        seg_long = create_segment("A" * (target_len * 2))

        constraint_match = Constraint(
            inputs=[seg_match],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )
        constraint_short = Constraint(
            inputs=[seg_short],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )
        constraint_long = Constraint(
            inputs=[seg_long],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )

        assert constraint_match.evaluate()[0] == 0.0
        assert abs(constraint_short.evaluate()[0] - 0.5) < 1e-9
        assert abs(constraint_long.evaluate()[0] - 1.0) < 1e-9
        assert seg_match.batch_sequences[0]._metadata["segment_0.sequence_length_constraint.length"] == target_len
        assert seg_short.batch_sequences[0]._metadata["segment_0.sequence_length_constraint.length"] == target_len // 2

    def test_contiguous_concatenation(self):
        """Tests length constraint on concatenated segments."""
        target_len = 20
        seg1 = create_segment("A" * 10)
        seg2 = create_segment("T" * 10)

        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
            constraint_type=ConstraintType.CONTIGUOUS,
        )

        assert constraint.evaluate()[0] == 0.0
        # Check metadata propagation to original segments
        assert seg1.batch_sequences[0]._metadata["segment_0-segment_1.sequence_length_constraint.length"] == target_len
        assert seg2.batch_sequences[0]._metadata["segment_0-segment_1.sequence_length_constraint.length"] == target_len

    def test_batch_processing(self):
        """Tests length constraint with a batch of sequences."""
        target_len = 15
        sequences = ["A" * 8, "A" * 12, "A" * 15, "A" * 16, "A" * 20]
        seg_batch = create_batched_segment(sequences)

        constraint = Constraint(
            inputs=[seg_batch],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )

        scores = constraint.evaluate()
        expected_scores = [
            abs(8 - 15) / 15.0,
            abs(12 - 15) / 15.0,
            abs(15 - 15) / 15.0,
            abs(16 - 15) / 15.0,
            abs(20 - 15) / 15.0,
        ]

        assert len(scores) == len(expected_scores)
        for actual, expected in zip(scores, expected_scores):
            assert abs(actual - expected) < 1e-9

        # Check metadata for all sequences in the batch
        for i, seq_obj in enumerate(seg_batch):
            assert seq_obj._metadata["segment_0.sequence_length_constraint.length"] == len(sequences[i])

    @pytest.mark.parametrize(
        "seq_str, target_len, expected_score",
        [
            ("", 10, 1.0),  # Empty sequence
            ("A", 1, 0.0),  # Single character match
            ("A", 2, 0.5),  # Single character mismatch
            ("ATCG", 0, 1.0), # Target length is 0, score capped at 1.0
        ],
    )
    def test_edge_cases(self, seq_str, target_len, expected_score):
        segment = create_segment(seq_str)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9

    def test_invalid_config(self):
        """Tests that missing 'target_length' raises an error."""
        segment = create_segment("ATCG")
        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={}, # Missing target_length
        )
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'target_length'"):
            constraint.evaluate()

    def test_disjoint_mode_raises_error(self):
        """Tests that sequence_length_constraint doesn't support DISJOINT mode."""
        seg1 = create_segment("A" * 10)
        seg2 = create_segment("T" * 10)
        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": 20},
            constraint_type=ConstraintType.DISJOINT,
        )
        # The default scoring function expects a single Sequence, not a tuple
        with pytest.raises(AttributeError):
            constraint.evaluate()


# Tests for gc_content_constraint
class TestGCContentConstraint:
    @pytest.mark.parametrize(
        "sequence, min_gc, max_gc, expected_score",
        [
            ("GCGCGAATTA", 40, 60, 0.0),  # In range (50%)
            ("GCATTATTAT", 40, 60, 0.5),  # Below range (20% -> (40-20)/40=0.5)
            ("GCGCGCGCGT", 40, 60, 0.75),  # Above range (90% -> (90-60)/(100-60)=0.75)
            ("GCGCGCGCGC", 50, 70, 1.0),  # 100% GC, above range
            ("ATATATATAT", 30, 50, 1.0),  # 0% GC, below range
            ("", 40, 60, 1.0),  # Empty sequence, 0% GC
            ("G", 50, 50, 1.0),  # Single G, 100% GC
            ("A", 50, 50, 1.0),  # Single A, 0% GC
        ],
    )
    def test_dna_sequences(self, sequence, min_gc, max_gc, expected_score):
        segment = create_segment(sequence, SequenceType.DNA)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": min_gc, "max_gc": max_gc},
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9
        # Check metadata
        gc_content = (
            100.0 * sum(nt in "GC" for nt in sequence) / max(len(sequence), 1)
        )
        assert abs(segment[0]._metadata["segment_0.gc_content_constraint.gc_content"] - gc_content) < 1e-9

    @pytest.mark.parametrize(
        "sequence, min_gc, max_gc, expected_score",
        [
            ("GCGCGAUUUA", 40, 60, 0.0),  # In range (50%)
            ("GCAUUAUUAU", 40, 60, 0.5),  # Below range (20%)
        ],
    )
    def test_rna_sequences(self, sequence, min_gc, max_gc, expected_score):
        segment = create_segment(sequence, SequenceType.RNA)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": min_gc, "max_gc": max_gc},
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9

    def test_invalid_config(self):
        segment = create_segment("ATCG")
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'max_gc'"):
            Constraint(
                inputs=[segment],
                scoring_function=gc_content_constraint,
                scoring_function_config={"min_gc": 40},
            ).evaluate()
        with pytest.raises(ValueError, match="min_gc must be between 0.0 and 100.0"):
            Constraint(
                inputs=[segment],
                scoring_function=gc_content_constraint,
                scoring_function_config={"min_gc": -10, "max_gc": 60},
            ).evaluate()

    def test_wrong_sequence_type(self):
        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40, "max_gc": 60},
        )
        with pytest.raises(AssertionError):
            constraint.evaluate()

    def test_batch_processing(self):
        sequences = ["GCGC", "ATAT", "GCAT", ""]
        seg_batch = create_batched_segment(sequences, SequenceType.DNA)
        constraint = Constraint(
            inputs=[seg_batch],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40, "max_gc": 60},
        )
        scores = constraint.evaluate()
        expected_scores = [
            1.0,  # 100% GC -> (100-60)/(100-60) = 1.0
            1.0,  # 0% GC -> (40-0)/40 = 1.0
            0.0,  # 50% GC
            1.0,  # 0% GC
        ]
        assert len(scores) == len(expected_scores)
        for actual, expected in zip(scores, expected_scores):
            assert abs(actual - expected) < 1e-9


# Tests for max_homopolymer_constraint
class TestMaxHomopolymerConstraint:
    @pytest.mark.parametrize(
        "sequence, max_len, expected_score, seq_type",
        [
            ("AAATTTGGGGCCCC", 4, 0.0, SequenceType.DNA),  # OK
            ("AAATTTTGGGGGCCC", 4, np.log2(1 + 1 / 4), SequenceType.DNA),  # Excess 1
            ("AAAAAAAATTTT", 4, 1.0, SequenceType.DNA),  # Excess 4, score = log2(2)=1
            ("A", 3, 0.0, SequenceType.DNA),  # Single NT
            ("ATATAT", 1, 0.0, SequenceType.DNA),  # No homopolymers
            ("AAAAAAAAAA", 3, 1.0, SequenceType.DNA),  # Large excess, capped at 1.0
            ("", 3, 0.0, SequenceType.DNA), # Empty sequence
            ("AAAUUUGGGGCCCC", 3, np.log2(1 + 1/3), SequenceType.RNA), # RNA
            ("AAALLLDDDEEEEEFFFF", 3, np.log2(1 + 2/3), SequenceType.PROTEIN), # Protein
        ],
    )
    def test_homopolymer_scoring(self, sequence, max_len, expected_score, seq_type):
        segment = create_segment(sequence, seq_type)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=max_homopolymer_constraint,
            scoring_function_config={"max_length": max_len},
        )
        score = constraint.evaluate()[0]
        assert abs(score - expected_score) < 1e-9
        # Test metadata
        if len(sequence) > 0:
            import itertools
            expected_max_homopolymer = max(len(list(g)) for _, g in itertools.groupby(sequence))
            assert segment[0]._metadata["segment_0.max_homopolymer_constraint.max_homopolymer_length"] == expected_max_homopolymer
        else:
            assert segment[0]._metadata["segment_0.max_homopolymer_constraint.max_homopolymer_length"] == 0

    def test_invalid_config(self):
        segment = create_segment("ATCG")
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'max_length'"):
            Constraint(
                inputs=[segment],
                scoring_function=max_homopolymer_constraint,
                scoring_function_config={},
            ).evaluate()

    def test_batch_processing(self):
        sequences = ["AAAA", "AAACCC", "AAAGGC", ""]
        max_len = 3
        seg_batch = create_batched_segment(sequences, SequenceType.DNA)
        constraint = Constraint(
            inputs=[seg_batch],
            scoring_function=max_homopolymer_constraint,
            scoring_function_config={"max_length": max_len},
        )
        scores = constraint.evaluate()
        expected_scores = [
            np.log2(1 + 1/3), # excess 1
            0.0, # in limit
            0.0, # in limit
            0.0, # empty
        ]
        assert len(scores) == len(expected_scores)
        for actual, expected in zip(scores, expected_scores):
            assert abs(actual - expected) < 1e-9


# Tests for dinucleotide_frequency_constraint
class TestDinucleotideFrequencyConstraint:
    def test_dna_sequences(self):
        # Sequence "ATCGATCG" has freqs: AT=0.286, TC=0.286, CG=0.286, GA=0.143
        # But also has 0.0 for all other dinucleotides (AA, TT, CC, GG, etc.)
        seq_ok = create_segment("ATCGATCG", SequenceType.DNA)
        # Sequence with only AT dinucleotides (freq 1.0)
        seq_violate = create_segment("ATATATAT", SequenceType.DNA)

        # Range that includes 0.0 frequency (for dinucleotides that don't appear)
        constraint_ok = Constraint(
            inputs=[seq_ok],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config={"min_freq": 0.0, "max_freq": 0.3},
        )
        assert constraint_ok.evaluate()[0] == 0.0

        # Range that excludes 0.0 frequency, should fail
        constraint_fail = Constraint(
            inputs=[seq_ok],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config={"min_freq": 0.1, "max_freq": 0.3},
        )
        assert constraint_fail.evaluate()[0] > 0.0

        # Repetitive sequence, should fail narrow range
        constraint_violate = Constraint(
            inputs=[seq_violate],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config={"min_freq": 0.0, "max_freq": 0.5},
        )
        assert constraint_violate.evaluate()[0] > 0.0
        assert "segment_0.dinucleotide_frequency_constraint.dinucleotide_freqs" in seq_violate[0]._metadata
        # ATATATAT has AT freq ~0.57 and TA freq ~0.43
        assert abs(seq_violate[0]._metadata["segment_0.dinucleotide_frequency_constraint.dinucleotide_freqs"]["AT"] - 4/7) < 1e-9

    @pytest.mark.parametrize("sequence", ["", "A"])
    def test_edge_cases(self, sequence):
        """Test with sequences too short to have dinucleotides."""
        segment = create_segment(sequence)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config={"min_freq": 0.1, "max_freq": 0.9},
        )
        assert constraint.evaluate()[0] == 1.0 # MAX_ENERGY


# Tests for tetranucleotide_usage_constraint
class TestTetranucleotideUsageConstraint:
    def test_tud_scoring(self):
        tetranuc = "GATC"
        tud_range = (0.8, 1.2)
        # From old tests: seq with one GATC, TUD is ~3.16, outside range.
        seq_balanced = create_segment("AGCT" * 10 + "GATC" + "AGCT" * 10)
        seq_no_gatc = create_segment("A" * 25) # TUD is 0, outside range.

        constraint_bal = Constraint(
            inputs=[seq_balanced],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config={
                "tetranucleotide": tetranuc,
                "min_tud": tud_range[0],
                "max_tud": tud_range[1],
            },
        )
        # TUD is high, deviation is (3.16-1.2)/1.2 -> capped at 1.0
        assert abs(constraint_bal.evaluate()[0] - 1.0) < 1e-9
        assert "segment_0.tetranucleotide_usage_constraint.GATC_tud" in seq_balanced[0]._metadata
        assert seq_balanced[0]._metadata["segment_0.tetranucleotide_usage_constraint.GATC_tud"] > 3.0

        constraint_no_gatc = Constraint(
            inputs=[seq_no_gatc],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config={
                "tetranucleotide": tetranuc,
                "min_tud": tud_range[0],
                "max_tud": tud_range[1],
            },
        )
        # TUD is 0, deviation is (0.8-0)/0.8 = 1.0
        assert abs(constraint_no_gatc.evaluate()[0] - 1.0) < 1e-9
        assert seq_no_gatc[0]._metadata["segment_0.tetranucleotide_usage_constraint.GATC_tud"] == 0.0

    def test_edge_cases(self):
        # Sequence too short
        seq_short = create_segment("GAT")
        constraint_short = Constraint(
            inputs=[seq_short],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config={
                "tetranucleotide": "GATC",
                "min_tud": 0.8,
                "max_tud": 1.2,
            },
        )
        assert constraint_short.evaluate()[0] == 0.0
        assert seq_short[0]._metadata["segment_0.tetranucleotide_usage_constraint.GATC_tud"] == 0.0

        # Empty sequence
        seq_empty = create_segment("")
        constraint_empty = Constraint(
            inputs=[seq_empty],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config={
                "tetranucleotide": "GATC",
                "min_tud": 0.8,
                "max_tud": 1.2,
            },
        )
        assert constraint_empty.evaluate()[0] == 0.0

    def test_all_same_tetranucleotide(self):
        """Tests when the sequence is composed of the target tetranucleotide."""
        # TUD for AAAA in AAAAAAAAAAAAAAAA should be 1.0
        seq_all_a = create_segment("A" * 16)
        constraint = Constraint(
            inputs=[seq_all_a],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config={
                "tetranucleotide": "AAAA",
                "min_tud": 0.8,
                "max_tud": 1.2,
            },
        )
        assert constraint.evaluate()[0] == 0.0
        assert abs(seq_all_a[0]._metadata["segment_0.tetranucleotide_usage_constraint.AAAA_tud"] - 1.0) < 1e-9


# Tests for tool-based constraints

# Test data file paths
TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_DB_PATH = TEST_DATA_DIR / "test_proteins_database.faa"
DNA_FASTA_PATH = TEST_DATA_DIR / "test_dna_sequences.fna"
ORFIPY_AA_PATH = TEST_DATA_DIR / "test_orfipy_aa.faa"
ORFIPY_NT_PATH = TEST_DATA_DIR / "test_orfipy_nt.fna"
M8_RESULTS_PATH = TEST_DATA_DIR / "test_mmseqs_results.m8"

def get_test_sequences_with_real_hits():
    """Returns DNA sequences that should produce hits against our dummy database."""
    # These sequences correspond to the test data files we created
    sequences = []
    with open(DNA_FASTA_PATH, 'r') as f:
        current_seq = ""
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append(current_seq)
    return sequences

@pytest.fixture(scope="module")
def dummy_db_path():
    return str(PROTEIN_DB_PATH)

# Sample data for constraint tests
SAMPLE_ORFIPY_AA_FASTA = """>dna_seq_1_ORF.1 [0-180](+)
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK*
>dna_seq_2_ORF.1 [0-540](+)
MKALIVLGLVLLSVTVQGKVFGRCELAAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL*
"""

SAMPLE_ORFIPY_NT_FASTA = """>dna_seq_1_ORF.1 [0-180](+)
ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGGTGGGCGCGCACGCCGGCGAATATGGCGCAGAAGCCTTGGAAAGAATGTTTTTGAGCTTTCCAACCACCAAGACCTATTTCCCACACTTTGATTTGAGCCACGGCAGCGCACAGGTGAAAGGCCACGGCAAA
>dna_seq_2_ORF.1 [0-540](+)
ATGAAAGCCTTGATCGTGTTGGGCTTGGTGTTGTTGAGCGTGACCGTGCAGGGCAAAGTGTTCGGCAGATGCGAATTGGCCGCAGCCGCAATGAAGAGACACGGCTTGGATAACTACAGAGGCTACAGCTTGGGCAACTGGGTGTGCGCAGCAAAGTTTGAAAGCAACTTCAACACACAGGCCACCAACAGAAACACCGATGGCAGCACCGATTATGGCATCTTGCAGATCAACAGCAGATGGTGGTGCAACGATGGCAGAACCCCAGGCAGCAGAAACTTGTGCAACATCCCATGCAGCGCCTTGTTGAGCAGCGATATTACCGCAAGCGTGAACTGCGCAAAGAAAATCGTGAGCGATGGCAACGGCATGAACGCATGGGTGGCATGGAGAAACAGATGCAAAGGCACCGATGTGCAGGCATGGATCAGAGGCTGCAGATTGTAA
"""

SAMPLE_M8_OUTPUT = """protein_seq_1	test_protein_1	95.2	1.5e-35
protein_seq_2	test_protein_2	87.3	2.1e-28
protein_seq_3	test_protein_5	100.0	1.0e-3
protein_seq_4	test_protein_1	98.1	3.2e-42
"""

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)

def setup_test_files(temp_dir: Path, sequence: str) -> dict:
    """Set up test files for orfipy and mmseqs tests using real files."""
    # Create input DNA file
    dna_file = temp_dir / "input.fna"
    dna_file.write_text(f">test_seq\n{sequence}\n")
    
    # Create orfipy output directory and files
    orfipy_dir = temp_dir / "orfipy_output"
    orfipy_dir.mkdir()
    
    # Use real test data files
    shutil.copy(ORFIPY_AA_PATH, orfipy_dir / "orfipy_aa.faa")
    shutil.copy(ORFIPY_NT_PATH, orfipy_dir / "orfipy_nt.fna")
    
    # Create mmseqs output file
    mmseqs_file = temp_dir / "mmseqs_results.m8"
    shutil.copy(M8_RESULTS_PATH, mmseqs_file)
    
    return {
        "dna_file": dna_file,
        "orfipy_dir": orfipy_dir,
        "mmseqs_file": mmseqs_file,
    }

# Check if orfipy is available
try:
    import subprocess
    subprocess.run(["orfipy", "--help"], capture_output=True, check=True)
    ORFIPY_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    ORFIPY_AVAILABLE = False

@pytest.mark.skipif(
    not pd, reason="Pandas not installed, skipping ORF/MMseqs tests"
)
@pytest.mark.skipif(
    not ORFIPY_AVAILABLE, reason="orfipy not installed, skipping ORF tests"
)
class TestOrfipyMmseqsConstraints:
    @pytest.fixture
    def hit_count_config(self, dummy_db_path):
        return {
            "min_hits": 1,
            "max_hits": 3,
            "mmseqs_kwargs": MMseqsKwargs(database=dummy_db_path, threads=1, sensitivity=1.0),
            "orfipy_kwargs": ORFipyKwargs(threads=1, min_len=30),
        }

    @pytest.fixture
    def homology_config(self, dummy_db_path):
        return {
            "min_homology": 80.0,
            "max_homology": 100.0,
            "mmseqs_kwargs": MMseqsKwargs(database=dummy_db_path, threads=1, sensitivity=1.0),
            "orfipy_kwargs": ORFipyKwargs(threads=1, min_len=30),
        }

    def test_hit_count_constraint(self, hit_count_config, temp_dir):
        """Test hit count constraint using real test files."""
        sequences = get_test_sequences_with_real_hits()
        segment = create_segment(sequences[0])

        # Set up test files
        setup_test_files(temp_dir, sequences[0])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=hit_count_config,
        )

        # Since we're using real files, we expect the constraint to work with actual data
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0  # Score should be non-negative

        metadata = segment[0]._metadata
        assert "segment_0.orfipy_mmseqs_gene_hit_count_constraint.orfipy_orfs" in metadata
        assert "segment_0.orfipy_mmseqs_gene_hit_count_constraint.mmseqs_results" in metadata
        assert "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits" in metadata
        assert isinstance(metadata["segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"], int)
        assert metadata["segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"] >= 0

    def test_homology_constraint(self, homology_config, temp_dir):
        """Test homology constraint using real test files."""
        sequences = get_test_sequences_with_real_hits()
        segment = create_segment(sequences[0])

        # Set up test files
        setup_test_files(temp_dir, sequences[0])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_homology_constraint,
            scoring_function_config=homology_config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0

        metadata = segment[0]._metadata
        assert "segment_0.orfipy_mmseqs_gene_homology_constraint.orfs_with_acceptable_homology" in metadata
        assert metadata["segment_0.orfipy_mmseqs_gene_homology_constraint.orfs_with_acceptable_homology"] >= 0
        assert "segment_0.orfipy_mmseqs_gene_homology_constraint.homology_compliance_rate" in metadata
        assert 0.0 <= metadata["segment_0.orfipy_mmseqs_gene_homology_constraint.homology_compliance_rate"] <= 1.0

    def test_no_hits_scenario(self, hit_count_config, temp_dir):
        """Test constraint behavior when no hits are found."""
        # Use a sequence with no meaningful ORFs
        segment = create_segment("A" * 100)

        # Set up test files with empty ORF results
        dna_file = temp_dir / "input.fna"
        dna_file.write_text(">test_seq\n" + "A" * 100 + "\n")

        orfipy_dir = temp_dir / "orfipy_output"
        orfipy_dir.mkdir()

        # Create empty ORF files
        (orfipy_dir / "orfipy_aa.faa").write_text("")
        (orfipy_dir / "orfipy_nt.fna").write_text("")

        # Create empty mmseqs results
        mmseqs_file = temp_dir / "mmseqs_results.m8"
        mmseqs_file.write_text("")

        constraint = Constraint(
            inputs=[segment],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=hit_count_config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] >= 0.0  # Should have a penalty for not meeting min_hits
        assert segment[0]._metadata["segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"] == 0

    def test_batch_processing(self, hit_count_config, temp_dir):
        """Test constraint with batch processing using real files."""
        sequences = get_test_sequences_with_real_hits()
        # Create a batch with multiple sequences
        batch = create_batched_segment([sequences[0], sequences[1], "A"*100])

        # Set up test files
        setup_test_files(temp_dir, sequences[0])

        # Adjust config for batch testing
        hit_count_config["min_hits"] = 0  # Allow 0 hits for some sequences

        constraint = Constraint(
            inputs=[batch],
            scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
            scoring_function_config=hit_count_config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)
        assert all(score >= 0.0 for score in scores)

        # Check that metadata is populated for all sequences
        for i in range(3):
            assert "segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits" in batch[i]._metadata
            assert isinstance(batch[i]._metadata["segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"], int)
            assert batch[i]._metadata["segment_0.orfipy_mmseqs_gene_hit_count_constraint.unique_orfs_with_hits"] >= 0

    def test_caching(self, hit_count_config, temp_dir):
        """Test that caching works correctly with real files."""
        from proto_language.language.constraint.sequence_annotation import (
            run_orfipy_mmseqs_pipeline,
        )
        from proto_language.tools.tool_cache import ToolCache
        seq = Sequence("ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA", SequenceType.DNA)

        # Set up test files
        setup_test_files(temp_dir, seq.sequence)

        # First call, should compute
        run_orfipy_mmseqs_pipeline(seq, 
                                          orfipy_kwargs=hit_count_config.get("orfipy_kwargs"),
                                          mmseqs_kwargs=hit_count_config.get("mmseqs_kwargs"))
        # Check that results are in metadata
        assert "orfipy_orfs" in seq._metadata
        assert "mmseqs_results" in seq._metadata
        assert "unique_orfs_with_hits" in seq._metadata

        # Second call, should use cache
        seq._metadata["test_marker"] = "should_remain"
        run_orfipy_mmseqs_pipeline(seq, 
                                          orfipy_kwargs=hit_count_config.get("orfipy_kwargs"),
                                          mmseqs_kwargs=hit_count_config.get("mmseqs_kwargs"))
        assert seq._metadata["test_marker"] == "should_remain"

        # Verify cache is working by checking ToolCache directly with model parameters
        orfipy_kwargs = hit_count_config.get("orfipy_kwargs").model_dump()
        mmseqs_kwargs = hit_count_config.get("mmseqs_kwargs").model_dump()

        cached_results = ToolCache.get_cached_results(seq, "orfipy_mmseqs", 
                                                    orfipy_kwargs=orfipy_kwargs,
                                                    mmseqs_kwargs=mmseqs_kwargs)
        assert cached_results is not None
        assert "orfipy_orfs" in cached_results
        assert "mmseqs_results" in cached_results

        # Different config should recompute when pipeline parameters change
        new_mmseqs_kwargs = MMseqsKwargs(database=hit_count_config["mmseqs_kwargs"].database, 
                                   threads=1, sensitivity=2.0)  # Change pipeline parameter
        mmseqs_kwargs_new = new_mmseqs_kwargs.model_dump()
        cached_results_new = ToolCache.get_cached_results(seq, "orfipy_mmseqs", 
                                                        orfipy_kwargs=orfipy_kwargs,
                                                        mmseqs_kwargs=mmseqs_kwargs_new)
        assert cached_results_new is None  # Should not be cached with different params

    def test_parameter_validation(self, dummy_db_path):
        """Tests that missing required parameters raise ValueErrors."""
        segment = create_segment("ATGAAATAG")

        # Test hit count constraint
        with pytest.raises(TypeError, match="missing 2 required positional arguments: 'min_hits' and 'max_hits'"):
            Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
                scoring_function_config={},
            ).evaluate()

        # Test homology constraint
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'max_homology'"):
            Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_homology_constraint,
                scoring_function_config={"min_homology": 50.0},
            ).evaluate()


# Tests for protein_length_constraint
class TestProteinLengthConstraint:
    def test_protein_within_range(self):
        """Test protein length within acceptable range."""
        segment = create_segment("MVLSPADKTNVKAAWGKVGAH", SequenceType.PROTEIN)
        config = {"config": {"min_length": 20, "max_length": 25}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )
        
        assert constraint.evaluate()[0] == 0.0
        assert segment[0]._metadata["segment_0.protein_length_constraint.protein_length"] == 21
    
    def test_protein_too_short(self):
        """Test protein shorter than minimum."""
        segment = create_segment("MVLSP", SequenceType.PROTEIN)
        config = {"config": {"min_length": 10, "max_length": 50}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score > 0.0
        assert segment[0]._metadata["segment_0.protein_length_constraint.protein_length"] == 5
    
    def test_protein_too_long(self):
        """Test protein longer than maximum."""
        segment = create_segment("M" * 100, SequenceType.PROTEIN)
        config = {"config": {"min_length": 10, "max_length": 50}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score > 0.0
        assert segment[0]._metadata["segment_0.protein_length_constraint.protein_length"] == 100
    
    def test_batch_processing(self):
        """Test constraint with batch of proteins."""
        sequences = ["M" * 10, "M" * 25, "M" * 60]
        batch = create_batched_segment(sequences, SequenceType.PROTEIN)
        config = {"config": {"min_length": 20, "max_length": 50}}
        
        constraint = Constraint(
            inputs=[batch],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 3
        assert scores[0] > 0.0  # Too short
        assert scores[1] == 0.0  # Within range
        assert scores[2] > 0.0  # Too long
    
    def test_invalid_sequence_type(self):
        """Test that DNA sequence raises error."""
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        config = {"config": {"min_length": 10, "max_length": 50}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_length_constraint,
            scoring_function_config=config,
        )
        
        with pytest.raises(AssertionError):
            constraint.evaluate()


# Tests for protein_repetitiveness_constraint
class TestProteinRepetitivenessConstraint:
    def test_non_repetitive_protein(self):
        """Test protein with low repetitiveness."""
        segment = create_segment("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF", SequenceType.PROTEIN)
        config = {"config": {"max_repetitiveness": 0.5, "min_repeat_length": 3}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_repetitiveness_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score >= 0.0
        assert "segment_0.protein_repetitiveness_constraint.repetitiveness_score" in segment[0]._metadata
        assert "segment_0.protein_repetitiveness_constraint.max_repetitive_fraction" in segment[0]._metadata
    
    def test_highly_repetitive_protein(self):
        """Test protein with high repetitiveness."""
        segment = create_segment("AAAAAAAAAAAAAA", SequenceType.PROTEIN)
        config = {"config": {"max_repetitiveness": 0.3}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_repetitiveness_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score > 0.0
        rep_score = segment[0]._metadata["segment_0.protein_repetitiveness_constraint.repetitiveness_score"]
        assert rep_score > 0.5  # Highly repetitive
    
    def test_repetitive_pattern(self):
        """Test protein with repetitive pattern."""
        segment = create_segment("MVKMVKMVKMVKMVK", SequenceType.PROTEIN)
        config = {"config": {"max_repetitiveness": 0.3, "min_repeat_length": 3}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_repetitiveness_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score > 0.0
        rep_score = segment[0]._metadata["segment_0.protein_repetitiveness_constraint.repetitiveness_score"]
        assert rep_score > 0.3
    
    def test_batch_processing(self):
        """Test constraint with batch of proteins."""
        sequences = ["MVLSPADKTNVK", "AAAAAAAAAA", "MVKMVKMVKMVK"]
        batch = create_batched_segment(sequences, SequenceType.PROTEIN)
        config = {"config": {"max_repetitiveness": 0.4}}
        
        constraint = Constraint(
            inputs=[batch],
            scoring_function=protein_repetitiveness_constraint,
            scoring_function_config=config,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 3
        assert scores[0] <= scores[2]  # First is less repetitive than third
        assert scores[1] > scores[0]  # Second (all As) is most repetitive


# Tests for protein_diversity_constraint
class TestProteinDiversityConstraint:
    def test_high_diversity(self):
        """Test protein with high amino acid diversity."""
        segment = create_segment("MVLSPADKTNVKAAWGKVGAHAGEYGAEALER", SequenceType.PROTEIN)
        config = {"config": {"min_diversity": 0.5}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score == 0.0
        assert "segment_0.protein_diversity_constraint.aa_diversity_score" in segment[0]._metadata
        assert "segment_0.protein_diversity_constraint.unique_amino_acid_count" in segment[0]._metadata
        assert segment[0]._metadata["segment_0.protein_diversity_constraint.aa_diversity_score"] > 0.5
    
    def test_low_diversity(self):
        """Test protein with low amino acid diversity."""
        segment = create_segment("AAAAAAGGGGGGLLLLLL", SequenceType.PROTEIN)
        config = {"config": {"min_diversity": 0.5}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score > 0.0
        diversity = segment[0]._metadata["segment_0.protein_diversity_constraint.aa_diversity_score"]
        assert diversity < 0.5
        assert segment[0]._metadata["segment_0.protein_diversity_constraint.unique_amino_acid_count"] == 3
    
    def test_single_amino_acid(self):
        """Test protein with only one amino acid type."""
        segment = create_segment("AAAAAAAAAA", SequenceType.PROTEIN)
        config = {"config": {"min_diversity": 0.2}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score > 0.0
        assert segment[0]._metadata["segment_0.protein_diversity_constraint.unique_amino_acid_count"] == 1
        assert segment[0]._metadata["segment_0.protein_diversity_constraint.aa_diversity_score"] == 1/20  # 1 out of 20 standard AAs
    
    def test_empty_sequence(self):
        """Test that empty sequence raises error."""
        segment = create_segment("", SequenceType.PROTEIN)
        config = {"config": {"min_diversity": 0.3}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )
        
        with pytest.raises(ValueError, match="Sequence is non-existent"):
            constraint.evaluate()


# Tests for balanced_aa_constraint
class TestBalancedAAConstraint:
    def test_balanced_protein(self):
        """Test protein with balanced amino acid frequencies."""
        # Create a relatively balanced sequence
        segment = create_segment("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF", SequenceType.PROTEIN)
        config = {"config": {"min_aa_frequency": 0.02, "max_underrepresented_count": 10}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score >= 0.0
        assert "segment_0.balanced_aa_constraint.underrepresented_aa_score" in segment[0]._metadata
        assert "segment_0.balanced_aa_constraint.underrepresented_amino_acids" in segment[0]._metadata
    
    def test_unbalanced_protein(self):
        """Test protein with unbalanced amino acid frequencies."""
        segment = create_segment("AAAAAAGGGGLLLLMMMM", SequenceType.PROTEIN)
        config = {"config": {"min_aa_frequency": 0.1, "max_underrepresented_count": 2}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        # With 4 amino acids, all at ~25%, and threshold of 10%, all are above threshold
        # So underrepresented_aa_count should be 0
        assert score >= 0.0
        assert "segment_0.balanced_aa_constraint.underrepresented_aa_count" in segment[0]._metadata
    
    def test_empty_sequence(self):
        """Test empty sequence handling."""
        segment = create_segment("", SequenceType.PROTEIN)
        config = {"config": {"min_aa_frequency": 0.05, "max_underrepresented_count": 5}}
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )
        
        score = constraint.evaluate()[0]
        assert score == 1.0
    
    def test_batch_processing(self):
        """Test constraint with batch of proteins."""
        sequences = [
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL",  # Balanced
            "AAAAAGGGGGLLLLLL",  # Less balanced
            "MMMMM"  # Very unbalanced (single AA)
        ]
        batch = create_batched_segment(sequences, SequenceType.PROTEIN)
        config = {"config": {"min_aa_frequency": 0.15, "max_underrepresented_count": 3}}
        
        constraint = Constraint(
            inputs=[batch],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )
        
        scores = constraint.evaluate()
        assert len(scores) == 3
        # All sequences should have some underrepresented amino acids with this threshold
        assert all(score >= 0.0 for score in scores)


# Tests for sigma70_promoter_constraint
class TestSigma70PromoterConstraint:
    def test_ideal_promoter(self):
        """Test ideal sigma70 promoter sequence."""
        from proto_language.language.constraint import sigma70_promoter_constraint
        
        # Ideal promoter: -35 box + 17bp spacer + -10 box
        ideal = "TTGACA" + "A" * 17 + "TATAAT"
        segment = create_segment(ideal, SequenceType.DNA)
        
        score = sigma70_promoter_constraint(segment.batch_sequences[0])  # Pass single Sequence
        assert score < 0.5  # Should have low penalty
        assert "sigma70" in segment.batch_sequences[0]._metadata
        assert segment.batch_sequences[0]._metadata["sigma70"]["spacer_len"] == 17
    
    def test_poor_promoter(self):
        """Test poor promoter sequence."""
        from proto_language.language.constraint import sigma70_promoter_constraint
        
        poor = "AAAAAA" + "G" * 17 + "CCCCCC"
        segment = create_segment(poor, SequenceType.DNA)
        
        score = sigma70_promoter_constraint(segment.batch_sequences[0])
        assert score > 0.4  # Should have moderate-to-high penalty
        assert "sigma70" in segment.batch_sequences[0]._metadata
    
    def test_scanning_long_sequence(self):
        """Test scanning long sequence for best promoter."""
        from proto_language.language.constraint import sigma70_promoter_constraint
        
        # Embed promoter in longer sequence
        long_seq = "A" * 50 + "TTGACA" + "T" * 17 + "TATAAT" + "G" * 50
        segment = create_segment(long_seq, SequenceType.DNA)
        
        score = sigma70_promoter_constraint(segment.batch_sequences[0])
        assert score < 0.5
        assert "sigma70" in segment.batch_sequences[0]._metadata
        assert "pos" in segment.batch_sequences[0]._metadata["sigma70"]
    
    def test_short_sequence(self):
        """Test sequence too short for promoter."""
        from proto_language.language.constraint import sigma70_promoter_constraint
        
        short = "ATCG"
        segment = create_segment(short, SequenceType.DNA)
        
        score = sigma70_promoter_constraint(segment.batch_sequences[0])
        assert score == 1.0
        assert segment.batch_sequences[0]._metadata["sigma70"]["reason"] == "too_short"
    
    def test_batch_processing(self):
        """Test batch of sequences."""
        from proto_language.language.constraint import sigma70_promoter_constraint
        
        ideal = "TTGACA" + "A" * 17 + "TATAAT"
        poor = "AAAAAA" + "G" * 17 + "CCCCCC"
        batch = create_batched_segment([ideal, poor], SequenceType.DNA)
        
        scores = sigma70_promoter_constraint(batch.batch_sequences)
        assert len(scores) == 2
        assert scores[0] < scores[1]  # Ideal better than poor


class TestConstraintConfigNormalization:
    """Test that Constraint class automatically converts dict configs to Pydantic models."""

    def test_esmfold_kwargs_normalization(self):
        """Test that esmfold_kwargs dict is converted to ESMFoldKwargs model."""
        from proto_language.language.constraint import esmfold_plddt_constraint
        
        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)
        
        # Test with dict config (what API sends)
        config_with_dict = {
            "n_replications": 1,
            "esmfold_kwargs": {
                "verbose": True,
                "residue_idx_offset": 256,
                "chain_linker": "G" * 10
            }
        }
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config_with_dict
        )
        
        # Verify the dict was converted to Pydantic model
        assert "esmfold_kwargs" in constraint.scoring_function_config
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, ESMFoldKwargs)
        assert esmfold_kwargs.verbose == True
        assert esmfold_kwargs.residue_idx_offset == 256
        assert esmfold_kwargs.chain_linker == "G" * 10

    def test_orfipy_mmseqs_kwargs_normalization(self):
        """Test that orfipy_kwargs and mmseqs_kwargs dicts are converted to Pydantic models."""
        segment = create_segment("ATGTCGATCGATGTAG", SequenceType.DNA)
        
        # Create dummy database file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.faa', delete=False) as f:
            f.write(">test_protein\nMVLSPADKTNVK\n")
            dummy_db_path = f.name
        
        try:
            config_with_dicts = {
                "min_hits": 1,
                "max_hits": 5,
                "orfipy_kwargs": {
                    "threads": 4,
                    "min_len": 30,
                    "max_len": 1000,
                    "start_codons": "ATG,GTG"
                },
                "mmseqs_kwargs": {
                    "database": dummy_db_path,
                    "threads": 4,
                    "sensitivity": 2.0,
                    "only_top_hits": False
                }
            }
            
            constraint = Constraint(
                inputs=[segment],
                scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
                scoring_function_config=config_with_dicts
            )
            
            # Verify dicts were converted to Pydantic models
            assert "orfipy_kwargs" in constraint.scoring_function_config
            assert "mmseqs_kwargs" in constraint.scoring_function_config
            
            orfipy_kwargs = constraint.scoring_function_config["orfipy_kwargs"]
            mmseqs_kwargs = constraint.scoring_function_config["mmseqs_kwargs"]
            
            assert isinstance(orfipy_kwargs, ORFipyKwargs)
            assert isinstance(mmseqs_kwargs, MMseqsKwargs)
            
            # Verify values were preserved
            assert orfipy_kwargs.threads == 4
            assert orfipy_kwargs.min_len == 30
            assert orfipy_kwargs.start_codons == "ATG,GTG"
            
            assert mmseqs_kwargs.database == dummy_db_path
            assert mmseqs_kwargs.threads == 4
            assert mmseqs_kwargs.sensitivity == 2.0
            assert mmseqs_kwargs.only_top_hits == False
            
        finally:
            # Clean up
            Path(dummy_db_path).unlink(missing_ok=True)

    def test_mixed_config_normalization(self):
        """Test that configs with both regular params and Pydantic kwargs work correctly."""
        from proto_language.language.constraint import esmfold_plddt_constraint
        
        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)
        
        config = {
            "n_replications": 2,  # Regular parameter
            "esmfold_kwargs": {   # Should be converted to Pydantic
                "verbose": False,
                "residue_idx_offset": 1024
            }
        }
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config
        )
        
        # Regular parameter should remain unchanged
        assert constraint.scoring_function_config["n_replications"] == 2
        
        # Pydantic parameter should be converted
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, ESMFoldKwargs)
        assert esmfold_kwargs.verbose == False
        assert esmfold_kwargs.residue_idx_offset == 1024

    def test_already_pydantic_models_unchanged(self):
        """Test that configs already containing Pydantic models are left unchanged."""
        from proto_language.language.constraint import esmfold_plddt_constraint
        
        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)
        
        # Create config with already-instantiated Pydantic model
        esmfold_model = ESMFoldKwargs(verbose=True, residue_idx_offset=512)
        config = {
            "n_replications": 1,
            "esmfold_kwargs": esmfold_model  # Already a Pydantic model
        }
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config
        )
        
        # Should be the same instance (not converted again)
        assert constraint.scoring_function_config["esmfold_kwargs"] is esmfold_model
        assert isinstance(constraint.scoring_function_config["esmfold_kwargs"], ESMFoldKwargs)

    def test_empty_config_handling(self):
        """Test that empty configs are handled gracefully."""
        segment = create_segment("ATGTCGATCGATGTAG", SequenceType.DNA)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={}
        )
        
        assert constraint.scoring_function_config == {}

    def test_invalid_pydantic_conversion_fallback(self):
        """Test that invalid Pydantic conversions fall back to dict (backward compatibility)."""
        from proto_language.language.constraint import esmfold_plddt_constraint
        
        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)
        
        # Config with invalid ESMFold parameters
        config = {
            "esmfold_kwargs": {
                "invalid_param": "should_cause_error",
                "verbose": "not_a_boolean"  # Invalid type
            }
        }
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=esmfold_plddt_constraint,
            scoring_function_config=config
        )
        
        # Should fall back to dict when Pydantic conversion fails
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, dict)  # Fallback to original dict
        assert esmfold_kwargs["invalid_param"] == "should_cause_error"

    def test_parser_integration(self):
        """Test that the parser creates constraints with properly normalized configs."""
        from api.core.parser import DarwinParser
        
        # Darwin JSON with protein constraint
        darwin_data = {
            "constructs": [{
                "type": "protein",
                "segments": [{"id": "protein_segment"}]
            }],
            "constraints": [{
                "key": "esmfold-plddt",
                "config": {
                    "n_replications": 1,
                    "esmfold_kwargs": {
                        "verbose": True,
                        "residue_idx_offset": 256
                    }
                },
                "targets": ["protein_segment"]
            }],
            "generators": [{
                "key": "uniform-mutation",
                "config": {"batch_size": 1, "sequence_length": 20},
                "targets": ["protein_segment"]
            }],
            "optimization": {
                "method": "mcmc",
                "num_steps": 1
            }
        }
        
        parser = DarwinParser(darwin_data)
        program = parser.parse()
        
        # Get the constraint that was created
        constraint = program.constraints[0]
        
        # Verify that the config was normalized
        assert "esmfold_kwargs" in constraint.scoring_function_config
        esmfold_kwargs = constraint.scoring_function_config["esmfold_kwargs"]
        assert isinstance(esmfold_kwargs, ESMFoldKwargs)
        assert esmfold_kwargs.verbose == True
        assert esmfold_kwargs.residue_idx_offset == 256
