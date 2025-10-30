import pytest
import sys
from typing import Tuple

sys.path.append(".")

from pydantic import BaseModel

from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from .utils import (
    create_segment,
    create_batched_segment,
    mock_single_input_scoring_function,
    mock_multi_input_scoring_function,
    mock_single_input_scoring_function_disjoint,
    mock_multi_input_scoring_function_disjoint,
)


# Empty config model for mock constraint functions
class MockConstraintConfig(BaseModel):
    """Empty config for mock constraints that don't need parameters."""
    pass


# Tests for Sequence and Segment basics
def test_sequence_validation():
    """Tests character validation for Sequence objects."""
    with pytest.warns(UserWarning, match=r"Invalid characters found: (X, Z|Z, X)"):
        Sequence("ATCGXZ", SequenceType.DNA)
    with pytest.warns(UserWarning, match="Invalid characters found: T"):
        Sequence("ACGUUUT", SequenceType.RNA)
    with pytest.warns(UserWarning, match=r"Invalid characters found: (J, O|O, J)"):
        Sequence("MVLSPADKTNVKJO", SequenceType.PROTEIN)
    # Test custom valid characters
    seq = Sequence("123", valid_chars=set("123"))
    assert seq.sequence == "123"
    with pytest.warns(UserWarning, match="Invalid characters found: 4"):
        seq.sequence = "1234"


def test_segment_batching():
    """Tests candidate pool creation for Segment (dual-pool API)."""
    segment = create_segment("ATCG")
    assert segment.num_candidates == 1  # create_segment populates 1 candidate
    segment.create_candidates(5)
    assert segment.num_candidates == 5
    assert all(s.sequence == "ATCG" for s in segment.candidate_sequences)
    segment.candidate_sequences[0].sequence = "GGGG"
    assert segment.candidate_sequences[0].sequence == "GGGG"
    assert segment.candidate_sequences[1].sequence == "ATCG"


def test_construct_concatenation():
    """Tests sequence concatenation in Construct objects (from selected pools)."""
    seg1 = Segment(sequence="ATG", sequence_type=SequenceType.DNA)
    seg2 = Segment(sequence="GGG", sequence_type=SequenceType.DNA)
    seg3 = Segment(sequence="TAA", sequence_type=SequenceType.DNA)
    construct = Construct([seg1, seg2, seg3])
    assert len(construct.joined_sequences) == 1
    assert construct.joined_sequences[0].sequence == "ATGGGGTAA"

    # Test with multiple selected sequences per segment
    batch_seg1 = Segment(sequence="ATG", sequence_type=SequenceType.DNA)
    batch_seg1.selected_sequences.append(Sequence(sequence="ATG", sequence_type=SequenceType.DNA))
    batch_seg2 = Segment(sequence="GGG", sequence_type=SequenceType.DNA)
    batch_seg2.selected_sequences.append(Sequence(sequence="CCC", sequence_type=SequenceType.DNA))
    batch_seg3 = Segment(sequence="TAA", sequence_type=SequenceType.DNA)
    batch_seg3.selected_sequences.append(Sequence(sequence="TGA", sequence_type=SequenceType.DNA))
    batch_construct = Construct([batch_seg1, batch_seg2, batch_seg3])
    assert len(batch_construct.joined_sequences) == 2
    assert batch_construct.joined_sequences[0].sequence == "ATGGGGTAA"
    assert batch_construct.joined_sequences[1].sequence == "ATG" + "CCC" + "TGA"


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

    # Empty config for mock functions
    empty_config = MockConstraintConfig()
    
    # Create a single-input constraint
    constraint_single_input = Constraint(
        inputs=[single_batch_input],
        scoring_function=mock_single_input_scoring_function,
        scoring_function_config=empty_config,
        vectorized=False,
    )
    scores_single_input = constraint_single_input.evaluate()
    constraint_multi_input = Constraint(
        inputs=[multi_batch_input],
        scoring_function=mock_multi_input_scoring_function,
        scoring_function_config=empty_config,
        vectorized=True,
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

        # Ensure metadata is propagated correctly (metadata stored in candidate_sequences)
        sequence_metadata = single_batch_input.candidate_sequences[i]._metadata
        sequence_metadata_multi = multi_batch_input.candidate_sequences[i]._metadata

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

    # Empty config for mock functions
    empty_config = MockConstraintConfig()

    # Create constraints with single sequence inputs
    constraint_single_input = Constraint(
        inputs=[single_seq_segment],
        scoring_function=mock_single_input_scoring_function,
        scoring_function_config=empty_config,
        vectorized=False,
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_seq_segment],
        scoring_function=mock_multi_input_scoring_function,
        scoring_function_config=empty_config,
        vectorized=True,
    )
    scores_multi_input = constraint_multi_input.evaluate()

    # Both should return a single score
    expected_score = 0.25  # 2 T's out of 8 characters

    assert (
        len(scores_single_input) == 1
    ), f"Expected 1 score, got {len(scores_single_input)}"
    assert (
        len(scores_multi_input) == 1
    ), f"Expected 1 score, got {len(scores_multi_input)}"

    assert (
        scores_single_input[0] == expected_score
    ), f"Single input score mismatch: {scores_single_input[0]} vs {expected_score}"
    assert (
        scores_multi_input[0] == expected_score
    ), f"Multi input score mismatch: {scores_multi_input[0]} vs {expected_score}"

    # Check metadata propagation for single sequence
    sequence_metadata = single_seq_segment.candidate_sequences[0]._metadata
    sequence_metadata_multi = multi_seq_segment.candidate_sequences[0]._metadata

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
    multi_total_length = sequence_metadata_multi[
        f"{expected_prefix_multi}.total_length"
    ]
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

    # Empty config for mock functions
    empty_config = MockConstraintConfig()

    # Create constraints with multiple segment inputs
    constraint_single_input = Constraint(
        inputs=[single_batch_input_a, single_batch_input_b],
        scoring_function=mock_single_input_scoring_function,
        scoring_function_config=empty_config,
        concatenate=True,
        vectorized=False,
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_batch_input_a, multi_batch_input_b],
        scoring_function=mock_multi_input_scoring_function,
        scoring_function_config=empty_config,
        concatenate=True,
        vectorized=True,
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
        sequence_metadata_a = single_batch_input_a.candidate_sequences[i]._metadata
        sequence_metadata_b = single_batch_input_b.candidate_sequences[i]._metadata
        sequence_metadata_multi_a = multi_batch_input_a.candidate_sequences[i]._metadata
        sequence_metadata_multi_b = multi_batch_input_b.candidate_sequences[i]._metadata

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

    # Empty config for mock functions
    empty_config = MockConstraintConfig()

    constraint_single_input = Constraint(
        inputs=[single_batch_input_a, single_batch_input_b],
        scoring_function=mock_single_input_scoring_function_disjoint,
        scoring_function_config=empty_config,
        concatenate=False,
        vectorized=False,
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_batch_input_a, multi_batch_input_b],
        scoring_function=mock_multi_input_scoring_function_disjoint,
        scoring_function_config=empty_config,
        concatenate=False,
        vectorized=True,
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
        metadata_a_single = single_batch_input_a.candidate_sequences[i]._metadata
        metadata_a_multi = multi_batch_input_a.candidate_sequences[i]._metadata

        # Check metadata in segment B
        metadata_b_single = single_batch_input_b.candidate_sequences[i]._metadata
        metadata_b_multi = multi_batch_input_b.candidate_sequences[i]._metadata

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


# =============================================================================
# TESTS FOR INPUT VALIDATION
# =============================================================================

def test_empty_inputs_raises_error():
    """Test that empty inputs list raises ValueError."""
    empty_config = MockConstraintConfig()
    with pytest.raises(ValueError, match="At least one segment must be provided"):
        Constraint(
            inputs=[],
            scoring_function=mock_single_input_scoring_function,
            scoring_function_config=empty_config,
        )


def test_mixed_batch_sizes_raises_error():
    """Test that inconsistent candidate pool sizes raise ValueError."""
    seg1 = create_batched_segment(["ATCG", "GGGG"])  # 2 candidates
    seg2 = create_batched_segment(["TTTT"])  # 1 candidate
    config = MockConstraintConfig()
    with pytest.raises(ValueError, match="All segments must have the same number of candidate sequences"):
        Constraint(
            inputs=[seg1, seg2],
            scoring_function=mock_single_input_scoring_function,
            scoring_function_config=config,
        )


def test_mixed_sequence_types_raises_error():
    """Test that inconsistent sequence types raise ValueError."""
    seg1 = create_segment("ATCG", SequenceType.DNA)
    seg2 = create_segment("MVLS", SequenceType.PROTEIN)
    config = MockConstraintConfig()
    with pytest.raises(ValueError, match="same sequence type"):
        Constraint(
            inputs=[seg1, seg2],
            scoring_function=mock_single_input_scoring_function,
            scoring_function_config=config,
        )


def test_mixed_valid_chars_raises_error():
    """Test that inconsistent alphabets raise ValueError."""
    seg1 = Segment(sequence="ATCG", sequence_type=SequenceType.DNA)
    seg2 = Segment(sequence="ATCG", sequence_type=SequenceType.DNA, 
                   valid_chars=set("ATCGN"))  # Different alphabet
    config = MockConstraintConfig()
    with pytest.raises(ValueError, match="same valid_chars"):
        Constraint(
            inputs=[seg1, seg2],
            scoring_function=mock_single_input_scoring_function,
            scoring_function_config=config,
        )


# =============================================================================
# TESTS FOR CUSTOM LABEL HANDLING
# =============================================================================

def test_custom_label_in_metadata():
    """Test that custom label overrides function name in metadata."""
    segment = create_segment("ATCGACTG", SequenceType.DNA)
    config = MockConstraintConfig()
    
    constraint = Constraint(
        inputs=[segment],
        scoring_function=mock_single_input_scoring_function,
        scoring_function_config=config,
        label="my_custom_label"
    )
    
    scores = constraint.evaluate()
    
    # Metadata should use custom label, not "mock_single_input_scoring_function"
    metadata_keys = [key for key in segment.candidate_sequences[0]._metadata.keys() 
                     if key not in ["sequence", "sequence_length"]]
    
    assert any("my_custom_label" in key for key in metadata_keys), \
        f"Custom label not found in metadata keys: {metadata_keys}"
    
    assert not any("mock_single_input_scoring_function" in key for key in metadata_keys), \
        f"Function name found in metadata instead of custom label: {metadata_keys}"
    
    # Verify specific metadata keys use custom label
    assert "segment_0.my_custom_label.t_count" in segment.candidate_sequences[0]._metadata
    assert "segment_0.my_custom_label.total_length" in segment.candidate_sequences[0]._metadata
    assert "segment_0.my_custom_label.t_fraction" in segment.candidate_sequences[0]._metadata


def test_custom_label_disjoint_mode():
    """Test that custom label works correctly in disjoint mode."""
    seg1 = create_segment("AAAA", SequenceType.DNA)
    seg2 = create_segment("CCCC", SequenceType.DNA)
    config = MockConstraintConfig()
    
    constraint = Constraint(
        inputs=[seg1, seg2],
        scoring_function=mock_single_input_scoring_function_disjoint,
        scoring_function_config=config,
        concatenate=False,
        label="disjoint_custom_label"
    )
    
    scores = constraint.evaluate()
    
    # Check both segments have metadata with custom label
    metadata_keys_seg1 = [key for key in seg1.candidate_sequences[0]._metadata.keys() 
                          if key not in ["sequence", "sequence_length"]]
    metadata_keys_seg2 = [key for key in seg2.candidate_sequences[0]._metadata.keys() 
                          if key not in ["sequence", "sequence_length"]]
    
    assert any("disjoint_custom_label" in key for key in metadata_keys_seg1)
    assert any("disjoint_custom_label" in key for key in metadata_keys_seg2)
    
    # Verify each segment has its own prefixed metadata
    assert "segment_0.disjoint_custom_label.t_percent" in seg1.candidate_sequences[0]._metadata
    assert "segment_1.disjoint_custom_label.c_percent" in seg2.candidate_sequences[0]._metadata


# =============================================================================
# TESTS FOR EDGE CASES
# =============================================================================

def test_large_batch_processing():
    """Test constraint with very large batch (100+ sequences)."""
    sequences = ["ATCG"] * 100
    segment = create_batched_segment(sequences, SequenceType.DNA)
    config = MockConstraintConfig()
    
    constraint = Constraint(
        inputs=[segment],
        scoring_function=mock_multi_input_scoring_function,
        scoring_function_config=config,
        vectorized=True,
    )
    
    scores = constraint.evaluate()
    assert len(scores) == 100
    
    # Verify all scores are valid
    for score in scores:
        assert 0.0 <= score <= 1.0
    
    # Verify metadata was propagated to all sequences
    for seq in segment.candidate_sequences:
        assert "segment_0.mock_multi_input_scoring_function.t_count" in seq._metadata


def test_three_or_more_segments_contiguous():
    """Test constraint with 3+ segments in contiguous mode."""
    seg1 = create_segment("ATCG")
    seg2 = create_segment("GGGG")
    seg3 = create_segment("TTTT")
    config = MockConstraintConfig()
    
    constraint = Constraint(
        inputs=[seg1, seg2, seg3],
        scoring_function=mock_single_input_scoring_function,
        scoring_function_config=config,
        concatenate=True,
    )
    
    scores = constraint.evaluate()
    assert len(scores) == 1
    
    # Check metadata propagation to all three segments
    expected_prefix = "segment_0-segment_1-segment_2.mock_single_input_scoring_function"
    
    for seg in [seg1, seg2, seg3]:
        metadata_keys = list(seg.candidate_sequences[0]._metadata.keys())
        assert any(expected_prefix in key for key in metadata_keys), \
            f"Expected prefix '{expected_prefix}' not found in segment metadata: {metadata_keys}"
        
        # Verify specific metadata keys
        assert f"{expected_prefix}.t_count" in seg.candidate_sequences[0]._metadata
        assert f"{expected_prefix}.total_length" in seg.candidate_sequences[0]._metadata
        assert f"{expected_prefix}.t_fraction" in seg.candidate_sequences[0]._metadata


def test_three_or_more_segments_disjoint():
    """Test constraint with 3+ segments in disjoint mode."""
    # Create a mock function that handles 3 segments
    def mock_triple_input_scoring_function(sequence_tuple: Tuple[Sequence, Sequence, Sequence], config=None) -> float:
        """Mock scoring function for 3 disjoint sequences."""
        a_count = sequence_tuple[0].sequence.count("A") / len(sequence_tuple[0])
        t_count = sequence_tuple[1].sequence.count("T") / len(sequence_tuple[1])
        g_count = sequence_tuple[2].sequence.count("G") / len(sequence_tuple[2])
        
        sequence_tuple[0]._metadata["a_percent"] = a_count
        sequence_tuple[1]._metadata["t_percent"] = t_count
        sequence_tuple[2]._metadata["g_percent"] = g_count
        
        return (a_count + t_count + g_count) / 3
    
    seg1 = create_segment("AAAA", SequenceType.DNA)
    seg2 = create_segment("TTTT", SequenceType.DNA)
    seg3 = create_segment("GGGG", SequenceType.DNA)
    config = MockConstraintConfig()
    
    constraint = Constraint(
        inputs=[seg1, seg2, seg3],
        scoring_function=mock_triple_input_scoring_function,
        scoring_function_config=config,
        concatenate=False,
    )
    
    scores = constraint.evaluate()
    assert len(scores) == 1
    # Each segment is 100% of its respective nucleotide, so score = (1+1+1)/3 ≈ 1.0
    assert abs(scores[0] - 1.0) < 1e-9
    
    # Check that each segment has its own prefixed metadata
    assert "segment_0.mock_triple_input_scoring_function.a_percent" in seg1.candidate_sequences[0]._metadata
    assert "segment_1.mock_triple_input_scoring_function.t_percent" in seg2.candidate_sequences[0]._metadata
    assert "segment_2.mock_triple_input_scoring_function.g_percent" in seg3.candidate_sequences[0]._metadata
    
    # Verify the metadata values are correct
    assert seg1.candidate_sequences[0]._metadata["segment_0.mock_triple_input_scoring_function.a_percent"] == 1.0
    assert seg2.candidate_sequences[0]._metadata["segment_1.mock_triple_input_scoring_function.t_percent"] == 1.0
    assert seg3.candidate_sequences[0]._metadata["segment_2.mock_triple_input_scoring_function.g_percent"] == 1.0


def test_empty_sequence_in_batch():
    """Test constraint with empty sequence in batch.
    
    Note: This test documents that empty sequences will cause issues with
    most scoring functions (division by zero). This is expected behavior -
    individual constraints should validate their inputs appropriately.
    """
    sequences = ["ATCG", "", "GGGG"]
    segment = create_batched_segment(sequences, SequenceType.DNA)
    config = MockConstraintConfig()
    
    constraint = Constraint(
        inputs=[segment],
        scoring_function=mock_multi_input_scoring_function,
        scoring_function_config=config,
        vectorized=True,
    )
    
    # Most scoring functions will fail on empty sequences (division by zero)
    # This is expected behavior - constraints should validate their inputs
    with pytest.raises(ZeroDivisionError):
        scores = constraint.evaluate()