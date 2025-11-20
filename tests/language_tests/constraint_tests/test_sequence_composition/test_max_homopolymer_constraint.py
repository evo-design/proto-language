import numpy as np
import pytest

from proto_language.language.core import Constraint, Segment, SequenceType
from proto_language.language.constraint import max_homopolymer_constraint
from proto_language.language.constraint.sequence_composition.max_homopolymer_constraint import MaxHomopolymerConfig


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
            ("", 3, 0.0, SequenceType.DNA),  # Empty sequence
            ("AAAUUUGGGGCCCC", 3, np.log2(1 + 1 / 3), SequenceType.RNA),  # RNA
            (
                "AAALLLDDDEEEEEFFFF",
                3,
                np.log2(1 + 2 / 3),
                SequenceType.PROTEIN,
            ),  # Protein
        ],
    )
    def test_homopolymer_scoring(self, sequence, max_len, expected_score, seq_type):
        segment = Segment(starting_sequence_or_desired_length=sequence, sequence_type=seq_type)
        config = MaxHomopolymerConfig(max_length=max_len)
        constraint = Constraint(
            inputs=[segment],
            function=max_homopolymer_constraint,
            function_config=config,
        )
        score = constraint.evaluate()[0]
        assert abs(score - expected_score) < 1e-9
        # Test metadata
        if len(sequence) > 0:
            import itertools

            expected_max_homopolymer = max(
                len(list(g)) for _, g in itertools.groupby(sequence)
            )
            assert (
                segment.candidate_sequences[0]._metadata[
                    "segment_0.max_homopolymer_constraint.max_homopolymer_length"
                ]
                == expected_max_homopolymer
            )
        else:
            assert (
                segment.candidate_sequences[0]._metadata[
                    "segment_0.max_homopolymer_constraint.max_homopolymer_length"
                ]
                == 0
            )
