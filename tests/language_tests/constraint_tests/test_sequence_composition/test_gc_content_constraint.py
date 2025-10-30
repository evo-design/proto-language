import pytest
import sys

sys.path.append(".")

from proto_language.language.core import Constraint, SequenceType
from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from ..utils import create_segment


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
        config = GCContentConfig(min_gc=min_gc, max_gc=max_gc)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9
        # Check metadata (stored in candidate sequences which constraints evaluate)
        gc_content = 100.0 * sum(nt in "GC" for nt in sequence) / max(len(sequence), 1)
        assert (
            abs(
                segment.candidate_sequences[0]._metadata["segment_0.gc_content_constraint.gc_content"]
                - gc_content
            )
            < 1e-9
        )

    @pytest.mark.parametrize(
        "sequence, min_gc, max_gc, expected_score",
        [
            ("GCGCGAUUUA", 40, 60, 0.0),  # In range (50%)
            ("GCAUUAUUAU", 40, 60, 0.5),  # Below range (20%)
        ],
    )
    def test_rna_sequences(self, sequence, min_gc, max_gc, expected_score):
        segment = create_segment(sequence, SequenceType.RNA)
        config = GCContentConfig(min_gc=min_gc, max_gc=max_gc)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9

    def test_wrong_sequence_type(self):
        """Test that protein sequences raise ValueError (constraint-specific check)."""
        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)
        config = GCContentConfig(min_gc=40, max_gc=60)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=config,
            vectorized=True,
        )
        with pytest.raises(ValueError):
            constraint.evaluate()