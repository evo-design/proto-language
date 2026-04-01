"""tests/language_tests/constraint_tests/test_sequence_composition/test_gc_content_constraint.py."""

import pytest
from pydantic import ValidationError

from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
    GCContentConfig,
)
from proto_language.language.core import Constraint, Segment


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
        segment = Segment(sequence=sequence, sequence_type="dna")
        config = GCContentConfig(min_gc=min_gc, max_gc=max_gc)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=config,
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9
        # Check metadata (stored in proposal sequences which constraints evaluate)
        gc_content = 100.0 * sum(nt in "GC" for nt in sequence) / max(len(sequence), 1)
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert abs(constraints["gc_content_constraint"]["data"]["gc_content"] - gc_content) < 1e-9

    @pytest.mark.parametrize(
        "sequence, min_gc, max_gc, expected_score",
        [
            ("GCGCGAUUUA", 40, 60, 0.0),  # In range (50%)
            ("GCAUUAUUAU", 40, 60, 0.5),  # Below range (20%)
        ],
    )
    def test_rna_sequences(self, sequence, min_gc, max_gc, expected_score):
        segment = Segment(sequence=sequence, sequence_type="rna")
        config = GCContentConfig(min_gc=min_gc, max_gc=max_gc)
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=config,
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9

    def test_wrong_sequence_type(self):
        """Test that protein sequences raise TypeError at construction (centralized validation)."""
        segment = Segment(sequence="MVLSPADKTNVK", sequence_type="protein")
        config = GCContentConfig(min_gc=40, max_gc=60)
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[segment],
                function=gc_content_constraint,
                function_config=config,
            )


class TestGCContentConfigValidation:
    """Tests for GCContentConfig cross-field validation."""

    def test_min_gc_greater_than_max_gc_raises(self):
        """Test that min_gc > max_gc raises ValidationError."""
        with pytest.raises(ValidationError, match=r"min_gc.*must be <= max_gc"):
            GCContentConfig(min_gc=80, max_gc=20)

    def test_min_gc_equal_max_gc_allowed(self):
        """Test that min_gc == max_gc is a valid (exact target) config."""
        config = GCContentConfig(min_gc=50, max_gc=50)
        assert config.min_gc == 50
        assert config.max_gc == 50
