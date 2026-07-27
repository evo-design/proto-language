"""Tests for the uracil (U) content constraint."""

import pytest
from pydantic import ValidationError

from proto_language.constraint.constraint_registry import ConstraintRegistry
from proto_language.constraint.sequence_composition.uracil_content_constraint import (
    UracilContentConfig,
    uracil_content_constraint,
)
from proto_language.core import Sequence


class TestForward:
    def test_within_range_scores_zero(self) -> None:
        # ATGTTT -> 3/6 = 50% U (T counted as U).
        out = uracil_content_constraint(
            [(Sequence("ATGTTT", sequence_type="dna"),)], UracilContentConfig(min_u=0, max_u=60)
        )
        assert out[0].score == 0.0
        assert out[0].metadata["uracil_content"] == pytest.approx(50.0)

    def test_above_max_is_penalized(self) -> None:
        out = uracil_content_constraint(
            [(Sequence("ATGTTT", sequence_type="dna"),)], UracilContentConfig(min_u=0, max_u=40)
        )
        assert out[0].score > 0.0  # 50% U exceeds the 40% ceiling

    def test_rna_u_equals_dna_t(self) -> None:
        dna = uracil_content_constraint(
            [(Sequence("ATGTTT", sequence_type="dna"),)], UracilContentConfig(min_u=0, max_u=100)
        )
        rna = uracil_content_constraint(
            [(Sequence("AUGUUU", sequence_type="rna"),)], UracilContentConfig(min_u=0, max_u=100)
        )
        assert dna[0].metadata["uracil_content"] == pytest.approx(rna[0].metadata["uracil_content"])

    def test_empty_sequence_max_penalty(self) -> None:
        out = uracil_content_constraint([(Sequence("", sequence_type="dna"),)], UracilContentConfig())
        assert out[0].score == 1.0


class TestConfigAndRegistry:
    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            UracilContentConfig(min_u=60, max_u=40)

    def test_registration(self) -> None:
        spec = ConstraintRegistry.get("uracil-content")
        assert spec.function is uracil_content_constraint
        assert spec.category == "sequence_composition"
        assert spec.supported_sequence_types == ["dna", "rna"]
