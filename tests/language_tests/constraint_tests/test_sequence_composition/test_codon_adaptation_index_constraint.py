"""Tests for the Codon Adaptation Index (CAI) constraint."""

import math

import pytest
from pydantic import ValidationError

from proto_language.constraint.constraint_registry import ConstraintRegistry
from proto_language.constraint.sequence_composition.codon_adaptation_index_constraint import (
    STANDARD_GENETIC_CODE,
    CodonAdaptationIndexConfig,
    codon_adaptation_index_constraint,
    relative_adaptiveness_from_sequences,
)
from proto_language.core import Sequence
from proto_language.utils import sigmoid_score


def test_genetic_code_is_complete() -> None:
    assert len(STANDARD_GENETIC_CODE) == 64
    assert sum(1 for aa in STANDARD_GENETIC_CODE.values() if aa == "*") == 3  # TAA, TAG, TGA
    assert STANDARD_GENETIC_CODE["ATG"] == "M" and STANDARD_GENETIC_CODE["TGG"] == "W"


class TestWeightsAndCAI:
    def test_relative_adaptiveness_from_reference(self) -> None:
        # Reference uses GCC for every Ala -> GCC is optimal, GCA absent (floored).
        weights = relative_adaptiveness_from_sequences(["ATGGCCGCC"])
        assert weights["GCC"] == 1.0
        assert weights["GCA"] == pytest.approx(1e-3)

    def test_optimal_sequence_scores_cai_one(self) -> None:
        config = CodonAdaptationIndexConfig(reference_sequences=["ATGGCCGCC"], direction="max")
        out = codon_adaptation_index_constraint([(Sequence("GCCGCC", sequence_type="dna"),)], config)
        cai = out[0].metadata["cai"]
        assert cai == pytest.approx(1.0)
        # direction="max": high CAI -> low energy.
        expected = 1.0 - sigmoid_score(cai, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        assert out[0].score == pytest.approx(expected)

    def test_reference_weights_table_geometric_mean(self) -> None:
        config = CodonAdaptationIndexConfig(reference_weights={"GCC": 1.0, "GCA": 0.5}, direction="max")
        out = codon_adaptation_index_constraint([(Sequence("GCCGCA", sequence_type="dna"),)], config)
        assert out[0].metadata["cai"] == pytest.approx(math.exp((math.log(1.0) + math.log(0.5)) / 2))

    def test_rna_input_and_direction_min(self) -> None:
        config = CodonAdaptationIndexConfig(reference_sequences=["ATGGCCGCC"], direction="min")
        out = codon_adaptation_index_constraint([(Sequence("GCCGCC", sequence_type="rna"),)], config)
        cai = out[0].metadata["cai"]
        assert out[0].score == pytest.approx(sigmoid_score(cai, config.sigmoid_center, slope=1.0 / config.sigmoid_scale))

    def test_only_single_codon_families_is_max_penalty(self) -> None:
        # ATG (Met) + TGG (Trp) are excluded -> no scorable codons -> CAI undefined.
        config = CodonAdaptationIndexConfig(reference_sequences=["ATGGCC"])
        out = codon_adaptation_index_constraint([(Sequence("ATGTGG", sequence_type="dna"),)], config)
        assert out[0].score == 1.0
        assert out[0].metadata["cai"] is None


class TestConfigAndRegistry:
    def test_requires_exactly_one_reference(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            CodonAdaptationIndexConfig()
        with pytest.raises(ValidationError, match="exactly one"):
            CodonAdaptationIndexConfig(reference_weights={"GCC": 1.0}, reference_sequences=["ATGGCC"])

    def test_rejects_bad_weights(self) -> None:
        with pytest.raises(ValidationError, match="sense codon"):
            CodonAdaptationIndexConfig(reference_weights={"TAA": 1.0})  # stop codon
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            CodonAdaptationIndexConfig(reference_weights={"GCC": 1.5})

    def test_registration(self) -> None:
        spec = ConstraintRegistry.get("codon-adaptation-index")
        assert spec.function is codon_adaptation_index_constraint
        assert spec.category == "sequence_composition"
        assert spec.supported_sequence_types == ["dna", "rna"]
