"""tests/language_tests/constraint_tests/test_sequence_composition/test_dinucleotide_composition_constraint.py."""

import pytest
from pydantic import ValidationError

from proto_language.constraint import dinucleotide_composition_constraint
from proto_language.constraint.sequence_composition.dinucleotide_composition_constraint import (
    DinucleotideCompositionConfig,
)
from proto_language.core import Constraint, Segment


class TestDinucleotideCompositionConstraint:
    def test_perfect_match_scores_zero(self):
        # "ATATAT" -> dinucleotides AT, TA, AT, TA, AT -> {AT: 3/5, TA: 2/5}
        segment = Segment(sequence="ATATAT", sequence_type="dna")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 0.6, "TA": 0.4})
        constraint = Constraint(
            inputs=[segment],
            function=dinucleotide_composition_constraint,
            function_config=config,
        )
        assert constraint.evaluate()[0] == pytest.approx(0.0, abs=1e-9)

    def test_total_variation_distance(self):
        # observed {AT: 0.6, TA: 0.4}; reference {AT: 1.0}
        # TV = 0.5 * (|0.6-1.0| + |0.4-0.0|) = 0.5 * 0.8 = 0.4
        segment = Segment(sequence="ATATAT", sequence_type="dna")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 1.0})
        constraint = Constraint(
            inputs=[segment],
            function=dinucleotide_composition_constraint,
            function_config=config,
        )
        assert constraint.evaluate()[0] == pytest.approx(0.4, abs=1e-9)

    def test_reference_is_renormalized(self):
        # Unnormalized reference {AT: 3, TA: 2} -> {AT: 0.6, TA: 0.4}; matches observed exactly.
        segment = Segment(sequence="ATATAT", sequence_type="dna")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 3, "TA": 2})
        constraint = Constraint(
            inputs=[segment],
            function=dinucleotide_composition_constraint,
            function_config=config,
        )
        assert constraint.evaluate()[0] == pytest.approx(0.0, abs=1e-9)

    def test_rna_sequence_folds_to_dna_alphabet(self):
        # RNA "AUAUAU" matches a T-keyed reference after U->T folding.
        segment = Segment(sequence="AUAUAU", sequence_type="rna")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 0.6, "TA": 0.4})
        constraint = Constraint(
            inputs=[segment],
            function=dinucleotide_composition_constraint,
            function_config=config,
        )
        assert constraint.evaluate()[0] == pytest.approx(0.0, abs=1e-9)

    def test_scale_amplifies_distance(self):
        segment = Segment(sequence="ATATAT", sequence_type="dna")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 1.0}, scale=2.0)
        constraint = Constraint(
            inputs=[segment],
            function=dinucleotide_composition_constraint,
            function_config=config,
        )
        # 0.4 distance * 2.0 scale = 0.8
        assert constraint.evaluate()[0] == pytest.approx(0.8, abs=1e-9)

    def test_short_sequence_max_penalty(self):
        segment = Segment(sequence="A", sequence_type="dna")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 1.0})
        constraint = Constraint(
            inputs=[segment],
            function=dinucleotide_composition_constraint,
            function_config=config,
        )
        assert constraint.evaluate()[0] == pytest.approx(1.0, abs=1e-9)

    def test_metadata_propagation(self):
        segment = Segment(sequence="ATATAT", sequence_type="dna")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 0.6, "TA": 0.4})
        constraint = Constraint(
            inputs=[segment],
            function=dinucleotide_composition_constraint,
            function_config=config,
        )
        constraint.evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["dinucleotide_composition_constraint"]["data"]
        assert data["dinucleotide_frequencies"]["AT"] == pytest.approx(0.6, abs=1e-9)
        assert data["dinucleotide_distance"] == pytest.approx(0.0, abs=1e-9)

    def test_wrong_sequence_type(self):
        segment = Segment(sequence="MVLSPADKTNVK", sequence_type="protein")
        config = DinucleotideCompositionConfig(reference_frequencies={"AT": 1.0})
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[segment],
                function=dinucleotide_composition_constraint,
                function_config=config,
            )


class TestDinucleotideCompositionConfigValidation:
    def test_empty_reference_raises(self):
        with pytest.raises(ValidationError):
            DinucleotideCompositionConfig(reference_frequencies={})

    def test_invalid_dinucleotide_key_raises(self):
        with pytest.raises(ValidationError, match="Invalid dinucleotide key"):
            DinucleotideCompositionConfig(reference_frequencies={"AXY": 1.0})

    def test_wrong_length_key_raises(self):
        with pytest.raises(ValidationError, match="Invalid dinucleotide key"):
            DinucleotideCompositionConfig(reference_frequencies={"A": 1.0})

    def test_negative_frequency_raises(self):
        with pytest.raises(ValidationError, match="nonnegative"):
            DinucleotideCompositionConfig(reference_frequencies={"AT": -0.5})

    def test_all_zero_reference_raises(self):
        with pytest.raises(ValidationError, match="at least one positive"):
            DinucleotideCompositionConfig(reference_frequencies={"AT": 0.0, "TA": 0.0})
