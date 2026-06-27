"""tests/language_tests/constraint_tests/test_sequence_composition/test_melting_temperature_constraint.py."""

import math

import pytest
from pydantic import ValidationError

from proto_language.constraint import melting_temperature_constraint
from proto_language.constraint.sequence_composition.melting_temperature_constraint import (
    MeltingTemperatureConfig,
    _tm_gc_based,
    _tm_wallace,
)
from proto_language.core import Constraint, Segment

# ---------------------------------------------------------------------------
# Helper: manually compute expected Tm values
# ---------------------------------------------------------------------------

def _tm_wallace_ref(seq: str) -> float:
    s = seq.upper()
    return 2.0 * (s.count("A") + s.count("T")) + 4.0 * (s.count("G") + s.count("C"))


# ---------------------------------------------------------------------------
# Wallace rule formula tests
# ---------------------------------------------------------------------------

class TestTmWallaceFormula:
    @pytest.mark.parametrize(
        "sequence, expected_tm",
        [
            ("AAAAAAAAAA", 20.0),   # 10 A's: 2*10 + 4*0 = 20
            ("GGGGGGGGGG", 40.0),   # 10 G's: 2*0 + 4*10 = 40
            ("ATCGATCG",   24.0),   # 4 AT + 4 GC: 2*4 + 4*4 = 24
            ("GCATGCAT",   24.0),   # same composition
        ],
    )
    def test_wallace_values(self, sequence: str, expected_tm: float) -> None:
        assert _tm_wallace(sequence.upper()) == pytest.approx(expected_tm, abs=1e-9)

    def test_single_nucleotide_a(self) -> None:
        assert _tm_wallace("A") == pytest.approx(2.0)

    def test_single_nucleotide_g(self) -> None:
        assert _tm_wallace("G") == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# GC-based formula tests
# ---------------------------------------------------------------------------

class TestTmGCBasedFormula:
    def test_all_gc_50mM(self) -> None:
        seq = "G" * 20  # 100% GC, N=20
        expected = 81.5 + 16.6 * math.log10(0.05) + 0.41 * 100.0 - 675.0 / 20
        assert _tm_gc_based(seq.upper(), 50.0) == pytest.approx(expected, abs=1e-6)

    def test_all_at_50mM(self) -> None:
        seq = "A" * 20  # 0% GC, N=20
        expected = 81.5 + 16.6 * math.log10(0.05) + 0.41 * 0.0 - 675.0 / 20
        assert _tm_gc_based(seq.upper(), 50.0) == pytest.approx(expected, abs=1e-6)

    def test_higher_salt_raises_tm(self) -> None:
        seq = "ATCGATCGATCGATCG"  # 16 nt
        tm_low_salt = _tm_gc_based(seq.upper(), 10.0)
        tm_high_salt = _tm_gc_based(seq.upper(), 200.0)
        assert tm_high_salt > tm_low_salt

    def test_longer_sequence_higher_tm(self) -> None:
        """Longer sequences approach the asymptote (remove 675/N penalty)."""
        short = _tm_gc_based("GCGCGCGCGCGCGCGC", 50.0)   # 16 nt, 100% GC
        long_seq = "GCGCGCGCGCGCGCGC" * 10               # 160 nt, 100% GC
        long_tm = _tm_gc_based(long_seq.upper(), 50.0)
        assert long_tm > short


# ---------------------------------------------------------------------------
# Constraint scoring (in-range, below-range, above-range)
# ---------------------------------------------------------------------------

class TestMeltingTemperatureConstraintScoring:
    @pytest.mark.parametrize(
        "sequence, min_tm, max_tm, expected_score",
        [
            # ── Wallace rule path (≤ 13 nt) ──────────────────────────────
            # "ATCGATCG" → Tm = 2*4 + 4*4 = 24 °C; range [20, 30] → in range
            ("ATCGATCG", 20.0, 30.0, 0.0),
            # "AAAAAAAA" → Tm = 2*8 + 4*0 = 16 °C; range [20, 30] → below range
            # deviation = (20 - 16) / 20 = 0.2
            ("AAAAAAAA", 20.0, 30.0, 0.2),
            # "GCGCGCGC" -> Tm = 2*0 + 4*8 = 32 degC; range [20, 30] -> above range
            # calculate_range_deviation: (32 - 30) / max(30, 1) = 2/30
            ("GCGCGCGC", 20.0, 30.0, (32.0 - 30.0) / 30.0),
            # Exact boundary: Tm == min_tm → score 0.0
            ("AAAAAAAA", 16.0, 30.0, 0.0),
            # Exact boundary: Tm == max_tm → score 0.0
            ("GCGCGCGC", 20.0, 32.0, 0.0),
        ],
    )
    def test_scoring_parametrized(
        self, sequence: str, min_tm: float, max_tm: float, expected_score: float
    ) -> None:
        segment = Segment(sequence=sequence, sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=min_tm, max_tm=max_tm, method="auto")
        cons = Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        )
        assert abs(cons.evaluate()[0] - expected_score) < 1e-6

    def test_empty_sequence_max_penalty(self) -> None:
        segment = Segment(sequence="", sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0)
        cons = Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        )
        assert cons.evaluate()[0] == pytest.approx(1.0)

    def test_score_capped_at_max_energy(self) -> None:
        """Sequence far outside the target range receives a high penalty."""
        # "AAAAAAAA" -> Tm = 16 (Wallace); range [90, 100] -> well below range
        # calculate_range_deviation(16, 90, 100): (90-16)/max(90,1) = 74/90 ~0.822
        segment = Segment(sequence="AAAAAAAA", sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=90.0, max_tm=100.0)
        cons = Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        )
        score = cons.evaluate()[0]
        # Score is capped at MAX_ENERGY = 1.0; here it's ~0.822 (high but not 1.0)
        assert score > 0.5
        assert score <= 1.0

    def test_batch_multiple_sequences(self) -> None:
        """Constraint handles a batch of proposals correctly."""
        segment = Segment(sequence="AAAAAAAA", sequence_type="dna")
        # Introduce extra proposals
        segment.proposal_sequences[0].sequence = "AAAAAAAA"
        from proto_language.core import Sequence
        segment.proposal_sequences.append(Sequence("GCGCGCGC", "dna"))
        config = MeltingTemperatureConfig(min_tm=20.0, max_tm=30.0)
        cons = Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        )
        scores = cons.evaluate()
        assert len(scores) == 2
        # "AAAAAAAA" Tm=16, below range
        assert scores[0] > 0.0
        # "GCGCGCGC" Tm=32, above range
        assert scores[1] > 0.0


# ---------------------------------------------------------------------------
# Method selection (auto / wallace / gc_based)
# ---------------------------------------------------------------------------

class TestMethodSelection:
    def test_auto_short_uses_wallace(self) -> None:
        """Auto method selects Wallace for sequences ≤ 13 nt."""
        seq = "ATCGATCG"  # 8 nt
        config = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="auto")
        segment = Segment(sequence=seq, sequence_type="dna")
        cons = Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        )
        cons.evaluate()
        metadata = segment.proposal_sequences[0]._constraints_metadata
        assert metadata["melting_temperature_constraint"]["data"]["method_used"] == "wallace"

    def test_auto_long_uses_gc_based(self) -> None:
        """Auto method selects gc_based for sequences > 13 nt."""
        seq = "ATCGATCGATCGATCG"  # 16 nt
        config = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="auto")
        segment = Segment(sequence=seq, sequence_type="dna")
        cons = Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        )
        cons.evaluate()
        metadata = segment.proposal_sequences[0]._constraints_metadata
        assert metadata["melting_temperature_constraint"]["data"]["method_used"] == "gc_based"

    def test_explicit_wallace_on_long_sequence(self) -> None:
        """Explicit method='wallace' applies Wallace even to long sequences."""
        seq = "ATCGATCGATCGATCG"  # 16 nt
        config_auto = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="auto")
        config_wall = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="wallace")
        segment_auto = Segment(sequence=seq, sequence_type="dna")
        segment_wall = Segment(sequence=seq, sequence_type="dna")
        Constraint(inputs=[segment_auto], function=melting_temperature_constraint, function_config=config_auto).evaluate()
        Constraint(inputs=[segment_wall], function=melting_temperature_constraint, function_config=config_wall).evaluate()
        tm_auto = segment_auto.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]["tm"]
        tm_wall = segment_wall.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]["tm"]
        # auto uses gc_based for 16 nt, explicit wallace uses Wallace formula → different values
        assert tm_auto != pytest.approx(tm_wall, abs=1e-3)

    def test_explicit_gc_based_on_short_sequence(self) -> None:
        """Explicit method='gc_based' applies GC formula even to short sequences."""
        seq = "ATCGATCG"  # 8 nt
        config = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="gc_based")
        segment = Segment(sequence=seq, sequence_type="dna")
        Constraint(inputs=[segment], function=melting_temperature_constraint, function_config=config).evaluate()
        metadata = segment.proposal_sequences[0]._constraints_metadata
        assert metadata["melting_temperature_constraint"]["data"]["method_used"] == "gc_based"

    def test_salt_concentration_affects_gc_based_tm(self) -> None:
        """Higher salt → higher GC-based Tm → possibly different score."""
        seq = "GCATGCATGCATGCAT"  # 16 nt, gc_based path
        config_low = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="gc_based", salt_mm=10.0)
        config_high = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="gc_based", salt_mm=500.0)
        seg_low = Segment(sequence=seq, sequence_type="dna")
        seg_high = Segment(sequence=seq, sequence_type="dna")
        Constraint(inputs=[seg_low], function=melting_temperature_constraint, function_config=config_low).evaluate()
        Constraint(inputs=[seg_high], function=melting_temperature_constraint, function_config=config_high).evaluate()
        tm_low = seg_low.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]["tm"]
        tm_high = seg_high.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]["tm"]
        assert tm_high > tm_low


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------

class TestMetadataPropagation:
    def test_tm_in_metadata(self) -> None:
        seq = "ATCGATCGATCG"  # 12 nt, Wallace path
        expected_tm = _tm_wallace_ref(seq)
        segment = Segment(sequence=seq, sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0)
        Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        ).evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]
        assert "tm" in data
        assert data["tm"] == pytest.approx(expected_tm, abs=1e-6)

    def test_method_used_in_metadata(self) -> None:
        segment = Segment(sequence="ATCG", sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="wallace")
        Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        ).evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]
        assert data["method_used"] == "wallace"

    def test_empty_sequence_metadata(self) -> None:
        segment = Segment(sequence="", sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0)
        Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        ).evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]
        assert data["tm"] == pytest.approx(0.0)
        assert data["method_used"] == "none"


# ---------------------------------------------------------------------------
# Wrong sequence type
# ---------------------------------------------------------------------------

class TestWrongSequenceType:
    def test_protein_raises_type_error(self) -> None:
        segment = Segment(sequence="MVLSPADKTNVK", sequence_type="protein")
        config = MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0)
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[segment],
                function=melting_temperature_constraint,
                function_config=config,
            )

    def test_rna_raises_type_error(self) -> None:
        segment = Segment(sequence="AUCGAUCG", sequence_type="rna")
        config = MeltingTemperatureConfig(min_tm=20.0, max_tm=40.0)
        with pytest.raises(TypeError, match="does not support sequence type 'rna'"):
            Constraint(
                inputs=[segment],
                function=melting_temperature_constraint,
                function_config=config,
            )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestMeltingTemperatureConfigValidation:
    def test_min_tm_greater_than_max_tm_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"min_tm.*must be <= max_tm"):
            MeltingTemperatureConfig(min_tm=70.0, max_tm=50.0)

    def test_min_tm_equal_max_tm_allowed(self) -> None:
        config = MeltingTemperatureConfig(min_tm=60.0, max_tm=60.0)
        assert config.min_tm == config.max_tm == 60.0

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"method must be one of"):
            MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0, method="nearest_neighbor")

    def test_zero_salt_raises(self) -> None:
        with pytest.raises(ValidationError):
            MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0, salt_mm=0.0)

    def test_negative_salt_raises(self) -> None:
        with pytest.raises(ValidationError):
            MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0, salt_mm=-10.0)

    def test_valid_gc_based_config(self) -> None:
        config = MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0, method="gc_based", salt_mm=150.0)
        assert config.method == "gc_based"
        assert config.salt_mm == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_nucleotide_g(self) -> None:
        """Single G: Tm = 4 (Wallace); far below range [50, 65] -> high penalty."""
        # calculate_range_deviation(4, 50, 65): (50-4)/max(50,1) = 0.92
        segment = Segment(sequence="G", sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=50.0, max_tm=65.0)
        cons = Constraint(
            inputs=[segment],
            function=melting_temperature_constraint,
            function_config=config,
        )
        score = cons.evaluate()[0]
        assert score > 0.8
        assert score <= 1.0

    def test_lowercase_sequence_handled(self) -> None:
        """Lowercase input should be handled identically to uppercase."""
        seg_lower = Segment(sequence="atcgatcg", sequence_type="dna")
        seg_upper = Segment(sequence="ATCGATCG", sequence_type="dna")
        config = MeltingTemperatureConfig(min_tm=20.0, max_tm=30.0)
        score_lower = Constraint(
            inputs=[seg_lower],
            function=melting_temperature_constraint,
            function_config=config,
        ).evaluate()[0]
        score_upper = Constraint(
            inputs=[seg_upper],
            function=melting_temperature_constraint,
            function_config=config,
        ).evaluate()[0]
        assert score_lower == pytest.approx(score_upper, abs=1e-9)

    def test_exactly_at_crossover_length(self) -> None:
        """13-nt sequence uses Wallace; 14-nt uses gc_based under 'auto'."""
        seq_13 = "ATCGATCGATCGA"  # 13 nt
        seq_14 = "ATCGATCGATCGAT"  # 14 nt
        config = MeltingTemperatureConfig(min_tm=0.0, max_tm=100.0, method="auto")
        for seq, expected_method in [(seq_13, "wallace"), (seq_14, "gc_based")]:
            seg = Segment(sequence=seq, sequence_type="dna")
            Constraint(inputs=[seg], function=melting_temperature_constraint, function_config=config).evaluate()
            method_used = seg.proposal_sequences[0]._constraints_metadata["melting_temperature_constraint"]["data"]["method_used"]
            assert method_used == expected_method, f"Expected {expected_method} for {len(seq)}-nt sequence, got {method_used}"
