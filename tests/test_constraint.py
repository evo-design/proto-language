import numpy as np
import pytest

import sys
sys.path.append(".")
from language.constraint import (
    DinucleotideFrequencyConstraint,
    GCContentConstraint,
    MaxHomopolymerConstraint,
    SequenceLengthConstraint,
    TetranucleotideUsageConstraint,
)
from language.sequence import (
    ProgramDNASequence,
    ProgramRNASequence,
    ProgramProteinSequence,
)
from language.base import ProgramGenerator


# Create a dummy generator for sequence initialization.
class DummyGenerator(ProgramGenerator):
    def register(self, outputs=None):
        pass
    def sample(self):
        pass

dummy_gen = DummyGenerator()


# Helper to create sequence objects easily.
def create_seq(seq_type, sequence_str):
    if seq_type == "dna":
        cls = ProgramDNASequence
    elif seq_type == "rna":
        cls = ProgramRNASequence
    elif seq_type == "protein":
        cls = ProgramProteinSequence
    else:
        raise ValueError("Invalid seq_type")
    return cls(generator=dummy_gen, generator_output_idx=0, sequence=sequence_str)


def test_sequence_length_constraint():
    """Tests SequenceLengthConstraint."""
    target_len = 20
    seq_match = create_seq("dna", "A" * target_len)
    seq_short = create_seq("dna", "A" * (target_len // 2))
    seq_long = create_seq("dna", "A" * (target_len * 2))

    constraint_match = SequenceLengthConstraint(
        inputs=seq_match, target_length=target_len
    )
    constraint_short = SequenceLengthConstraint(
        inputs=seq_short, target_length=target_len
    )
    constraint_long = SequenceLengthConstraint(
        inputs=seq_long, target_length=target_len
    )

    assert constraint_match.evaluate() == 0.0
    # Deviation = abs(10 - 20) / 20 = 10 / 20 = 0.5.
    assert abs(constraint_short.evaluate() - 0.5) < 1e-9
    # Deviation = abs(40 - 20) / 20 = 20 / 20 = 1.0.
    assert abs(constraint_long.evaluate() - 1.0) < 1e-9
    assert seq_match._metadata["length"] == target_len
    assert seq_short._metadata["length"] == target_len // 2


def test_gc_content_constraint():
    """Tests GCContentConstraint."""
    target_range = (40.0, 60.0)
    seq_len = 10
    seq_in_range = create_seq("dna", "GCGCGAATTA")  # 5/10 = 50% GC.
    seq_below = create_seq("dna", "GCATTATTAT")  # 2/10 = 20% GC.
    seq_above = create_seq("dna", "GCGCGCGCGT")  # 9/10 = 90% GC.

    constraint_in = GCContentConstraint(inputs=seq_in_range, target_range=target_range)
    constraint_below = GCContentConstraint(inputs=seq_below, target_range=target_range)
    constraint_above = GCContentConstraint(inputs=seq_above, target_range=target_range)

    assert constraint_in.evaluate() == 0.0
    # Deviation = (40 - 20) / 40 = 0.5.
    assert abs(constraint_below.evaluate() - 0.5) < 1e-9
    # Deviation = (90 - 60) / (100 - 60) = 30 / 40 = 0.75.
    assert abs(constraint_above.evaluate() - 0.75) < 1e-9
    assert abs(seq_in_range._metadata["gc_content"] - 50.0) < 1e-9


def test_max_homopolymer_constraint():
    """Tests MaxHomopolymerConstraint."""
    max_len = 4
    seq_ok = create_seq("dna", "AAATTTGGGGCCCC")  # Max is 4.
    seq_long = create_seq("dna", "AAATTTTGGGGGCCC")  # Max T is 5.
    seq_very_long = create_seq("dna", "AAAAAAAATTTT")  # Max A is 8.

    constraint_ok = MaxHomopolymerConstraint(inputs=seq_ok, max_length=max_len)
    constraint_long = MaxHomopolymerConstraint(inputs=seq_long, max_length=max_len)
    constraint_very_long = MaxHomopolymerConstraint(
        inputs=seq_very_long, max_length=max_len
    )

    assert constraint_ok.evaluate() == 0.0
    # Excess = 5 - 4 = 1. Score = log2(1 + 1/4) = log2(1.25) approx 0.32.
    assert abs(constraint_long.evaluate() - np.log2(1 + 1 / 4)) < 1e-9
    # Excess = 8 - 4 = 4. Score = log2(1 + 4/4) = log2(2) = 1.0.
    assert abs(constraint_very_long.evaluate() - 1.0) < 1e-9
    assert seq_ok._metadata["max_homopolymer_length"] == 4
    assert seq_long._metadata["max_homopolymer_length"] == 5


def test_dinucleotide_frequency_constraint():
    """Tests DinucleotideFrequencyConstraint."""
    freq_range_wide = (0., 1.)
    freq_range_narrow = (0.03, 0.08)
    seq_wide = create_seq("dna", "ACGT" * 5)
    seq_narrow = create_seq("dna", "ACGT" * 5)

    constraint_wide = DinucleotideFrequencyConstraint(
        inputs=seq_wide, freq_range=freq_range_wide,
    )
    constraint_narrow = DinucleotideFrequencyConstraint(
        inputs=seq_narrow, freq_range=freq_range_narrow,
    )

    assert constraint_wide.evaluate() == 0.0
    assert constraint_narrow.evaluate() == 1.0


def test_tetranucleotide_usage_constraint():
    """Tests TetranucleotideUsageConstraint."""
    tetranuc = "GATC"
    # Target TUD range: 0.8 to 1.2.
    tud_range = (0.8, 1.2)

    # Sequence with roughly equal base frequencies (should result in TUD near 1).
    seq_balanced = create_seq(
        "dna", "AGCT" * 10 + "GATC" + "AGCT" * 10
    )  # Len 84. One GATC.
    # Sequence with zero GATC occurrences.
    seq_no_gatc = create_seq("dna", "AAAAAAAAAAAAAAAAAAAAAAAAA")  # Len 25.

    constraint_bal = TetranucleotideUsageConstraint(
        inputs=seq_balanced, tetranucleotide=tetranuc, tud_range=tud_range
    )
    constraint_no_gatc = TetranucleotideUsageConstraint(
        inputs=seq_no_gatc, tetranucleotide=tetranuc, tud_range=tud_range
    )

    # Calculate expected TUD for balanced sequence.
    seq_len_bal = len(seq_balanced)
    freq_A = str(seq_balanced).count("A") / seq_len_bal  # 21/84 = 0.25.
    freq_T = str(seq_balanced).count("T") / seq_len_bal  # 21/84 = 0.25.
    freq_C = str(seq_balanced).count("C") / seq_len_bal  # 21/84 = 0.25.
    freq_G = str(seq_balanced).count("G") / seq_len_bal  # 21/84 = 0.25.
    expected_freq = freq_G * freq_A * freq_T * freq_C  # (0.25)^4 = 0.00390625.
    expected_occurrences = expected_freq * (seq_len_bal - 3)  # 0.00390625 * 81 ~ 0.316.
    actual_occurrences = 1
    tud_bal = (
        actual_occurrences / expected_occurrences
    )  # 1 / 0.316 ~ 3.16 (Outside range [0.8, 1.2]).
    # Expected deviation = (tud - max_tud) / max_tud = (3.16 - 1.2) / 1.2 ~ 1.96 / 1.2 ~ 1.63 -> capped at 1.0.
    assert abs(constraint_bal.evaluate() - 1.0) < 1e-9
    assert abs(seq_balanced._metadata["GATC_tud"] - tud_bal) < 1e-9

    # Sequence with no GATC should have TUD of 0, which is outside range [0.8, 1.2].
    # Expected deviation = (min_tud - tud) / min_tud = (0.8 - 0) / 0.8 = 1.0.
    assert abs(constraint_no_gatc.evaluate() - 1.0) < 1e-9
    assert abs(seq_no_gatc._metadata["GATC_tud"] - 0.0) < 1e-9

    # Simple edge case.
    seq_edge_case = create_seq("dna", "GAT")  # len < 4.
    constraint_edge = TetranucleotideUsageConstraint(
        inputs=seq_edge_case, tetranucleotide=tetranuc, tud_range=tud_range
    )
    assert constraint_edge.evaluate() == 0.0  # Score is 0 for len < 4.
    assert seq_edge_case._metadata["GATC_tud"] == 0.0
