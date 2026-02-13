"""Tests for MSAGenerator."""

from __future__ import annotations

import copy
import random

import pytest
from proto_tools import MSA

from proto_language.language.core import Segment
from proto_language.language.generator import MSAGenerator, MSAGeneratorConfig


class TestMSAGeneratorConfig:
    """Tests for MSAGeneratorConfig validation."""

    def test_valid_config_with_list(self):
        """Test valid configuration with list of sequences (auto-coerced to MSA)."""
        config = MSAGeneratorConfig(
            msa=["MVLS", "AVLS", "MVLS"],
            num_mutations=1,
        )
        assert config.msa.num_sequences == 3
        assert config.num_mutations == 1
        assert config.include_gaps is False

    def test_valid_config_with_msa(self):
        """Test valid configuration with MSA object."""
        msa = MSA(["MVLS", "AVLS", "MVLS"])
        config = MSAGeneratorConfig(msa=msa, num_mutations=1)
        assert config.msa.num_sequences == 3
        assert config.msa is msa

    def test_default_values(self):
        """Test default configuration values."""
        config = MSAGeneratorConfig(msa=["MVLS", "AVLS"])
        assert config.num_mutations == 1
        assert config.include_gaps is False

    def test_include_gaps_option(self):
        """Test include_gaps configuration option."""
        config = MSAGeneratorConfig(
            msa=["MV-S", "AVLS"],
            include_gaps=True,
        )
        assert config.include_gaps is True

    @pytest.mark.parametrize(
        "sequences,error_match",
        [
            ([], "MSA must contain at least two sequences"),
            (["MVLS", 123], "is not a string"),
            (["MVLS", ""], "same length"),
            (["MVLS", "AV"], "same length"),
        ],
    )
    def test_invalid_aligned_sequences(self, sequences, error_match):
        """Test validation errors for invalid aligned sequences."""
        with pytest.raises(ValueError, match=error_match):
            MSAGeneratorConfig(msa=sequences)

    def test_invalid_num_mutations(self):
        """Test validation for num_mutations."""
        with pytest.raises(ValueError):
            MSAGeneratorConfig(msa=["MVLS"], num_mutations=0)


class TestMSAGeneratorProbabilityCalculation:
    """Tests for position probability calculation."""

    def test_uniform_distribution(self):
        """Test probability calculation with uniform character distribution."""
        config = MSAGeneratorConfig(
            msa=["AAAA", "CCCC", "GGGG", "TTTT"],
        )
        gen = MSAGenerator(config)

        # Each position should have 25% probability for each character
        for pos in range(4):
            probs = gen.position_probs[pos]
            assert len(probs) == 4
            assert probs["A"] == pytest.approx(0.25)
            assert probs["C"] == pytest.approx(0.25)
            assert probs["G"] == pytest.approx(0.25)
            assert probs["T"] == pytest.approx(0.25)

    def test_conserved_position(self):
        """Test probability calculation for a conserved position."""
        config = MSAGeneratorConfig(
            msa=["MVLS", "MVLS", "MVLS"],
        )
        gen = MSAGenerator(config)

        # Position 0 should have 100% M
        assert gen.position_probs[0] == {"M": 1.0}
        # Position 1 should have 100% V
        assert gen.position_probs[1] == {"V": 1.0}

    def test_variable_position(self):
        """Test probability calculation for a variable position."""
        config = MSAGeneratorConfig(
            msa=["MVLS", "AVLS", "MVLS"],  # Position 0: 2 M, 1 A
        )
        gen = MSAGenerator(config)

        assert gen.position_probs[0]["M"] == pytest.approx(2 / 3)
        assert gen.position_probs[0]["A"] == pytest.approx(1 / 3)

    def test_gaps_excluded_by_default(self):
        """Test that gaps are excluded from probability calculation by default."""
        config = MSAGeneratorConfig(
            msa=["M-LS", "A-LS", "M-LS"],  # Position 1 is all gaps
        )
        gen = MSAGenerator(config)

        # Position 1 should be None (all gaps)
        assert gen.position_probs[1] is None
        # Position 1 should not be in mutable_positions
        assert 1 not in gen.mutable_positions

    def test_gaps_included_when_enabled(self):
        """Test that gaps are included when include_gaps=True."""
        config = MSAGeneratorConfig(
            msa=["M-LS", "AVLS"],
            include_gaps=True,
        )
        gen = MSAGenerator(config)

        # Position 1 should include gap
        assert gen.position_probs[1]["-"] == pytest.approx(0.5)
        assert gen.position_probs[1]["V"] == pytest.approx(0.5)

    def test_all_gaps_position(self):
        """Test handling of position with all gaps (division by zero prevention)."""
        config = MSAGeneratorConfig(
            msa=["M-LS", "A-LS", "M-LS"],
        )
        gen = MSAGenerator(config)

        # Position 1 (all gaps) should be None
        assert gen.position_probs[1] is None
        assert 1 not in gen.mutable_positions

    def test_mutable_positions_computed(self):
        """Test that mutable_positions is correctly computed."""
        config = MSAGeneratorConfig(
            msa=["M--S", "A--S"],  # Positions 1, 2 are all gaps
        )
        gen = MSAGenerator(config)

        assert gen.mutable_positions == [0, 3]
        assert len(gen.mutable_positions) == 2


class TestMSAGeneratorAssign:
    """Tests for MSAGenerator.assign method."""

    def test_assign_matching_length(self):
        """Test successful assignment with matching lengths."""
        config = MSAGeneratorConfig(msa=["MVLS", "AVLS"])
        gen = MSAGenerator(config)
        segment = Segment(sequence="MVLS", sequence_type="protein")

        gen.assign(segment)

        assert gen._assigned_segment is segment

    def test_assign_length_mismatch(self):
        """Test that assign raises error for length mismatch."""
        config = MSAGeneratorConfig(msa=["MVLS", "AVLS"])  # Length 4
        gen = MSAGenerator(config)
        segment = Segment(sequence="MVLSPADKTN", sequence_type="protein")  # Length 10

        with pytest.raises(ValueError, match="alignment length.*must match.*segment length"):
            gen.assign(segment)

    def test_assign_all_gaps_rejected(self):
        """Test that MSA with all positions being gaps is rejected."""
        config = MSAGeneratorConfig(msa=["----", "----"])
        gen = MSAGenerator(config)
        segment = Segment(sequence="MVLS", sequence_type="protein")

        with pytest.raises(ValueError, match="No mutable positions"):
            gen.assign(segment)


class TestMSAGeneratorSample:
    """Tests for MSAGenerator.sample method."""

    def test_sample_single_mutation(self):
        """Test that sample introduces exactly one mutation."""
        config = MSAGeneratorConfig(
            msa=["AAAA", "CCCC"],
            num_mutations=1,
        )
        gen = MSAGenerator(config)
        segment = Segment(sequence="GGGG", sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence)]
        initial = segment.candidate_sequences[0].sequence
        gen.sample()
        mutated = segment.candidate_sequences[0].sequence

        diff_count = sum(1 for a, b in zip(initial, mutated) if a != b)
        assert diff_count == 1

    def test_sample_multiple_mutations(self):
        """Test that sample introduces the specified number of mutations."""
        config = MSAGeneratorConfig(
            msa=["AAAA", "CCCC"],
            num_mutations=3,
        )
        gen = MSAGenerator(config)
        segment = Segment(sequence="GGGG", sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence)]
        initial = segment.candidate_sequences[0].sequence
        gen.sample()
        mutated = segment.candidate_sequences[0].sequence

        diff_count = sum(1 for a, b in zip(initial, mutated) if a != b)
        assert diff_count == 3

    def test_sample_respects_distribution(self):
        """Test that sampling respects the MSA distribution."""
        # Create MSA where position 0 is 90% A, 10% C
        aligned_seqs = ["A" + "GGG"] * 9 + ["C" + "GGG"]
        config = MSAGeneratorConfig(msa=aligned_seqs, num_mutations=1)
        gen = MSAGenerator(config)
        segment = Segment(sequence="MMMM", sequence_type="protein")
        gen.assign(segment)

        # Sample many times and count outcomes at position 0
        random.seed(42)
        a_count = 0
        c_count = 0
        trials = 1000

        for _ in range(trials):
            segment.candidate_sequences = [copy.deepcopy(segment.original_sequence)]
            gen.sample()
            mutated = segment.candidate_sequences[0].sequence
            # Find which position was mutated
            for i, (orig, mut) in enumerate(zip("XGGG", mutated)):
                if orig != mut:
                    if i == 0:
                        if mut == "A":
                            a_count += 1
                        elif mut == "C":
                            c_count += 1
                    break

        # A should be much more common than C (roughly 9:1 ratio)
        # Position 0 should be mutated 25% of the time (1 in 4 positions)
        # So we expect roughly 0.25 * 1000 * 0.9 = 225 A's
        total_pos0_mutations = a_count + c_count
        if total_pos0_mutations > 0:
            a_ratio = a_count / total_pos0_mutations
            assert a_ratio > 0.7  # Should be close to 0.9 but with some variance

    def test_sample_batch_candidates(self):
        """Test that sample mutates all candidates independently."""
        config = MSAGeneratorConfig(
            msa=["AAAA", "CCCC"],
            num_mutations=1,
        )
        gen = MSAGenerator(config)
        segment = Segment(sequence="GGGG", sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [
            copy.deepcopy(segment.original_sequence) for _ in range(5)
        ]
        gen.sample()
        mutated_seqs = [s.sequence for s in segment.candidate_sequences]

        # Each should have exactly 1 mutation
        for seq in mutated_seqs:
            diff_count = sum(1 for a, b in zip("GGGG", seq) if a != b)
            assert diff_count == 1

        # Mutations should be independent (high probability of different outcomes)
        assert len(set(mutated_seqs)) > 1

    def test_sample_skips_all_gap_positions(self):
        """Test that sample only mutates positions with non-gap characters."""
        config = MSAGeneratorConfig(
            msa=["A--G", "C--G"],  # Positions 1, 2 are all gaps
            num_mutations=2,
        )
        gen = MSAGenerator(config)
        segment = Segment(sequence="GGGG", sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence)]
        gen.sample()
        mutated = segment.candidate_sequences[0].sequence

        # Positions 1 and 2 should be unchanged (all gaps in MSA)
        assert mutated[1] == "G"
        assert mutated[2] == "G"
        # Positions 0 and 3 should be mutated (to A/C and G)
        assert mutated[0] in ["A", "C"]
        assert mutated[3] == "G"

    def test_sample_caps_mutations_to_mutable_positions(self):
        """Test that num_mutations is capped at available mutable positions."""
        config = MSAGeneratorConfig(
            msa=["A---", "C---"],  # Only position 0 is mutable
            num_mutations=10,
        )
        gen = MSAGenerator(config)
        segment = Segment(sequence="GGGG", sequence_type="dna")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence)]
        gen.sample()
        mutated = segment.candidate_sequences[0].sequence

        # Only position 0 can be mutated
        diff_count = sum(1 for a, b in zip("GGGG", mutated) if a != b)
        assert diff_count == 1
        assert mutated[0] in ["A", "C"]

    def test_deterministic_with_seed(self):
        """Test reproducibility with fixed random seed."""

        def run_with_seed(seed):
            random.seed(seed)
            config = MSAGeneratorConfig(
                msa=["AAAA", "CCCC", "GGGG", "TTTT"],
                num_mutations=2,
            )
            gen = MSAGenerator(config)
            segment = Segment(sequence="MMMM", sequence_type="protein")
            gen.assign(segment)
            segment.candidate_sequences = [copy.deepcopy(segment.original_sequence)]
            gen.sample()
            return segment.candidate_sequences[0].sequence

        seq1 = run_with_seed(42)
        seq2 = run_with_seed(42)
        seq3 = run_with_seed(123)

        assert seq1 == seq2
        assert seq1 != seq3


class TestMSAGeneratorSequenceTypes:
    """Tests for sequence type compatibility."""

    def test_accepts_dna_segment(self):
        """MSAGenerator should accept DNA segments."""
        config = MSAGeneratorConfig(msa=["ACGT", "ACGT"])
        gen = MSAGenerator(config)
        segment = Segment(sequence="ACGT", sequence_type="dna")

        gen.assign(segment)
        assert gen._assigned_segment is segment

    def test_accepts_rna_segment(self):
        """MSAGenerator should accept RNA segments."""
        config = MSAGeneratorConfig(msa=["ACGU", "ACGU"])
        gen = MSAGenerator(config)
        segment = Segment(sequence="ACGU", sequence_type="rna")

        gen.assign(segment)
        assert gen._assigned_segment is segment

    def test_accepts_protein_segment(self):
        """MSAGenerator should accept protein segments."""
        config = MSAGeneratorConfig(msa=["MVLS", "AVLS"])
        gen = MSAGenerator(config)
        segment = Segment(sequence="MVLS", sequence_type="protein")

        gen.assign(segment)
        assert gen._assigned_segment is segment


class TestMSAGeneratorRegistry:
    """Tests for generator registry integration."""

    def test_registered_in_registry(self):
        """Test that MSAGenerator is registered in GeneratorRegistry."""
        from proto_language.language.generator import GeneratorRegistry

        spec = GeneratorRegistry.get("msa")
        assert spec.key == "msa"
        assert spec.label == "MSA Generator"
        assert spec.category == "mutation"
        assert spec.requires_gpu is False

    def test_create_from_registry(self):
        """Test creating MSAGenerator via registry."""
        from proto_language.language.generator import GeneratorRegistry

        gen = GeneratorRegistry.create(
            "msa",
            {"msa": ["MVLS", "AVLS"], "num_mutations": 2},
        )
        assert isinstance(gen, MSAGenerator)
        assert gen.num_mutations == 2


class TestMSAModel:
    """Tests for the base MSA model in tool_io."""

    def test_msa_creation(self):
        """Test basic MSA creation."""
        msa = MSA(["MVLS", "AVLS", "MVLS"])
        assert msa.num_sequences == 3
        assert msa.alignment_length == 4

    def test_msa_iteration(self):
        """Test iterating over MSA sequences."""
        msa = MSA(["MVLS", "AVLS"])
        seqs = list(msa)
        assert seqs == ["MVLS", "AVLS"]

    def test_msa_indexing(self):
        """Test indexing MSA sequences."""
        msa = MSA(["MVLS", "AVLS"])
        assert msa[0] == "MVLS"
        assert msa[1] == "AVLS"

    def test_msa_get_column(self):
        """Test get_column method."""
        msa = MSA(["MVLS", "AVLS", "MVLS"])
        assert msa.get_column(0) == ["M", "A", "M"]
        assert msa.get_column(1) == ["V", "V", "V"]

    def test_msa_get_conservation(self):
        """Test conservation calculation."""
        msa = MSA(["MVLS", "AVLS", "MVLS"])
        # Position 0: M=2, A=1, conservation = 2/3
        assert msa.get_conservation(0) == pytest.approx(2 / 3)
        # Position 1: all V, conservation = 1.0
        assert msa.get_conservation(1) == 1.0

    def test_msa_get_position_frequencies(self):
        """Test position frequency calculation."""
        msa = MSA(["M-LS", "AVLS"])
        # Position 1 without gaps
        freqs = msa.get_position_frequencies(1, include_gaps=False)
        assert freqs == {"V": 1.0}
        # Position 1 with gaps
        freqs = msa.get_position_frequencies(1, include_gaps=True)
        assert freqs["-"] == pytest.approx(0.5)
        assert freqs["V"] == pytest.approx(0.5)

    def test_msa_gap_statistics(self):
        """Test gap statistics."""
        msa = MSA(["M--S", "AVLS"])
        assert msa.total_gaps == 2
        # Seq 0: 2/4 = 0.5 gaps, Seq 1: 0/4 = 0 gaps, avg = 0.25
        assert msa.average_gap_fraction == pytest.approx(0.25)

    def test_msa_to_fasta(self):
        """Test FASTA conversion."""
        # Default sequence IDs (seq_0, seq_1, etc.)
        msa = MSA(["MVLS", "AVLS"])
        fasta = msa.to_fasta_string()
        assert fasta == ">seq_0\nMVLS\n>seq_1\nAVLS"
        # With custom IDs set at construction
        msa_with_ids = MSA(
            ["MVLS", "AVLS"],
            sequence_ids=["protein_a", "protein_b"],
        )
        fasta = msa_with_ids.to_fasta_string()
        assert fasta == ">protein_a\nMVLS\n>protein_b\nAVLS"
