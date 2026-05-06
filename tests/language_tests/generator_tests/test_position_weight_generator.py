"""Tests for the PositionWeightGenerator."""

import copy

import numpy as np
import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import (
    GeneratorRegistry,
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    SequenceLogitBiasConfig,
)


class TestPositionWeightGenerator:
    @pytest.mark.parametrize(
        ("sequence_type", "logits", "expected_sequence"),
        [
            ("dna", np.array([[5.0, 0, 0, 0], [0, 4.0, 0, 0]]), "AC"),
            ("rna", np.array([[0, 0, 5.0, 0], [0, 0, 0, 4.0]]), "GU"),
            ("protein", np.pad(np.array([[7.0, 0], [0, 6.0]]), ((0, 0), (0, 18))), "AC"),
        ],
    )
    def test_argmax_decoding(self, sequence_type, logits, expected_sequence):
        """Argmax decoding maps columns to canonical vocab."""
        segment = Segment(sequence="AA", sequence_type=sequence_type)
        segment.proposal_sequences[0].logits = logits
        generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
        generator.assign(segment)

        generator.sample()

        assert segment.proposal_sequences[0].sequence == expected_sequence

    def test_categorical_sampling(self):
        """Categorical sampling is seeded, fills all proposals, and converges to argmax at low temperature."""
        logits = np.array([[0.1, 2.0, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [2.0, 0.1, 0.1, 0.1]])
        results = []
        for _ in range(2):
            segment = Segment(sequence="AAA", sequence_type="dna")
            segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(3)]
            for seq in segment.proposal_sequences:
                seq.logits = logits.copy()
            gen = PositionWeightGenerator(PositionWeightGeneratorConfig(sampling_mode="categorical"))
            gen._set_program_seed(7)
            gen.assign(segment)
            gen.sample()
            results.append([p.sequence for p in segment.proposal_sequences])

        assert results[0] == results[1]
        assert len(results[0]) == 3
        assert all(len(s) == 3 for s in results[0])
        assert len(set(results[0])) > 1

        # Low temperature should converge to argmax
        logits = np.array([[1.0, 2.0, 0.5, 0.1], [0.3, 0.1, 3.0, 0.5]])

        argmax_seg = Segment(sequence="AA", sequence_type="dna")
        argmax_seg.proposal_sequences[0].logits = logits
        argmax_gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        argmax_gen.assign(argmax_seg)
        argmax_gen.sample()

        low_temp_seg = Segment(sequence="AA", sequence_type="dna")
        low_temp_seg.proposal_sequences[0].logits = logits
        low_temp_gen = PositionWeightGenerator(
            PositionWeightGeneratorConfig(sampling_mode="categorical", temperature=0.01)
        )
        low_temp_gen._set_program_seed(42)
        low_temp_gen.assign(low_temp_seg)
        low_temp_gen.sample()

        assert argmax_seg.proposal_sequences[0].sequence == low_temp_seg.proposal_sequences[0].sequence

    def test_argmax_decoding_respects_logit_bias_and_scale(self):
        """Decoding uses the configured effective logits, not raw proposal logits."""
        logits = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0]])
        bias = [[0, 0, 3.0, 0], [3.0, 0, 0, 0]]

        biased_seg = Segment(sequence="AA", sequence_type="dna")
        biased_seg.proposal_sequences[0].logits = logits
        biased_gen = PositionWeightGenerator(PositionWeightGeneratorConfig(logit_bias=bias))
        biased_gen.assign(biased_seg)
        biased_gen.sample()

        scaled_seg = Segment(sequence="AA", sequence_type="dna")
        scaled_seg.proposal_sequences[0].logits = logits
        scaled_gen = PositionWeightGenerator(PositionWeightGeneratorConfig(logit_bias=bias, logit_scale=2.0))
        scaled_gen.assign(scaled_seg)
        scaled_gen.sample()

        assert biased_seg.proposal_sequences[0].sequence == "GA"
        assert scaled_seg.proposal_sequences[0].sequence == "AC"

    def test_sequence_bias_applies_to_cross_type_decoding(self):
        """Declarative sequence_bias works with the cross-type position-weight generator."""
        dna_segment = Segment(sequence="AA", sequence_type="dna")
        dna_segment.proposal_sequences[0].logits = np.zeros((2, 4))
        dna_gen = PositionWeightGenerator(
            PositionWeightGeneratorConfig(
                sequence_bias=SequenceLogitBiasConfig(
                    reference_sequence="AT",
                    reference_bias=5.0,
                    unbiased_positions=[1],
                    excluded_symbols=["A"],
                )
            )
        )
        dna_gen.assign(dna_segment)
        dna_gen.sample()

        rna_segment = Segment(sequence="AA", sequence_type="rna")
        rna_segment.proposal_sequences[0].logits = np.zeros((2, 4))
        rna_gen = PositionWeightGenerator(
            PositionWeightGeneratorConfig(
                sequence_bias=SequenceLogitBiasConfig(reference_sequence="AU", reference_bias=5.0)
            )
        )
        rna_gen.assign(rna_segment)
        rna_gen.sample()

        assert dna_segment.proposal_sequences[0].sequence == "AC"
        assert rna_segment.proposal_sequences[0].sequence == "AU"

    def test_registry_and_ligand_rejection(self):
        """Generator is discoverable via registry and rejects ligand segments."""
        generator = GeneratorRegistry.create("position-weight", {"sampling_mode": "argmax"})
        assert isinstance(generator, PositionWeightGenerator)

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            generator.assign(Segment(sequence="CCC", sequence_type="ligand"))

    def test_argmax_identical_across_proposals(self):
        """Argmax is deterministic — all proposals receive the same sequence."""
        segment = Segment(sequence="ACG", sequence_type="dna")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        logits = np.array([[5, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0]])
        for seq in segment.proposal_sequences:
            seq.logits = logits.copy()
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(segment)

        gen.sample()

        sequences = [p.sequence for p in segment.proposal_sequences]
        assert all(s == sequences[0] for s in sequences)

    @pytest.mark.parametrize(
        ("logits", "error_match"),
        [
            (np.ones((2, 2)), "expected shape"),
            (np.ones((8,)), "2D array"),
            (np.array([[1.0, np.inf, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]), "finite values"),
            (np.array([[1.0, np.nan, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]), "finite values"),
        ],
    )
    def test_sample_validation(self, logits, error_match):
        """Sampling fails fast on invalid logits."""
        segment = Segment(sequence="AA", sequence_type="dna")
        segment.proposal_sequences[0]._logits = logits  # bypass setter to test generator-level checks
        generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
        generator.assign(segment)

        with pytest.raises(ValueError, match=error_match):
            generator.sample()

    def test_no_logits_raises(self):
        """sample() raises when proposal has no logits."""
        segment = Segment(sequence="AA", sequence_type="dna")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(segment)

        with pytest.raises(RuntimeError, match="has no logits"):
            gen.sample()

    def test_custom_valid_chars(self):
        """Custom valid_chars outside the canonical alphabet are included in vocab ordering."""
        segment = Segment(sequence="AX", sequence_type="dna", valid_chars={"A", "C", "G", "T", "X"})
        segment.proposal_sequences[0].logits = np.array([[0, 0, 0, 0, 10.0], [0, 0, 0, 0, 10.0]])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(segment)

        gen.sample()

        assert segment.proposal_sequences[0].sequence == "XX"

    def test_program_seed_reproducibility(self):
        """_set_program_seed resets categorical sampling for reproducible re-runs."""
        logits = np.zeros((3, 4))  # uniform after softmax
        segment = Segment(sequence="AAA", sequence_type="dna")
        segment.proposal_sequences[0].logits = logits
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig(sampling_mode="categorical"))
        gen.assign(segment)
        results = []
        for _ in range(2):
            gen._set_program_seed(42)
            gen.sample()
            results.append(segment.proposal_sequences[0].sequence)
        assert results[0] == results[1]

    def test_sample_before_assign(self):
        """Calling sample() before assign() raises RuntimeError."""
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        with pytest.raises(RuntimeError):
            gen.sample()

    @pytest.mark.parametrize(
        ("entropy_positions", "expected"),
        [(None, 0.5), ([1, 2], 0.25)],
        ids=["all_positions", "restricted"],
    )
    def test_mean_peak_probability_respects_entropy_positions(self, entropy_positions, expected):
        """sample() writes per-proposal mean peak-probability, optionally restricted by entropy_positions."""
        # Row 0 sharp (peak ≈ 0.99995); rows 1-2 uniform (peak = 0.25 for 4-base DNA).
        segment = Segment(sequence="AAA", sequence_type="dna")
        segment.proposal_sequences[0].logits = np.array([[10.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig(entropy_positions=entropy_positions))
        gen.assign(segment)
        gen.sample()
        assert segment.proposal_sequences[0]._generator_metadata["position-weight"][
            "mean_peak_probability"
        ] == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize("bad_value", [[-1], []], ids=["negative", "empty"])
    def test_entropy_positions_config_validation(self, bad_value):
        """Pydantic rejects negative + empty entropy_positions before the generator is built."""
        with pytest.raises(ValueError, match="entropy_positions"):
            PositionWeightGeneratorConfig(entropy_positions=bad_value)

    @pytest.mark.parametrize("bad_value", [[5], [2]], ids=["beyond_len", "at_boundary"])
    def test_entropy_positions_out_of_range_raises_at_sample_time(self, bad_value):
        """Segment-length bounds are checked at sample() with a strict ``>=`` (not ``>``)."""
        segment = Segment(sequence="AA", sequence_type="dna")  # sequence_length=2 → valid rows are [0, 1]
        segment.proposal_sequences[0].logits = np.zeros((2, 4))
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig(entropy_positions=bad_value))
        gen.assign(segment)
        with pytest.raises(ValueError, match="entropy_positions"):
            gen.sample()
