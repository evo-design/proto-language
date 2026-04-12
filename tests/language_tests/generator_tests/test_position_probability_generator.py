"""Tests for the PositionProbabilityGenerator."""

import copy

import numpy as np
import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import (
    GeneratorRegistry,
    PositionProbabilityGenerator,
    PositionProbabilityGeneratorConfig,
)


class TestPositionProbabilityGenerator:
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
        generator = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig())
        generator.assign(segment)

        generator.sample(logits=logits)

        assert segment.proposal_sequences[0].sequence == expected_sequence

    def test_categorical_sampling(self):
        """Categorical sampling is seeded, fills all proposals, and converges to argmax at low temperature."""
        logits = np.array([[0.1, 2.0, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [2.0, 0.1, 0.1, 0.1]])
        results = []
        for _ in range(2):
            segment = Segment(sequence="AAA", sequence_type="dna")
            segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(3)]
            gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig(sampling_mode="categorical"))
            gen._set_program_seed(7)
            gen.assign(segment)
            gen.sample(logits=logits)
            results.append([p.sequence for p in segment.proposal_sequences])

        assert results[0] == results[1]
        assert len(results[0]) == 3
        assert all(len(s) == 3 for s in results[0])
        assert len(set(results[0])) > 1

        # Low temperature should converge to argmax
        logits = np.array([[1.0, 2.0, 0.5, 0.1], [0.3, 0.1, 3.0, 0.5]])

        argmax_seg = Segment(sequence="AA", sequence_type="dna")
        argmax_gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig())
        argmax_gen.assign(argmax_seg)
        argmax_gen.sample(logits=logits)

        low_temp_seg = Segment(sequence="AA", sequence_type="dna")
        low_temp_gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig(sampling_mode="categorical"))
        low_temp_gen._set_program_seed(42)
        low_temp_gen.assign(low_temp_seg)
        low_temp_gen.sample(logits=logits, temperature=0.01)

        assert argmax_seg.proposal_sequences[0].sequence == low_temp_seg.proposal_sequences[0].sequence

    def test_registry_and_ligand_rejection(self):
        """Generator is discoverable via registry and rejects ligand segments."""
        generator = GeneratorRegistry.create("position-probability", {"sampling_mode": "argmax"})
        assert isinstance(generator, PositionProbabilityGenerator)

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            generator.assign(Segment(sequence="CCC", sequence_type="ligand"))

    def test_argmax_identical_across_proposals(self):
        """Argmax is deterministic — all proposals receive the same sequence."""
        segment = Segment(sequence="ACG", sequence_type="dna")
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(5)]
        gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig())
        gen.assign(segment)

        gen.sample(logits=np.array([[5, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0]]))

        sequences = [p.sequence for p in segment.proposal_sequences]
        assert all(s == sequences[0] for s in sequences)

    @pytest.mark.parametrize(
        ("sample_kwargs", "error_match"),
        [
            ({"logits": np.ones((2, 2))}, "expected shape"),
            ({"logits": np.ones((8,))}, "2D array"),
            ({"logits": np.array([[1.0, np.inf, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])}, "finite values"),
            ({"logits": np.array([[1.0, np.nan, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])}, "finite values"),
            ({"logits": np.ones((2, 4)), "temperature": 0.0}, "positive"),
            ({"logits": np.ones((2, 4)), "temperature": -1.0}, "positive"),
        ],
    )
    def test_sample_validation(self, sample_kwargs, error_match):
        """Sampling fails fast on invalid inputs."""
        generator = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig())
        generator.assign(Segment(sequence="AA", sequence_type="dna"))

        with pytest.raises(ValueError, match=error_match):
            generator.sample(**sample_kwargs)

    def test_custom_valid_chars(self):
        """Custom valid_chars outside the canonical alphabet are included in vocab ordering."""
        segment = Segment(sequence="AX", sequence_type="dna", valid_chars={"A", "C", "G", "T", "X"})
        gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig())
        gen.assign(segment)

        # Vocab should be [A, C, G, T, X] — canonical first, then extras sorted
        # Column 4 (index 4) maps to "X"
        logits = np.array([[0, 0, 0, 0, 10.0], [0, 0, 0, 0, 10.0]])
        gen.sample(logits=logits)

        assert segment.proposal_sequences[0].sequence == "XX"

    def test_program_seed_reproducibility(self):
        """_set_program_seed resets categorical sampling for reproducible re-runs."""
        logits = np.zeros((3, 4))  # uniform after softmax
        results = []
        gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig(sampling_mode="categorical"))
        segment = Segment(sequence="AAA", sequence_type="dna")
        gen.assign(segment)
        for _ in range(2):
            gen._set_program_seed(42)
            gen.sample(logits=logits)
            results.append(segment.proposal_sequences[0].sequence)
        assert results[0] == results[1]

    def test_sample_before_assign(self):
        """Calling sample() before assign() raises RuntimeError."""
        gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig())
        with pytest.raises(RuntimeError):
            gen.sample(logits=np.ones((2, 4)))
