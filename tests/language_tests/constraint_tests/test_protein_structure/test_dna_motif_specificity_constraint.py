"""tests/language_tests/constraint_tests/test_protein_structure/test_dna_motif_specificity_constraint.py."""

from itertools import pairwise

import numpy as np
import pytest
from pydantic import ValidationError

from proto_language.constraint.protein_structure.dna_motif_specificity_constraint import (
    DeepPBSMotifSpecificityConfig,
    NAMPNNMotifSpecificityConfig,
    _normalized_cross_entropy,
    _slide_best_logprob,
    _to_ppm_matrix,
    sliding_logprob_advantage,
    sliding_logprob_score,
)


def _onehot_ppm(motif: str) -> np.ndarray:
    """Build a near-deterministic PPM that strongly prefers ``motif`` at each position."""
    order = "ACGT"
    rows = []
    for base in motif:
        row = [0.01, 0.01, 0.01, 0.01]
        row[order.index(base)] = 0.97
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class TestPPMHelpers:
    def test_to_ppm_normalizes_rows(self):
        ppm = _to_ppm_matrix([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 0.0, 0.0]])
        assert np.allclose(ppm.sum(axis=1), 1.0)
        assert ppm[0, 0] == pytest.approx(0.25)
        assert ppm[1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_to_ppm_rejects_wrong_width(self):
        with pytest.raises(ValueError, match=r"\(L, 4\)"):
            _to_ppm_matrix([[0.5, 0.5, 0.0]])

    def test_cross_entropy_zero_for_perfect_match(self):
        ppm = _onehot_ppm("ACGT")
        ce = _normalized_cross_entropy(ppm, "ACGT", [0, 1, 2, 3])
        assert ce == pytest.approx(0.0, abs=0.05)

    def test_cross_entropy_higher_for_mismatch(self):
        ppm = _onehot_ppm("ACGT")
        ce_match = _normalized_cross_entropy(ppm, "ACGT", [0, 1, 2, 3])
        ce_mismatch = _normalized_cross_entropy(ppm, "TGCA", [0, 1, 2, 3])
        assert ce_mismatch > ce_match

    def test_margin_scoring_logic(self):
        # Replicates the constraint's score formula: target should beat off-target.
        ppm = _onehot_ppm("ACGTAC")
        indices = [0, 1, 2, 3, 4, 5]
        target_ce = _normalized_cross_entropy(ppm, "ACGTAC", indices)
        off_ce = _normalized_cross_entropy(ppm, "TGCATG", indices)
        advantage = off_ce - target_ce
        desired_margin = 0.2
        score = float(np.clip((desired_margin - advantage) / desired_margin, 0.0, 1.0))
        assert advantage > 0  # target reads better than the off-target
        assert score < 0.5  # specificity advantage -> low (good) score


class TestConfigValidation:
    def _kwargs(self, **overrides):
        base = {
            "target_motif": "ACGTAC",
            "off_target_motifs": ["TGCATG"],
            "dna_indices": [0, 1, 2, 3, 4, 5],
            "structure_tool": "alphafold3",
        }
        base.update(overrides)
        return base

    def test_valid_config(self):
        cfg = NAMPNNMotifSpecificityConfig(**self._kwargs())
        assert cfg.target_motif == "ACGTAC"
        assert cfg.desired_margin == 1.0  # default tuned for sliding_logprob

    def test_default_structure_tool_is_dna_capable(self):
        # Bare default must not inherit esmfold (cannot fold DNA).
        cfg = NAMPNNMotifSpecificityConfig(target_motif="ACGTAC", off_target_motifs=["TGCATG"])
        assert cfg.structure_tool == "alphafold3"

    def test_target_motif_lowercased_normalized(self):
        cfg = DeepPBSMotifSpecificityConfig(**self._kwargs(target_motif="acgtac"))
        assert cfg.target_motif == "ACGTAC"

    def test_bad_alphabet_raises(self):
        with pytest.raises(ValidationError, match="A, C, G, T"):
            NAMPNNMotifSpecificityConfig(**self._kwargs(target_motif="ACGTAX"))

    def test_dna_indices_optional_for_sliding_logprob(self):
        # Default mode ignores dna_indices, so they may be omitted.
        cfg = NAMPNNMotifSpecificityConfig(target_motif="ACGTAC", off_target_motifs=["TGCATG"])
        assert cfg.scoring_mode == "sliding_logprob"
        assert cfg.dna_indices == []

    def test_dna_indices_required_for_cross_entropy(self):
        with pytest.raises(ValidationError, match="dna_indices is required"):
            NAMPNNMotifSpecificityConfig(
                target_motif="ACGTAC", off_target_motifs=["TGCATG"], scoring_mode="cross_entropy"
            )

    def test_motif_index_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="dna_indices length"):
            NAMPNNMotifSpecificityConfig(**self._kwargs(dna_indices=[0, 1, 2], scoring_mode="cross_entropy"))

    def test_off_target_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="match target_motif length"):
            NAMPNNMotifSpecificityConfig(**self._kwargs(off_target_motifs=["TGC"]))

    def test_duplicate_dna_indices_raises(self):
        with pytest.raises(ValidationError, match="unique"):
            NAMPNNMotifSpecificityConfig(**self._kwargs(dna_indices=[0, 1, 2, 3, 4, 4]))

    def test_non_positive_margin_raises(self):
        with pytest.raises(ValidationError):
            NAMPNNMotifSpecificityConfig(**self._kwargs(desired_margin=0.0))


class TestSlidingLogprob:
    def test_slide_best_logprob_finds_best_window(self):
        # Motif embedded mid-PPM; best window should align to that offset (near log(0.97)).
        ppm = _onehot_ppm("TTACGTTT")
        lp = _slide_best_logprob(ppm, "ACGT")
        assert lp == pytest.approx(float(np.log(0.97)), abs=0.01)

    def test_slide_best_logprob_mismatch_is_lower(self):
        ppm = _onehot_ppm("ACGTAC")
        assert _slide_best_logprob(ppm, "ACGT") > _slide_best_logprob(ppm, "TTTT")

    def test_slide_best_logprob_empty_motif(self):
        assert _slide_best_logprob(_onehot_ppm("ACGT"), "") == float("-inf")

    def test_slide_best_logprob_motif_longer_than_ppm(self):
        assert _slide_best_logprob(_onehot_ppm("AC"), "ACGT") == float("-inf")

    def test_advantage_positive_when_target_reads_better(self):
        ppm = _onehot_ppm("ACGTAC")
        adv = sliding_logprob_advantage(ppm, "ACGTAC", ["TGCATG"])
        assert adv["advantage"] > 0
        assert adv["target_lp"] == pytest.approx(float(np.log(0.97)), abs=0.01)

    def test_score_monotonic_decreasing_in_advantage(self):
        scores = [sliding_logprob_score(adv, desired_margin=1.0) for adv in (-2.0, -0.5, 0.0, 0.5, 2.0)]
        assert all(earlier > later for earlier, later in pairwise(scores))

    def test_score_clamped_to_unit_interval(self):
        assert sliding_logprob_score(50.0, desired_margin=1.0) == 0.0
        assert sliding_logprob_score(-50.0, desired_margin=1.0) == 1.0

    def test_score_half_at_zero_advantage(self):
        assert sliding_logprob_score(0.0, desired_margin=1.0) == pytest.approx(0.5)
