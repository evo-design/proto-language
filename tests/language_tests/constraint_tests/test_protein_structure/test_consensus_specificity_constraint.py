"""tests/language_tests/constraint_tests/test_protein_structure/test_consensus_specificity_constraint.py."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from proto_language.constraint.protein_structure import consensus_specificity_constraint as csc
from proto_language.constraint.protein_structure.consensus_specificity_constraint import (
    ConsensusSpecificityConfig,
    consensus_operator_specificity_constraint,
    consensus_score,
    load_reference_stats,
    z_score,
)
from proto_language.constraint.protein_structure.dna_motif_specificity_constraint import (
    sliding_logprob_advantage,
    sliding_logprob_score,
)
from proto_language.core import Sequence


def _onehot_ppm(motif: str) -> np.ndarray:
    """Build a near-deterministic PPM that strongly prefers ``motif`` at each position."""
    order = "ACGT"
    rows = []
    for base in motif:
        row = [0.01, 0.01, 0.01, 0.01]
        row[order.index(base)] = 0.97
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class TestSlidingLogProb:
    def test_advantage_positive_for_target_match(self):
        ppm = _onehot_ppm("ACGTAC")
        adv = sliding_logprob_advantage(ppm, "ACGTAC", ["TGCATG"])
        assert adv["advantage"] > 0.0
        assert adv["target_lp"] > adv["best_off_lp"]

    def test_advantage_negative_when_offtarget_reads_better(self):
        ppm = _onehot_ppm("TGCATG")
        adv = sliding_logprob_advantage(ppm, "ACGTAC", ["TGCATG"])
        assert adv["advantage"] < 0.0

    def test_slide_finds_best_window(self):
        # Motif sits in the middle of a longer PPM; sliding should find it.
        ppm = _onehot_ppm("AAACGTAA")
        adv = sliding_logprob_advantage(ppm, "ACGT", ["TTTT"])
        # Perfect window exists for ACGT -> avg log-prob ~ log(0.97).
        assert adv["target_lp"] == pytest.approx(float(np.log(0.97)), abs=0.01)

    def test_motif_longer_than_ppm_is_neg_inf(self):
        ppm = _onehot_ppm("AC")
        adv = sliding_logprob_advantage(ppm, "ACGT", ["ACGT"])
        assert adv["target_lp"] == float("-inf")

    def test_score_monotonic_in_advantage(self):
        s_pos = sliding_logprob_score(2.0, 1.0)
        s_zero = sliding_logprob_score(0.0, 1.0)
        s_neg = sliding_logprob_score(-2.0, 1.0)
        assert s_pos < s_zero < s_neg
        assert s_zero == pytest.approx(0.5)
        assert 0.0 <= s_pos <= 1.0 and 0.0 <= s_neg <= 1.0


class TestZScore:
    def test_basic_z(self):
        assert z_score(2.0, 0.0, 1.0) == pytest.approx(2.0)
        assert z_score(0.5, 0.5, 0.25) == pytest.approx(0.0)

    def test_zero_std_returns_zero(self):
        assert z_score(5.0, 1.0, 0.0) == 0.0
        assert z_score(5.0, 1.0, -1.0) == 0.0

    def test_nan_std_returns_zero(self):
        assert z_score(5.0, 1.0, float("nan")) == 0.0


class TestConsensusScore:
    def test_large_positive_consensus_low_score(self):
        assert consensus_score(10.0) < 0.05

    def test_negative_consensus_high_score(self):
        assert consensus_score(-10.0) > 0.95

    def test_zero_consensus_half(self):
        assert consensus_score(0.0) == pytest.approx(0.5)

    def test_monotonic(self):
        assert consensus_score(3.0) < consensus_score(1.0) < consensus_score(-1.0)


class TestLoadReferenceStats:
    def test_loads_mean_std(self, tmp_path):
        ref = {"na_mpnn": {"mean": 0.5, "std": 1.2}, "deeppbs": {"mean": 0.3, "std": 0.33}}
        path = tmp_path / "ref.json"
        path.write_text(json.dumps(ref))
        stats = load_reference_stats(str(path))
        assert stats["na_mpnn"]["mean"] == pytest.approx(0.5)
        assert stats["deeppbs"]["std"] == pytest.approx(0.33)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_reference_stats(str(tmp_path / "nope.json"))

    def test_missing_key_raises(self, tmp_path):
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({"na_mpnn": {"mean": 0.0, "std": 1.0}}))
        with pytest.raises(ValueError, match="deeppbs"):
            load_reference_stats(str(path))

    def test_shipped_reference_is_valid(self):
        cfg = ConsensusSpecificityConfig(
            target_motif="ACGTAC",
            off_target_motifs=["TGCATG"],
            dna_indices=[0, 1, 2, 3, 4, 5],
            structure_tool="alphafold3",
        )
        stats = load_reference_stats(cfg.reference_path)
        assert stats["na_mpnn"]["std"] > 0.0
        assert stats["deeppbs"]["std"] > 0.0


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
        cfg = ConsensusSpecificityConfig(**self._kwargs())
        assert cfg.target_motif == "ACGTAC"
        assert cfg.reference_path.endswith("consensus_specificity_reference.json")

    def test_bad_alphabet_raises(self):
        with pytest.raises(ValidationError, match="A, C, G, T"):
            ConsensusSpecificityConfig(**self._kwargs(target_motif="ACGTAX"))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="dna_indices length"):
            ConsensusSpecificityConfig(**self._kwargs(dna_indices=[0, 1, 2]))

    def test_off_target_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="match target_motif length"):
            ConsensusSpecificityConfig(**self._kwargs(off_target_motifs=["TGC"]))

    def test_structure_tool_defaults_to_dna_capable(self):
        cfg = ConsensusSpecificityConfig(**self._kwargs())
        assert cfg.structure_tool == "alphafold3"


def _result_for_motif(motif: str) -> SimpleNamespace:
    """Tool result whose PPM strongly reads ``motif`` (high target advantage)."""
    return SimpleNamespace(predicted_ppm=_onehot_ppm(motif).tolist())


def _config(tmp_path, **overrides) -> ConsensusSpecificityConfig:
    """Config wired to a synthetic reference with unit mean/std for both models."""
    ref = {"na_mpnn": {"mean": 0.0, "std": 1.0}, "deeppbs": {"mean": 0.0, "std": 1.0}}
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref))
    base = {
        "target_motif": "ACGTAC",
        "off_target_motifs": ["TGCATG"],
        "dna_indices": [0, 1, 2, 3, 4, 5],
        "structure_tool": "alphafold3",
        "reference_path": str(ref_path),
    }
    base.update(overrides)
    return ConsensusSpecificityConfig(**base)


class TestConstraintScoring:
    """End-to-end scoring with the structure resolver and both tools mocked."""

    def _candidate(self):
        protein = Sequence(sequence="MKQ", sequence_type="protein")
        dna = Sequence(sequence="ACGTAC", sequence_type="dna")
        return (protein, dna)

    def test_empty_input(self, tmp_path):
        """No candidates returns an empty result list."""
        assert consensus_operator_specificity_constraint([], _config(tmp_path)) == []

    def test_specific_scores_low_nonspecific_high(self, tmp_path):
        """A target-reading candidate scores low; an off-target-reading one scores high."""
        config = _config(tmp_path)
        na = SimpleNamespace(results=[_result_for_motif("ACGTAC"), _result_for_motif("TGCATG")])
        pbs = SimpleNamespace(results=[_result_for_motif("ACGTAC"), _result_for_motif("TGCATG")])
        with (
            patch.object(csc, "resolve_structure_paths", return_value=["a.pdb", "b.pdb"]),
            patch.object(csc, "run_na_mpnn_specificity", return_value=na),
            patch.object(csc, "run_deeppbs_specificity", return_value=pbs),
        ):
            specific, nonspecific = consensus_operator_specificity_constraint(
                [self._candidate(), self._candidate()], config
            )
        assert 0.0 <= specific.score <= 1.0 and 0.0 <= nonspecific.score <= 1.0
        assert specific.score < nonspecific.score
        assert specific.metadata["consensus"] > nonspecific.metadata["consensus"]

    def test_deeppbs_failure_soft_fails_candidate(self, tmp_path, caplog):
        """DeepPBS failure degrades to z_deeppbs=0 with a logged warning and error metadata."""
        config = _config(tmp_path)
        na = SimpleNamespace(results=[_result_for_motif("ACGTAC")])
        with (
            patch.object(csc, "resolve_structure_paths", return_value=["a.pdb"]),
            patch.object(csc, "run_na_mpnn_specificity", return_value=na),
            patch.object(csc, "run_deeppbs_specificity", side_effect=RuntimeError("DeepPBS boom")),
        ):
            with caplog.at_level("WARNING", logger=csc.__name__):
                (result,) = consensus_operator_specificity_constraint([self._candidate()], config)
        assert result.metadata["deeppbs_z"] == 0.0
        assert np.isnan(result.metadata["deeppbs_margin"])
        assert result.metadata["deeppbs_error"] == "DeepPBS boom"
        assert "DeepPBS failed" in caplog.text

    def test_metadata_propagation(self, tmp_path):
        """Per-candidate metadata round-trips the config and both models' margins/z-scores."""
        config = _config(tmp_path)
        na = SimpleNamespace(results=[_result_for_motif("ACGTAC")])
        pbs = SimpleNamespace(results=[_result_for_motif("ACGTAC")])
        with (
            patch.object(csc, "resolve_structure_paths", return_value=["a.pdb"]),
            patch.object(csc, "run_na_mpnn_specificity", return_value=na),
            patch.object(csc, "run_deeppbs_specificity", return_value=pbs),
        ):
            (result,) = consensus_operator_specificity_constraint([self._candidate()], config)
        meta = result.metadata
        assert meta["pdb_path"] == "a.pdb"
        assert meta["source_method"] == "consensus"
        assert meta["target_motif"] == "ACGTAC"
        assert meta["off_target_motifs"] == ["TGCATG"]
        assert meta["reference_path"] == config.reference_path
        assert meta["consensus"] == pytest.approx(meta["na_mpnn_z"] + meta["deeppbs_z"])
        assert "deeppbs_error" not in meta
