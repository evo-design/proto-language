"""Tests for the minimum-free-energy (MFE) constraint."""

import importlib
import math
from types import SimpleNamespace

import pytest

from proto_language.constraint.constraint_registry import ConstraintRegistry
from proto_language.constraint.rna_secondary_structure.mfe_constraint import MFEConfig, mfe_constraint
from proto_language.core import Sequence
from proto_language.utils import sigmoid_score

# Patch the module object: `proto_language.constraint` is the registration decorator, not the
# package, so monkeypatch cannot resolve a dotted string through it.
_MODULE = importlib.import_module("proto_language.constraint.rna_secondary_structure.mfe_constraint")
_SEQ = "AUGGUGAGCAAGGGC"


def _fake_output(*mfes: float) -> SimpleNamespace:
    return SimpleNamespace(results=[SimpleNamespace(mfe=m, structure="." * 15) for m in mfes])


class TestForward:
    def test_direction_min_rewards_more_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def fake_run(inputs, config):
            captured["inputs"] = inputs
            captured["config"] = config
            return _fake_output(-40.0)

        monkeypatch.setattr(_MODULE, "run_viennarna", fake_run)

        config = MFEConfig(direction="min", sigmoid_center=-30.0, sigmoid_scale=10.0, temperature=37.0)
        out = mfe_constraint([(Sequence(_SEQ, sequence_type="rna"),)], config)

        expected = sigmoid_score(-40.0, -30.0, slope=1.0 / 10.0)  # more-negative MFE -> low energy
        assert out[0].score == pytest.approx(expected)
        assert out[0].metadata["mfe"] == -40.0
        assert captured["config"].temperature == 37.0
        assert captured["inputs"].sequences == [_SEQ]

    def test_direction_max_reverses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_viennarna", lambda inputs, config: _fake_output(-40.0))
        config = MFEConfig(direction="max", sigmoid_center=-30.0, sigmoid_scale=10.0)
        out = mfe_constraint([(Sequence(_SEQ, sequence_type="rna"),)], config)
        assert out[0].score == pytest.approx(1.0 - sigmoid_score(-40.0, -30.0, slope=1.0 / 10.0))

    def test_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_viennarna", lambda inputs, config: _fake_output(-10.0, -50.0))
        out = mfe_constraint(
            [(Sequence(_SEQ, sequence_type="rna"),), (Sequence(_SEQ, sequence_type="rna"),)], MFEConfig()
        )
        assert [o.metadata["mfe"] for o in out] == [-10.0, -50.0]
        for o in out:
            assert 0.0 <= o.score <= 1.0

    def test_non_finite_mfe_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_viennarna", lambda inputs, config: _fake_output(math.nan))
        with pytest.raises(ValueError, match="non-finite MFE"):
            mfe_constraint([(Sequence(_SEQ, sequence_type="rna"),)], MFEConfig())


class TestRegistry:
    def test_registration(self) -> None:
        spec = ConstraintRegistry.get("mfe")
        assert spec.function is mfe_constraint
        assert spec.category == "rna_secondary_structure"
        assert spec.tools_called == ["viennarna"]
        assert spec.supported_sequence_types == ["dna", "rna"]
