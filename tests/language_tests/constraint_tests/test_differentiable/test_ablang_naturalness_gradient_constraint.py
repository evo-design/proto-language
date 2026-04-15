"""Tests for the AbLang naturalness gradient constraints."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.language.constraint.differentiable.ablang_naturalness_gradient_constraint import (
    AbLangGradientConstraintConfig,
    ablang_scfv_gradient_backward,
    ablang_vhh_gradient_backward,
)
from proto_language.language.core import Sequence

_TOOL_MODULE = "proto_language.language.constraint.differentiable.ablang_naturalness_gradient_constraint"


def _seq_with_logits(logits: np.ndarray) -> Sequence:
    seq = Sequence("A" * logits.shape[0], "protein")
    seq.logits = logits
    return seq


class TestVHH:
    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_passes_heavy_chain_and_returns_1_tuple(self, mock_run: object) -> None:
        mock_run.return_value = SimpleNamespace(gradient=[[0.1] * 20] * 5, loss=0.5, metrics={})
        seq = _seq_with_logits(np.ones((5, 20)) / 20.0)

        result = ablang_vhh_gradient_backward((seq,), temperature=1.0, config=AbLangGradientConstraintConfig())

        assert len(result.gradient) == 1
        assert result.gradient[0].shape == (5, 20)
        assert result.loss == 0.5
        ab = mock_run.call_args[0][0].antibody
        assert ab.heavy_chain is not None and ab.light_chain is None

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_uses_config_temperature_and_ste(self, mock_run: object) -> None:
        mock_run.return_value = SimpleNamespace(gradient=[[0.0] * 20] * 5, loss=0.0, metrics={})
        seq = _seq_with_logits(np.zeros((5, 20)))
        cfg = AbLangGradientConstraintConfig(temperature=0.8, use_ste=False)

        ablang_vhh_gradient_backward((seq,), temperature=999.0, config=cfg)

        assert mock_run.call_args[0][0].temperature == 0.8
        assert mock_run.call_args[0][1].use_ste is False

    def test_registry(self) -> None:
        spec = ConstraintRegistry.get("ablang-vhh-gradient")
        assert spec.function is None and spec.backward is ablang_vhh_gradient_backward
        assert spec.input_labels == ["Sequence"]


class TestScFv:
    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_splits_gradient_per_segment(self, mock_run: object) -> None:
        mock_run.return_value = SimpleNamespace(gradient=[[0.1] * 20] * 7, loss=0.5, metrics={})
        vh = _seq_with_logits(np.ones((4, 20)) / 20.0)
        vl = _seq_with_logits(np.ones((3, 20)) / 20.0)

        result = ablang_scfv_gradient_backward((vh, vl), temperature=1.0, config=AbLangGradientConstraintConfig())

        assert len(result.gradient) == 2
        assert result.gradient[0].shape == (4, 20)  # VH
        assert result.gradient[1].shape == (3, 20)  # VL
        ab = mock_run.call_args[0][0].antibody
        assert len(ab.heavy_chain) == 4 and len(ab.light_chain) == 3

    def test_registry_and_create(self) -> None:
        spec = ConstraintRegistry.get("ablang-scfv-gradient")
        assert spec.function is None and spec.backward is ablang_scfv_gradient_backward
        assert spec.input_labels == ["Heavy Chain (VH)", "Light Chain (VL)"]

        from proto_language.language.core import Segment

        vh_seg = Segment(sequence="EVQLVESG", sequence_type="protein")
        vl_seg = Segment(sequence="DIQMTQS", sequence_type="protein")
        c = ConstraintRegistry.create("ablang-scfv-gradient", [vh_seg, vl_seg], {})
        assert c.supports_gradient and not c.supports_discrete


@pytest.mark.uses_gpu
class TestGPU:
    def test_vhh_different_inputs_different_gradients(self) -> None:
        config = AbLangGradientConstraintConfig()
        uniform = _seq_with_logits(np.zeros((20, 20), dtype=np.float64))
        biased = _seq_with_logits(np.zeros((20, 20), dtype=np.float64))
        biased.logits[:, 0] = 5.0

        r_uniform = ablang_vhh_gradient_backward((uniform,), temperature=1.0, config=config)
        r_biased = ablang_vhh_gradient_backward((biased,), temperature=1.0, config=config)

        assert np.isfinite(r_uniform.gradient[0]).all()
        assert r_uniform.loss != r_biased.loss
        assert not np.allclose(r_uniform.gradient[0], r_biased.gradient[0])

    def test_vhh_onehot_logits_produce_sequence_dependent_loss(self) -> None:
        """One-hot-like logits for different sequences produce different, finite losses."""
        config = AbLangGradientConstraintConfig()
        aa_order = "ACDEFGHIKLMNPQRSTVWY"

        def make_seq(sequence: str) -> Sequence:
            logits = np.zeros((len(sequence), 20), dtype=np.float64)
            for i, aa in enumerate(sequence):
                logits[i, aa_order.index(aa)] = 10.0
            seq = Sequence(sequence, "protein")
            seq.logits = logits
            return seq

        r_natural = ablang_vhh_gradient_backward((make_seq("EVQLVESGGGLVQPGGSLRL"),), temperature=0.1, config=config)
        r_polyala = ablang_vhh_gradient_backward((make_seq("A" * 20),), temperature=0.1, config=config)

        assert np.isfinite(r_natural.gradient[0]).all()
        assert np.isfinite(r_polyala.gradient[0]).all()
        assert np.any(r_natural.gradient[0] != 0.0)
        assert np.any(r_polyala.gradient[0] != 0.0)
        assert r_natural.loss != r_polyala.loss

    def test_vhh_gradient_nonzero_and_finite(self) -> None:
        config = AbLangGradientConstraintConfig()
        seq = _seq_with_logits(np.zeros((15, 20), dtype=np.float64))
        result = ablang_vhh_gradient_backward((seq,), temperature=1.0, config=config)

        assert np.isfinite(result.gradient[0]).all()
        assert np.any(result.gradient[0] != 0.0)

    def test_scfv_produces_per_segment_gradients(self) -> None:
        """Paired VH + VL produces finite, nonzero gradients for both chains."""
        config = AbLangGradientConstraintConfig()
        vh = _seq_with_logits(np.zeros((15, 20), dtype=np.float64))
        vl = _seq_with_logits(np.zeros((12, 20), dtype=np.float64))

        result = ablang_scfv_gradient_backward((vh, vl), temperature=1.0, config=config)

        assert len(result.gradient) == 2
        assert result.gradient[0].shape == (15, 20)
        assert result.gradient[1].shape == (12, 20)
        assert np.isfinite(result.gradient[0]).all() and np.isfinite(result.gradient[1]).all()
        assert np.any(result.gradient[0] != 0.0) and np.any(result.gradient[1] != 0.0)
        assert np.isfinite(result.loss)
