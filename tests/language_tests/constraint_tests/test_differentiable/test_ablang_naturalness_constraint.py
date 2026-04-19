"""Tests for the AbLang naturalness constraints (dual-mode: forward + backward)."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proto_language.language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.language.constraint.differentiable.ablang_naturalness_constraint import (
    AbLangConstraintConfig,
    ablang_scfv_forward,
    ablang_scfv_gradient_backward,
    ablang_vhh_forward,
    ablang_vhh_gradient_backward,
)
from proto_language.language.core import Segment, Sequence

_TOOL_MODULE = "proto_language.language.constraint.differentiable.ablang_naturalness_constraint"


def _seq_with_logits(logits: np.ndarray) -> Sequence:
    seq = Sequence("A" * logits.shape[0], "protein")
    seq.logits = logits
    return seq


def _mock_tool_output(
    *, gradient: list[list[float]] | None, loss: float = 0.5, log_likelihood: float = -0.5
) -> SimpleNamespace:
    """Build a mock AbLangGradientOutput-shaped object."""
    return SimpleNamespace(gradient=gradient, loss=loss, metrics={"log_likelihood": log_likelihood})


class TestBackward:
    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_vhh_dispatches_with_config_and_returns_1_tuple(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=[[0.1] * 20] * 5, loss=0.5)
        binder = _seq_with_logits(np.ones((5, 20)) / 20.0)
        config = AbLangConstraintConfig(temperature=0.8, use_ste=False, device="cpu")

        result = ablang_vhh_gradient_backward((binder,), config=config)

        tool_input, tool_config = mock_run.call_args[0]
        assert tool_input.antibody.heavy_chain is not None and tool_input.antibody.light_chain is None
        assert tool_input.temperature == 0.8
        assert tool_config.use_ste is False
        assert tool_config.compute_gradient is True
        assert tool_config.device == "cpu"
        assert len(result.gradient) == 1
        assert result.gradient[0].shape == (5, 20)
        assert result.loss == 0.5

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_scfv_splits_gradient_per_segment(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=[[0.1] * 20] * 7, loss=0.5)
        vh = _seq_with_logits(np.ones((4, 20)) / 20.0)
        vl = _seq_with_logits(np.ones((3, 20)) / 20.0)

        result = ablang_scfv_gradient_backward((vh, vl), config=AbLangConstraintConfig(temperature=0.6, device="cpu"))

        assert result.gradient[0].shape == (4, 20)
        assert result.gradient[1].shape == (3, 20)
        ab = mock_run.call_args[0][0].antibody
        assert len(ab.heavy_chain) == 4 and len(ab.light_chain) == 3
        assert mock_run.call_args[0][1].device == "cpu"


class TestForward:
    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_returns_monotone_increasing_energy(self, mock_run: object) -> None:
        def score_for(loss: float) -> float:
            mock_run.return_value = _mock_tool_output(gradient=None, loss=loss, log_likelihood=-loss)
            return ablang_vhh_forward([(Sequence("EV", "protein"),)], config=AbLangConstraintConfig(temperature=0.6))[0]

        assert score_for(0.0) == pytest.approx(0.5)
        assert score_for(-2.0) < score_for(0.0) < score_for(2.0)

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_writes_metadata_and_sets_forward_flag(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=None, loss=2.0, log_likelihood=-2.0)
        binder = Sequence("EVQLV", "protein")

        ablang_vhh_forward([(binder,)], config=AbLangConstraintConfig(temperature=0.6, device="cpu"))

        assert mock_run.call_args[0][1].compute_gradient is False
        assert mock_run.call_args[0][1].device == "cpu"
        assert binder._metadata["ablang_log_likelihood"] == -2.0
        assert binder._metadata["ablang_loss"] == 2.0

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_scfv_one_hot_encodes_both_chains(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=None, loss=1.0)
        ablang_scfv_forward(
            [(Sequence("EVQL", "protein"), Sequence("DIQ", "protein"))],
            config=AbLangConstraintConfig(temperature=0.6, device="cpu"),
        )

        ab = mock_run.call_args[0][0].antibody
        assert len(ab.heavy_chain) == 4 and len(ab.light_chain) == 3
        assert mock_run.call_args[0][1].compute_gradient is False
        assert mock_run.call_args[0][1].device == "cpu"


class TestRegistry:
    @pytest.mark.parametrize(
        "key,forward,backward,slots",
        [
            (
                "ablang-vhh",
                ablang_vhh_forward,
                ablang_vhh_gradient_backward,
                [InputSlot(label="VHH Chain", requires_logits=True)],
            ),
            (
                "ablang-scfv",
                ablang_scfv_forward,
                ablang_scfv_gradient_backward,
                [
                    InputSlot(label="Heavy Chain (VH)", requires_logits=True),
                    InputSlot(label="Light Chain (VL)", requires_logits=True),
                ],
            ),
        ],
    )
    def test_registers_as_dual_mode(self, key, forward, backward, slots) -> None:
        spec = ConstraintRegistry.get(key)
        assert spec.mode == "dual"
        assert spec.function is forward
        assert spec.backward is backward
        assert spec.input_labels == slots

    def test_factory_builds_dual_capable_constraint(self) -> None:
        vh_seg = Segment(sequence="EVQLVESG", sequence_type="protein")
        vl_seg = Segment(sequence="DIQMTQS", sequence_type="protein")
        c = ConstraintRegistry.create("ablang-scfv", [vh_seg, vl_seg], {"temperature": 0.6})
        assert c.supports_gradient and c.supports_discrete


@pytest.mark.uses_gpu
class TestGPU:
    def test_different_inputs_produce_different_gradients(self) -> None:
        uniform = _seq_with_logits(np.zeros((20, 20), dtype=np.float64))
        biased = _seq_with_logits(np.zeros((20, 20), dtype=np.float64))
        biased.logits[:, 0] = 5.0

        r1 = ablang_vhh_gradient_backward((uniform,), config=AbLangConstraintConfig(temperature=0.6))
        r2 = ablang_vhh_gradient_backward((biased,), config=AbLangConstraintConfig(temperature=0.6))

        assert np.isfinite(r1.gradient[0]).all() and np.isfinite(r2.gradient[0]).all()
        assert r1.loss != r2.loss
        assert not np.allclose(r1.gradient[0], r2.gradient[0])

    def test_forward_matches_backward_loss_on_discrete_sequence(self) -> None:
        """Forward path one-hots a discrete sequence → same argmax as backward input → same loss."""
        config = AbLangConstraintConfig(temperature=0.6)
        sequence = "EVQLVESGGGLVQPGGSLRL"
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        logits = np.zeros((len(sequence), 20), dtype=np.float64)
        for i, aa in enumerate(sequence):
            logits[i, aa_order.index(aa)] = 1.0  # proto_language.utils.one_hot_protein_matrix

        backward = ablang_vhh_gradient_backward((_seq_with_logits(logits),), config=config)
        forward_score = ablang_vhh_forward([(Sequence(sequence, "protein"),)], config=config)[0]

        assert math.isclose(forward_score, 1.0 / (1.0 + math.exp(-backward.loss)), rel_tol=1e-3)

    def test_scfv_produces_per_segment_gradients(self) -> None:
        vh = _seq_with_logits(np.zeros((15, 20), dtype=np.float64))
        vl = _seq_with_logits(np.zeros((12, 20), dtype=np.float64))

        result = ablang_scfv_gradient_backward((vh, vl), config=AbLangConstraintConfig(temperature=0.6))

        assert result.gradient[0].shape == (15, 20) and result.gradient[1].shape == (12, 20)
        assert np.isfinite(result.gradient[0]).all() and np.isfinite(result.gradient[1]).all()
        assert np.any(result.gradient[0] != 0.0) and np.any(result.gradient[1] != 0.0)
