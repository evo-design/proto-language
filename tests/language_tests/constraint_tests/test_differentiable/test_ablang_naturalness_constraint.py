"""Tests for the AbLang naturalness constraint (dual-mode: forward + backward; VHH and scFv)."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proto_language.language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.language.constraint.differentiable.ablang_naturalness_constraint import (
    AbLangConstraintConfig,
    ablang_naturalness_forward,
    ablang_naturalness_gradient_backward,
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


class TestVHHMode:
    """No slices: full binder is scored as a heavy-only chain (default / nanobody)."""

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_backward_passes_full_binder_as_heavy_chain(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=[[0.1] * 20] * 5, loss=0.5)
        binder = _seq_with_logits(np.ones((5, 20)) / 20.0)

        result = ablang_naturalness_gradient_backward(
            (binder,), config=AbLangConstraintConfig(temperature=0.8, use_ste=False, device="cpu")
        )

        tool_input, tool_config = mock_run.call_args[0]
        assert tool_input.antibody.heavy_chain is not None and tool_input.antibody.light_chain is None
        assert tool_input.temperature == 0.8
        assert tool_config.use_ste is False and tool_config.compute_gradient is True
        assert len(result.gradient) == 1 and result.gradient[0].shape == (5, 20)

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_forward_returns_monotone_increasing_energy(self, mock_run: object) -> None:
        def score_for(loss: float) -> float:
            mock_run.return_value = _mock_tool_output(gradient=None, loss=loss, log_likelihood=-loss)
            return ablang_naturalness_forward(
                [(Sequence("EV", "protein"),)], config=AbLangConstraintConfig(temperature=0.6)
            )[0]

        assert score_for(0.0) == pytest.approx(0.5)
        assert score_for(-2.0) < score_for(0.0) < score_for(2.0)

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_forward_writes_metadata_and_sets_compute_gradient_false(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=None, loss=2.0, log_likelihood=-2.0)
        binder = Sequence("EVQLV", "protein")
        ablang_naturalness_forward([(binder,)], config=AbLangConstraintConfig(temperature=0.6, device="cpu"))
        assert binder._metadata == {"ablang_log_likelihood": -2.0, "ablang_loss": 2.0}
        assert mock_run.call_args[0][1].compute_gradient is False


class TestScFvMode:
    """Slices set: extract VH/VL from the single binder Segment, scatter gradient back."""

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_backward_extracts_slices_and_scatters_gradient(self, mock_run: object) -> None:
        # Binder length 20: VH at [0:8], linker at [8:12] (zeros), VL at [12:20].
        # Tool returns concatenated (vh_len + vl_len, 20) gradient with distinct per-row values.
        vh_len, vl_len, total = 8, 8, 20
        full_paired = [[float(i + 1)] * 20 for i in range(vh_len + vl_len)]
        mock_run.return_value = _mock_tool_output(gradient=full_paired, loss=0.3)
        binder = _seq_with_logits(np.zeros((total, 20)))

        result = ablang_naturalness_gradient_backward(
            (binder,),
            config=AbLangConstraintConfig(temperature=0.6, device="cpu", heavy_slice=(0, 8), light_slice=(12, 20)),
        )

        # Tool received heavy/light extracted from the slices.
        ab = mock_run.call_args[0][0].antibody
        assert len(ab.heavy_chain) == vh_len and len(ab.light_chain) == vl_len
        # Returned 1-tuple gradient with full binder shape; VH at [0:8], VL at [12:20], linker zero.
        assert len(result.gradient) == 1
        full = result.gradient[0]
        assert full.shape == (total, 20)
        assert full[0, 0] == 1.0 and full[7, 0] == float(vh_len)  # VH rows 1..8
        assert np.all(full[8:12] == 0.0)  # linker untouched
        assert full[12, 0] == float(vh_len + 1) and full[19, 0] == float(vh_len + vl_len)  # VL rows 9..16

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_forward_extracts_slices_for_one_hot(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=None, loss=1.0)
        binder = Sequence("EVQLAAAA" + "GGGG" + "DIQAAAAA", "protein")  # vh=8, linker=4, vl=8
        ablang_naturalness_forward(
            [(binder,)],
            config=AbLangConstraintConfig(temperature=0.6, device="cpu", heavy_slice=(0, 8), light_slice=(12, 20)),
        )
        ab = mock_run.call_args[0][0].antibody
        assert len(ab.heavy_chain) == 8 and len(ab.light_chain) == 8
        assert mock_run.call_args[0][1].compute_gradient is False

    @pytest.mark.parametrize("mode", ["forward", "backward"])
    def test_runtime_rejects_slice_past_binder_length(self, mode: str) -> None:
        config = AbLangConstraintConfig(temperature=0.6, heavy_slice=(0, 4), light_slice=(6, 12))
        with pytest.raises(ValueError, match="extend past binder length"):
            if mode == "backward":
                ablang_naturalness_gradient_backward((_seq_with_logits(np.zeros((10, 20))),), config=config)
            else:
                ablang_naturalness_forward([(Sequence("A" * 10, "protein"),)], config=config)


class TestConfig:
    """Slice-config validation: both-or-neither, non-empty, non-overlapping."""

    @pytest.mark.parametrize(
        "heavy,light,error_match",
        [
            ((0, 8), None, "must be set together"),
            (None, (0, 8), "must be set together"),
            ((5, 5), (10, 15), "non-empty"),  # empty range
            ((-1, 5), (10, 15), "non-empty"),  # negative start
            ((0, 10), (5, 15), "overlaps"),
        ],
    )
    def test_rejects_invalid_slices(
        self, heavy: tuple[int, int] | None, light: tuple[int, int] | None, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            AbLangConstraintConfig(temperature=0.6, heavy_slice=heavy, light_slice=light)


class TestRegistry:
    """Constraint registers as dual-mode with one binder slot; factory builds for both modes."""

    def test_registers_as_dual_mode_with_one_slot(self) -> None:
        spec = ConstraintRegistry.get("ablang-naturalness")
        assert spec.mode == "dual"
        assert spec.function is ablang_naturalness_forward
        assert spec.backward is ablang_naturalness_gradient_backward
        assert spec.input_labels == [InputSlot(label="Binder", requires_logits=True)]

    def test_factory_builds_vhh_constraint(self) -> None:
        binder = Segment(sequence="EVQLVESG", sequence_type="protein")
        c = ConstraintRegistry.create("ablang-naturalness", [binder], {"temperature": 0.6})
        assert c.supports_gradient and c.supports_discrete

    def test_factory_builds_scfv_constraint_with_slices(self) -> None:
        binder = Segment(sequence="A" * 20, sequence_type="protein")
        c = ConstraintRegistry.create(
            "ablang-naturalness",
            [binder],
            {"temperature": 0.6, "heavy_slice": (0, 8), "light_slice": (12, 20)},
        )
        assert c.supports_gradient and c.supports_discrete


@pytest.mark.uses_gpu
class TestGPU:
    def test_vhh_different_inputs_produce_different_gradients(self) -> None:
        uniform = _seq_with_logits(np.zeros((20, 20), dtype=np.float64))
        biased = _seq_with_logits(np.zeros((20, 20), dtype=np.float64))
        biased.logits[:, 0] = 5.0

        r1 = ablang_naturalness_gradient_backward((uniform,), config=AbLangConstraintConfig(temperature=0.6))
        r2 = ablang_naturalness_gradient_backward((biased,), config=AbLangConstraintConfig(temperature=0.6))

        assert np.isfinite(r1.gradient[0]).all() and np.isfinite(r2.gradient[0]).all()
        assert r1.loss != r2.loss and not np.allclose(r1.gradient[0], r2.gradient[0])

    def test_vhh_forward_matches_backward_loss_on_discrete_sequence(self) -> None:
        """Forward path one-hots a discrete sequence → same argmax as backward input → same loss."""
        config = AbLangConstraintConfig(temperature=0.6)
        sequence = "EVQLVESGGGLVQPGGSLRL"
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        logits = np.zeros((len(sequence), 20), dtype=np.float64)
        for i, aa in enumerate(sequence):
            logits[i, aa_order.index(aa)] = 1.0

        backward = ablang_naturalness_gradient_backward((_seq_with_logits(logits),), config=config)
        forward_score = ablang_naturalness_forward([(Sequence(sequence, "protein"),)], config=config)[0]

        assert math.isclose(forward_score, 1.0 / (1.0 + math.exp(-backward.loss)), rel_tol=1e-3)

    def test_scfv_mode_produces_full_binder_gradient_with_zero_linker(self) -> None:
        binder = _seq_with_logits(np.zeros((30, 20), dtype=np.float64))
        result = ablang_naturalness_gradient_backward(
            (binder,),
            config=AbLangConstraintConfig(temperature=0.6, heavy_slice=(0, 12), light_slice=(18, 30)),
        )
        full = result.gradient[0]
        assert full.shape == (30, 20)
        assert np.isfinite(full).all()
        assert np.any(full[0:12] != 0.0) and np.any(full[18:30] != 0.0)
        assert np.all(full[12:18] == 0.0)  # linker untouched
