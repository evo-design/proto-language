"""Tests for the AbLang perplexity constraint."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proto_language.language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.language.constraint.sequence_scoring.ablang_perplexity_constraint import (
    AbLangPerplexityConfig,
    ablang_perplexity_constraint,
    ablang_perplexity_gradient_backward,
)
from proto_language.language.core import Segment, Sequence

_TOOL_MODULE = "proto_language.language.constraint.sequence_scoring.ablang_perplexity_constraint"


def _seq_with_logits(logits: np.ndarray) -> Sequence:
    seq = Sequence("A" * logits.shape[0], "protein")
    seq.logits = logits
    return seq


def _mock_gradient_output(
    *, gradient: list[list[float]] | None, loss: float = 0.5, log_likelihood: float = -0.5
) -> SimpleNamespace:
    """Build a mock AbLangGradientOutput-shaped object."""
    return SimpleNamespace(gradient=gradient, loss=loss, metrics={"log_likelihood": log_likelihood})


class TestVHHMode:
    """No slices: full binder is scored as a heavy-only chain."""

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_backward_passes_full_binder_as_heavy_chain(self, mock_run: object) -> None:
        mock_run.return_value = _mock_gradient_output(gradient=[[0.1] * 20] * 5, loss=0.5)
        binder = _seq_with_logits(np.ones((5, 20)) / 20.0)

        (result,) = ablang_perplexity_gradient_backward(
            [(binder,)], config=AbLangPerplexityConfig(temperature=0.8, use_ste=False, device="cpu")
        )

        tool_input, tool_config = mock_run.call_args[0]
        assert tool_input.antibody.heavy_chain is not None and tool_input.antibody.light_chain is None
        assert tool_input.temperature == 0.8
        assert tool_config.use_ste is False and tool_config.compute_gradient is True
        assert len(result.gradient) == 1 and result.gradient[0].shape == (5, 20)


class TestForward:
    @pytest.mark.parametrize(("score_mode", "expected_score"), [("nll", 2.0), ("ppl", pytest.approx(np.exp(2.0)))])
    def test_returns_requested_score_and_metadata(self, score_mode: str, expected_score: object) -> None:
        binder = Sequence("EVQLV", "protein")

        with patch(f"{_TOOL_MODULE}.run_ablang_gradient") as mock_run:
            mock_run.return_value = _mock_gradient_output(gradient=None, loss=2.0, log_likelihood=-2.0)
            (result,) = ablang_perplexity_constraint(
                [(binder,)], config=AbLangPerplexityConfig(temperature=0.6, device="cpu", score_mode=score_mode)
            )

        assert result.score == expected_score
        assert result.metadata["ablang_log_likelihood"] == -2.0
        assert result.metadata["ablang_loss"] == 2.0
        assert result.metadata["ablang_nll"] == 2.0
        assert result.metadata["ablang_perplexity"] == pytest.approx(np.exp(2.0))
        assert result.metadata["ablang_score_mode"] == score_mode
        assert mock_run.call_args[0][1].compute_gradient is False


class TestScFvMode:
    """Slices set: extract VH/VL from the single binder Segment."""

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_backward_extracts_slices_and_scatters_gradient(self, mock_run: object) -> None:
        vh_len, vl_len, total = 8, 8, 20
        full_paired = [[float(i + 1)] * 20 for i in range(vh_len + vl_len)]
        mock_run.return_value = _mock_gradient_output(gradient=full_paired, loss=0.3)
        binder = _seq_with_logits(np.zeros((total, 20)))

        (result,) = ablang_perplexity_gradient_backward(
            [(binder,)],
            config=AbLangPerplexityConfig(temperature=0.6, device="cpu", heavy_slice=(0, 8), light_slice=(12, 20)),
        )

        ab = mock_run.call_args[0][0].antibody
        assert len(ab.heavy_chain) == vh_len and len(ab.light_chain) == vl_len
        full = result.gradient[0]
        assert full.shape == (total, 20)
        assert full[0, 0] == 1.0 and full[7, 0] == float(vh_len)
        assert np.all(full[8:12] == 0.0)
        assert full[12, 0] == float(vh_len + 1) and full[19, 0] == float(vh_len + vl_len)

    @patch(f"{_TOOL_MODULE}.run_ablang_gradient")
    def test_ppl_backward_scales_nll_gradient_by_perplexity(self, mock_run: object) -> None:
        mock_run.return_value = _mock_gradient_output(gradient=[[0.5] * 20] * 5, loss=2.0)
        binder = _seq_with_logits(np.zeros((5, 20)))

        (result,) = ablang_perplexity_gradient_backward(
            [(binder,)], config=AbLangPerplexityConfig(temperature=0.6, device="cpu", score_mode="ppl")
        )

        assert result.loss == pytest.approx(np.exp(2.0))
        assert result.gradient[0][0, 0] == pytest.approx(0.5 * np.exp(2.0))
        assert result.metrics["ablang_nll"] == 2.0
        assert result.metrics["ablang_score_mode"] == "ppl"

    @pytest.mark.parametrize("mode", ["forward", "backward"])
    def test_runtime_rejects_slice_past_binder_length(self, mode: str) -> None:
        config = AbLangPerplexityConfig(temperature=0.6, heavy_slice=(0, 4), light_slice=(6, 12))
        with pytest.raises(ValueError, match="extend past binder length"):
            if mode == "backward":
                ablang_perplexity_gradient_backward([(_seq_with_logits(np.zeros((10, 20))),)], config=config)
            else:
                ablang_perplexity_constraint([(Sequence("A" * 10, "protein"),)], config=config)


class TestConfig:
    """Slice-config validation: both-or-neither, non-empty, non-overlapping."""

    @pytest.mark.parametrize(
        "heavy,light,error_match",
        [
            ((0, 8), None, "must be set together"),
            (None, (0, 8), "must be set together"),
            ((5, 5), (10, 15), "non-empty"),
            ((-1, 5), (10, 15), "non-empty"),
            ((0, 10), (5, 15), "overlaps"),
        ],
    )
    def test_rejects_invalid_slices(
        self, heavy: tuple[int, int] | None, light: tuple[int, int] | None, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            AbLangPerplexityConfig(temperature=0.6, heavy_slice=heavy, light_slice=light)


class TestRegistry:
    """Constraint registers as dual-mode with one binder slot."""

    def test_registers_as_dual_mode_with_one_slot(self) -> None:
        spec = ConstraintRegistry.get("ablang-perplexity")
        assert spec.mode == "dual"
        assert spec.function is ablang_perplexity_constraint
        assert spec.backward is ablang_perplexity_gradient_backward
        assert spec.input_labels == [InputSlot(label="Binder", requires_logits=True)]

    def test_factory_builds_vhh_constraint(self) -> None:
        binder = Segment(sequence="EVQLVESG", sequence_type="protein")
        c = ConstraintRegistry.create("ablang-perplexity", [binder], {"temperature": 0.6})
        assert c.supports_gradient and c.supports_discrete
