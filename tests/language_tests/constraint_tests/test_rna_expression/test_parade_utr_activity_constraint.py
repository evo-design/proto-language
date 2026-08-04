"""Tests for the PARADE UTR activity constraint (forward scoring + gradient backward)."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from proto_tools import ParadeGradientSampleMetrics

from proto_language import (
    ParadeUTRActivityConfig,
    parade_utr_activity_constraint,
    parade_utr_activity_gradient_backward,
)
from proto_language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.core import Constraint, Segment, Sequence
from proto_language.utils import sigmoid_score

_MODULE = "proto_language.constraint.rna_expression.parade_utr_activity_constraint"


def _seq(sequence: str, logits: np.ndarray | None = None) -> Sequence:
    seq = Sequence(sequence, "dna")
    if logits is not None:
        seq.logits = logits
    return seq


def _activity_output(cell_scores: dict[str, float]) -> SimpleNamespace:
    """Mock a ParadeActivityOutput with one per-sequence result."""
    return SimpleNamespace(results=[SimpleNamespace(scores=cell_scores)])


def _gradient_output(gradient: list, loss: float = 0.5, metrics: dict | None = None) -> SimpleNamespace:
    """Mock a ParadeGradientOutput (gradient is B x L x 4)."""
    sample_metrics = [{"loss": loss, **(metrics or {})} for _ in gradient]
    return SimpleNamespace(gradient=gradient, loss=loss * len(gradient), sample_metrics=sample_metrics, metrics={})


class TestForward:
    @pytest.mark.parametrize(
        ("activity", "direction", "expected"),
        [
            (5.0, "max", 1.0 - sigmoid_score(5.0, 2.0, slope=1.0)),
            (5.0, "min", sigmoid_score(5.0, 2.0, slope=1.0)),
            (-1.0, "max", 1.0 - sigmoid_score(-1.0, 2.0, slope=1.0)),
        ],
    )
    def test_parametrized_scoring_formula(self, activity: float, direction: str, expected: float) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = _activity_output({"c2": activity})
            (result,) = parade_utr_activity_constraint(
                [(_seq("ACGT"),)],
                ParadeUTRActivityConfig(cell_type="c2", direction=direction, device="cpu"),
            )
        assert result.score == pytest.approx(expected)

    def test_max_rewards_high_activity_with_low_energy(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = _activity_output({"c2": 10.0})
            (result,) = parade_utr_activity_constraint(
                [(_seq("ACGT" * 12 + "AC"),)],
                ParadeUTRActivityConfig(construct_type="utr5", cell_type="c2", direction="max", device="cpu"),
            )
        assert result.score < 0.01  # very high activity under 'max' -> energy near 0
        assert result.metadata == {"cell_type": "c2", "direction": "max", "activity": 10.0}
        # forward requests exactly the one configured cell code from the tool
        _, tool_config = mock_run.call_args[0]
        assert tool_config.cell_types == ["c2"] and tool_config.construct_type == "utr5"

    def test_min_flips_direction(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = _activity_output({"c2": 10.0})
            (result,) = parade_utr_activity_constraint(
                [(_seq("ACGTACGT"),)],
                ParadeUTRActivityConfig(cell_type="c2", direction="min", device="cpu"),
            )
        assert result.score > 0.99  # high activity under 'min' -> high energy

    def test_empty_input_returns_empty(self) -> None:
        assert parade_utr_activity_constraint([], ParadeUTRActivityConfig()) == []

    def test_extreme_activity_is_numerically_stable(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = _activity_output({"c2": -1000.0})
            (result,) = parade_utr_activity_constraint(
                [(_seq("ACGT"),)],
                ParadeUTRActivityConfig(cell_type="c2", direction="max", sigmoid_scale=0.001, device="cpu"),
            )
        assert result.score == 1.0

    def test_minimum_float32_scale_scores_center_as_half(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = _activity_output({"c2": 2.0})
            (result,) = parade_utr_activity_constraint(
                [(_seq("ACGT"),)],
                ParadeUTRActivityConfig(sigmoid_scale=1.1754943508222875e-38, device="cpu"),
            )
        assert result.score == 0.5

    def test_rejects_nonfinite_tool_activity(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = _activity_output({"c2": float("nan")})
            with pytest.raises(ValueError, match="non-finite activity"):
                parade_utr_activity_constraint([(_seq("ACGT"),)], ParadeUTRActivityConfig(device="cpu"))

    def test_rna_evaluate_propagates_metadata(self) -> None:
        segment = Segment(sequence="ACGU", sequence_type="rna")
        constraint = Constraint(
            inputs=[segment],
            function=parade_utr_activity_constraint,
            function_config=ParadeUTRActivityConfig(device="cpu"),
        )
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = _activity_output({"c2": 3.0})
            constraint.evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["parade_utr_activity_constraint"]["data"]
        assert data["activity"] == 3.0


class TestBackward:
    def test_gradient_shape_and_single_loss_term(self) -> None:
        grad_LxC = [[0.1, 0.2, 0.3, 0.4]] * 5
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = _gradient_output(gradient=[grad_LxC], loss=0.7, metrics={"c2": 3.0})
            (result,) = parade_utr_activity_gradient_backward(
                [(_seq("ACGTA", np.ones((5, 4)) / 4.0),)],
                config=ParadeUTRActivityConfig(
                    construct_type="utr5", cell_type="c2", direction="max", temperature=0.7, device="cpu"
                ),
            )
        tool_input, tool_config = mock_run.call_args[0]
        # logits wrapped as B x L x 4
        assert len(tool_input.logits) == 1 and len(tool_input.logits[0]) == 5 and len(tool_input.logits[0][0]) == 4
        assert tool_input.temperature == 0.7
        assert tool_config.compute_gradient is True and tool_config.construct_type == "utr5"
        assert len(tool_config.loss_terms) == 1
        assert tool_config.loss_terms[0].cell_type == "c2" and tool_config.loss_terms[0].direction == "max"
        # gradient is a one-tuple matching the segment's (L, 4) logits
        assert len(result.gradient) == 1 and result.gradient[0].shape == (5, 4)
        assert result.loss == 0.7
        assert result.metrics["cell_type"] == "c2" and result.metrics["activity"] == 3.0

    def test_optimizer_relaxation_and_batching_are_forwarded(self) -> None:
        gradients = [
            [[0.1, 0.2, 0.3, 0.4]] * 4,
            [[0.4, 0.3, 0.2, 0.1]] * 5,
            [[0.2, 0.1, 0.4, 0.3]] * 4,
        ]
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.side_effect = [
                SimpleNamespace(
                    gradient=[gradients[0], gradients[2]],
                    loss=0.4,
                    sample_metrics=[{"loss": 0.1, "c2": 1.0}, {"loss": 0.3, "c2": 3.0}],
                    metrics={},
                ),
                SimpleNamespace(
                    gradient=[gradients[1]],
                    loss=0.2,
                    sample_metrics=[{"loss": 0.2, "c2": 2.0}],
                    metrics={},
                ),
            ]
            results = parade_utr_activity_gradient_backward(
                [
                    (_seq("ACGT", np.zeros((4, 4))),),
                    (_seq("ACGTA", np.zeros((5, 4))),),
                    (_seq("TGCA", np.zeros((4, 4))),),
                ],
                config=ParadeUTRActivityConfig(batch_size=2, temperature=0.9, device="cpu"),
                temperature=0.4,
                soft=0.3,
                hard=0.8,
            )
        assert mock_run.call_count == 2
        first_input, first_config = mock_run.call_args_list[0].args
        second_input, second_config = mock_run.call_args_list[1].args
        assert [len(logits) for logits in first_input.logits] == [4, 4]
        assert [len(logits) for logits in second_input.logits] == [5]
        assert first_input.temperature == second_input.temperature == 0.4
        assert first_config.soft == second_config.soft == 0.3
        assert first_config.hard == second_config.hard == 0.8
        assert [result.loss for result in results] == [0.1, 0.2, 0.3]
        assert [result.metrics["activity"] for result in results] == [1.0, 2.0, 3.0]
        assert [result.gradient[0].shape for result in results] == [(4, 4), (5, 4), (4, 4)]

    def test_preserves_declared_loss_term_metadata(self) -> None:
        sample_metrics = ParadeGradientSampleMetrics(
            loss=0.4,
            c2=2.5,
            loss_terms=[{"cell_type": "c2", "direction": "max", "weighted_score": 0.4}],
        )
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[0.0, 0.0, 0.0, 0.0]] * 4],
                loss=0.4,
                sample_metrics=[sample_metrics],
                metrics={},
            )
            (result,) = parade_utr_activity_gradient_backward(
                [(_seq("ACGT", np.zeros((4, 4))),)], config=ParadeUTRActivityConfig(device="cpu")
            )
        assert result.metrics["loss_terms"] == sample_metrics.loss_terms
        assert result.metrics["activity"] == 2.5

    @pytest.mark.parametrize(
        ("gradient_value", "loss", "message"),
        [
            (0.0, float("nan"), "non-finite activity loss"),
            (float("inf"), 0.5, "non-finite activity gradient"),
        ],
    )
    def test_rejects_nonfinite_gradient_outputs(self, gradient_value: float, loss: float, message: str) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[gradient_value, 0.0, 0.0, 0.0]] * 4],
                loss=loss,
                sample_metrics=[{"loss": loss}],
                metrics={},
            )
            with pytest.raises(ValueError, match=message):
                parade_utr_activity_gradient_backward(
                    [(_seq("ACGT", np.zeros((4, 4))),)], config=ParadeUTRActivityConfig(device="cpu")
                )

    def test_rejects_noncanonical_logit_width(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(L, 4\)"):
            parade_utr_activity_gradient_backward(
                [(_seq("ACGT", np.zeros((4, 5))),)], config=ParadeUTRActivityConfig(device="cpu")
            )

    def test_rejects_logit_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="L matches the sequence length"):
            parade_utr_activity_gradient_backward(
                [(_seq("ACGT", np.zeros((5, 4))),)], config=ParadeUTRActivityConfig(device="cpu")
            )

    def test_rejects_gradient_shape_mismatch(self) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = _gradient_output(
                gradient=[[[0.0, 0.0, 0.0, 0.0]] * 3], loss=0.5, metrics={"c2": 2.0}
            )
            with pytest.raises(ValueError, match=r"activity gradient with shape \(3, 4\).+expected \(4, 4\)"):
                parade_utr_activity_gradient_backward(
                    [(_seq("ACGT", np.zeros((4, 4))),)], config=ParadeUTRActivityConfig(device="cpu")
                )

    @pytest.mark.parametrize(("gradient_count", "metrics_count"), [(0, 1), (1, 0)])
    def test_rejects_gradient_output_cardinality_mismatch(self, gradient_count: int, metrics_count: int) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[0.0] * 4] * 4] * gradient_count,
                loss=0.5,
                sample_metrics=[{"loss": 0.5, "c2": 2.0}] * metrics_count,
                metrics={},
            )
            with pytest.raises(ValueError, match="gradient output cardinality mismatch"):
                parade_utr_activity_gradient_backward(
                    [(_seq("ACGT", np.zeros((4, 4))),)], config=ParadeUTRActivityConfig(device="cpu")
                )

    def test_accepts_rna_logits_in_acgu_order(self) -> None:
        sequence = Sequence("ACGU", "rna")
        sequence.logits = np.zeros((4, 4))
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = _gradient_output(
                gradient=[[[0.0, 0.0, 0.0, 0.0]] * 4], loss=0.5, metrics={"c2": 2.0}
            )
            (result,) = parade_utr_activity_gradient_backward(
                [(sequence,)], config=ParadeUTRActivityConfig(device="cpu")
            )
        assert mock_run.call_args.args[0].logits == [np.zeros((4, 4)).tolist()]
        assert result.gradient[0].shape == (4, 4)

    def test_rejects_subnormal_scheduled_temperature(self) -> None:
        with pytest.raises(ValueError, match="gradient temperature"):
            parade_utr_activity_gradient_backward(
                [(_seq("ACGT", np.zeros((4, 4))),)],
                config=ParadeUTRActivityConfig(device="cpu"),
                temperature=5e-324,
            )

    def test_rejects_four_column_noncanonical_vocabulary(self) -> None:
        seq = Sequence("ACGN", "dna", valid_chars={"A", "C", "G", "N"})
        seq.logits = np.zeros((4, 4))
        with pytest.raises(ValueError, match=r"vocabulary \['A', 'C', 'G', 'N'\]"):
            parade_utr_activity_gradient_backward([(seq,)], config=ParadeUTRActivityConfig(device="cpu"))

    def test_empty_input_returns_empty(self) -> None:
        assert parade_utr_activity_gradient_backward([], config=ParadeUTRActivityConfig()) == []


class TestConfigAndRegistry:
    def test_rejects_offpanel_cell(self) -> None:
        with pytest.raises(ValueError, match="not in the utr5 panel"):
            ParadeUTRActivityConfig(construct_type="utr5", cell_type="c13")  # c13 is utr3-only

    def test_rejects_subnormal_sigmoid_scale(self) -> None:
        with pytest.raises(ValueError):
            ParadeUTRActivityConfig(sigmoid_scale=5e-324)

    def test_rejects_subnormal_config_temperature(self) -> None:
        with pytest.raises(ValueError):
            ParadeUTRActivityConfig(temperature=5e-324)

    @pytest.mark.parametrize("field", ["sigmoid_center", "sigmoid_scale", "temperature"])
    def test_rejects_nonfinite_gradient_config(self, field: str) -> None:
        with pytest.raises(ValueError):
            ParadeUTRActivityConfig(**{field: float("nan")})

    def test_rejects_protein_input(self) -> None:
        segment = Segment(sequence="MKT", sequence_type="protein")
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[segment],
                function=parade_utr_activity_constraint,
                function_config=ParadeUTRActivityConfig(),
            )

    def test_registers_dual_mode_with_one_logits_slot(self) -> None:
        spec = ConstraintRegistry.get("parade-utr-activity")
        assert spec.mode == "dual"
        assert spec.function is parade_utr_activity_constraint
        assert spec.backward is parade_utr_activity_gradient_backward
        assert spec.input_labels == [InputSlot(label="Sequence", requires_logits=True)]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_real_model_forward_backward_consistency() -> None:
    """Forward sequence scoring and hard-forward gradient mode share one activity objective."""
    sequence = "ACGT" * 12 + "AC"
    config = ParadeUTRActivityConfig(construct_type="utr5", cell_type="c2", direction="max", device="cuda")
    (forward,) = parade_utr_activity_constraint([(Sequence(sequence, "dna"),)], config)

    gradient_sequence = Sequence(sequence, "dna")
    logits = np.full((len(sequence), 4), -20.0)
    nucleotide_index = {base: idx for idx, base in enumerate("ACGT")}
    for position, base in enumerate(sequence):
        logits[position, nucleotide_index[base]] = 20.0
    gradient_sequence.logits = logits
    (backward,) = parade_utr_activity_gradient_backward(
        [(gradient_sequence,)], config=config, temperature=1.0, soft=1.0, hard=1.0
    )

    # The two paths run the model separately, so they agree to float32 noise rather than exactly.
    assert forward.score == pytest.approx(backward.loss, rel=1e-4, abs=1e-6)
    assert backward.metrics["activity"] == pytest.approx(forward.metadata["activity"], rel=1e-4)
    assert np.isfinite(backward.gradient[0]).all()
    assert np.any(backward.gradient[0] != 0.0)
