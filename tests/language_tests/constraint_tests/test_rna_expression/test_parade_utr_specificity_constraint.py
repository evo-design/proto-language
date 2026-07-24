"""Tests for the PARADE UTR cell-type-specificity constraint (forward + gradient)."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from proto_tools import ParadeGradientSampleMetrics

from proto_language import (
    ParadeUTRSpecificityConfig,
    parade_utr_specificity_constraint,
    parade_utr_specificity_gradient_backward,
)
from proto_language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.core import Constraint, Segment, Sequence
from proto_language.utils import sigmoid_score

_MODULE = "proto_language.constraint.rna_expression.parade_utr_specificity_constraint"


def _seq(sequence: str, logits: np.ndarray | None = None) -> Sequence:
    seq = Sequence(sequence, "dna")
    if logits is not None:
        seq.logits = logits
    return seq


class TestForward:
    @pytest.mark.parametrize(
        ("on_activity", "off_activity"),
        [(5.0, 1.0), (1.0, 5.0), (2.0, 2.0)],
    )
    def test_parametrized_scoring_formula(self, on_activity: float, off_activity: float) -> None:
        config = ParadeUTRSpecificityConfig(device="cpu")
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(
                results=[SimpleNamespace(scores={"c2": on_activity, "c6": off_activity})]
            )
            (result,) = parade_utr_specificity_constraint([(_seq("ACGT"),)], config)
        expected = 0.5 * (
            (1.0 - sigmoid_score(on_activity, config.sigmoid_center, slope=1.0 / config.sigmoid_scale))
            + sigmoid_score(off_activity, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        )
        assert result.score == pytest.approx(expected)

    def test_high_on_minus_off_gives_low_energy(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(results=[SimpleNamespace(scores={"c2": 5.0, "c6": 1.0})])
            (result,) = parade_utr_specificity_constraint(
                [(_seq("ACGT" * 12 + "AC"),)],
                ParadeUTRSpecificityConfig(construct_type="utr5", on_target="c2", off_target="c6", device="cpu"),
            )
        expected = 0.5 * ((1.0 - sigmoid_score(5.0, 2.0, slope=1.0)) + sigmoid_score(1.0, 2.0, slope=1.0))
        assert result.score == pytest.approx(expected)
        assert result.metadata["on_activity"] == 5.0
        assert result.metadata["off_activity"] == 1.0
        assert result.metadata["specificity"] == 4.0
        # forward requests both cell codes from the tool
        _, tool_config = mock_run.call_args[0]
        assert tool_config.cell_types == ["c2", "c6"]

    def test_negative_gap_gives_high_energy(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(results=[SimpleNamespace(scores={"c2": 1.0, "c6": 5.0})])
            (result,) = parade_utr_specificity_constraint(
                [(_seq("ACGTACGT"),)],
                ParadeUTRSpecificityConfig(on_target="c2", off_target="c6", device="cpu"),
            )
        assert result.score > 0.8  # off > on -> high (bad) energy

    def test_extreme_activities_are_numerically_stable(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(results=[SimpleNamespace(scores={"c2": -1000.0, "c6": 1000.0})])
            (result,) = parade_utr_specificity_constraint(
                [(_seq("ACGT"),)],
                ParadeUTRSpecificityConfig(sigmoid_scale=0.001, device="cpu"),
            )
        assert result.score == 1.0

    def test_rejects_nonfinite_tool_activity(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(results=[SimpleNamespace(scores={"c2": float("nan"), "c6": 1.0})])
            with pytest.raises(ValueError, match="non-finite specificity activities"):
                parade_utr_specificity_constraint([(_seq("ACGT"),)], ParadeUTRSpecificityConfig(device="cpu"))

    def test_rejects_overflowed_specificity_difference(self) -> None:
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(results=[SimpleNamespace(scores={"c2": 1e308, "c6": -1e308})])
            with pytest.raises(ValueError, match="specificity difference is non-finite"):
                parade_utr_specificity_constraint([(_seq("ACGT"),)], ParadeUTRSpecificityConfig(device="cpu"))

    def test_empty_input_returns_empty(self) -> None:
        assert parade_utr_specificity_constraint([], ParadeUTRSpecificityConfig()) == []

    def test_evaluate_propagates_metadata(self) -> None:
        segment = Segment(sequence="ACGT", sequence_type="dna")
        constraint = Constraint(
            inputs=[segment],
            function=parade_utr_specificity_constraint,
            function_config=ParadeUTRSpecificityConfig(device="cpu"),
        )
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(results=[SimpleNamespace(scores={"c2": 4.0, "c6": 1.0})])
            constraint.evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["parade_utr_specificity_constraint"]["data"]
        assert data["specificity"] == 3.0

    def test_rna_evaluate_propagates_metadata(self) -> None:
        segment = Segment(sequence="ACGU", sequence_type="rna")
        constraint = Constraint(
            inputs=[segment],
            function=parade_utr_specificity_constraint,
            function_config=ParadeUTRSpecificityConfig(device="cpu"),
        )
        with patch(f"{_MODULE}.run_parade_activity") as mock_run:
            mock_run.return_value = SimpleNamespace(results=[SimpleNamespace(scores={"c2": 4.0, "c6": 1.0})])
            constraint.evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["parade_utr_specificity_constraint"]["data"]
        assert data["specificity"] == 3.0


class TestBackward:
    def test_two_loss_terms_on_max_off_min(self) -> None:
        grad_LxC = [[0.05, -0.05, 0.1, -0.1]] * 4
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[grad_LxC],
                loss=1.2,
                sample_metrics=[{"loss": 1.2, "c2": 3.0, "c6": 1.0}],
                metrics={},
            )
            (result,) = parade_utr_specificity_gradient_backward(
                [(_seq("ACGT", np.ones((4, 4)) / 4.0),)],
                config=ParadeUTRSpecificityConfig(
                    construct_type="utr5", on_target="c2", off_target="c6", temperature=0.9, device="cpu"
                ),
            )
        tool_input, tool_config = mock_run.call_args[0]
        assert tool_input.temperature == 0.9
        assert tool_config.compute_gradient is True
        terms = {(t.cell_type, t.direction) for t in tool_config.loss_terms}
        assert terms == {("c2", "max"), ("c6", "min")}
        assert [term.weight for term in tool_config.loss_terms] == [0.5, 0.5]
        assert len(result.gradient) == 1 and result.gradient[0].shape == (4, 4)
        assert result.loss == 1.2
        assert result.metrics["on_target"] == "c2" and result.metrics["off_target"] == "c6"

    def test_optimizer_relaxation_is_forwarded(self) -> None:
        grad = [[[0.0, 0.0, 0.0, 0.0]] * 4]
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=grad,
                loss=0.5,
                sample_metrics=[{"loss": 0.5, "c2": 3.0, "c6": 1.0}],
                metrics={},
            )
            parade_utr_specificity_gradient_backward(
                [(_seq("ACGT", np.zeros((4, 4))),)],
                config=ParadeUTRSpecificityConfig(device="cpu"),
                temperature=0.6,
                soft=0.2,
                hard=0.7,
            )
        tool_input, tool_config = mock_run.call_args.args
        assert tool_input.temperature == 0.6
        assert tool_config.soft == 0.2 and tool_config.hard == 0.7

    def test_batches_and_preserves_per_proposal_losses(self) -> None:
        gradients = [
            [[0.1, 0.0, 0.0, 0.0]] * 4,
            [[0.0, 0.1, 0.0, 0.0]] * 4,
        ]
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=gradients,
                loss=0.9,
                sample_metrics=[
                    {"loss": 0.7, "c2": 1.0, "c6": 3.0},
                    {"loss": 0.2, "c2": 4.0, "c6": 1.0},
                ],
                metrics={},
            )
            results = parade_utr_specificity_gradient_backward(
                [(_seq("ACGT", np.zeros((4, 4))),), (_seq("TGCA", np.zeros((4, 4))),)],
                config=ParadeUTRSpecificityConfig(batch_size=2, device="cpu"),
            )
        assert mock_run.call_count == 1
        assert [result.loss for result in results] == [0.7, 0.2]
        assert [result.metrics["c2"] for result in results] == [1.0, 4.0]

    def test_normalizes_real_sample_metrics(self) -> None:
        sample_metrics = ParadeGradientSampleMetrics(
            loss=0.25,
            c2=4.0,
            c6=1.5,
            loss_terms=[{"cell_type": "c2"}, {"cell_type": "c6"}],
        )
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[0.0, 0.0, 0.0, 0.0]] * 4],
                loss=0.25,
                sample_metrics=[sample_metrics],
                metrics={},
            )
            (result,) = parade_utr_specificity_gradient_backward(
                [(_seq("ACGT", np.zeros((4, 4))),)],
                config=ParadeUTRSpecificityConfig(device="cpu"),
            )
        assert result.metrics["on_activity"] == 4.0
        assert result.metrics["off_activity"] == 1.5
        assert result.metrics["specificity"] == 2.5
        assert result.metrics["loss_terms"] == sample_metrics.loss_terms

    def test_mixed_length_grouping_preserves_order(self) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.side_effect = [
                SimpleNamespace(
                    gradient=[[[0.0] * 4] * 4, [[0.0] * 4] * 4],
                    loss=0.4,
                    sample_metrics=[
                        {"loss": 0.1, "c2": 4.0, "c6": 1.0},
                        {"loss": 0.3, "c2": 2.0, "c6": 1.0},
                    ],
                    metrics={},
                ),
                SimpleNamespace(
                    gradient=[[[0.0] * 4] * 5],
                    loss=0.2,
                    sample_metrics=[{"loss": 0.2, "c2": 3.0, "c6": 1.0}],
                    metrics={},
                ),
            ]
            results = parade_utr_specificity_gradient_backward(
                [
                    (_seq("ACGT", np.zeros((4, 4))),),
                    (_seq("ACGTA", np.zeros((5, 4))),),
                    (_seq("TGCA", np.zeros((4, 4))),),
                ],
                config=ParadeUTRSpecificityConfig(batch_size=2, device="cpu"),
            )
        assert mock_run.call_count == 2
        assert [result.loss for result in results] == [0.1, 0.2, 0.3]
        assert [result.gradient[0].shape for result in results] == [(4, 4), (5, 4), (4, 4)]

    def test_rejects_noncanonical_vocabulary(self) -> None:
        seq = Sequence("ACGN", "dna", valid_chars={"A", "C", "G", "N"})
        seq.logits = np.zeros((4, 4))
        with pytest.raises(ValueError, match="canonical DNA/RNA logits"):
            parade_utr_specificity_gradient_backward([(seq,)], config=ParadeUTRSpecificityConfig(device="cpu"))

    def test_rejects_logit_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="L matches the sequence length"):
            parade_utr_specificity_gradient_backward(
                [(_seq("ACGT", np.zeros((5, 4))),)], config=ParadeUTRSpecificityConfig(device="cpu")
            )

    def test_rejects_gradient_shape_mismatch(self) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[0.0, 0.0, 0.0, 0.0]] * 3],
                loss=0.5,
                sample_metrics=[{"loss": 0.5, "c2": 3.0, "c6": 1.0}],
                metrics={},
            )
            with pytest.raises(ValueError, match=r"specificity gradient with shape \(3, 4\).+expected \(4, 4\)"):
                parade_utr_specificity_gradient_backward(
                    [(_seq("ACGT", np.zeros((4, 4))),)], config=ParadeUTRSpecificityConfig(device="cpu")
                )

    @pytest.mark.parametrize(("gradient_count", "metrics_count"), [(0, 1), (1, 0)])
    def test_rejects_gradient_output_cardinality_mismatch(self, gradient_count: int, metrics_count: int) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[0.0] * 4] * 4] * gradient_count,
                loss=0.5,
                sample_metrics=[{"loss": 0.5, "c2": 3.0, "c6": 1.0}] * metrics_count,
                metrics={},
            )
            with pytest.raises(ValueError, match="gradient output cardinality mismatch"):
                parade_utr_specificity_gradient_backward(
                    [(_seq("ACGT", np.zeros((4, 4))),)], config=ParadeUTRSpecificityConfig(device="cpu")
                )

    def test_accepts_rna_logits_in_acgu_order(self) -> None:
        sequence = Sequence("ACGU", "rna")
        sequence.logits = np.zeros((4, 4))
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[0.0, 0.0, 0.0, 0.0]] * 4],
                loss=0.5,
                sample_metrics=[{"loss": 0.5, "c2": 3.0, "c6": 1.0}],
                metrics={},
            )
            (result,) = parade_utr_specificity_gradient_backward(
                [(sequence,)], config=ParadeUTRSpecificityConfig(device="cpu")
            )
        assert mock_run.call_args.args[0].logits == [np.zeros((4, 4)).tolist()]
        assert result.gradient[0].shape == (4, 4)

    def test_rejects_subnormal_scheduled_temperature(self) -> None:
        with pytest.raises(ValueError, match="gradient temperature"):
            parade_utr_specificity_gradient_backward(
                [(_seq("ACGT", np.zeros((4, 4))),)],
                config=ParadeUTRSpecificityConfig(device="cpu"),
                temperature=5e-324,
            )

    def test_rejects_nonfinite_gradient_loss(self) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[0.0, 0.0, 0.0, 0.0]] * 4],
                loss=float("inf"),
                sample_metrics=[{"loss": float("inf")}],
                metrics={},
            )
            with pytest.raises(ValueError, match="non-finite specificity loss"):
                parade_utr_specificity_gradient_backward(
                    [(_seq("ACGT", np.zeros((4, 4))),)],
                    config=ParadeUTRSpecificityConfig(device="cpu"),
                )

    @pytest.mark.parametrize(
        ("gradient_value", "on_activity", "off_activity", "message"),
        [
            (float("inf"), 3.0, 1.0, "non-finite specificity gradient"),
            (0.0, float("nan"), 1.0, "non-finite gradient specificity activities"),
            (0.0, 1e308, -1e308, "non-finite gradient specificity activities"),
        ],
    )
    def test_rejects_nonfinite_gradient_outputs(
        self, gradient_value: float, on_activity: float, off_activity: float, message: str
    ) -> None:
        with patch(f"{_MODULE}.run_parade_gradient") as mock_run:
            mock_run.return_value = SimpleNamespace(
                gradient=[[[gradient_value, 0.0, 0.0, 0.0]] * 4],
                loss=0.5,
                sample_metrics=[{"loss": 0.5, "c2": on_activity, "c6": off_activity}],
                metrics={},
            )
            with pytest.raises(ValueError, match=message):
                parade_utr_specificity_gradient_backward(
                    [(_seq("ACGT", np.zeros((4, 4))),)],
                    config=ParadeUTRSpecificityConfig(device="cpu"),
                )

    def test_empty_input_returns_empty(self) -> None:
        assert parade_utr_specificity_gradient_backward([], config=ParadeUTRSpecificityConfig()) == []


class TestConfigAndRegistry:
    def test_rejects_identical_targets(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            ParadeUTRSpecificityConfig(on_target="c2", off_target="c2")

    def test_rejects_offpanel_target(self) -> None:
        with pytest.raises(ValueError, match="not in the utr5 panel"):
            ParadeUTRSpecificityConfig(construct_type="utr5", on_target="c13", off_target="c2")

    @pytest.mark.parametrize("field", ["sigmoid_center", "sigmoid_scale", "temperature"])
    def test_rejects_nonfinite_gradient_config(self, field: str) -> None:
        with pytest.raises(ValueError):
            ParadeUTRSpecificityConfig(**{field: float("nan")})

    def test_rejects_protein_input(self) -> None:
        segment = Segment(sequence="MKT", sequence_type="protein")
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[segment],
                function=parade_utr_specificity_constraint,
                function_config=ParadeUTRSpecificityConfig(),
            )

    def test_registers_dual_mode_with_one_logits_slot(self) -> None:
        spec = ConstraintRegistry.get("parade-utr-specificity")
        assert spec.mode == "dual"
        assert spec.function is parade_utr_specificity_constraint
        assert spec.backward is parade_utr_specificity_gradient_backward
        assert spec.input_labels == [InputSlot(label="Sequence", requires_logits=True)]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_real_model_forward_backward_consistency() -> None:
    """Forward sequence scoring and hard-forward gradient mode share one specificity objective."""
    sequence = "ACGT" * 12 + "AC"
    config = ParadeUTRSpecificityConfig(construct_type="utr5", on_target="c2", off_target="c6", device="cuda")
    (forward,) = parade_utr_specificity_constraint([(Sequence(sequence, "dna"),)], config)

    gradient_sequence = Sequence(sequence, "dna")
    logits = np.full((len(sequence), 4), -20.0)
    nucleotide_index = {base: idx for idx, base in enumerate("ACGT")}
    for position, base in enumerate(sequence):
        logits[position, nucleotide_index[base]] = 20.0
    gradient_sequence.logits = logits
    (backward,) = parade_utr_specificity_gradient_backward(
        [(gradient_sequence,)], config=config, temperature=1.0, soft=1.0, hard=1.0
    )

    assert forward.score == pytest.approx(backward.loss, rel=1e-5, abs=1e-6)
    assert backward.metrics["on_activity"] == pytest.approx(forward.metadata["on_activity"], rel=1e-5)
    assert backward.metrics["off_activity"] == pytest.approx(forward.metadata["off_activity"], rel=1e-5)
    assert np.isfinite(backward.gradient[0]).all()
    assert np.any(backward.gradient[0] != 0.0)
