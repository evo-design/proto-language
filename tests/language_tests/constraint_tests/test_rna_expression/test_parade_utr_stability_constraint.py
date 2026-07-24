"""Tests for the PARADE 3' UTR mRNA-stability constraint (forward scoring)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proto_language import ParadeUTRStabilityConfig, parade_utr_stability_constraint
from proto_language.constraint.constraint_registry import ConstraintRegistry
from proto_language.core import Constraint, Segment, Sequence

_MODULE = "proto_language.constraint.rna_expression.parade_utr_stability_constraint"


def _stability_output(log_ratios: list[float]) -> SimpleNamespace:
    """Mock a ParadeStabilityOutput with one result per sequence."""
    return SimpleNamespace(results=[SimpleNamespace(log_ratio=lr) for lr in log_ratios])


class TestForward:
    @pytest.mark.parametrize(
        ("log_ratio", "direction", "expected"),
        [(5.0, "max", 0.0), (5.0, "min", 1.0), (0.0, "max", 0.5)],
    )
    def test_parametrized_scoring_formula(self, log_ratio: float, direction: str, expected: float) -> None:
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([log_ratio])
            (result,) = parade_utr_stability_constraint(
                [(Sequence("ACGT", "dna"),)],
                ParadeUTRStabilityConfig(direction=direction, sigmoid_scale=0.01, device="cpu"),
            )
        assert result.score == pytest.approx(expected, abs=1e-12)

    def test_max_rewards_high_stability_with_low_energy(self) -> None:
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([5.0])
            (result,) = parade_utr_stability_constraint(
                [(Sequence("ACGT" * 46 + "AC", "dna"),)],
                ParadeUTRStabilityConfig(direction="max", device="cpu"),
            )
        assert result.score < 0.01  # high log-ratio under 'max' -> low energy
        assert result.metadata == {"direction": "max", "log_ratio": 5.0}

    def test_min_flips_direction(self) -> None:
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([5.0])
            (result,) = parade_utr_stability_constraint(
                [(Sequence("ACGTACGT", "dna"),)],
                ParadeUTRStabilityConfig(direction="min", device="cpu"),
            )
        assert result.score > 0.99

    def test_batches_all_proposals_and_preserves_order(self) -> None:
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([5.0, -5.0])
            results = parade_utr_stability_constraint(
                [(Sequence("ACGT", "dna"),), (Sequence("TTTT", "dna"),)],
                ParadeUTRStabilityConfig(direction="max", device="cpu"),
            )
        assert len(results) == 2
        assert results[0].score < 0.01 and results[1].score > 0.99  # first stable, second unstable
        tool_input = mock_run.call_args[0][0]  # a single batched tool call
        assert tool_input.sequences == ["ACGT", "TTTT"]

    def test_empty_input_returns_empty(self) -> None:
        assert parade_utr_stability_constraint([], ParadeUTRStabilityConfig()) == []

    def test_extreme_log_ratio_is_numerically_stable(self) -> None:
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([-1000.0])
            (result,) = parade_utr_stability_constraint(
                [(Sequence("ACGT", "dna"),)],
                ParadeUTRStabilityConfig(direction="max", sigmoid_scale=0.001, device="cpu"),
            )
        assert result.score == 1.0

    def test_minimum_float32_scale_scores_center_as_half(self) -> None:
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([0.0])
            (result,) = parade_utr_stability_constraint(
                [(Sequence("ACGT", "dna"),)],
                ParadeUTRStabilityConfig(sigmoid_scale=1.1754943508222875e-38, device="cpu"),
            )
        assert result.score == 0.5

    def test_rejects_nonfinite_tool_output(self) -> None:
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([float("nan")])
            with pytest.raises(ValueError, match="non-finite stability log-ratio"):
                parade_utr_stability_constraint([(Sequence("ACGT", "dna"),)], ParadeUTRStabilityConfig(device="cpu"))

    def test_rna_evaluate_propagates_metadata(self) -> None:
        segment = Segment(sequence="ACGU", sequence_type="rna")
        constraint = Constraint(
            inputs=[segment],
            function=parade_utr_stability_constraint,
            function_config=ParadeUTRStabilityConfig(device="cpu"),
        )
        with patch(f"{_MODULE}.run_parade_stability") as mock_run:
            mock_run.return_value = _stability_output([1.25])
            constraint.evaluate()
        data = segment.proposal_sequences[0]._constraints_metadata["parade_utr_stability_constraint"]["data"]
        assert data["log_ratio"] == 1.25


class TestRegistry:
    def test_registers_forward_only(self) -> None:
        spec = ConstraintRegistry.get("parade-utr-stability")
        assert spec.mode == "discrete"  # forward-only; parade-gradient is activity-only
        assert spec.function is parade_utr_stability_constraint
        assert spec.backward is None

    @pytest.mark.parametrize("field", ["sigmoid_center", "sigmoid_scale"])
    def test_rejects_nonfinite_config(self, field: str) -> None:
        with pytest.raises(ValueError):
            ParadeUTRStabilityConfig(**{field: float("nan")})

    def test_rejects_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError):
            ParadeUTRStabilityConfig(batch_size=0)

    def test_rejects_protein_input(self) -> None:
        segment = Segment(sequence="MKT", sequence_type="protein")
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[segment],
                function=parade_utr_stability_constraint,
                function_config=ParadeUTRStabilityConfig(),
            )
