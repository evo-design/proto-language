"""Tests for DifferentiableConstraint and GradientResult."""

import copy

import numpy as np
import pytest
from pydantic import BaseModel

from constraint_tests.utils import mock_single_input_scoring_function
from proto_language.language.core import Constraint, Segment
from proto_language.language.core.differentiable_constraint import (
    DifferentiableConstraint,
    GradientResult,
)


class MockConfig(BaseModel):
    """Empty config for mock constraint functions."""


class MockBackwardConfig(BaseModel):
    """Separate config for mock backward function."""

    loss_type: str = "plddt"


def _make_segment(sequences: list[str], seq_type: str = "dna") -> Segment:
    segment = Segment(sequence=sequences[0], sequence_type=seq_type)
    segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(len(sequences))]
    for i, seq_str in enumerate(sequences):
        segment.proposal_sequences[i].sequence = seq_str
    return segment


def _mock_backward(logits: np.ndarray, temperature: float, *, config: BaseModel) -> GradientResult:
    return GradientResult(
        gradient=-logits * temperature, loss=float(np.mean(logits**2)), metrics={"temperature": temperature}
    )


def _make_dc(segment: Segment | None = None, **kwargs: object) -> DifferentiableConstraint:
    if segment is None:
        segment = _make_segment(["ACTGACTG"])
    defaults: dict[str, object] = {
        "inputs": [segment],
        "function": mock_single_input_scoring_function,
        "function_config": MockConfig(),
        "backward": _mock_backward,
    }
    defaults.update(kwargs)
    return DifferentiableConstraint(**defaults)


class TestGradientResult:
    def test_construction_and_defaults(self) -> None:
        result = GradientResult(gradient=np.array([[1.0, 2.0], [3.0, 4.0]]), loss=0.5)
        assert result.loss == 0.5
        assert result.metrics == {}
        assert result.gradient.shape == (2, 2)

    def test_custom_metrics_and_repr(self) -> None:
        result = GradientResult(
            gradient=np.zeros((5, 20)),
            loss=0.5,
            metrics={"plddt": 0.85, "ptm": 0.72},
        )
        assert result.metrics["plddt"] == pytest.approx(0.85)
        assert result.metrics["ptm"] == pytest.approx(0.72)
        assert repr(result) == "GradientResult(gradient=ndarray(5, 20), loss=0.5, metrics={'plddt': 0.85, 'ptm': 0.72})"

    def test_frozen(self) -> None:
        result = GradientResult(gradient=np.zeros((5, 20)), loss=1.0)
        with pytest.raises(AttributeError):
            result.loss = 2.0  # type: ignore[misc]


class TestDifferentiableConstraint:
    def test_backward_config_defaults_to_function_config(self) -> None:
        config = MockConfig()
        dc = _make_dc(function_config=config)
        assert dc.backward_config is config

    def test_separate_backward_config(self) -> None:
        bwd_config = MockBackwardConfig(loss_type="ptm")
        dc = _make_dc(backward_config=bwd_config)
        assert dc.backward_config is bwd_config
        assert dc.backward_config.loss_type == "ptm"

    def test_backward_property(self) -> None:
        dc = _make_dc()
        assert dc.backward is _mock_backward

    def test_weight_and_label(self) -> None:
        dc = _make_dc(weight=2.5, label="plddt_gradient")
        assert dc.weight == 2.5
        assert dc.label == "plddt_gradient"

    def test_threshold_and_weight_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match=r"Both threshold.*and weight.*are set"):
            _make_dc(threshold=0.5, weight=2.0)

    def test_isinstance_discovery(self) -> None:
        segment = _make_segment(["ACTGACTG"])
        regular = Constraint(
            inputs=[segment], function=mock_single_input_scoring_function, function_config=MockConfig()
        )
        diff = _make_dc(segment=segment)

        assert isinstance(diff, Constraint)
        assert isinstance(diff, DifferentiableConstraint)
        assert not isinstance(regular, DifferentiableConstraint)

        found = [c for c in [regular, diff] if isinstance(c, DifferentiableConstraint)]
        assert found == [diff]

    def test_compute_gradient(self) -> None:
        dc = _make_dc()
        logits = np.random.randn(8, 4)
        result = dc.compute_gradient(logits, temperature=1.0)
        assert isinstance(result, GradientResult)
        assert result.gradient.shape == (8, 4)
        np.testing.assert_array_almost_equal(result.gradient, -logits)

    def test_temperature_affects_gradient(self) -> None:
        dc = _make_dc()
        logits = np.ones((8, 4))
        hot = dc.compute_gradient(logits, temperature=2.0)
        cold = dc.compute_gradient(logits, temperature=0.5)
        assert np.abs(hot.gradient).mean() > np.abs(cold.gradient).mean()

    def test_backward_config_forwarded(self) -> None:
        received: list[BaseModel] = []

        def capturing_backward(logits: np.ndarray, temperature: float, *, config: BaseModel) -> GradientResult:
            received.append(config)
            return GradientResult(gradient=np.zeros_like(logits), loss=0.0)

        bwd_config = MockBackwardConfig(loss_type="ptm")
        dc = _make_dc(backward=capturing_backward, backward_config=bwd_config)
        dc.compute_gradient(np.zeros((8, 4)), temperature=1.0)
        assert received == [bwd_config]

    @pytest.mark.parametrize(
        ("logits", "temperature", "match"),
        [
            (np.zeros(8), 1.0, r"logits must have shape"),
            (np.zeros((8, 4)), 0.0, r"temperature must be positive"),
        ],
    )
    def test_compute_gradient_validates_inputs(self, logits: np.ndarray, temperature: float, match: str) -> None:
        dc = _make_dc()
        with pytest.raises(ValueError, match=match):
            dc.compute_gradient(logits, temperature=temperature)

    @pytest.mark.parametrize(
        ("backward", "expected_exception", "match"),
        [
            (
                lambda logits, temperature, *, config: {"gradient": np.zeros_like(logits), "loss": 0.0},
                TypeError,
                r"must return GradientResult",
            ),
            (
                lambda logits, temperature, *, config: GradientResult(gradient=np.zeros((4, 8)), loss=0.0),
                ValueError,
                r"returned gradient with shape",
            ),
        ],
    )
    def test_compute_gradient_validates_backward_result(
        self,
        backward: object,
        expected_exception: type[Exception],
        match: str,
    ) -> None:
        dc = _make_dc(backward=backward)
        with pytest.raises(expected_exception, match=match):
            dc.compute_gradient(np.zeros((8, 4)), temperature=1.0)

    def test_evaluate_inherited(self) -> None:
        segment = _make_segment(["ACTGACTG", "TTTTTTTT"])
        dc = _make_dc(segment=segment)
        scores = dc.evaluate()
        assert scores[0] == pytest.approx(0.25)
        assert scores[1] == pytest.approx(1.0)

    def test_evaluate_as_filter(self) -> None:
        segment = _make_segment(["ACTGACTG", "TTTTTTTT"])
        dc = _make_dc(segment=segment, threshold=0.5)
        results = dc.evaluate()
        assert results[0] is True
        assert results[1] is False

    def test_metadata_propagation(self) -> None:
        segment = _make_segment(["ACTGACTG"])
        dc = _make_dc(segment=segment)
        dc.evaluate()
        metadata = segment.proposal_sequences[0]._constraints_metadata
        assert metadata["mock_single_input_scoring_function"]["score"] == pytest.approx(0.25)
