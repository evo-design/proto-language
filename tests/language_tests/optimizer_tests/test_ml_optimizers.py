"""Tests for pluggable ML optimizers (SGD, Adam)."""

import numpy as np
import pytest

from proto_language.utils.ml_optimizers import ML_OPTIMIZERS, SGD, Adam, AdamConfig


def test_sgd_update() -> None:
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = SGD().step(logits, grad, lr=0.5, trajectory=0, step=1)
    np.testing.assert_allclose(result, logits - 0.5 * grad)


def test_adam_exact_values() -> None:
    adam = Adam(AdamConfig(beta1=0.9, beta2=0.999, eps=1e-8))
    logits = np.zeros((1, 2))
    grad = np.array([[1.0, 2.0]])
    result = adam.step(logits, grad, lr=0.1, trajectory=0, step=1)
    # m = 0.1*[1,2], v = 0.001*[1,4]; bias-corrected: m_hat=[1,2], v_hat=[1,4]
    expected = -0.1 * np.array([[1.0, 2.0]]) / (np.sqrt(np.array([[1.0, 4.0]])) + 1e-8)
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_adam_per_trajectory_independence() -> None:
    adam = Adam()
    z = np.zeros((2, 5))
    adam.step(z.copy(), np.ones((2, 5)), lr=0.1, trajectory=0, step=1)
    adam.step(z.copy(), -np.ones((2, 5)), lr=0.1, trajectory=1, step=1)
    assert not np.array_equal(adam._m[0], adam._m[1])


def test_adam_reset() -> None:
    adam = Adam()
    grad = np.ones((2, 5))
    r1 = adam.step(np.zeros((2, 5)), grad, lr=0.1, trajectory=0, step=1)
    adam.step(np.zeros((2, 5)), grad, lr=0.1, trajectory=0, step=2)
    adam.reset()
    r1_after = adam.step(np.zeros((2, 5)), grad, lr=0.1, trajectory=0, step=1)
    np.testing.assert_array_equal(r1, r1_after)


@pytest.mark.parametrize("name", list(ML_OPTIMIZERS.keys()))
def test_registry_entries_produce_correct_shape(name: str) -> None:
    result = ML_OPTIMIZERS[name]().step(np.zeros((2, 3)), np.ones((2, 3)), lr=0.1, trajectory=0, step=1)
    assert result.shape == (2, 3)
