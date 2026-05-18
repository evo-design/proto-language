"""Numerical optimizers for gradient-based sequence design (SGD, Adam)."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from proto_language.base_config import BaseConfig, ConfigField

MLOptimizerType = Literal["sgd", "adam"]


class AdamConfig(BaseConfig):
    """Adam optimizer hyperparameters.

    Attributes:
        beta1 (float): First moment decay rate.
        beta2 (float): Second moment decay rate.
        eps (float): Numerical stability term.
    """

    beta1: float = ConfigField(default=0.9, ge=0.0, lt=1.0, title="β₁", description="First moment decay rate.")
    beta2: float = ConfigField(default=0.999, ge=0.0, lt=1.0, title="β₂", description="Second moment decay rate.")
    eps: float = ConfigField(default=1e-8, gt=0.0, title="ε", description="Numerical stability term.")


class MLOptimizer(ABC):
    """Applies a gradient update to logits."""

    def __init__(self, config: BaseConfig | None = None) -> None:  # noqa: B027
        """Initialize from config. Subclasses override to read their specific fields."""

    @abstractmethod
    def step(self, logits: np.ndarray, gradient: np.ndarray, lr: float, trajectory: int, step: int) -> np.ndarray:
        """Apply one update step, return new logits. ``step`` is 1-indexed."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (e.g. Adam moments)."""


class SGD(MLOptimizer):
    """Vanilla stochastic gradient descent."""

    def step(
        self,
        logits: np.ndarray,
        gradient: np.ndarray,
        lr: float,
        trajectory: int,  # noqa: ARG002
        step: int,  # noqa: ARG002
    ) -> np.ndarray:
        """Return ``logits - lr * gradient``."""
        return logits - lr * gradient

    def reset(self) -> None:
        """No-op — SGD is stateless."""


class Adam(MLOptimizer):
    """Adam optimizer with per-trajectory state and bias correction."""

    def __init__(self, config: AdamConfig | None = None) -> None:
        """Initialize from config (defaults used if None)."""
        cfg = config or AdamConfig()
        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.eps = cfg.eps
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self, logits: np.ndarray, gradient: np.ndarray, lr: float, trajectory: int, step: int) -> np.ndarray:
        """Apply Adam update with bias correction for the given trajectory."""
        if step < 1:
            raise ValueError(f"step must be >= 1 (got {step})")
        if trajectory not in self._m:
            self._m[trajectory] = np.zeros_like(gradient)
            self._v[trajectory] = np.zeros_like(gradient)

        self._m[trajectory] = self.beta1 * self._m[trajectory] + (1 - self.beta1) * gradient
        self._v[trajectory] = self.beta2 * self._v[trajectory] + (1 - self.beta2) * gradient**2

        m_hat = self._m[trajectory] / (1 - self.beta1**step)
        v_hat = self._v[trajectory] / (1 - self.beta2**step)

        return logits - lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        """Clear moment buffers for all trajectories."""
        self._m.clear()
        self._v.clear()


ML_OPTIMIZERS: dict[MLOptimizerType, type[MLOptimizer]] = {
    "sgd": SGD,
    "adam": Adam,
}
