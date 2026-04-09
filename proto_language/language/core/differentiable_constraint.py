"""Differentiable constraint support for gradient-based sequence optimization.

Extends the base Constraint with a backward callable that computes gradients
of a scalar objective with respect to optimizer-owned relaxed logits. Used by
the gradient optimizer for activation maximization through frozen models while
preserving discrete `Constraint.evaluate()` behavior for existing optimizers.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

from proto_language.language.core.constraint import Constraint
from proto_language.language.core.segment import Segment


@dataclass(frozen=True)
class GradientResult:
    """Result of a gradient computation through a differentiable model.

    Attributes:
        gradient (np.ndarray): Gradient of the scalar objective with respect to
            the relaxed optimization input. For the current sequence-design
            path this typically matches the input logits shape.
        loss (float): Scalar objective value returned by the differentiable
            backend. This may already be a weighted combination of multiple
            model-local loss terms.
        metrics (dict[str, Any]): Optional model-specific auxiliary metrics
            (e.g., pLDDT, pTM) reported alongside ``loss``.
    """

    gradient: np.ndarray
    loss: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a compact repr that does not dump the full gradient array."""
        return f"GradientResult(gradient=ndarray{self.gradient.shape}, loss={self.loss}, metrics={self.metrics})"


class DifferentiableConstraint(Constraint):
    """A constraint that supports gradient computation for continuous optimization.

    Extends :class:`Constraint` with a ``backward`` callable that computes
    gradients of a scalar objective with respect to relaxed input logits. This
    enables gradient-based sequence optimization (activation maximization)
    while retaining full compatibility with discrete optimizers (MCMC, beam
    search, etc.) via the inherited :meth:`evaluate` method.

    The gradient optimizer discovers differentiable constraints via
    ``isinstance(constraint, DifferentiableConstraint)``.

    Note:
        ``evaluate()`` still operates on discrete proposal ``Sequence`` tuples,
        exactly like ``Constraint``. ``compute_gradient()`` is intentionally
        separate and consumes optimizer-owned relaxed logits instead. The future
        gradient optimizer is responsible for weighting and merging gradients
        across multiple differentiable constraints. Any weighting or
        combination of model-local loss terms inside a single differentiable
        backend should be reflected in the returned scalar ``loss``.

    Examples:
        >>> def mock_backward(logits, temperature, *, config):
        ...     return GradientResult(gradient=np.zeros_like(logits), loss=0.5)
        >>>
        >>> constraint = DifferentiableConstraint(
        ...     inputs=[segment],
        ...     function=scoring_function,
        ...     function_config=config,
        ...     backward=mock_backward,
        ...     weight=2.0,
        ... )
        >>> result = constraint.compute_gradient(logits, temperature=1.0)
        >>> result.gradient.shape == logits.shape
        >>> result.loss  # scalar

    Attributes:
        backward_config (BaseModel | dict[str, Any]): Configuration passed to the
            backward callable. Defaults to ``function_config`` when not provided.
    """

    def __init__(
        self,
        inputs: list[Segment],
        function: Callable[..., Any],
        function_config: BaseModel | dict[str, Any],
        backward: Callable[..., GradientResult],
        backward_config: BaseModel | dict[str, Any] | None = None,
        label: str | None = None,
        threshold: float | None = None,
        weight: float | None = None,
    ) -> None:
        """Initialize a differentiable constraint.

        Args:
            inputs (list[Segment]): List of Segment objects to evaluate.
            function (Callable[..., Any]): Discrete scoring function (inherited from Constraint).
            function_config (BaseModel | dict[str, Any]): Configuration for the scoring function.
            backward (Callable[..., GradientResult]): Gradient computation callable with signature
                ``(logits: np.ndarray, temperature: float, *, config: BaseModel, **kwargs) -> GradientResult``.
                In production, this calls a tool's gradient operation through IPC
                and may internally combine multiple model-local loss terms into
                the returned scalar objective.
            backward_config (BaseModel | dict[str, Any] | None): Configuration for the backward callable.
                Defaults to ``function_config`` when ``None``.
            label (str | None): Optional label for metadata tracking.
            threshold (float | None): Optional threshold for filtering mode.
            weight (float | None): Optional weight multiplier for scores.
        """
        super().__init__(
            inputs=inputs,
            function=function,
            function_config=function_config,
            label=label,
            threshold=threshold,
            weight=weight,
        )
        self._backward_fn = backward
        self._backward_config: BaseModel | dict[str, Any] = (
            backward_config if backward_config is not None else self._function_config
        )

    @property
    def backward_config(self) -> BaseModel | dict[str, Any]:
        """Configuration for the backward callable (read-only)."""
        return self._backward_config

    @property
    def backward(self) -> Callable[..., GradientResult]:
        """Backward callable used to compute gradients (read-only)."""
        return self._backward_fn

    def compute_gradient(self, logits: np.ndarray, temperature: float) -> GradientResult:
        """Compute gradient of this constraint's score with respect to input logits.

        Calls the ``backward`` callable to perform a forward and backward pass
        through the underlying differentiable model. The gradient flows from
        the scalar objective through the model back to the input logits.

        Args:
            logits (np.ndarray): Input logits for the relaxed sequence state.
            temperature (float): Softmax temperature for continuous relaxation.

        Returns:
            GradientResult: Gradient, loss, and optional metrics from the backward pass.

        Raises:
            TypeError: If the backward callable does not return ``GradientResult``.
            ValueError: If ``logits`` is not 2D, ``temperature`` is not positive,
                or the returned gradient shape does not match ``logits.shape``.
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (L, vocab_size), got array with shape {logits.shape}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        result = self._backward_fn(logits, temperature, config=self._backward_config)
        if not isinstance(result, GradientResult):
            raise TypeError(
                f"backward callable for differentiable constraint '{self.label}' must return GradientResult, "
                f"got {type(result).__name__}"
            )
        if result.gradient.shape != logits.shape:
            raise ValueError(
                f"backward callable for differentiable constraint '{self.label}' returned gradient with shape "
                f"{result.gradient.shape}, expected {logits.shape}"
            )
        return result
