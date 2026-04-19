"""Gradient math utilities: mergers, norm alignment, gradient normalization.

Mergers expect pre-weighted gradients — the optimizer multiplies by ``Constraint.weight``
before calling ``merge()``.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

GradientMergerName = Literal["weighted_sum", "pcgrad", "mgda"]


class GradientMerger(ABC):
    """Merge pre-weighted gradients into a single update direction."""

    @abstractmethod
    def merge(self, gradients: list[np.ndarray]) -> np.ndarray:
        """Merge multiple gradients into a single update direction."""

    @staticmethod
    def _prepare_inputs(gradients: list[np.ndarray]) -> np.ndarray:
        """Validate and stack merger inputs."""
        if not gradients:
            raise ValueError("gradients must contain at least one array")

        try:
            stacked = np.stack([np.asarray(gradient, dtype=float) for gradient in gradients])
        except ValueError as exc:
            raise ValueError("all gradients must have the same shape") from exc

        return stacked

    @classmethod
    def _flatten_inputs(cls, gradients: list[np.ndarray]) -> tuple[tuple[int, ...], np.ndarray]:
        """Return flattened gradients plus the original gradient shape."""
        stacked = cls._prepare_inputs(gradients)
        return stacked.shape[1:], stacked.reshape(stacked.shape[0], -1)


class WeightedSumMerger(GradientMerger):
    """Simple sum merger (gradients should be pre-weighted by the caller)."""

    def merge(self, gradients: list[np.ndarray]) -> np.ndarray:
        """Merge gradients via element-wise summation."""
        stacked = self._prepare_inputs(gradients)
        return np.asarray(stacked.sum(axis=0), dtype=float)


class PCGradMerger(GradientMerger):
    """Pairwise-conflict merger using deterministic projection order."""

    def merge(self, gradients: list[np.ndarray]) -> np.ndarray:
        """Merge gradients via pairwise conflict projection."""
        shape, flattened = self._flatten_inputs(gradients)
        projected = []

        for index, gradient in enumerate(flattened):
            current = gradient.copy()
            for other_index, other in enumerate(flattened):
                if index == other_index:
                    continue
                dot_product = float(np.dot(current, other))
                other_norm = float(np.dot(other, other))
                if dot_product < 0 and other_norm > 0:
                    current -= (dot_product / other_norm) * other
            projected.append(current)

        return np.asarray(np.sum(projected, axis=0).reshape(shape), dtype=float)


class MGDAMerger(GradientMerger):
    """Pareto-style gradient merger via Frank-Wolfe on the gradient Gram matrix."""

    def __init__(self, max_iter: int = 250, tolerance: float = 1e-8) -> None:
        """Initialize with Frank-Wolfe iteration limits."""
        self.max_iter = max_iter
        self.tolerance = tolerance

    def merge(self, gradients: list[np.ndarray]) -> np.ndarray:
        """Merge gradients via Pareto-optimal convex combination."""
        shape, flattened = self._flatten_inputs(gradients)
        if flattened.shape[0] == 1:
            return np.asarray(flattened[0].reshape(shape), dtype=float)

        gram = flattened @ flattened.T
        coefficients = self._solve_simplex_qp(gram)
        return np.asarray((coefficients @ flattened).reshape(shape), dtype=float)

    def _solve_simplex_qp(self, gram: np.ndarray) -> np.ndarray:
        """Solve ``min 0.5 * a^T G a`` over the probability simplex."""
        task_count = gram.shape[0]
        coefficients = np.full(task_count, 1.0 / task_count, dtype=float)

        for _ in range(self.max_iter):
            gradient = gram @ coefficients
            vertex_index = int(np.argmin(gradient))
            vertex = np.zeros(task_count, dtype=float)
            vertex[vertex_index] = 1.0

            direction = vertex - coefficients
            quadratic = float(direction @ gram @ direction)
            if quadratic <= self.tolerance:
                break

            step_size = float(-(direction @ gradient) / quadratic)
            step_size = min(max(step_size, 0.0), 1.0)
            updated = coefficients + (step_size * direction)
            if np.linalg.norm(updated - coefficients, ord=1) <= self.tolerance:
                coefficients = updated
                break
            coefficients = updated

        return coefficients


MERGERS: dict[GradientMergerName, type[GradientMerger]] = {
    "weighted_sum": WeightedSumMerger,
    "pcgrad": PCGradMerger,
    "mgda": MGDAMerger,
}


# =============================================================================
# Pre-merge and post-merge gradient transforms
# =============================================================================


def align_norms(grads: list[np.ndarray], mode: Literal["none", "unit", "match_first"]) -> list[np.ndarray]:
    """Align gradient norms before merging."""
    if mode == "none" or len(grads) <= 1:
        return grads
    if mode == "unit":
        return [g / (np.linalg.norm(g) + 1e-7) for g in grads]
    if mode == "match_first":
        target_norm = np.linalg.norm(grads[0])
        return [grads[0]] + [g * (target_norm / (np.linalg.norm(g) + 1e-7)) for g in grads[1:]]
    raise ValueError(f"Unknown norm_alignment mode: {mode}")


def normalize_gradient(
    gradient: np.ndarray, mode: Literal["unit", "sqrt_length"], zero_thr: float = 1e-7
) -> np.ndarray:
    """Normalize merged gradient before update.

    Args:
        gradient (np.ndarray): Merged gradient array (L, vocab_size).
        mode (Literal["unit", "sqrt_length"]): ``"unit"`` = L2 normalize to magnitude 1.0.
            ``"sqrt_length"`` = Germinal-compatible: ``g * sqrt(eff_L) / ||g||``
            where ``eff_L`` = positions with non-zero gradient norm.
        zero_thr (float): Threshold for considering a position's gradient as zero.
    """
    gn = np.linalg.norm(gradient)
    if gn <= zero_thr:
        return gradient

    if mode == "unit":
        return np.asarray(gradient / gn, dtype=float)

    if mode == "sqrt_length":
        per_position_norm = np.sum(gradient**2, axis=-1)
        eff_l = float(np.sum(per_position_norm > zero_thr))
        return np.asarray(gradient * np.sqrt(eff_l) / (gn + 1e-7), dtype=float)

    raise ValueError(f"Unknown normalize_mode: {mode}")
