"""Scheduling utilities for optimizer temperature annealing and progress tracking."""

import math
from collections.abc import Callable
from typing import Literal

Schedule = Callable[[int, int], float]
"""Callable that maps ``(step, total_steps)`` to a scalar value."""

ScheduleName = Literal["constant", "cosine", "exponential", "hinge", "linear", "quadratic"]


def progress(step: int, total_steps: int) -> float:
    """Return clamped optimization progress in the unit interval."""
    for name, value in (("step", step), ("total_steps", total_steps)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    return 1.0 if total_steps == 0 else min(step / total_steps, 1.0)


def _with_progress(transform: Callable[[float], float]) -> Schedule:
    """Build a schedule from a progress-space transform."""

    def schedule(step: int, total_steps: int) -> float:
        return float(transform(progress(step, total_steps)))

    return schedule


def constant_schedule(start: float, end: float = 0.0) -> Schedule:  # noqa: ARG001
    """Return a schedule that always emits *start* (ignores *end*)."""
    if not math.isfinite(start):
        raise ValueError(f"start must be finite, got {start}")
    return _with_progress(lambda _: start)


def linear_decay(start: float, end: float) -> Schedule:
    """Return a linearly interpolated schedule from *start* to *end*."""
    delta = end - start
    return _with_progress(lambda t: start + (delta * t))


def cosine_anneal(start: float, end: float) -> Schedule:
    """Return a cosine-annealed schedule from *start* to *end*."""
    amplitude = 0.5 * (start - end)
    return _with_progress(lambda t: end + amplitude * (1.0 + math.cos(math.pi * t)))


def exponential_decay(start: float, end: float) -> Schedule:
    """Return an exponentially interpolated schedule from *start* to *end*."""
    for name, value in (("start", start), ("end", end)):
        if value <= 0:
            raise ValueError(f"{name} must be > 0 for exponential decay, got {value}")
    ratio = end / start
    return _with_progress(lambda t: start * (ratio**t))


def hinge_schedule(start: float, end: float) -> Schedule:
    """Return a hinge schedule: ``max(end * t, start)`` — flat at *start* then linear to *end*."""
    return _with_progress(lambda t: max(end * t, start))


def quadratic_decay(start: float, end: float) -> Schedule:
    """Return a quadratically decaying schedule: ``end + (start - end) * (1 - t)²``."""
    return _with_progress(lambda t: end + (start - end) * (1.0 - t) ** 2)


SCHEDULES: dict[ScheduleName, Callable[[float, float], Schedule]] = {
    "constant": constant_schedule,
    "cosine": cosine_anneal,
    "exponential": exponential_decay,
    "hinge": hinge_schedule,
    "linear": linear_decay,
    "quadratic": quadratic_decay,
}
