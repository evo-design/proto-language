"""Abstract base types shared by the compiler and its backend providers.

These types form the contract between the optimizer-facing compiler
(``compiler.py``) and per-backend provider implementations such as
``alphafold2_multimer_provider.py``. Hosting them in a leaf module lets both
callers import from one place without circularity, replacing the lazy
function-local imports that would otherwise be needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from proto_language.language.core import Constraint

EffectiveWeight = Callable[[Constraint, int], float]


def raise_for_failed_tool_output(output: Any, label: str) -> None:
    """Raise the captured tool error before provider-specific output validation."""
    if getattr(output, "success", None) is not False:
        return

    errors = getattr(output, "errors", None) or []
    detail = "\n".join(str(error).strip() for error in errors if str(error).strip())
    if not detail:
        detail = "remote tool returned success=False without error details."
    raise RuntimeError(f"{label} failed: {detail}")


@dataclass(frozen=True)
class CompiledConstraint:
    """A public constraint paired with the backend objective that implements it.

    The compiler keeps this pairing private because the public constraint name
    and backend objective name may be intentionally different vocabularies. For
    example, a public ``structure-iplddt`` constraint can compile to the AF2M
    ``iplddt`` loss key.

    Attributes:
        constraint (Constraint): User-authored constraint object. Its label, weight schedule,
            metadata sink, and proposal inputs remain the source of truth.
        objective_key (str): Backend-specific objective identifier. Only the matching
            model adapter should interpret this value.
    """

    constraint: Constraint
    objective_key: str


@dataclass(frozen=True)
class GradientProviderOutput:
    """Gradients and losses returned by a provider for one optimizer step.

    ``gradients`` and ``losses`` are proposal-aligned: index ``k`` corresponds
    to proposal ``k`` in the optimizer's target segment. Losses are already
    weighted for the current optimization step so the optimizer can sum provider
    outputs directly.

    Attributes:
        label (str): Human-readable provider label used in errors and logging.
        gradients (list[np.ndarray]): One gradient matrix per proposal, each shaped like the
            target proposal logits.
        losses (list[float]): One weighted scalar loss per proposal.
        weight (float): Scalar multiplier to apply to ``gradients`` when the provider
            wraps a single direct constraint. Grouped providers usually return
            gradients that already include their internal term weights and leave
            this at ``1.0``.
    """

    label: str
    gradients: list[np.ndarray]
    losses: list[float]
    weight: float = 1.0


class GradientProvider(ABC):
    """Common optimizer contract for direct and compiled gradient sources.

    A provider can represent one direct differentiable constraint or a compiled
    backend group containing several public constraints. Either way, the
    gradient optimizer calls ``compute`` once per optimization step and receives
    proposal-aligned gradients for the target segment.

    Implementations may call a normal constraint ``backward`` function, combine
    several constraints into one model invocation, write per-constraint metadata,
    or update proposal structures. Those details remain hidden behind this
    interface.
    """

    label: str

    @abstractmethod
    def compute(
        self,
        *,
        temperature: float,
        soft: float,
        hard: float,
        step: int,
        effective_weight: EffectiveWeight,
    ) -> GradientProviderOutput:
        """Compute proposal gradients and weighted losses for this step.

        Args:
            temperature (float): Sampling temperature used to materialize relaxed
                proposal sequences.
            soft (float): Soft sequence interpolation coefficient for differentiable
                structure tools.
            hard (float): Hard sequence interpolation coefficient for differentiable
                structure tools.
            step (int): Zero-based optimizer step. Providers pass this to
                ``effective_weight`` for step-dependent schedules.
            effective_weight (EffectiveWeight): Callback returning the current scalar weight for a
                public constraint.

        Returns:
            GradientProviderOutput: Proposal-aligned gradients and weighted losses.
        """
