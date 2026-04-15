"""AbLang antibody naturalness gradient constraints.

Two registered constraints sharing the same underlying tool:

- ``ablang-vhh-gradient``: single-domain antibody (VHH/nanobody), 1 segment
- ``ablang-scfv-gradient``: scFv with separate VH + VL chains, 2 labeled segments
"""

from typing import Any

import numpy as np
from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang import (
    AbLangGradientConfig,
    AbLangGradientInput,
    run_ablang_gradient,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.language.core.constraint import GradientResult


class AbLangGradientConstraintConfig(BaseConfig):
    """Configuration for AbLang naturalness gradient constraints.

    Attributes:
        temperature (float): Softmax temperature for AbLang gradient computation.
            Germinal uses a fixed 0.6 regardless of the AF2 optimizer's temperature
            schedule.
        use_ste (bool): Use Straight-Through Estimator (hard one-hot forward pass
            with gradients through soft probabilities). Germinal always uses STE.
    """

    temperature: float = ConfigField(
        title="AbLang Temperature",
        default=0.6,
        gt=0.0,
        description="Softmax temperature for AbLang (fixed, not varied per step like AF2).",
    )
    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=True,
        description="Hard one-hot forward pass with soft-probability gradients. Matches Germinal's default.",
    )


@constraint(
    key="ablang-vhh-gradient",
    label="AbLang VHH Naturalness Gradient",
    config=AbLangGradientConstraintConfig,
    description="Differentiable antibody naturalness gradient for single-domain antibodies (VHH/nanobody)",
    tools_called=["ablang-gradient"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
)
def ablang_vhh_gradient_backward(
    inputs: tuple[Sequence, ...],
    *,
    config: AbLangGradientConstraintConfig,
    **kwargs: Any,  # noqa: ARG001
) -> GradientResult:
    """Compute AbLang naturalness gradient for a single-domain antibody (VHH/nanobody)."""
    logits = inputs[0].logits
    if logits is None:
        raise RuntimeError("Input sequence has no logits set")
    output = run_ablang_gradient(
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=logits.tolist()), temperature=config.temperature),
        AbLangGradientConfig(use_ste=config.use_ste),
    )
    gradient = np.array(output.gradient, dtype=np.float64)
    return GradientResult(gradient=(gradient,), loss=output.loss, metrics=output.metrics)


@constraint(
    key="ablang-scfv-gradient",
    label="AbLang scFv Naturalness Gradient",
    config=AbLangGradientConstraintConfig,
    description="Differentiable antibody naturalness gradient for scFv (separate VH + VL chains)",
    tools_called=["ablang-gradient"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
    input_labels=["Heavy Chain (VH)", "Light Chain (VL)"],
)
def ablang_scfv_gradient_backward(
    inputs: tuple[Sequence, ...],
    *,
    config: AbLangGradientConstraintConfig,
    **kwargs: Any,  # noqa: ARG001
) -> GradientResult:
    """Compute AbLang naturalness gradient for scFv (paired VH + VL chains)."""
    vh, vl = inputs[0].logits, inputs[1].logits
    if vh is None or vl is None:
        raise RuntimeError("Both input segments must have logits set")
    output = run_ablang_gradient(
        AbLangGradientInput(
            antibody=AntibodyLogits(heavy_chain=vh.tolist(), light_chain=vl.tolist()),
            temperature=config.temperature,
        ),
        AbLangGradientConfig(use_ste=config.use_ste),
    )
    gradient = np.array(output.gradient, dtype=np.float64)
    vh_grad, vl_grad = gradient[: len(vh)], gradient[len(vh) :]
    return GradientResult(gradient=(vh_grad, vl_grad), loss=output.loss, metrics=output.metrics)
