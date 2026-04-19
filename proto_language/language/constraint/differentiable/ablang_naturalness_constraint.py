"""AbLang antibody naturalness constraints (dual-mode: discrete scoring + gradient).

Two registered constraints sharing the same underlying tool:

- ``ablang-vhh``: single-domain antibody (VHH/nanobody), 1 segment
- ``ablang-scfv``: scFv with separate VH + VL chains, 2 labeled segments
"""

import math
from typing import Any

import numpy as np
from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang import (
    AbLangGradientConfig,
    AbLangGradientInput,
    run_ablang_gradient,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import InputSlot, constraint
from proto_language.language.core import Sequence
from proto_language.language.core.constraint import GradientResult
from proto_language.utils import one_hot_protein_matrix


class AbLangConstraintConfig(BaseConfig):
    """Configuration for AbLang naturalness constraints (forward scoring + gradient).

    Attributes:
        temperature (float): Softmax temperature for AbLang. Required, no default:
            AbLang's fixed temperature is a scientific parameter of the pipeline (Germinal
            VHH uses 0.6, ``vhh.yaml:46``), not a framework default, so callers must
            choose it deliberately.
        use_ste (bool): Use Straight-Through Estimator (hard one-hot forward pass with
            gradients through soft probabilities). Germinal always uses STE.
        device (str): Execution device for AbLang, for example ``"cuda"`` or ``"cpu"``.
    """

    temperature: float = ConfigField(
        title="AbLang Temperature",
        gt=0.0,
        description="Softmax temperature for AbLang (fixed, not varied per step like AF2).",
    )
    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=True,
        description="Hard one-hot forward pass with soft-probability gradients. Matches Germinal's default.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Execution device for AbLang, for example 'cuda' or 'cpu'.",
        hidden=True,
    )


AbLangForwardConstraintConfig = AbLangConstraintConfig
AbLangBackwardConstraintConfig = AbLangConstraintConfig


def ablang_vhh_gradient_backward(
    inputs: tuple[Sequence, ...],
    *,
    config: AbLangBackwardConstraintConfig,
    **kwargs: Any,  # noqa: ARG001
) -> GradientResult:
    """Compute AbLang naturalness gradient for a single-domain antibody (VHH/nanobody)."""
    logits = inputs[0].logits
    assert logits is not None  # noqa: S101 -- input_labels slot check guarantees it
    output = run_ablang_gradient(
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=logits.tolist()), temperature=config.temperature),
        AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=True, device=config.device),
    )
    assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
    gradient = np.array(output.gradient, dtype=np.float64)
    return GradientResult(gradient=(gradient,), loss=output.loss, metrics=output.metrics)


def ablang_scfv_gradient_backward(
    inputs: tuple[Sequence, ...],
    *,
    config: AbLangBackwardConstraintConfig,
    **kwargs: Any,  # noqa: ARG001
) -> GradientResult:
    """Compute AbLang naturalness gradient for scFv (paired VH + VL chains)."""
    vh, vl = inputs[0].logits, inputs[1].logits
    assert vh is not None and vl is not None  # noqa: S101 -- input_labels slot checks guarantee both
    output = run_ablang_gradient(
        AbLangGradientInput(
            antibody=AntibodyLogits(heavy_chain=vh.tolist(), light_chain=vl.tolist()),
            temperature=config.temperature,
        ),
        AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=True, device=config.device),
    )
    assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
    gradient = np.array(output.gradient, dtype=np.float64)
    vh_grad, vl_grad = gradient[: len(vh)], gradient[len(vh) :]
    return GradientResult(gradient=(vh_grad, vl_grad), loss=output.loss, metrics=output.metrics)


@constraint(
    key="ablang-vhh",
    label="AbLang VHH Naturalness",
    config=AbLangForwardConstraintConfig,
    description="AbLang VHH naturalness: scores antibody-likeness (discrete) or gradient w.r.t. logits (differentiable).",
    tools_called=["ablang-gradient"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
    input_labels=[InputSlot(label="VHH Chain", requires_logits=True)],
    backward=ablang_vhh_gradient_backward,
    backward_config=AbLangBackwardConstraintConfig,
)
def ablang_vhh_forward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AbLangForwardConstraintConfig,
) -> list[float]:
    """Forward AbLang VHH naturalness scoring for discrete optimizers.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal ``(binder_seq,)``.
        config (AbLangForwardConstraintConfig): Forward-mode config.

    Returns:
        list[float]: Per-proposal energy ``sigmoid(loss)`` in ``(0, 1)``; lower is better.
    """
    scores: list[float] = []
    for (binder_seq,) in input_sequences:
        output = run_ablang_gradient(
            AbLangGradientInput(
                antibody=AntibodyLogits(heavy_chain=one_hot_protein_matrix(binder_seq.sequence)),
                temperature=config.temperature,
            ),
            AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=False, device=config.device),
        )
        binder_seq._metadata["ablang_log_likelihood"] = output.metrics["log_likelihood"]
        binder_seq._metadata["ablang_loss"] = output.loss
        scores.append(1.0 / (1.0 + math.exp(-output.loss)))
    return scores


@constraint(
    key="ablang-scfv",
    label="AbLang scFv Naturalness",
    config=AbLangForwardConstraintConfig,
    description="AbLang scFv naturalness: scores antibody-likeness (discrete) or gradient w.r.t. logits (differentiable).",
    tools_called=["ablang-gradient"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
    input_labels=[
        InputSlot(label="Heavy Chain (VH)", requires_logits=True),
        InputSlot(label="Light Chain (VL)", requires_logits=True),
    ],
    backward=ablang_scfv_gradient_backward,
    backward_config=AbLangBackwardConstraintConfig,
)
def ablang_scfv_forward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AbLangForwardConstraintConfig,
) -> list[float]:
    """Forward AbLang scFv naturalness scoring for discrete optimizers.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal ``(vh_seq, vl_seq)``.
        config (AbLangForwardConstraintConfig): Forward-mode config.

    Returns:
        list[float]: Per-proposal energy ``sigmoid(loss)`` in ``(0, 1)``; lower is better.
    """
    scores: list[float] = []
    for vh_seq, vl_seq in input_sequences:
        output = run_ablang_gradient(
            AbLangGradientInput(
                antibody=AntibodyLogits(
                    heavy_chain=one_hot_protein_matrix(vh_seq.sequence),
                    light_chain=one_hot_protein_matrix(vl_seq.sequence),
                ),
                temperature=config.temperature,
            ),
            AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=False, device=config.device),
        )
        # Joint VH+VL log-likelihood; write to both chains so either is a valid read site.
        for seq in (vh_seq, vl_seq):
            seq._metadata["ablang_log_likelihood"] = output.metrics["log_likelihood"]
            seq._metadata["ablang_loss"] = output.loss
        scores.append(1.0 / (1.0 + math.exp(-output.loss)))
    return scores
