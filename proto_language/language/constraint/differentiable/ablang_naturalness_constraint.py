"""AbLang antibody naturalness constraint (dual-mode: discrete scoring + gradient).

Takes one binder ``Segment``. Optional ``heavy_slice`` / ``light_slice`` config
fields enable single-chain scFv mode (paired VH+VL call); unset scores the whole
binder as a heavy-only chain (VHH / nanobody mode).
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
from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import InputSlot, constraint
from proto_language.language.core import Sequence
from proto_language.language.core.constraint import GradientResult
from proto_language.utils import one_hot_protein_matrix


class AbLangConstraintConfig(BaseConfig):
    """Configuration for AbLang naturalness scoring (forward + gradient).

    Attributes:
        temperature (float): Softmax temperature for AbLang. Required, no default:
            AbLang's fixed temperature is a scientific parameter of the pipeline (Germinal
            VHH uses 0.6, ``vhh.yaml:46``), not a framework default, so callers must
            choose it deliberately.
        use_ste (bool): Use Straight-Through Estimator (hard one-hot forward pass with
            gradients through soft probabilities). Germinal always uses STE.
        device (str): Execution device for AbLang, for example ``"cuda"`` or ``"cpu"``.
        heavy_slice (tuple[int, int] | None): Optional half-open ``(start, end)`` over the
            binder Segment for the VH region. Set together with ``light_slice`` to enable
            single-chain scFv mode; leave both ``None`` for VHH (heavy-only) scoring.
        light_slice (tuple[int, int] | None): Optional half-open ``(start, end)`` over the
            binder Segment for the VL region. Set together with ``heavy_slice``.
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
    heavy_slice: tuple[int, int] | None = ConfigField(
        title="Heavy Chain Slice",
        default=None,
        description="VH region (start, end) within the binder; set with light_slice for scFv mode.",
    )
    light_slice: tuple[int, int] | None = ConfigField(
        title="Light Chain Slice",
        default=None,
        description="VL region (start, end) within the binder; set with heavy_slice for scFv mode.",
    )

    @model_validator(mode="after")
    def _validate_slices(self) -> "AbLangConstraintConfig":
        """Slices must be both set or both None; each non-empty; non-overlapping."""
        heavy, light = self.heavy_slice, self.light_slice
        if (heavy is None) != (light is None):
            raise ValueError("heavy_slice and light_slice must be set together (both None for VHH mode).")
        if heavy is None or light is None:
            return self
        for name, (start, end) in (("heavy_slice", heavy), ("light_slice", light)):
            if start < 0 or end <= start:
                raise ValueError(f"{name}={(start, end)} must be a non-empty range with start >= 0 and end > start.")
        if heavy[0] < light[1] and light[0] < heavy[1]:
            raise ValueError(f"heavy_slice {heavy} overlaps light_slice {light}.")
        return self


def ablang_naturalness_gradient_backward(
    inputs: tuple[Sequence, ...],
    *,
    config: AbLangConstraintConfig,
    **kwargs: Any,  # noqa: ARG001
) -> GradientResult:
    """Compute AbLang naturalness gradient w.r.t. binder logits.

    VHH mode (no slices): the whole binder is one heavy chain. scFv mode (slices set):
    slice VH and VL out of the binder, call AbLang paired, scatter per-chain gradients
    back into a full-binder-shaped array with linker rows zero.
    """
    logits = inputs[0].logits
    assert logits is not None  # noqa: S101 -- input_labels slot check guarantees it

    if config.heavy_slice is None:
        output = run_ablang_gradient(
            AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=logits.tolist()), temperature=config.temperature),
            AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=True, device=config.device),
        )
        assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
        return GradientResult(
            gradient=(np.array(output.gradient, dtype=np.float64),), loss=output.loss, metrics=output.metrics
        )

    assert config.light_slice is not None  # noqa: S101 -- validator guarantees both-or-neither
    h_start, h_end = config.heavy_slice
    l_start, l_end = config.light_slice
    if max(h_end, l_end) > logits.shape[0]:
        raise ValueError(
            f"slices (heavy={config.heavy_slice}, light={config.light_slice}) extend past binder length {logits.shape[0]}."
        )
    vh_logits, vl_logits = logits[h_start:h_end], logits[l_start:l_end]
    output = run_ablang_gradient(
        AbLangGradientInput(
            antibody=AntibodyLogits(heavy_chain=vh_logits.tolist(), light_chain=vl_logits.tolist()),
            temperature=config.temperature,
        ),
        AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=True, device=config.device),
    )
    assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
    paired_grad = np.array(output.gradient, dtype=np.float64)
    full_grad = np.zeros_like(logits, dtype=np.float64)
    full_grad[h_start:h_end] = paired_grad[: h_end - h_start]
    full_grad[l_start:l_end] = paired_grad[h_end - h_start :]
    return GradientResult(gradient=(full_grad,), loss=output.loss, metrics=output.metrics)


@constraint(
    key="ablang-naturalness",
    label="AbLang Naturalness",
    config=AbLangConstraintConfig,
    description="AbLang naturalness on a single binder Segment (VHH/nanobody by default; set heavy_slice/light_slice to score a single-chain scFv as paired VH+VL). Discrete scoring or gradient w.r.t. logits.",
    tools_called=["ablang-gradient"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
    input_labels=[InputSlot(label="Binder", requires_logits=True)],
    backward=ablang_naturalness_gradient_backward,
    backward_config=AbLangConstraintConfig,
)
def ablang_naturalness_forward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AbLangConstraintConfig,
) -> list[float]:
    """Forward AbLang naturalness scoring for discrete optimizers.

    The ``ablang_log_likelihood`` / ``ablang_loss`` metadata values are VHH
    (heavy-only) scores when slices are unset and paired (VH+VL) scores when
    slices are set.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal ``(binder_seq,)``.
        config (AbLangConstraintConfig): Forward-mode config; slice fields control mode.

    Returns:
        list[float]: Per-proposal energy ``sigmoid(loss)`` in ``(0, 1)``; lower is better.
    """
    scores: list[float] = []
    for (binder_seq,) in input_sequences:
        if config.heavy_slice is None:
            antibody = AntibodyLogits(heavy_chain=one_hot_protein_matrix(binder_seq.sequence))
        else:
            assert config.light_slice is not None  # noqa: S101 -- validator guarantees both-or-neither
            h_start, h_end = config.heavy_slice
            l_start, l_end = config.light_slice
            if max(h_end, l_end) > len(binder_seq.sequence):
                raise ValueError(
                    f"slices (heavy={config.heavy_slice}, light={config.light_slice}) extend past binder length {len(binder_seq.sequence)}."
                )
            antibody = AntibodyLogits(
                heavy_chain=one_hot_protein_matrix(binder_seq.sequence[h_start:h_end]),
                light_chain=one_hot_protein_matrix(binder_seq.sequence[l_start:l_end]),
            )
        output = run_ablang_gradient(
            AbLangGradientInput(antibody=antibody, temperature=config.temperature),
            AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=False, device=config.device),
        )
        binder_seq._metadata["ablang_log_likelihood"] = output.metrics["log_likelihood"]
        binder_seq._metadata["ablang_loss"] = output.loss
        scores.append(1.0 / (1.0 + math.exp(-output.loss)))
    return scores
