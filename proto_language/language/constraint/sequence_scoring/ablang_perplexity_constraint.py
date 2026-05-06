"""AbLang antibody perplexity constraint (dual-mode: discrete scoring + gradient).

Takes one binder ``Segment``. Optional ``heavy_slice`` / ``light_slice`` config
fields enable single-chain scFv mode, where VH and VL are sliced out of one
binder and scored as a paired antibody. When both slices are unset, the whole
binder is scored as a heavy-only chain, matching VHH / nanobody usage.

The score is AbLang mean negative log-likelihood (lower is more natural under
the antibody language model). Perplexity is reported in metadata as
``exp(mean_nll)`` for interpretability. The default score remains raw mean NLL
for Germinal compatibility, while ``score_mode="ppl"`` makes forward and
gradient modes optimize/report perplexity instead.
"""

from typing import Any, Literal

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
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.language.core.constraint import GradientConstraintOutput
from proto_language.utils import one_hot_protein_matrix


class AbLangPerplexityConfig(BaseConfig):
    """Configuration for AbLang perplexity scoring (forward + gradient).

    Attributes:
        temperature (float): Softmax temperature for AbLang. Required, no
            default: this is a scientific parameter of the optimization
            program rather than a framework default, so callers must choose it
            deliberately.
        use_ste (bool): Use straight-through estimation: a hard one-hot forward
            pass with gradients through soft probabilities. This matches the
            Germinal-style antibody LM path.
        device (str): Device for AbLang execution, for example ``"cuda"`` or
            ``"cuda:0"``. Hidden from public JSON forms because most programs
            should set it at the execution environment level.
        heavy_slice (tuple[int, int] | None): Optional half-open ``(start, end)``
            over the binder Segment for the VH region. Set together with
            ``light_slice`` to enable single-chain scFv mode; leave both
            ``None`` for VHH / heavy-only scoring.
        light_slice (tuple[int, int] | None): Optional half-open ``(start, end)``
            over the binder Segment for the VL region. Must be set together
            with ``heavy_slice``.
        score_mode (Literal["nll", "ppl"]): Score/loss returned by the
            constraint. ``"nll"`` preserves Germinal compatibility;
            ``"ppl"`` returns perplexity and scales gradients by ``exp(NLL)``.
        logit_scale (float): Multiplier applied to raw optimizer logits before
            AbLang. The backward pass multiplies returned gradients by the same
            scale via the chain rule.
        logit_bias (list[list[float]] | None): Optional additive bias with shape
            ``L x 20`` applied after scaling and before AbLang. This is useful
            for carrying persistent sequence priors into the antibody LM term.
    """

    temperature: float = ConfigField(
        title="AbLang Temperature",
        gt=0.0,
        description="Softmax temperature for AbLang (fixed for this constraint, not varied per optimizer step).",
    )
    use_ste: bool = ConfigField(
        default=True,
        title="Straight-Through Estimator",
        description="Hard one-hot forward pass with soft-probability gradients.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for AbLang execution, e.g. 'cuda' or 'cuda:0'.",
        hidden=True,
    )
    heavy_slice: tuple[int, int] | None = ConfigField(
        default=None,
        title="Heavy Chain Slice",
        description="VH region (start, end) within the binder; set with light_slice for scFv mode.",
    )
    light_slice: tuple[int, int] | None = ConfigField(
        default=None,
        title="Light Chain Slice",
        description="VL region (start, end) within the binder; set with heavy_slice for scFv mode.",
    )
    score_mode: Literal["nll", "ppl"] = ConfigField(
        default="nll",
        title="Score Mode",
        description="Return raw mean NLL by default, or AbLang perplexity when set to 'ppl'.",
        advanced=True,
    )
    logit_scale: float = ConfigField(
        default=1.0,
        title="Logit Scale",
        description="Pre-scale raw logits before AbLang; gradients are scaled back by the same factor.",
        gt=0.0,
    )
    logit_bias: list[list[float]] | None = ConfigField(
        default=None,
        title="Logit Bias",
        description="Additive bias (L x 20) applied after scaling, before AbLang.",
        hidden=True,
    )

    @model_validator(mode="after")
    def _validate_slices(self) -> "AbLangPerplexityConfig":
        """Slices must be both set or both omitted, non-empty, and non-overlapping."""
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


def ablang_perplexity_gradient_backward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AbLangPerplexityConfig,
    **kwargs: Any,  # noqa: ARG001
) -> list[GradientConstraintOutput]:
    """Compute AbLang mean-NLL gradients with perplexity metadata.

    VHH mode (no slices): the whole binder is one heavy chain. scFv mode
    (slices set): slice VH and VL out of the binder, call AbLang in paired mode,
    and scatter per-chain gradients back into a full-binder-shaped array with
    linker or non-antibody rows set to zero. The returned loss follows
    ``score_mode``; metadata always includes log-likelihood, NLL, and
    perplexity.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal binder inputs.
        config (AbLangPerplexityConfig): AbLang scoring configuration.
        kwargs (Any): Optimizer-supplied values ignored by AbLang.

    Returns:
        list[GradientConstraintOutput]: Per-proposal loss, gradient, and metrics.
    """
    results: list[GradientConstraintOutput] = []
    for (binder_seq,) in input_sequences:
        raw_logits = binder_seq.logits
        assert raw_logits is not None  # noqa: S101 -- input slot validation guarantees logits
        bias = np.asarray(config.logit_bias, dtype=np.float64) if config.logit_bias is not None else 0.0
        logits = config.logit_scale * raw_logits + bias

        if config.heavy_slice is None:
            output = run_ablang_gradient(
                AbLangGradientInput(
                    antibody=AntibodyLogits(heavy_chain=logits.tolist()),
                    temperature=config.temperature,
                ),
                AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=True, device=config.device),
            )
            assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
            grad = np.array(output.gradient, dtype=np.float64) * config.logit_scale
        else:
            assert config.light_slice is not None  # noqa: S101 -- validator guarantees both-or-neither
            h_start, h_end = config.heavy_slice
            l_start, l_end = config.light_slice
            if max(h_end, l_end) > logits.shape[0]:
                raise ValueError(
                    f"slices (heavy={config.heavy_slice}, light={config.light_slice}) "
                    f"extend past binder length {logits.shape[0]}."
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
            paired_grad = np.array(output.gradient, dtype=np.float64) * config.logit_scale
            grad = np.zeros_like(raw_logits, dtype=np.float64)
            grad[h_start:h_end] = paired_grad[: h_end - h_start]
            grad[l_start:l_end] = paired_grad[h_end - h_start :]

        perplexity = float(np.exp(output.loss))
        score = output.loss if config.score_mode == "nll" else perplexity
        if config.score_mode == "ppl":
            if not np.isfinite(perplexity):
                raise ValueError(
                    f"AbLang perplexity {perplexity} is non-finite (mean NLL={output.loss}); "
                    "cannot scale gradient for score_mode='ppl'."
                )
            grad = grad * perplexity
        results.append(
            GradientConstraintOutput(
                gradient=(grad,),
                loss=score,
                metrics={
                    **output.metrics,
                    "ablang_log_likelihood": output.metrics.get("log_likelihood"),
                    "ablang_loss": output.loss,
                    "ablang_nll": output.loss,
                    "ablang_perplexity": perplexity,
                    "ablang_score_mode": config.score_mode,
                },
            )
        )
    return results


@constraint(
    key="ablang-perplexity",
    label="AbLang Perplexity",
    config=AbLangPerplexityConfig,
    description="Score antibody naturalness by AbLang mean NLL and report perplexity.",
    tools_called=["ablang-gradient"],
    uses_gpu=True,
    category="sequence scoring",
    supported_sequence_types=["protein"],
    input_labels=[InputSlot(label="Binder", requires_logits=True)],
    backward=ablang_perplexity_gradient_backward,
    backward_config=AbLangPerplexityConfig,
)
def ablang_perplexity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AbLangPerplexityConfig,
) -> list[ConstraintOutput]:
    """Forward AbLang antibody scoring via masked pseudo-log-likelihood.

    The returned score is raw AbLang mean negative log-likelihood by default,
    so lower is better and the forward score matches the loss used by
    ``ablang_perplexity_gradient_backward``. Set ``score_mode="ppl"`` to return
    perplexity instead. Metadata always includes ``ablang_log_likelihood``,
    ``ablang_nll``, ``ablang_loss``, and ``ablang_perplexity``.

    With no slices, the whole binder is scored as a heavy-only VHH/nanobody.
    With ``heavy_slice`` and ``light_slice`` set, the function extracts VH and
    VL from a single binder Segment and scores them as a paired scFv.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal binder inputs.
        config (AbLangPerplexityConfig): Forward-mode config; slice fields
            control VHH versus paired scFv mode.

    Returns:
        list[ConstraintOutput]: Per-proposal AbLang scores with log-likelihood,
            NLL, loss, and perplexity metadata.
    """
    results: list[ConstraintOutput] = []
    for (binder_seq,) in input_sequences:
        if config.heavy_slice is None:
            antibody = AntibodyLogits(heavy_chain=one_hot_protein_matrix(binder_seq.sequence))
        else:
            assert config.light_slice is not None  # noqa: S101 -- validator guarantees both-or-neither
            h_start, h_end = config.heavy_slice
            l_start, l_end = config.light_slice
            if max(h_end, l_end) > len(binder_seq.sequence):
                raise ValueError(
                    f"slices (heavy={config.heavy_slice}, light={config.light_slice}) "
                    f"extend past binder length {len(binder_seq.sequence)}."
                )
            antibody = AntibodyLogits(
                heavy_chain=one_hot_protein_matrix(binder_seq.sequence[h_start:h_end]),
                light_chain=one_hot_protein_matrix(binder_seq.sequence[l_start:l_end]),
            )
        output = run_ablang_gradient(
            AbLangGradientInput(antibody=antibody, temperature=config.temperature),
            AbLangGradientConfig(use_ste=config.use_ste, compute_gradient=False, device=config.device),
        )
        perplexity = float(np.exp(output.loss))
        score = output.loss if config.score_mode == "nll" else perplexity
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    **output.metrics,
                    "ablang_log_likelihood": output.metrics.get("log_likelihood"),
                    "ablang_loss": output.loss,
                    "ablang_nll": output.loss,
                    "ablang_perplexity": perplexity,
                    "ablang_score_mode": config.score_mode,
                },
            )
        )
    return results


# Germinal-style semigreedy ranking uses the raw antibody-LM loss, so this
# intentional exception keeps discrete scoring aligned with the gradient target.
ablang_perplexity_constraint._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
