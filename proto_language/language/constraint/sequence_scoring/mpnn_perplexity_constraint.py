"""ProteinMPNN perplexity constraint (forward + differentiable, backbone-conditioned).

Calls ``proteinmpnn-gradient`` directly against a fixed backbone for both
forward scoring and gradient computation, returning either mean NLL or
perplexity. Works as a discrete scorer in rejection sampling / MCMC and
as a differentiable scorer in ``GradientOptimizer``.
"""

from typing import Any, Literal

import numpy as np
from proto_tools import InverseFoldingStructureInput, Structure
from proto_tools.tools.inverse_folding.proteinmpnn import (
    ProteinMPNNGradientConfig,
    ProteinMPNNGradientInput,
    run_proteinmpnn_gradient,
)
from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_gradient import ProteinMPNNModelChoice
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import InputSlot, constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.language.core.constraint import GradientConstraintOutput
from proto_language.utils import one_hot_protein_matrix
from proto_language.utils.sequence_logit_bias import (
    SequenceLogitBiasConfig,
    build_sequence_logit_bias_matrix_from_sequence,
)


class MpnnPerplexityConfig(BaseConfig):
    """Configuration for ProteinMPNN perplexity scoring.

    Attributes:
        structure_input (InverseFoldingStructureInput): Backbone structure
            plus optional ``chains_to_redesign`` and fixed positions.
        model_choice (ProteinMPNNModelChoice): ProteinMPNN weight variant used
            for forward scoring and gradient computation.
        temperature (float): Softmax temperature for relaxing optimizer logits
            before ProteinMPNN scoring.
        use_ste (bool): Whether to use a hard one-hot forward pass with
            soft-probability gradients.
        device (str): Device for ProteinMPNN execution, for example ``"cuda"``
            or ``"cuda:0"``.
        seed (int | None): Optional random seed for ProteinMPNN decoding-order
            sampling.
        score_mode (Literal["nll", "ppl"]): Objective returned by the
            constraint, either mean negative log-likelihood or perplexity.
        logit_scale (float): Scale applied to raw optimizer logits before
            ProteinMPNN; gradients are scaled back by the same factor.
        sequence_bias (SequenceLogitBiasConfig | None): Optional declarative
            per-position symbol bias resolved against the binder's 20-AA
            protein vocabulary; added to logits after scaling and before
            ProteinMPNN.
    """

    structure_input: InverseFoldingStructureInput = ConfigField(
        title="Structure Input",
        description=(
            "Backbone structure, optional chains_to_redesign, and fixed positions for direct ProteinMPNN scoring."
        ),
    )
    model_choice: ProteinMPNNModelChoice = ConfigField(
        default="proteinmpnn",
        title="Model Choice",
        description="Weights: proteinmpnn (=v_48_020), v_48_{002,010,030} noise variants, abmpnn, soluble.",
    )
    temperature: float = ConfigField(
        default=1.0,
        title="ProteinMPNN Temperature",
        gt=0.0,
        description="Softmax temperature for relaxing optimizer logits before ProteinMPNN scoring.",
    )
    use_ste: bool = ConfigField(
        default=True,
        title="Straight-Through Estimator",
        description="Hard one-hot forward pass with soft-probability gradients.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for ProteinMPNN execution, e.g. 'cuda' or 'cuda:0'.",
    )
    seed: int | None = ConfigField(
        default=None,
        title="Random Seed",
        description="Seed for ProteinMPNN decoding-order sampling. None lets proto-tools choose a fresh seed.",
        ge=0,
    )
    score_mode: Literal["nll", "ppl"] = ConfigField(
        default="ppl",
        title="Score Mode",
        description="Return ProteinMPNN perplexity by default, or raw mean NLL when set to 'nll'.",
    )
    logit_scale: float = ConfigField(
        default=1.0,
        title="Logit Scale",
        description="Pre-scale raw logits before ProteinMPNN; gradients are scaled back by the same factor.",
        gt=0.0,
    )
    sequence_bias: SequenceLogitBiasConfig | None = ConfigField(
        default=None,
        title="Sequence Bias",
        description="Declarative sequence-symbol bias (canonical 20-AA protein) added before ProteinMPNN.",
    )

    @field_validator("structure_input", mode="before")
    @classmethod
    def normalize_structure_input(cls, value: Any) -> Any:
        """Accept the same structure shorthands as ProteinMPNNGeneratorConfig."""
        if isinstance(value, InverseFoldingStructureInput):
            return value
        if isinstance(value, (str, Structure)):
            return {"structure": value}
        if isinstance(value, dict) and "chain_ids" in value and "chains_to_redesign" not in value:
            value = dict(value)
            value["chains_to_redesign"] = value.pop("chain_ids")
        return value


def _metadata_from_nll(nll: float, config: MpnnPerplexityConfig, extra: dict[str, Any]) -> dict[str, Any]:
    """Build normalized ProteinMPNN metadata from mean NLL."""
    perplexity = float(np.exp(nll))
    return {
        **extra,
        "mpnn_log_likelihood": -extra.get("effective_sequence_length", 1.0) * nll,
        "mpnn_avg_log_likelihood": -nll,
        "mpnn_loss": nll,
        "mpnn_nll": nll,
        "mpnn_perplexity": perplexity,
        "mpnn_score_mode": config.score_mode,
        "mpnn_model_choice": config.model_choice,
        "perplexity": perplexity,
    }


def _score_from_nll(nll: float, config: MpnnPerplexityConfig) -> float:
    """Return the configured score/loss from a mean NLL."""
    if config.score_mode == "nll":
        return nll
    perplexity = float(np.exp(nll))
    if not np.isfinite(perplexity):
        raise ValueError(
            f"ProteinMPNN perplexity {perplexity} is non-finite (mean NLL={nll}); cannot use score_mode='ppl'."
        )
    return perplexity


def mpnn_perplexity_gradient_backward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: MpnnPerplexityConfig,
    **kwargs: Any,  # noqa: ARG001
) -> list[GradientConstraintOutput]:
    """Compute ProteinMPNN mean-NLL/perplexity gradients against a fixed backbone."""
    structure_input = config.structure_input
    results: list[GradientConstraintOutput] = []
    if not input_sequences:
        return results
    # Bias resolves from binder length + sequence_type + valid_chars — invariant across proposals.
    seq_bias = build_sequence_logit_bias_matrix_from_sequence(config.sequence_bias, input_sequences[0][0])
    bias: np.ndarray | float = seq_bias if seq_bias is not None else 0.0
    for (binder_seq,) in input_sequences:
        raw_logits = binder_seq.logits
        assert raw_logits is not None  # noqa: S101 -- input slot validation guarantees logits
        logits = config.logit_scale * raw_logits + bias

        output = run_proteinmpnn_gradient(
            ProteinMPNNGradientInput(
                logits=logits.tolist(),
                structure=structure_input.structure,
                chains_to_redesign=structure_input.chains_to_redesign,
                fixed_positions=structure_input.fixed_positions,
                temperature=config.temperature,
            ),
            ProteinMPNNGradientConfig(
                model_choice=config.model_choice,
                use_ste=config.use_ste,
                compute_gradient=True,
                device=config.device,
                seed=config.seed,
            ),
        )
        assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
        grad = np.array(output.gradient, dtype=np.float64) * config.logit_scale

        nll = float(output.loss)
        score = _score_from_nll(nll, config)
        if config.score_mode == "ppl":
            grad = grad * score
        results.append(
            GradientConstraintOutput(
                gradient=(grad,),
                loss=score,
                metrics=_metadata_from_nll(nll, config, dict(output.metrics)),
            )
        )
    return results


@constraint(
    key="mpnn-perplexity",
    label="MPNN Perplexity",
    config=MpnnPerplexityConfig,
    description="Score protein sequences by ProteinMPNN perplexity against a fixed backbone; differentiable.",
    tools_called=["proteinmpnn-gradient"],
    uses_gpu=True,
    category="sequence scoring",
    supported_sequence_types=["protein"],
    input_labels=[InputSlot(label="Sequence", requires_logits=True)],
    backward=mpnn_perplexity_gradient_backward,
    backward_config=MpnnPerplexityConfig,
)
def mpnn_perplexity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: MpnnPerplexityConfig,
) -> list[ConstraintOutput]:
    """Score proposals by ProteinMPNN perplexity against the configured backbone."""
    structure_input = config.structure_input
    results = []
    for (binder_seq,) in input_sequences:
        output = run_proteinmpnn_gradient(
            ProteinMPNNGradientInput(
                logits=one_hot_protein_matrix(binder_seq.sequence),
                structure=structure_input.structure,
                chains_to_redesign=structure_input.chains_to_redesign,
                fixed_positions=structure_input.fixed_positions,
                temperature=config.temperature,
            ),
            ProteinMPNNGradientConfig(
                model_choice=config.model_choice,
                use_ste=config.use_ste,
                compute_gradient=False,
                device=config.device,
                seed=config.seed,
            ),
        )
        nll = float(output.loss)
        results.append(
            ConstraintOutput(
                score=_score_from_nll(nll, config),
                metadata=_metadata_from_nll(nll, config, dict(output.metrics)),
            )
        )
    return results


mpnn_perplexity_constraint._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
