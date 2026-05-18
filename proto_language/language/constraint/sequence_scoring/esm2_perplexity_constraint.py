"""ESM2 protein perplexity constraint (dual-mode: discrete scoring + gradient).

Takes one protein ``Segment`` and scores it by ESM2 masked pseudo-log-likelihood.
The default score is mean negative log-likelihood (lower is more natural under
the language model). Perplexity is reported in metadata as ``exp(mean_nll)`` for
interpretability, and ``score_mode="ppl"`` makes forward and gradient modes
optimize/report perplexity instead.
"""

from typing import Any, Literal

import numpy as np
from proto_tools.tools.masked_models.esm2.esm2_gradient import (
    ESM2GradientConfig,
    ESM2GradientInput,
    run_esm2_gradient,
)
from proto_tools.tools.masked_models.esm2.esm2_sample import ESM2_MODEL_CHECKPOINTS

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import InputSlot, constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.language.core.constraint import GradientConstraintOutput
from proto_language.utils import one_hot_protein_matrix
from proto_language.utils.sequence_logit_bias import (
    SequenceLogitBiasConfig,
    build_sequence_logit_bias_matrix_from_sequence,
)


class ESM2PerplexityConfig(BaseConfig):
    """Configuration for ESM2 perplexity scoring (forward + gradient).

    Attributes:
        model_checkpoint (ESM2_MODEL_CHECKPOINTS): ESM2 checkpoint to use.
        temperature (float): Softmax temperature for ESM2. Required, no default:
            callers must choose it deliberately as part of the optimization program.
        use_ste (bool): Use straight-through estimation: a hard one-hot forward
            pass with gradients through soft probabilities.
        device (str): Device for ESM2 execution, for example ``"cuda"`` or ``"cuda:0"``.
        batch_size (int | None): Masked positions per ESM2 forward pass. ``None``
            selects the proto-tools backend default.
        score_mode (Literal["nll", "ppl"]): Score/loss returned by the constraint.
            ``"nll"`` preserves raw mean-NLL behavior; ``"ppl"`` returns perplexity
            and scales gradients by ``exp(NLL)``.
        logit_scale (float): Multiplier applied to raw optimizer logits before ESM2.
            The backward pass multiplies returned gradients by the same scale via
            the chain rule.
        sequence_bias (SequenceLogitBiasConfig | None): Optional declarative
            per-position symbol bias resolved against the binder's 20-AA
            protein vocabulary; added to logits after scaling and before ESM2.
    """

    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        default="esm2_t33_650M_UR50D",
        title="Model Checkpoint",
        description="ESM2 model checkpoint to use",
    )
    temperature: float = ConfigField(
        title="ESM2 Temperature",
        gt=0.0,
        description="Softmax temperature for ESM2 (fixed for this constraint, not varied per optimizer step).",
    )
    use_ste: bool = ConfigField(
        default=True,
        title="Straight-Through Estimator",
        description="Hard one-hot forward pass with soft-probability gradients.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for ESM2 execution, e.g. 'cuda' or 'cuda:0'.",
    )
    batch_size: int | None = ConfigField(
        default=None,
        title="PLL Batch Size",
        description="AA positions per ESM2 forward pass. Lower if OOM, higher for throughput.",
        gt=0,
    )
    score_mode: Literal["nll", "ppl"] = ConfigField(
        default="nll",
        title="Score Mode",
        description="Return raw mean NLL by default, or ESM2 perplexity when set to 'ppl'.",
    )
    logit_scale: float = ConfigField(
        default=1.0,
        title="Logit Scale",
        description="Pre-scale raw logits before ESM2; gradients are scaled back by the same factor.",
        gt=0.0,
    )
    sequence_bias: SequenceLogitBiasConfig | None = ConfigField(
        default=None,
        title="Sequence Bias",
        description="Declarative sequence-symbol bias (canonical 20-AA protein) added before ESM2.",
    )


def _esm2_metadata(output: Any, config: ESM2PerplexityConfig, perplexity: float) -> dict[str, Any]:
    """Return normalized ESM2 metadata while preserving backend metrics."""
    metrics = output.metrics
    return {
        **metrics,
        "esm2_log_likelihood": metrics.get("log_likelihood"),
        "esm2_avg_log_likelihood": metrics.get("avg_log_likelihood"),
        "esm2_loss": output.loss,
        "esm2_nll": output.loss,
        "esm2_perplexity": perplexity,
        "esm2_score_mode": config.score_mode,
        "esm2_model_checkpoint": config.model_checkpoint,
    }


def esm2_perplexity_gradient_backward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: ESM2PerplexityConfig,
    **kwargs: Any,  # noqa: ARG001
) -> list[GradientConstraintOutput]:
    """Compute ESM2 mean-NLL gradients with perplexity metadata."""
    results: list[GradientConstraintOutput] = []
    if not input_sequences:
        return results
    # Bias resolves from binder length + sequence_type + valid_chars — all invariant across proposals.
    seq_bias = build_sequence_logit_bias_matrix_from_sequence(config.sequence_bias, input_sequences[0][0])
    bias: np.ndarray | float = seq_bias if seq_bias is not None else 0.0
    for (binder_seq,) in input_sequences:
        raw_logits = binder_seq.logits
        assert raw_logits is not None  # noqa: S101 -- input slot validation guarantees logits
        logits = config.logit_scale * raw_logits + bias

        output = run_esm2_gradient(
            ESM2GradientInput(logits=logits.tolist(), temperature=config.temperature),
            ESM2GradientConfig(
                model_checkpoint=config.model_checkpoint,
                use_ste=config.use_ste,
                compute_gradient=True,
                batch_size=config.batch_size,
                device=config.device,
            ),
        )
        assert output.gradient is not None  # noqa: S101 -- compute_gradient=True guarantees it
        grad = np.array(output.gradient, dtype=np.float64) * config.logit_scale

        perplexity = float(np.exp(output.loss))
        score = output.loss if config.score_mode == "nll" else perplexity
        if config.score_mode == "ppl":
            if not np.isfinite(perplexity):
                raise ValueError(
                    f"ESM2 perplexity {perplexity} is non-finite (mean NLL={output.loss}); "
                    "cannot scale gradient for score_mode='ppl'."
                )
            grad = grad * perplexity
        results.append(
            GradientConstraintOutput(
                gradient=(grad,),
                loss=score,
                metrics=_esm2_metadata(output, config, perplexity),
            )
        )
    return results


@constraint(
    key="esm2-perplexity",
    label="ESM2 Perplexity",
    config=ESM2PerplexityConfig,
    description="Score protein naturalness by ESM2 mean NLL and report perplexity.",
    tools_called=["esm2-gradient"],
    uses_gpu=True,
    category="sequence scoring",
    supported_sequence_types=["protein"],
    input_labels=[InputSlot(label="Sequence", requires_logits=True)],
    backward=esm2_perplexity_gradient_backward,
    backward_config=ESM2PerplexityConfig,
)
def esm2_perplexity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: ESM2PerplexityConfig,
) -> list[ConstraintOutput]:
    """Forward ESM2 protein scoring via masked pseudo-log-likelihood."""
    results: list[ConstraintOutput] = []
    for (binder_seq,) in input_sequences:
        output = run_esm2_gradient(
            ESM2GradientInput(logits=one_hot_protein_matrix(binder_seq.sequence), temperature=config.temperature),
            ESM2GradientConfig(
                model_checkpoint=config.model_checkpoint,
                use_ste=config.use_ste,
                compute_gradient=False,
                batch_size=config.batch_size,
                device=config.device,
            ),
        )
        perplexity = float(np.exp(output.loss))
        score = output.loss if config.score_mode == "nll" else perplexity
        results.append(
            ConstraintOutput(
                score=score,
                metadata=_esm2_metadata(output, config, perplexity),
            )
        )
    return results


# Intentional exception: semigreedy ranking can use the raw LM objective directly.
esm2_perplexity_constraint._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
