"""Taggability constraint: favor proteins that tolerate fluorescent knock-in tagging.

Bridges the ``gfp-taggability`` classifier into optimization programs. The
classifier returns ``P(success)``; this constraint maps it into proto's
lower-is-better energy convention, either directly (``score_mode="raw"``) or
through the endpoint's reference percentile (``score_mode="percentile"``).

Three caveats belong with every use of this signal:

1. It predicts whether the **experimental pipeline** succeeds. The training
   labels absorb failures from bad guide RNA, low expression, poor repair, and
   failed sorts, which is broader than intrinsic taggability — a low score is not
   proof that a protein cannot be tagged.
2. It has been measured on **one assay**: 1,757 attempts, HEK293T cells,
   split-mNeonGreen knock-in. Other cell lines, organisms, and tagging chemistries
   are untested rather than disproven.
3. It is **terminus-agnostic**. Choosing an N- or C-terminal tag is a separate
   model and is not served here.

Examples:
    >>> constraint = ConstraintRegistry.create("taggability", [protein_segment], {})  # doctest: +SKIP
    >>> constraint.evaluate()  # doctest: +SKIP
    [0.0228]
"""

import logging
from typing import Any, Literal

from proto_language.classifiers.base import ClassifierEndpointError, ClassifierOutput
from proto_language.classifiers.classifier_registry import ClassifierRegistry
from proto_language.classifiers.protein_tagging.gfp_taggability_classifier import (
    TAGGABILITY_API_KEY_ENV,
)
from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField

__all__ = ["TaggabilityConfig", "taggability_constraint"]

logger = logging.getLogger(__name__)

TAGGABILITY_CLASSIFIER_KEY = "gfp-taggability"
"""Registry key of the classifier this constraint wraps."""


class TaggabilityConfig(BaseConfig):
    """Configuration for taggability scoring.

    The endpoint returns both a raw probability and its rank among 1,757 reference
    proteins. ``score_mode`` selects which drives the energy; both are always
    recorded in metadata.

    Raw scores are not centred on 0.5. Because 74.6% of reference tagging attempts
    succeeded, a raw 0.64 is still bottom-quartile and 0.83 is the median, so
    ``"raw"`` energies compress near the top of the range. That is harmless for
    threshold-based selection, where the two modes rank proposals identically, but
    ``"percentile"`` spreads the scale evenly if this signal is ever summed against
    other weighted constraints.

    Attributes:
        score_mode (Literal["raw", "percentile"]): Which endpoint value drives the
            energy. "raw" uses 1 - P(success); "percentile" uses 1 - percentile.
        strict (bool): Raise on endpoint failure instead of scoring the batch worst.
        base_url (str | None): Endpoint base URL. When None, read from the
            ``FLUORESCE_API_URL`` environment variable.
        api_key_env (str): Environment variable holding the bearer token.
        timeout_s (float): Per-request timeout in seconds.
        max_retries (int): Additional attempts after a server error or timeout.
        cache_size (int): Retained cached responses keyed by sequence; 0 disables.
    """

    score_mode: Literal["raw", "percentile"] = ConfigField(
        default="raw",
        title="Score Mode",
        description="Energy source: 'raw' uses 1 - P(success); 'percentile' uses 1 - percentile.",
    )
    strict: bool = ConfigField(
        default=False,
        title="Strict Errors",
        description="Raise on endpoint failure instead of scoring the whole batch worst.",
    )
    base_url: str | None = ConfigField(
        default=None,
        title="Endpoint Base URL",
        description="Taggability endpoint base URL. When unset, read from FLUORESCE_API_URL.",
    )
    api_key_env: str = ConfigField(
        default=TAGGABILITY_API_KEY_ENV,
        title="API Key Env Var",
        description="Environment variable holding the endpoint bearer token.",
    )
    timeout_s: float = ConfigField(
        default=60.0,
        title="Request Timeout",
        description="Per-request timeout in seconds; absorbs the cold-start model load.",
        gt=0.0,
    )
    max_retries: int = ConfigField(
        default=1,
        title="Max Retries",
        description="Additional attempts after a server error or timeout.",
        ge=0,
    )
    cache_size: int = ConfigField(
        default=1024,
        title="Cache Size",
        description="Retained cached responses keyed by sequence; 0 disables caching.",
        ge=0,
    )

    def classifier_config_dict(self) -> dict[str, Any]:
        """Return the subset of fields that configure the wrapped classifier.

        Returns:
            dict[str, Any]: Config dict accepted by ``GFPTaggabilityConfig``.
        """
        return {
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "cache_size": self.cache_size,
        }


def taggability_energy(output: ClassifierOutput, config: TaggabilityConfig) -> float:
    """Map one classifier output into proto's lower-is-better energy convention.

    Args:
        output (ClassifierOutput): Native classifier output for one proposal.
        config (TaggabilityConfig): Validated configuration object.

    Returns:
        float: Energy in ``[0.0, 1.0]``, where 0.0 is maximally taggable. Sequences
            the endpoint rejected, and percentile mode without a reported
            percentile, score ``MAX_ENERGY``.
    """
    if output.metadata.get("taggability_invalid"):
        return MAX_ENERGY

    if config.score_mode == "percentile":
        percentile = output.metadata.get("percentile")
        if percentile is None:
            logger.warning("Taggability endpoint reported no percentile; scoring this proposal worst.")
            return MAX_ENERGY
        value = float(percentile)
    else:
        value = output.score

    return min(MAX_ENERGY, max(MIN_ENERGY, 1.0 - value))


def _error_outputs(input_sequences: list[tuple[Sequence, ...]], message: str) -> list[ConstraintOutput]:
    """Score every proposal worst after an endpoint failure.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): The batch that failed to score.
        message (str): Endpoint error message to surface in metadata.

    Returns:
        list[ConstraintOutput]: One worst-score output per proposal.
    """
    return [
        ConstraintOutput(
            score=MAX_ENERGY,
            metadata={"classifier_error": True, "classifier_error_message": message},
        )
        for _ in input_sequences
    ]


@constraint(
    key="taggability",
    label="Taggability",
    config=TaggabilityConfig,
    description="Favor proteins predicted to tolerate fluorescent knock-in tagging.",
    uses_gpu=False,
    tools_called=[],
    category="protein_tagging",
    supported_sequence_types=["protein"],
    input_labels=["Protein"],
)
def taggability_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: TaggabilityConfig,
) -> list[ConstraintOutput]:
    """Score proteins by predicted fluorescent-tag compatibility.

    The single input slot is the bare protein. The wrapped model is trained on
    bare proteins, so passing a tag/linker construct would shift the pooled
    embedding off the training distribution and invalidate the score.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One tuple per proposal, each
            holding the protein Sequence for the "Protein" slot.
        config (TaggabilityConfig): Validated configuration object.

    Returns:
        list[ConstraintOutput]: One output per proposal. ``score`` is in
            ``[0.0, 1.0]`` where 0.0 is maximally taggable; ``metadata`` carries the
            raw probability, percentile, and model version.

    Raises:
        ClassifierEndpointError: If the endpoint fails and ``strict`` is True.
    """
    if not input_sequences:
        return []

    sequences = [protein.sequence for (protein,) in input_sequences]
    bound = ClassifierRegistry.create(TAGGABILITY_CLASSIFIER_KEY, config.classifier_config_dict())

    try:
        outputs = bound.predict(sequences)
    except ClassifierEndpointError as exc:
        if config.strict:
            raise
        logger.warning("Taggability endpoint failed; scoring the batch worst: %s", exc)
        return _error_outputs(input_sequences, str(exc))

    results: list[ConstraintOutput] = []
    for output in outputs:
        energy = taggability_energy(output, config)
        results.append(
            ConstraintOutput(
                score=energy,
                metadata={**output.metadata, "taggability_energy": energy, "score_mode": config.score_mode},
            )
        )
    return results
