"""Core types for the classifier component family.

A classifier is a trained predictor that maps a sequence to a continuous score.
It is deliberately *not* part of ``proto_language.core``: the core abstractions are
there because ``Program`` orchestrates them into ``Construct`` objects, whereas a
classifier is never orchestrated directly — a constraint wraps one and bridges it
into a program. Keeping the family self-contained also keeps its native score out
of the energy convention, so ``ClassifierOutput.score`` stays the model's raw
output and each constraint decides how to map it into ``[0, 1]`` energy.

Examples:
    >>> output = ClassifierOutput(score=0.9772, metadata={"percentile": 0.936})
    >>> output.score  # 0.9772
"""

from collections.abc import Callable
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Classifier",
    "ClassifierEndpointError",
    "ClassifierError",
    "ClassifierInvalidSequenceError",
    "ClassifierOutput",
    "PredictFunction",
]


class ClassifierError(Exception):
    """Base error for the classifier family."""


class ClassifierEndpointError(ClassifierError):
    """A remote classifier endpoint was unreachable, unauthorized, or failing.

    Raised for transport failures, authentication failures, and server-side
    errors — conditions that describe the endpoint rather than the sequence.
    """


class ClassifierInvalidSequenceError(ClassifierError):
    """A sequence was rejected by the classifier as unscoreable.

    Distinct from :class:`ClassifierEndpointError` because the endpoint is
    healthy and the *proposal* is at fault, so callers can score the remaining
    proposals and penalize only this one.

    Attributes:
        detail (str): Endpoint-supplied explanation of why the sequence was rejected.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        """Store ``detail`` alongside the human-readable ``message``.

        Args:
            message (str): Human-readable error message.
            detail (str): Endpoint-supplied rejection reason.
        """
        super().__init__(message)
        self.detail = detail


class ClassifierOutput(BaseModel):
    """Typed result of scoring a single sequence with a classifier.

    Attributes:
        score (float): The classifier's native output for this sequence. Not
            energy-normalized — mapping into proto's lower-is-better ``[0, 1]``
            convention is the wrapping constraint's responsibility.
        metadata (dict[str, Any]): Flat per-sequence auxiliary data, such as model
            version or endpoint diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    score: float = Field(
        title="Score",
        description="Native classifier output for one sequence; not energy-normalized.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        title="Metadata",
        description="Per-sequence auxiliary data returned alongside the score.",
    )


PredictFunction: TypeAlias = Callable[[list[str], Any], list[ClassifierOutput]]
"""Registered predict callable: ``(sequences, config) -> list[ClassifierOutput]``."""


class Classifier:
    """A registered classifier bound to a validated configuration.

    Instances are produced by ``ClassifierRegistry.create()`` rather than
    constructed directly. ``predict_scores`` exposes the batched
    ``Callable[[list[str]], list[float]]`` shape that external design pipelines
    expect from a predictor.

    Attributes:
        key (str): Registry key this classifier was created from.
        config (BaseModel): Validated configuration for the predict function.

    Examples:
        >>> classifier = ClassifierRegistry.create("gfp-taggability", {})  # doctest: +SKIP
        >>> classifier.predict_scores(["MKTAYIAKQRQISFVKSHFSRQ"])  # doctest: +SKIP
        [0.83]
    """

    def __init__(self, key: str, function: PredictFunction, config: BaseModel) -> None:
        """Bind a registered predict function to its validated config.

        Args:
            key (str): Registry key this classifier was created from.
            function (PredictFunction): Registered batched predict callable.
            config (BaseModel): Validated configuration instance.
        """
        self.key = key
        self.config = config
        self._function = function

    def predict(self, sequences: list[str]) -> list[ClassifierOutput]:
        """Score sequences, returning one output per input in the same order.

        Args:
            sequences (list[str]): Sequences to score.

        Returns:
            list[ClassifierOutput]: One output per input sequence, order-aligned.

        Raises:
            ValueError: If the predict function returns a mismatched number of results.
        """
        if not sequences:
            return []
        outputs = self._function(sequences, self.config)
        if len(outputs) != len(sequences):
            raise ValueError(f"Classifier '{self.key}' returned {len(outputs)} results for {len(sequences)} sequences.")
        return outputs

    def predict_scores(self, sequences: list[str]) -> list[float]:
        """Score sequences and return the bare floats.

        This is the batched predictor contract external pipelines inject into a
        design program: ``Callable[[list[str]], list[float]]``.

        Args:
            sequences (list[str]): Sequences to score.

        Returns:
            list[float]: Native classifier scores, order-aligned with ``sequences``.
        """
        return [output.score for output in self.predict(sequences)]

    def __repr__(self) -> str:
        """Return a debug representation naming the registry key."""
        return f"Classifier(key={self.key!r})"
