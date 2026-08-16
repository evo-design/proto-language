"""Decorator-based registry for classifiers, plus a factory for bound instances.

Mirrors ``ConstraintRegistry`` in shape: ``@classifier`` records a spec at import
time, ``create()`` validates a plain config dict and returns a usable
``Classifier``, and the spec serializes to JSON Schema for docs and UI discovery.

Examples:
    >>> ClassifierRegistry.count() >= 1  # doctest: +SKIP
    True
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import pydantic
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from proto_language.classifiers.base import Classifier
from proto_language.utils.base import BaseRegistry, BaseSpec
from proto_language.utils.serialization import format_pydantic_error

if TYPE_CHECKING:
    from proto_language.utils.docs_api import ComponentDoc, ConfigModelDoc

__all__ = ["ClassifierRegistry", "ClassifierSpec", "classifier"]


class ClassifierSpec(BaseSpec):
    """Specification for a registered classifier.

    Attributes:
        category (str | None): Optional grouping; matches the subdirectory name.
        supported_sequence_types (list[str]): Sequence types this classifier accepts;
            must be non-empty.
        output_range (tuple[float, float] | None): Inclusive bounds of the native
            score, or None when the output is unbounded.
        output_description (str): What the native score means, for docs and UI.
        endpoint_env_var (str | None): Environment variable naming the remote
            endpoint, or None for classifiers that run locally.
        function (SkipJsonSchema[Callable[..., Any] | None]): Registered predict
            callable; excluded from serialization.
    """

    category: str | None = Field(
        default=None,
        title="Category",
        description="Optional grouping (e.g. 'protein_tagging'); matches the subdirectory name",
    )
    supported_sequence_types: list[str] = Field(
        title="Supported Sequence Types",
        description="Sequence types this classifier accepts (e.g. ['protein']); must be non-empty",
    )
    output_range: tuple[float, float] | None = Field(
        default=None,
        title="Output Range",
        description="Inclusive bounds of the native score, or None when unbounded",
    )
    output_description: str = Field(
        title="Output Description",
        description="What the native score means (e.g. 'P(knock-in tagging succeeds)')",
    )
    endpoint_env_var: str | None = Field(
        default=None,
        title="Endpoint Env Var",
        description="Environment variable naming the remote endpoint; None when run locally",
    )

    function: SkipJsonSchema[Callable[..., Any] | None] = Field(default=None, exclude=True)


class ClassifierRegistry(BaseRegistry[ClassifierSpec]):
    """Registry for classifier discovery, schema export, and instantiation.

    All predict functions use a standardized batched signature:
        ``(sequences: list[str], config) -> list[ClassifierOutput]``

    Batching lives in the signature even when a backend scores one sequence at a
    time, so swapping in a batch-capable backend never changes call sites.

    Public Methods:
    - register(): Decorator to register predict functions
    - list_all(): List classifiers with metadata
    - create(): Factory to create Classifier instances from config dicts
    - get(): Get classifier spec by key (inherited)
    - get_schema(): Get JSON schema for classifier configuration (inherited)
    - count(): Get number of registered classifiers (inherited)

    Examples:
        Registration:
        >>> @classifier(
        ...     key="gfp-taggability",
        ...     label="Taggability",
        ...     config=GFPTaggabilityConfig,
        ...     description="Predict whether a fluorescent knock-in tag succeeds",
        ...     supported_sequence_types=["protein"],
        ...     output_description="P(knock-in tagging succeeds)",
        ... )
        ... def gfp_taggability_classifier(
        ...     sequences: list[str], config: GFPTaggabilityConfig
        ... ) -> list[ClassifierOutput]:
        ...     return [ClassifierOutput(score=score_one(s, config)) for s in sequences]

        API Usage:
        >>> classifiers = ClassifierRegistry.list_all()
        >>> schema = ClassifierRegistry.get_schema("gfp-taggability")
        >>> bound = ClassifierRegistry.create("gfp-taggability", {"timeout_s": 30.0})
    """

    # Each registry subclass must have its own _registry dict
    _registry: ClassVar[dict[str, ClassifierSpec]] = {}

    @classmethod
    def register(  # type: ignore[override]
        cls,
        key: str,
        label: str,
        config: type[BaseModel],
        description: str,
        output_description: str,
        uses_gpu: bool = False,
        category: str | None = None,
        supported_sequence_types: list[str] | None = None,
        output_range: tuple[float, float] | None = None,
        endpoint_env_var: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a classifier predict function.

        Args:
            key (str): Unique kebab-case identifier (e.g. "gfp-taggability").
            label (str): Readable external name (e.g. "Taggability").
            config (type[BaseModel]): Pydantic model class for configuration validation.
            description (str): Readable description.
            output_description (str): What the native score means.
            uses_gpu (bool): If True, the classifier requires local GPU resources.
                Remote endpoints that use a GPU server-side stay False.
            category (str | None): Optional category for organization.
            supported_sequence_types (list[str] | None): Supported sequence types
                (e.g. ``["protein"]``).
            output_range (tuple[float, float] | None): Inclusive bounds of the native score.
            endpoint_env_var (str | None): Environment variable naming the remote endpoint.

        Returns:
            Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator that registers the function.

        Raises:
            ValueError: If ``key`` is already registered, or if
                ``supported_sequence_types`` is empty.

        Examples:
            >>> @classifier(key="my-classifier", ...)  # doctest: +SKIP
            ... def my_classifier(sequences, config) -> list[ClassifierOutput]: ...
        """
        resolved_sequence_types = supported_sequence_types if supported_sequence_types is not None else []

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cls._check_duplicate(key, func.__name__)

            if not resolved_sequence_types:
                raise ValueError(f"supported_sequence_types must be non-empty for classifier '{key}'")

            cls._registry[key] = ClassifierSpec(
                key=key,
                label=label,
                config_model=config,
                description=description,
                function=func,
                uses_gpu=uses_gpu,
                category=category,
                supported_sequence_types=resolved_sequence_types,
                output_range=output_range,
                output_description=output_description,
                endpoint_env_var=endpoint_env_var,
            )
            return func

        return decorator

    @classmethod
    def create(cls, key: str, config_dict: dict[str, Any]) -> Classifier:
        """Factory method to create a Classifier from a JSON-compatible config.

        Args:
            key (str): Registered classifier identifier (e.g. "gfp-taggability").
            config_dict (dict[str, Any]): Configuration as a plain dict.

        Returns:
            Classifier: Configured classifier ready to predict.

        Raises:
            ValueError: If ``key`` is not registered, if the spec has no predict
                function, or if ``config_dict`` fails Pydantic validation.
                ValidationError is reformatted as
                ``classifier '<key>' config invalid — <field>: <msg>; ...``.

        Examples:
            >>> bound = ClassifierRegistry.create("gfp-taggability", {})  # doctest: +SKIP
            >>> bound.predict_scores(["MKTAYIAKQRQISFVKSHFSRQ"])  # doctest: +SKIP
            [0.83]
        """
        spec = cls.get(key)

        if spec.function is None:
            raise ValueError(f"Registered classifier '{key}' has no predict function")

        try:
            validated_config = spec.config_model(**config_dict)
        except pydantic.ValidationError as e:
            raise ValueError(format_pydantic_error(e, f"classifier {key!r} config invalid")) from e

        return Classifier(key=key, function=spec.function, config=validated_config)

    @classmethod
    def list_all(cls) -> list[ClassifierSpec]:
        """List all registered classifiers as Pydantic models."""
        return list(cls._registry.values())

    @classmethod
    def get_docs(cls, identifier: str) -> "ComponentDoc":
        """Return a ``ComponentDoc`` for the classifier resolved from ``identifier``."""
        from proto_language.utils.docs_api import ComponentDoc, get_classifier_doc

        doc: ComponentDoc = get_classifier_doc(identifier)
        return doc

    @classmethod
    def get_config_doc(cls, identifier: str) -> "ConfigModelDoc":
        """Return a ``ConfigModelDoc`` for the classifier's config model."""
        from proto_language.utils.docs_api import ConfigModelDoc, get_config_doc, resolve_key

        spec = cls.get(resolve_key("classifier", identifier))
        doc: ConfigModelDoc = get_config_doc(spec.config_model)
        return doc


# Alias for simpler decorator syntax: @classifier(...)
classifier = ClassifierRegistry.register
