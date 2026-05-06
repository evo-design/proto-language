"""Provides a decorator-based API for registering constraint functions and.

a factory method for creating Constraint instances.
"""

import typing
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import pydantic
from pydantic import BaseModel, Field, field_serializer
from pydantic.json_schema import SkipJsonSchema

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Constraint, Segment
from proto_language.language.core.constraint import GradientConstraintOutput, InputSlot
from proto_language.utils.helpers import format_pydantic_error

__all__ = ["ConstraintRegistry", "ConstraintSpec", "InputSlot", "constraint"]


class ConstraintSpec(BaseSpec):
    """Specification for a registered constraint."""

    tools_called: list[str] = Field(
        description="List of tool keys this constraint calls (e.g., ['esmfold-prediction', 'prodigal-prediction']). Helps agent find relevant tool documentation."
    )
    category: str | None = Field(
        default=None,
        description="Optional category for organization (e.g., 'protein_structure', 'sequence_composition'). Not required for custom constraints.",
    )
    supported_sequence_types: list[str] = Field(
        description="List of supported sequence types (e.g., ['dna', 'protein']). Must be non-empty."
    )
    input_labels: list[str | InputSlot] | None = Field(  # type: ignore[assignment]
        default_factory=lambda: ["Sequence"],
        description="Per-slot labels; strings for plain labels, InputSlot for swap-detection. "
        "None means any number of interchangeable inputs.",
    )

    requires_generators: list[str] | None = Field(
        default=None,
        description="Generator registry keys required in the same optimizer stage. "
        "Validated at optimizer construction time. None means no generator dependency.",
    )

    # Constraint mode — set during registration, exposed in API
    mode: Literal["discrete", "gradient", "dual"] = Field(
        default="discrete",
        description="Whether this constraint uses discrete scoring ('discrete'), gradient computation ('gradient'), or both ('dual').",
    )

    # Separate config model for backward callable (None = uses config_model)
    backward_config_model: SkipJsonSchema[type[BaseModel] | None] = Field(
        default=None,
        description="Pydantic model for backward/gradient configuration. If None, uses config_model.",
    )

    @field_serializer("backward_config_model")
    def serialize_backward_config_model(self, v: type[BaseModel] | None) -> dict[str, Any] | None:
        """Serialize backward_config_model as standard JSON Schema, or None if absent."""
        if v is None:
            return None
        return self.serialize_config_model(v)

    # Private fields - excluded from serialization
    function: SkipJsonSchema[Callable[..., Any] | None] = Field(default=None, exclude=True)
    backward: SkipJsonSchema[Callable[..., Any] | None] = Field(default=None, exclude=True)


class ConstraintRegistry(BaseRegistry[ConstraintSpec]):
    """Registry for constraint discovery and API/client integration.

    All forward constraint functions use a standardized signature:
        (input_sequences: list[tuple[Sequence, ...]], config) -> list[ConstraintOutput]

    Public Methods:
    - register(): Decorator to register constraint functions
    - list_all(): List constraints with metadata (uses_gpu, etc.)
    - create(): Factory to create Constraint instances from config dicts
    - get(): Get constraint spec by key (inherited)
    - get_schema(): Get JSON schema for constraint configuration (inherited)
    - count(): Get number of registered constraints (inherited)

    Examples:
        Registration:
        >>> @constraint(
        ...     key="gc-content",
        ...     label="GC Content",
        ...     config=GCContentConfig,
        ...     description="Enforce GC content within range",
        ...     supported_sequence_types=["dna", "rna"],
        ... )
        ... def gc_content_constraint(
        ...     input_sequences: list[tuple[Sequence, ...]], config: GCContentConfig
        ... ) -> list[ConstraintOutput]:
        ...     return [
        ...         ConstraintOutput(score=calculate_penalty(seq_tuple[0], config)) for seq_tuple in input_sequences
        ...     ]

        API/Client Usage:
        >>> # List all available constraints
        >>> constraints = ConstraintRegistry.list_all()
        >>>
        >>> # Get form schema
        >>> schema = ConstraintRegistry.get_schema("gc-content")
        >>>
        >>> # Create from user input
        >>> constraint = ConstraintRegistry.create(
        ...     key="gc-content", segments=[segment], config_dict={"min_gc": 40, "max_gc": 60}
        ... )

        Direct Library Usage (no registry needed):
        >>> # Users can bypass registry entirely
        >>> constraint = Constraint(
        ...     inputs=[segment], function=gc_content_constraint, function_config=GCContentConfig(min_gc=40, max_gc=60)
        ... )
    """

    # Each registry subclass must have its own _registry dict
    _registry: ClassVar[dict[str, ConstraintSpec]] = {}

    @classmethod
    def register(  # type: ignore[override]
        cls,
        key: str,
        label: str,
        config: type[BaseModel],
        description: str,
        uses_gpu: bool = False,
        tools_called: list[str] | None = None,
        category: str | None = None,
        supported_sequence_types: list[str] | None = None,
        input_labels: list[str | InputSlot] | None = ("Sequence",),  # type: ignore[assignment]
        requires_generators: list[str] | None = None,
        backward: Callable[..., Any] | None = None,
        backward_config: type[BaseModel] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a constraint function or backward callable.

        The decorated function's role is auto-detected from its return type
        annotation: ``-> list[GradientConstraintOutput]`` registers as the backward
        callable, anything else registers as the scoring function.

        Args:
            key (str): Unique identifier (e.g., "gc-content", "protein-length").
            label (str): Readable external name (e.g., "GC Content Range", "Protein Length").
            config (type[BaseModel]): Pydantic model class for configuration validation.
            description (str): Readable description.
            uses_gpu (bool): If True, constraint requires GPU for computation.
            tools_called (list[str] | None): Tool keys this constraint calls.
            category (str | None): Optional category for organization.
            supported_sequence_types (list[str] | None): Supported sequence types (e.g., ``["dna", "protein"]``).
            input_labels (list[str | InputSlot] | None): Per-slot labels; strings become plain
                ``InputSlot(label=s)``. Use ``InputSlot(..., requires_logits=True)`` /
                ``requires_structure=True`` to enable swap-detection. ``None`` means any number
                of interchangeable inputs.
            requires_generators (list[str] | None): Generator registry keys required in
                the same optimizer stage. Validated at optimizer construction time.
            backward (Callable[..., Any] | None): Explicit backward callable to pair with
                a scoring function. Cannot be used when the decorated function itself
                returns ``list[GradientConstraintOutput]``.
            backward_config (type[BaseModel] | None): Pydantic model class for backward
                configuration. If None, the backward callable uses ``config`` instead.
                Only meaningful when ``backward`` is provided or the decorated function
                returns ``list[GradientConstraintOutput]``.

        Returns:
            Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator that registers the function.

        Raises:
            ValueError: If the decorated function returns ``list[GradientConstraintOutput]``
                and ``backward`` is also provided.

        Examples:
            Scoring function (single segment, default label):

            >>> @constraint(key="gc-content", ...)
            ... def gc_content(input_sequences, config) -> list[ConstraintOutput]: ...

            Multi-segment with labeled slots:

            >>> @constraint(key="gap-gini", input_labels=["Query", "Reference"], ...)
            ... def gap_gini(input_sequences, config) -> list[ConstraintOutput]: ...

            Gradient constraint (auto-detected from return type):

            >>> @constraint(key="ablang-perplexity", ...)
            ... def ablang_backward(input_sequences, *, config, temperature, **kwargs) -> list[GradientConstraintOutput]: ...

            Scoring function with explicit backward callable:

            >>> @constraint(key="ablang", backward=ablang_backward, ...)
            ... def ablang_score(input_sequences, config) -> list[ConstraintOutput]: ...

            Scoring + backward with separate config models:

            >>> @constraint(key="ablang", backward=ablang_backward, backward_config=AbLangGradientConfig, ...)
            ... def ablang_score(input_sequences, config) -> list[ConstraintOutput]: ...
        """
        if supported_sequence_types is None:
            supported_sequence_types = []
        if tools_called is None:
            tools_called = []

        slot_count = len(input_labels) if input_labels is not None else None

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, func.__name__)

            # Validate supported_sequence_types is non-empty
            if not supported_sequence_types:
                raise ValueError(f"supported_sequence_types must be non-empty for constraint '{key}'")

            is_backward_fn = typing.get_type_hints(func).get("return") == list[GradientConstraintOutput]
            if is_backward_fn and backward is not None:
                raise ValueError(
                    f"Constraint '{key}': decorated function returns list[GradientConstraintOutput] but backward= was also provided"
                )
            if backward_config is not None and not (is_backward_fn or backward is not None):
                raise ValueError(
                    f"Constraint '{key}': backward_config= requires backward= or -> list[GradientConstraintOutput]"
                )

            # Store metadata as function attributes for Constraint class to use
            func._constraint_config_class = config  # type: ignore[attr-defined]
            func._constraint_supported_sequence_types = supported_sequence_types  # type: ignore[attr-defined]
            func._constraint_num_input_sequences_per_tuple = slot_count  # type: ignore[attr-defined]

            if is_backward_fn:
                mode: Literal["discrete", "gradient", "dual"] = "gradient"
            elif backward is not None:
                mode = "dual"
            else:
                mode = "discrete"

            cls._registry[key] = ConstraintSpec(
                key=key,
                label=label,
                config_model=config,
                description=description,
                function=None if is_backward_fn else func,
                backward=func if is_backward_fn else backward,
                uses_gpu=uses_gpu,
                tools_called=tools_called,
                category=category,
                supported_sequence_types=supported_sequence_types,
                input_labels=input_labels,
                requires_generators=requires_generators,
                mode=mode,
                backward_config_model=backward_config,
            )
            return func

        return decorator

    @classmethod
    def create(
        cls,
        key: str,
        segments: list[Segment],
        config_dict: dict[str, Any],
        backward_config_dict: dict[str, Any] | None = None,
        label: str | None = None,
        threshold: float | None = None,
        weight: float | None = None,
        gradient_positions: list[int] | None = None,
    ) -> Constraint:
        """Factory method to create a Constraint from JSON-compatible config.

        This is the primary integration point with API/client layers. When the
        registered constraint has a ``backward`` callable, the returned
        ``Constraint`` supports ``compute_gradient()`` (discoverable via
        ``constraint.supports_gradient``).

        Args:
            key (str): Registered constraint identifier (e.g., "gc-content")
            segments (list[Segment]): List of Segment objects to evaluate
            config_dict (dict[str, Any]): Configuration as plain dict (from JSON/client).
                Used for the scoring function config; also used for backward config
                when ``backward_config_dict`` is not provided.
            backward_config_dict (dict[str, Any] | None): Configuration for the backward
                callable as plain dict. If None, falls back to ``config_dict``. Validated
                against ``backward_config_model`` if one was registered, otherwise against
                ``config_model``.
            label (str | None): Optional label for metadata tracking
            threshold (float | None): Optional threshold for filtering. If provided,
                constraint acts as a filter: scores <= threshold are accepted (True),
                scores > threshold are rejected (False). If None, returns raw float
                scores for optimization.
            weight (float | None): Optional weight to scale constraint scores.
                Defaults to 1.0 if not provided.
            gradient_positions (list[int] | None): Optional zero-based positions
                allowed to receive gradient when the registered constraint has a
                backward callable.

        Returns:
            Constraint: Configured Constraint instance ready to evaluate

        Raises:
            ValueError: If key is not registered or if ``config_dict``/
                ``backward_config_dict`` fails Pydantic validation. ValidationError
                is reformatted as ``constraint '<key>' config invalid — <field>: <msg>; ...``.

        Examples:
            >>> # Scoring mode (default)
            >>> constraint = ConstraintRegistry.create(
            ...     key="gc-content",
            ...     segments=[dna_segment],
            ...     config_dict={"min_gc": 40, "max_gc": 60},
            ...     label="promoter_gc",
            ... )
            >>> scores = constraint.evaluate()  # Returns list[float]
            >>>
            >>> # Filtering mode (with threshold)
            >>> filter_constraint = ConstraintRegistry.create(
            ...     key="gc-content", segments=[dna_segment], config_dict={"min_gc": 40, "max_gc": 60}, threshold=0.5
            ... )
            >>> passed = filter_constraint.evaluate()  # Returns list[bool]
        """
        spec = cls.get(key)

        if spec.function is None and spec.backward is None:
            raise ValueError(f"Registered constraint '{key}' has neither function nor backward")

        try:
            validated_config = spec.config_model(**config_dict)
        except pydantic.ValidationError as e:
            raise ValueError(format_pydantic_error(e, f"constraint {key!r} config invalid")) from e

        # Backward config: backward_config_model if registered else config_model;
        # backward_config_dict if provided else config_dict.
        validated_backward_config = None
        if spec.backward is not None:
            bw_model = spec.backward_config_model or spec.config_model
            bw_dict = backward_config_dict if backward_config_dict is not None else config_dict
            try:
                validated_backward_config = bw_model(**bw_dict)
            except pydantic.ValidationError as e:
                raise ValueError(format_pydantic_error(e, f"constraint {key!r} backward config invalid")) from e

        return Constraint(
            inputs=segments,
            function=spec.function,
            function_config=validated_config,
            backward=spec.backward,
            backward_config=validated_backward_config,
            label=label,
            threshold=threshold,
            weight=weight,
            input_slots=None
            if spec.input_labels is None
            else [s if isinstance(s, InputSlot) else InputSlot(label=s) for s in spec.input_labels],
            gradient_positions=gradient_positions,
        )

    @classmethod
    def find_key(cls, constraint: Constraint) -> str | None:
        """Get registry key for a constraint instance, or ``None`` if not registered."""
        for key, spec in cls._registry.items():
            if spec.function is not None and spec.function == constraint.function:
                return key
            if spec.backward is not None and spec.backward == constraint.backward:
                return key
        return None

    @classmethod
    def get_key(cls, constraint: Constraint) -> str:
        """Get registry key for a constraint instance. Raises ``ValueError`` if not registered."""
        key = cls.find_key(constraint)
        if key is None:
            raise ValueError(f"Constraint '{constraint.label}' is not registered")
        return key

    @classmethod
    def list_all(cls) -> list[ConstraintSpec]:
        """List all registered constraints as Pydantic models."""
        return list(cls._registry.values())


# Alias for simpler decorator syntax: @constraint(...) instead of @constraint(...)
constraint = ConstraintRegistry.register
