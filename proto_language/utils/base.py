"""Base classes for proto-language configs and registries."""

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer


def _require_title_and_description(field_helper: str, title: str | None, description: str | None) -> None:
    """Ensure ``title=`` and ``description=`` are non-empty.

    Both are surfaced via ``model_json_schema()`` and must be non-empty strings.
    """
    missing = []
    if not title:
        missing.append("title=")
    if not description:
        missing.append("description=")
    if not missing:
        return
    names = " and ".join(missing)
    raise TypeError(f"{field_helper} requires {names} as non-empty kwarg(s).")


def ConfigField(
    default: Any = ...,
    *,
    title: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Any:
    """Thin alias over ``pydantic.Field`` so call sites keep using ``ConfigField`` consistently.

    Args:
        default (Any): Default value for the configuration field.
        title (str | None): Short user-readable title; must be a non-empty string.
        description (str | None): Field description; must be a non-empty string.
        kwargs: All other standard Pydantic Field arguments (passed through
            to ``pydantic.Field``).

    Usage:
        param: int = ConfigField(default=42, title="Param", description="...", ge=0)
    """
    _require_title_and_description("ConfigField", title, description)
    return Field(default, title=title, description=description, **kwargs)


class BaseConfig(BaseModel):
    """Base configuration class for consistent behavior across all configs (tools, constraints, and generators).

    Example:
        >>> class MyToolConfig(BaseConfig):
        ...     param1: int
        ...     param2: str
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True,  # Validate on field updates
        use_enum_values=True,  # Serialize enums as values
        validate_default=True,  # Validate default values
    )


class BaseOptimizerConfig(BaseConfig):
    """Shared base config for all optimizers.

    Optimizer instances single-source their effective ``seed`` from this config.
    Program-level seeds overwrite this field with optimizer-specific child
    seeds during program initialization.
    """

    seed: int | None = ConfigField(
        default=None,
        title="Random Seed",
        description="Random seed for reproducible optimization, generator, and constraint tool streams.",
        ge=0,
    )
    tracking_interval: int = ConfigField(
        default=1,
        ge=1,
        title="Tracking Interval",
        description="Save history and log progress every N steps. Step 0 and final step always saved.",
    )
    track_proposals: bool = ConfigField(
        default=False,
        title="Track Proposals",
        description="Save granular per-proposal results (accept/reject) in history snapshots.",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Emit per-step debug information about proposals, scores, and acceptance through the logger.",
    )


SpecType = TypeVar("SpecType", bound="BaseSpec")


class BaseSpec(BaseModel):
    """Base specification for registered components.

    Subclasses extend this to add component-specific metadata.

    Attributes:
        key (str): Unique kebab-case registry identifier.
        label (str): Human-readable display name.
        description (str): Short description of the component.
        uses_gpu (bool): Whether this component requires GPU resources.
        config_model (type[BaseModel]): Pydantic model class for the component configuration.
    """

    key: str = Field(
        title="Registry Key",
        description="Internal kebab-case identifier (e.g. 'mcmc', 'gc-content')",
    )
    label: str = Field(
        title="Display Label",
        description="External display name (e.g. 'MCMC Optimizer', 'GC Content Range')",
    )
    description: str = Field(
        title="Component Description",
        description="Detailed description of component functionality",
    )
    uses_gpu: bool = Field(
        default=False,
        title="Uses GPU",
        description="Whether this component requires GPU resources",
    )
    config_model: type[BaseModel] = Field(
        title="Config Model",
        description="Pydantic model for configuration validation and schema generation",
    )

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }

    @field_serializer("config_model")
    def serialize_config_model(self, config_model: type[BaseModel]) -> dict[str, Any]:
        """Serialize ``config_model`` as standard JSON Schema.

        Args:
            config_model (type[BaseModel]): Pydantic model class for the component configuration.

        Returns:
            dict[str, Any]: Standard JSON Schema dict produced by Pydantic.
        """
        return config_model.model_json_schema()


class BaseRegistry(ABC, Generic[SpecType]):
    """Base registry for decorator-based component registration.

    Provides discovery, schema generation, and factory methods for constraints,
    generators, and tools. Registration happens at import time via decorators.

    Abstract Methods (implemented by subclasses):
    - register(): Decorator to register components
    - list_all(): List all components with metadata

    Public Methods:
    - get(): Retrieve component spec by key
    - get_schema(): Get JSON schema for component configuration
    - count(): Get number of registered components
    - snapshot() / restore(): Transactional rollback of registry state
    - unregister(): Remove a key from the registry
    """

    _registry: ClassVar[dict[str, Any]] = {}

    @classmethod
    @abstractmethod
    def register(cls, key: str, **kwargs: Any) -> Any:
        """Decorator to register a component. Implemented by subclasses."""
        raise NotImplementedError(f"{cls.__name__}.register() must be implemented by subclass")

    @classmethod
    @abstractmethod
    def list_all(cls) -> list[SpecType]:
        """List all components as Pydantic models. Implemented by subclasses."""
        raise NotImplementedError(f"{cls.__name__}.list_all() must be implemented by subclass")

    @classmethod
    def get(cls, key: str) -> SpecType:
        """Get component spec by key.

        Args:
            key (str): Component identifier.

        Returns:
            SpecType: Component specification object.

        Raises:
            ValueError: If ``key`` is not found in the registry.
        """
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            component_type = cls._component_type()
            raise ValueError(f"Unknown {component_type}: '{key}'. Available {component_type}s: {available}")
        return cls._registry[key]  # type: ignore[no-any-return]

    @classmethod
    def get_schema(cls, key: str) -> dict[str, Any]:
        """Get the JSON schema for a specific component's configuration.

        Args:
            key (str): Component identifier.

        Returns:
            dict[str, Any]: JSON Schema dict produced by Pydantic.
        """
        spec = cls.get(key)
        return spec.config_model.model_json_schema()

    @classmethod
    def count(cls) -> int:
        """Return the number of registered components."""
        return len(cls._registry)

    @classmethod
    def snapshot(cls) -> dict[str, SpecType]:
        """Deep-copy the registry for transactional rollback (pair with :meth:`restore`)."""
        return copy.deepcopy(cls._registry)

    @classmethod
    def restore(cls, snapshot: dict[str, SpecType]) -> None:
        """Replace the registry contents with ``snapshot`` in place."""
        cls._registry.clear()
        cls._registry.update(snapshot)

    @classmethod
    def unregister(cls, key: str) -> None:
        """Remove ``key`` from the registry. No-op if not present."""
        cls._registry.pop(key, None)

    @classmethod
    def _check_duplicate(cls, key: str, attempted_component_name: str | None = None) -> None:
        """Raise ``ValueError`` if ``key`` is already registered.

        Args:
            key (str): Component identifier to check.
            attempted_component_name (str | None): Name of the component
                attempting registration, used to enrich the error message.
        """
        if key in cls._registry:
            component_type = cls._component_type()
            existing_spec = cls._registry[key]
            existing_name = getattr(existing_spec, "label", "unknown")

            error_msg = (
                f"{component_type.capitalize()} '{key}' is already registered. Duplicate registration is not allowed."
            )
            if attempted_component_name:
                error_msg += f"\nExisting: {existing_name}, Attempted: {attempted_component_name}"
            else:
                error_msg += f"\nExisting component: {existing_name}"
            raise ValueError(error_msg)

    @classmethod
    def _component_type(cls) -> str:
        """Component type derived from the registry class name (e.g. ``MyRegistry`` → ``my``)."""
        return cls.__name__.replace("Registry", "").lower()
