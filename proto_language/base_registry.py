"""Provides shared infrastructure for ConstraintRegistry, GeneratorRegistry, and ToolRegistry."""

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field, field_serializer

SpecType = TypeVar("SpecType", bound="BaseSpec")

_UI_PRESENTATION_SCHEMA_KEYS = frozenset({"advanced", "hidden", "depends_on", "x-depends-on"})


def strip_ui_presentation_schema_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` without backend-deprecated UI metadata keys.

    Nested ``proto-tools`` config models can still carry legacy UI presentation
    metadata in their JSON Schema. Language registry outputs are the public
    schema boundary for proto-language components, so they strip those keys
    before exposing schemas to downstream consumers.
    """

    def strip(value: Any, *, in_properties: bool = False) -> Any:
        if isinstance(value, dict):
            stripped: dict[str, Any] = {}
            for key, child in value.items():
                if not in_properties and key in _UI_PRESENTATION_SCHEMA_KEYS:
                    continue
                stripped[key] = strip(child, in_properties=key == "properties")
            return stripped
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    stripped_schema = strip(schema)
    if not isinstance(stripped_schema, dict):
        raise TypeError("Expected JSON schema root to be a dictionary.")
    return stripped_schema


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

    # Public fields - exposed in API
    key: str = Field(description="Internal identifier (e.g., 'mcmc', 'gc-content')")
    label: str = Field(description="External UI display name (e.g., 'MCMC Optimizer', 'GC Content Range')")
    description: str = Field(description="Detailed description of component functionality")
    uses_gpu: bool = Field(default=False, description="Whether this component requires GPU")

    # Configuration model
    config_model: type[BaseModel] = Field(
        description="Pydantic model for configuration validation and schema generation"
    )

    model_config = {
        "extra": "allow",  # Allow subclasses to add fields
        "arbitrary_types_allowed": True,  # Allow Type[BaseModel] in config_model
    }

    @field_serializer("config_model")
    def serialize_config_model(self, config_model: type[BaseModel]) -> dict[str, Any]:
        """Serialize config_model as standard JSON Schema.

        Returns the full Pydantic JSON Schema including properties, required fields,
        and metadata. Provides a standard format for tooling and validation
        and validation.

        Args:
            config_model (type[BaseModel]): Pydantic model class for the component configuration.

        Returns:
            dict[str, Any]: Standard JSON Schema dict with structure::

                {
                    "properties": {
                        "param_name": {"type": "number", "description": "...", ...}
                    },
                    "required": ["param1", "param2"],
                    "title": "ConfigModelName",
                    "type": "object"
                }
        """
        return strip_ui_presentation_schema_keys(config_model.model_json_schema())


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

    # Any instead of SpecType — mypy can't handle ClassVar with generic type params
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
            key (str): Component identifier

        Returns:
            SpecType: Component specification object

        Raises:
            ValueError: If key not found in registry
        """
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))  # List all registered keys
            component_type = cls._component_type()  # Get the component type (e.g. "constraint", "generator", "tool")
            raise ValueError(f"Unknown {component_type}: '{key}'. Available {component_type}s: {available}")
        return cls._registry[key]  # type: ignore[no-any-return]

    @classmethod
    def get_schema(cls, key: str) -> dict[str, Any]:
        """Get the JSON schema for a specific component's configuration.

        The schema includes parameter names, types, defaults, validation rules,
        and descriptions for tooling and validation.

        Args:
            key (str): Component identifier

        Returns:
            dict[str, Any]: JSON Schema dict with structure::

                {
                    "properties": {
                        "param_name": {
                            "type": "number",
                            "description": "Parameter description",
                            "default": 42,
                            ...
                        },
                        ...
                    },
                    "required": ["param1", "param2"],
                    "title": "ConfigModelName",
                    ...
                }

        Examples:
            >>> schema = MyRegistry.get_schema("my_component")
            >>> # Client uses this to generate form fields:
            >>> for param_name, param_info in schema["properties"].items():
            ...     print(f"{param_name}: {param_info['type']}")
        """
        spec = cls.get(key)
        return strip_ui_presentation_schema_keys(spec.config_model.model_json_schema())

    @classmethod
    def count(cls) -> int:
        """Get count of registered components.

        Returns:
            int: Number of registered components
        """
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
        """Check for duplicate registration.

        Args:
            key (str): Component identifier to check
            attempted_component_name (str | None): Name of component attempting registration (optional)

        Raises:
            ValueError: If key already exists in registry
        """
        if key in cls._registry:
            component_type = cls._component_type()
            existing_spec = cls._registry[key]

            # Try to get name from the existing spec label
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
        """Get component type derived from registry class name.

        Returns:
            str: Component type string (e.g., 'constraint', 'generator', 'tool')
        """
        return cls.__name__.replace("Registry", "").lower()
