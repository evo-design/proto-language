"""
proto_language/base_registry.py

Provides shared infrastructure for ConstraintRegistry, GeneratorRegistry, and ToolRegistry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Type, TypeVar

from pydantic import BaseModel, Field, field_serializer

SpecType = TypeVar('SpecType', bound='BaseSpec')


class BaseSpec(BaseModel):
    """
    Base specification for registered components.

    This Pydantic model serves dual purposes:
    1. Internal: Stores component metadata in registry
    2. API: Automatically serialized by FastAPI to JSON

    Subclasses extend this to add component-specific metadata

    Attributes:
        key (str): Unique kebab-case registry identifier.
        label (str): Human-readable display name.
        description (str): Short description shown in the client UI.
        uses_gpu (bool): Whether this component requires GPU resources.
        config_model (type[BaseModel]): Pydantic model class for the component configuration.
    """

    # Public fields - exposed in API
    key: str = Field(description="Internal identifier (e.g., 'mcmc', 'gc-content')")
    label: str = Field(description="External UI display name (e.g., 'MCMC Optimizer', 'GC Content Range')")
    description: str = Field(description="Detailed description of component functionality")
    uses_gpu: bool = Field(default=False, description="Whether this component requires GPU")

    # Configuration model
    config_model: Type[BaseModel] = Field(
        description="Pydantic model for configuration validation and schema generation"
    )

    model_config = {
        "extra": "allow",  # Allow subclasses to add fields
        "arbitrary_types_allowed": True,  # Allow Type[BaseModel] in config_model
    }

    @field_serializer('config_model')
    def serialize_config_model(self, config_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Serialize config_model as standard JSON Schema.

        Returns the full Pydantic JSON Schema including properties, required fields,
        and metadata. This provides a standard format for client form generation
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
        return config_model.model_json_schema()


class BaseRegistry(ABC, Generic[SpecType]):
    """
    Base registry for decorator-based component registration.

    Provides discovery, schema generation, and factory methods for constraints,
    generators, and tools. Registration happens at import time via decorators.

    Abstract Methods (implemented by subclasses):
    - register(): Decorator to register components
    - list_all(): List all components with metadata

    Public Methods:
    - get(): Retrieve component spec by key
    - get_schema(): Get JSON schema for component configuration
    - count(): Get number of registered components
    """

    # Subclasses must define their own _registry class variable
    _registry: Dict[str, SpecType] = {}

    @classmethod
    @abstractmethod
    def register(cls, key: str, **kwargs):
        """Decorator to register a component. Implemented by subclasses."""
        raise NotImplementedError(f"{cls.__name__}.register() must be implemented by subclass")

    @classmethod
    @abstractmethod
    def list_all(cls) -> List[SpecType]:
        """List all components as Pydantic models. Implemented by subclasses."""
        raise NotImplementedError(f"{cls.__name__}.list_all() must be implemented by subclass")

    @classmethod
    def get(cls, key: str) -> SpecType:
        """
        Get component spec by key.

        Args:
            key (str): Component identifier

        Returns:
            SpecType: Component specification object

        Raises:
            ValueError: If key not found in registry
        """
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) # List all registered keys
            component_type = cls._component_type() # Get the component type (e.g. "constraint", "generator", "tool")
            raise ValueError(f"Unknown {component_type}: '{key}'. Available {component_type}s: {available}")
        return cls._registry[key]

    @classmethod
    def get_schema(cls, key: str) -> Dict[str, Any]:
        """
        Get the JSON schema for a specific component's configuration.

        The schema includes parameter names, types, defaults, validation rules,
        and descriptions - everything needed to generate a client form.

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
        return spec.config_model.model_json_schema()

    @classmethod
    def count(cls) -> int:
        """
        Get count of registered components.

        Returns:
            int: Number of registered components
        """
        return len(cls._registry)

    @classmethod
    def _check_duplicate(cls, key: str, attempted_component_name: str = None) -> None:
        """
        Check for duplicate registration.

        Args:
            key (str): Component identifier to check
            attempted_component_name (str): Name of component attempting registration (optional)

        Raises:
            ValueError: If key already exists in registry
        """
        if key in cls._registry:
            component_type = cls._component_type()
            existing_spec = cls._registry[key]

            # Try to get name from the existing spec label
            existing_name = getattr(existing_spec, 'label', 'unknown')

            error_msg = (
                f"{component_type.capitalize()} '{key}' is already registered. "
                "Duplicate registration is not allowed."
            )

            if attempted_component_name:
                error_msg += f"\nExisting: {existing_name}, Attempted: {attempted_component_name}"
            else:
                error_msg += f"\nExisting component: {existing_name}"

            raise ValueError(error_msg)

    @classmethod
    def _component_type(cls) -> str:
        """
        Get component type derived from registry class name.

        Returns:
            str: Component type string (e.g., 'constraint', 'generator', 'tool')
        """
        return cls.__name__.replace('Registry', '').lower()
