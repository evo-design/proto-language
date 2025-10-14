"""
Base registry pattern for decorator-based component registration.

Provides shared infrastructure for ConstraintRegistry, GeneratorRegistry, and ToolRegistry.
"""

from typing import Dict, Type, Any, TypeVar, Generic, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel


SpecType = TypeVar('SpecType')


@dataclass
class BaseSpec:
    """Base specification for registered components."""
    config_model: Type[BaseModel]
    description: str


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
    def list_all(cls) -> Dict[str, dict]:
        """List all components with descriptions and schemas. Implemented by subclasses."""
        raise NotImplementedError(f"{cls.__name__}.list_all() must be implemented by subclass")
    
    @classmethod
    def get(cls, key: str) -> SpecType:
        """
        Get component spec by key.
        
        Args:
            key: Component identifier
            
        Returns:
            Component specification object
            
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
            key: Component identifier
            
        Returns:
            JSON Schema dict with structure:
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
            Number of registered components
        """
        return len(cls._registry)
    
    @classmethod
    def _check_duplicate(cls, key: str, attempted_component_name: str = None) -> None:
        """
        Check for duplicate registration.
        
        Args:
            key: Component identifier to check
            attempted_component_name: Name of component attempting registration (optional)
            
        Raises:
            ValueError: If key already exists in registry
        """
        if key in cls._registry:
            component_type = cls._component_type()
            existing_spec = cls._registry[key]
            
            # Try to get function name from the existing spec if available
            existing_name = getattr(existing_spec, 'function', None)
            if existing_name:
                existing_name = getattr(existing_name, '__name__', str(existing_name))
            else:
                existing_name = "unknown"
            
            error_msg = (
                f"{component_type.capitalize()} '{key}' is already registered. "
                f"Duplicate registration is not allowed."
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
            Component type string (e.g., 'constraint', 'generator', 'tool')
        """
        return cls.__name__.replace('Registry', '').lower()
