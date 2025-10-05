"""
Base registry pattern for decorator-based component registration.

Provides shared infrastructure for ConstraintRegistry, GeneratorRegistry, etc.
"""

from typing import Dict, Type, Any, TypeVar, Generic, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel


# Type variable for the specification type (ConstraintSpec, GeneratorSpec, etc.)
SpecType = TypeVar('SpecType')


@dataclass
class BaseSpec:
    """Base specification for registered components."""
    config_model: Type[BaseModel]
    description: str


class BaseRegistry(ABC, Generic[SpecType]):
    """
    Base registry for decorator-based component registration.
    
    This abstract base class provides common infrastructure for:
    - Discovery: List all registered components
    - Schema generation: Get JSON schemas for client forms
    - Validation: Parse and validate user input via Pydantic
    - Factory: Create instances from dict config
    
    Subclasses implement:
    - Specific registration decorators with domain-specific parameters
    - Factory methods that create domain objects
    - Custom list_all() output formats if needed
    
    Design Pattern:
    ---------------
    The registry pattern is implemented using class variables and decorators.
    Registration happens at import time when decorated functions/classes are loaded.
    
    Import-Time Registration:
    -------------------------
    WARNING: Decorators run at import time. Components are only registered when
    their modules are imported. To ensure all builtin components are available:
    
    1. Import from the package level (not individual files):
       ✅ from proto_language.language.constraint import ConstraintRegistry
       ❌ from proto_language.language.constraint.registry import ConstraintRegistry
    
    2. Ensure __init__.py files import all component modules
    
    3. Call ensure_loaded() in tests to validate registration
    
    Custom Components:
    ------------------
    Users can define custom components in two ways:
    
    1. With Registry (discoverable via API):
       @MyRegistry.register(key="custom", ...)
       def custom_component(...):
           pass
    
    2. Without Registry (direct usage):
       component = MyComponent(...)  # Works without registration
    
    Examples:
        Subclass implementation:
        >>> class MySpec(BaseSpec):
        ...     function: Callable
        ...     vectorized: bool = False
        
        >>> class MyRegistry(BaseRegistry[MySpec]):
        ...     _registry: Dict[str, MySpec] = {}
        ...     
        ...     @classmethod
        ...     def register(cls, key: str, config: Type[BaseModel], **kwargs):
        ...         def decorator(func):
        ...             cls._registry[key] = MySpec(
        ...                 config_model=config,
        ...                 description=kwargs['description'],
        ...                 function=func,
        ...                 vectorized=kwargs.get('vectorized', False)
        ...             )
        ...             return func
        ...         return decorator
    """
    
    # Subclasses must define their own _registry class variable
    _registry: Dict[str, SpecType] = {}
    
    @classmethod
    def get(cls, key: str) -> SpecType:
        """
        Get registered component specification by key.
        
        Args:
            key: Component identifier
            
        Returns:
            Component specification with metadata
            
        Raises:
            ValueError: If key is not registered
        """
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            component_type = cls.__name__.replace('Registry', '').lower()
            raise ValueError(
                f"Unknown {component_type}: '{key}'. "
                f"Available {component_type}s: {available}"
            )
        return cls._registry[key]
    
    @classmethod
    def list_all(cls) -> Dict[str, dict]:
        """
        List all registered components with metadata.
        
        Returns:
            Dict mapping component keys to their specifications including
            JSON schemas for client form generation.
            
        Note:
            Subclasses may override this to customize the output format.
        """
        return {
            key: {
                "description": spec.description,
                "config_schema": spec.config_model.model_json_schema(),
            }
            for key, spec in cls._registry.items()
        }
    
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
    def get_defaults(cls, key: str) -> Dict[str, Any]:
        """
        Get the default values for a component's configuration parameters.
        
        Useful for pre-filling client forms with sensible defaults.
        
        Args:
            key: Component identifier
            
        Returns:
            Dict of parameter names to default values (only includes params with defaults)
        
        Examples:
            >>> defaults = MyRegistry.get_defaults("my_component")
            >>> print(defaults)
            {"threshold": 0.5, "iterations": 100}
        """
        spec = cls.get(key)
        schema = spec.config_model.model_json_schema()
        
        defaults = {}
        for param_name, param_info in schema.get("properties", {}).items():
            if "default" in param_info:
                defaults[param_name] = param_info["default"]
        
        return defaults
    
    @classmethod
    def list_keys(cls) -> list[str]:
        """
        Get list of all registered component keys.
        
        Returns:
            Sorted list of component keys
        """
        return sorted(cls._registry.keys())
    
    @classmethod
    def count(cls) -> int:
        """
        Get count of registered components.
        
        Returns:
            Number of registered components
        """
        return len(cls._registry)
    
    @classmethod
    def ensure_loaded(cls, expected_count: Optional[int] = None) -> None:
        """
        Verify that components are properly loaded.
        
        This method helps catch import-time registration issues where some
        components aren't registered because their modules weren't imported.
        
        Args:
            expected_count: Expected number of builtin components.
                          If provided and actual count differs, raises a warning.
        
        Raises:
            ImportWarning: If expected_count is provided and doesn't match actual count
        
        Examples:
            >>> # In tests or application startup:
            >>> ConstraintRegistry.ensure_loaded(expected_count=22)
            >>> # Warns if not all constraints are registered
        """
        actual_count = cls.count()
        
        if expected_count is not None and actual_count != expected_count:
            import warnings
            component_type = cls.__name__.replace('Registry', '').lower()
            warnings.warn(
                f"Expected {expected_count} {component_type}s but only {actual_count} are registered. "
                f"Some {component_type} modules may not be imported. "
                f"Make sure to import from the package level (e.g., 'from proto_language.language.constraint import ConstraintRegistry'). "
                f"Registered keys: {cls.list_keys()}",
                ImportWarning
            )
    
    @classmethod
    def _check_duplicate(cls, key: str, attempted_component_name: str = None) -> None:
        """
        Check if a key is already registered and raise error if so.
        
        This is a protected helper method that subclasses should call in their
        register() implementation to prevent duplicate registration.
        
        Args:
            key: Component identifier to check
            attempted_component_name: Name of component attempting to register (for error message)
            
        Raises:
            ValueError: If key is already registered
        """
        if key in cls._registry:
            component_type = cls.__name__.replace('Registry', '').lower()
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
    @abstractmethod
    def register(cls, key: str, **kwargs):
        """
        Decorator to register a component.
        
        Subclasses must implement this with domain-specific parameters.
        Subclasses should call cls._check_duplicate(key) to prevent duplicate registration.
        
        Args:
            key: Unique identifier for the component
            **kwargs: Domain-specific registration parameters
            
        Returns:
            Decorator function that registers and returns the component
        """
        pass
