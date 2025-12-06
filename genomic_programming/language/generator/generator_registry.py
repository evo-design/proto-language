"""
Generator registry for managing generator discovery and schema generation.

Provides a decorator-based API for registering generator classes with metadata and
automatic schema generation for API/client integration.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Type

from pydantic import BaseModel, Field

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Generator


class GeneratorSpec(BaseSpec):
    """
    Specification for a registered generator.

    Extends BaseSpec with generator-specific metadata for discovery and schema generation.
    """

    category: Literal["autoregressive", "mutation"] = Field(description="Generator category: 'autoregressive' (left-to-right, e.g. Evo2) or 'mutation' (bidirectional/masked, e.g. ESM2)")
    requires_gpu: bool = Field(description="Whether generator requires GPU")
    tools_called: List[str] = Field(description="List of tool keys this generator calls (e.g., ['esm3', 'evo2']). Helps agent find relevant tool documentation.")

    # Private field - excluded from serialization
    generator_class: Type[Generator] = Field(exclude=True)


class GeneratorRegistry(BaseRegistry[GeneratorSpec]):
    """
    Registry for generator discovery and schema generation.
    
    Inherits common registry functionality from BaseRegistry and adds
    generator-specific metadata (category, requires_gpu).
    
    Public Methods:
    - register(): Decorator to register generator classes
    - list_all(): List generators with metadata and schemas
    - create(): Factory to create generator instances from config dicts
    - get(): Get generator spec by key (inherited)
    - get_schema(): Get JSON schema for generator configuration (inherited)
    - count(): Get number of registered generators (inherited)
    
    Examples:
        Registration (in generator files):
        >>> @GeneratorRegistry.register(
        ...     key="uniform-mutation",
        ...     config=UniformMutationConfig,
        ...     description="Random point mutations",
        ...     category="mutation",
        ...     requires_gpu=False,
        ... )
        ... class UniformMutationGenerator(Generator):
        ...     def __init__(self, config: UniformMutationConfig):
        ...         super().__init__(batch_size=config.batch_size)
        ...         # Implementation
        
        API/Client Usage:
        >>> # List all available generators
        >>> generators = GeneratorRegistry.list_all()
        >>> 
        >>> # Get form schema
        >>> schema = GeneratorRegistry.get_schema("uniform-mutation")
        >>> 
        >>> # Create from config dict
        >>> config_dict = {"batch_size": 5, "num_mutations": 2}
        >>> generator = GeneratorRegistry.create("uniform-mutation", config_dict)
        
        Direct Usage:
        >>> # Call generator class directly
        >>> from proto_language.language.generator import UniformMutationGenerator, UniformMutationConfig
        >>> config = UniformMutationConfig(batch_size=5, num_mutations=2)
        >>> generator = UniformMutationGenerator(config)
    """
    
    # Each registry subclass must have its own _registry dict
    _registry: Dict[str, GeneratorSpec] = {}
    
    @classmethod
    def register(
        cls,
        key: str,
        label: str,
        config: Type[BaseModel],
        description: str,
        category: Literal["autoregressive", "mutation"],
        requires_gpu: bool,
        tools_called: List[str] = [],
    ):
        """
        Decorator to register a generator class.

        This is the generator-specific implementation of the abstract register()
        method from BaseRegistry.

        Args:
            key: Unique identifier (e.g., "uniform-mutation", "evo2")
            label: Readable external name (e.g., "Uniform Mutation Generator", "EVO2 Generator")
            config: Pydantic model class for configuration validation
            description: Readable description
            category: "autoregressive" (left-to-right) or "mutation" (bidirectional/masked)
            requires_gpu: If True, generator requires GPU for computation

        Returns:
            Decorator that registers the class and returns it unchanged

        Examples:
            >>> @GeneratorRegistry.register(
            ...     key="uniform-mutation",
            ...     label="Uniform Mutation",
            ...     config=UniformMutationConfig,
            ...     description="Random point mutations",
            ...     category="mutation",
            ...     requires_gpu=False,
            ... )
            ... class UniformMutationGenerator(Generator):
            ...     def __init__(self, config: UniformMutationConfig):
            ...         # Implementation
            ...         pass
        """
        def decorator(generator_class: Type[Generator]):
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, generator_class.__name__)

            cls._registry[key] = GeneratorSpec(
                key=key,
                label=label,
                description=description,
                config_model=config,
                generator_class=generator_class,
                category=category,
                requires_gpu=requires_gpu,
                tools_called=tools_called,
            )
            return generator_class
        return decorator
    
    @classmethod
    def create(
        cls,
        key: str,
        config_dict: Dict[str, Any],
    ) -> Generator:
        """
        Factory method to create Generator instance from JSON-compatible config.
        
        This is the primary integration point with API/client layers. It:
        1. Retrieves the registered generator specification
        2. Validates config_dict using Pydantic (catches errors early)
        3. Creates a Generator instance with validated config
        
        Args:
            key: Registered generator identifier (e.g., "uniform-mutation")
            config_dict: Configuration as plain dict (from JSON/client)
            
        Returns:
            Configured Generator instance ready to use
            
        Raises:
            ValueError: If key is not registered
            pydantic.ValidationError: If config_dict has invalid values
            
        Examples:
            >>> # From API endpoint receiving JSON
            >>> generator = GeneratorRegistry.create(
            ...     key="uniform-mutation",
            ...     config_dict={"batch_size": 5, "num_mutations": 2, "sequence_length": 100}
            ... )
            >>> generator.assign(segment)
            >>> generator.sample()
        """
        spec = cls.get(key)

        # Validate config with Pydantic (raises ValidationError if invalid)
        validated_config = spec.config_model(**config_dict)

        # Create Generator with validated Pydantic model
        return spec.generator_class(validated_config)
    
    @classmethod
    def list_all(cls) -> List[GeneratorSpec]:
        """
        List all registered generators as Pydantic models.

        Returns list of GeneratorSpec models that FastAPI automatically serializes to JSON.
        Each spec includes key, label, description, config_model (serialized as JSON Schema), category, and requires_gpu.

        Returns:
            List of GeneratorSpec Pydantic models

        Examples:
            >>> generators = GeneratorRegistry.list_all()
            >>> for spec in generators:
            ...     print(f"{spec.label} ({spec.key})")
            ...     print(f"  Category: {spec.category}")
            ...     print(f"  GPU Required: {spec.requires_gpu}")
        """
        return list(cls._registry.values())
