"""
Generator registry for managing generator discovery and schema generation.

Provides a decorator-based API for registering generator classes with metadata and
automatic schema generation for API/client integration.
"""

from typing import Dict, Type, Any
from dataclasses import dataclass

from pydantic import BaseModel

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Generator


@dataclass
class GeneratorSpec(BaseSpec):
    """
    Specification for a registered generator.
    
    Extends BaseSpec with generator-specific metadata for discovery and schema generation.
    """
    generator_class: Type[Generator]  # The generator class
    category: str  # Generator category (e.g., "mutation", "language_model", "optimization")
    requires_gpu: bool = False  # Whether generator requires GPU
    supports_batch: bool = True  # Whether generator supports batch processing


class GeneratorRegistry(BaseRegistry[GeneratorSpec]):
    """
    Registry for generator discovery and schema generation.
    
    Inherits common registry functionality from BaseRegistry and adds
    generator-specific metadata (category, requires_gpu, supports_batch).
    
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
        ...     supports_batch=True
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
        config: Type[BaseModel],
        description: str,
        category: str,
        requires_gpu: bool = False,
        supports_batch: bool = True,
    ):
        """
        Decorator to register a generator class.
        
        This is the generator-specific implementation of the abstract register()
        method from BaseRegistry.
        
        Args:
            key: Unique identifier (e.g., "uniform-mutation", "evo2")
            config: Pydantic model class for configuration validation
            description: Human-readable description for UI display
            category: Generator category (e.g., "mutation", "language_model",
                     "optimization", "pipeline")
            requires_gpu: If True, generator requires GPU for computation
            supports_batch: If True, generator supports batch processing
        
        Returns:
            Decorator that registers the class and returns it unchanged
        
        Examples:
            >>> @GeneratorRegistry.register(
            ...     key="uniform-mutation",
            ...     config=UniformMutationConfig,
            ...     description="Random point mutations",
            ...     category="mutation",
            ...     requires_gpu=False,
            ...     supports_batch=True
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
                generator_class=generator_class,
                config_model=config,
                description=description,
                category=category,
                requires_gpu=requires_gpu,
                supports_batch=supports_batch,
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
    def list_all(cls) -> Dict[str, dict]:
        """
        List all registered generators with metadata and schemas.
        
        Overrides BaseRegistry.list_all() to include generator-specific fields
        (category, requires_gpu, supports_batch).
        
        Returns:
            Dict mapping generator keys to specifications:
            {
                "generator-key": {
                    "description": "Human-readable description",
                    "category": "mutation",
                    "requires_gpu": False,
                    "supports_batch": True,
                    "config_schema": {...}  # JSON Schema
                }
            }
        
        Examples:
            >>> generators = GeneratorRegistry.list_all()
            >>> for key, info in generators.items():
            ...     print(f"{key}: {info['description']}")
            ...     print(f"  Category: {info['category']}")
            ...     print(f"  GPU Required: {info['requires_gpu']}")
        """
        return {
            key: {
                "description": spec.description,
                "category": spec.category,
                "requires_gpu": spec.requires_gpu,
                "supports_batch": spec.supports_batch,
                "config_schema": spec.config_model.model_json_schema(),
            }
            for key, spec in cls._registry.items()
        }

