"""
proto_language/language/generator/generator_registry.py

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

    Attributes:
        key (str): Unique kebab-case registry identifier.
        label (str): Human-readable display name.
        description (str): Short description shown in the client UI.
        uses_gpu (bool): Whether this component requires GPU resources.
        config_model (type[BaseModel]): Pydantic model class for the component configuration.
        category (Literal['autoregressive', 'mutation', 'inverse_folding']): Generator category grouping (e.g. 'language_model', 'mutation').
        tools_called (list[str]): List of external tool keys this generator invokes.
        supported_sequence_types (list[str]): Sequence types this generator can produce (e.g. 'protein', 'dna').
        generator_class (type[Generator]): Generator subclass implementing the generation logic.
    """

    category: Literal["autoregressive", "mutation", "inverse_folding"] = Field(description="Generator category: 'autoregressive' (left-to-right, e.g. Evo2), 'mutation' (bidirectional/masked, e.g. ESM2), or 'inverse_folding' (structure-conditioned, e.g. ProteinMPNN)")
    tools_called: List[str] = Field(description="List of tool keys this generator calls (e.g., ['esm3-sample', 'evo2-sample']). Helps agent find relevant tool documentation.")
    supported_sequence_types: List[str] = Field(description="List of supported sequence types (e.g., ['dna', 'protein']). Empty list means supports all types.")

    # Private field - excluded from serialization
    generator_class: Type[Generator] = Field(exclude=True)


class GeneratorRegistry(BaseRegistry[GeneratorSpec]):
    """
    Registry for generator discovery and schema generation.

    Inherits common registry functionality from BaseRegistry and adds
    generator-specific metadata (category, uses_gpu).

    Public Methods:
    - register(): Decorator to register generator classes
    - list_all(): List generators with metadata and schemas
    - create(): Factory to create generator instances from config dicts
    - get(): Get generator spec by key (inherited)
    - get_schema(): Get JSON schema for generator configuration (inherited)
    - count(): Get number of registered generators (inherited)

    Examples:
        Registration:
        >>> @generator(
        ...     key="random-nucleotide",
        ...     config=RandomNucleotideGeneratorConfig,
        ...     description="Random nucleotide mutations",
        ...     category="mutation",
        ...     uses_gpu=False,
        ... )
        ... class RandomNucleotideGenerator(Generator):
        ...     def __init__(self, config: RandomNucleotideGeneratorConfig):
        ...         super().__init__(batch_size=config.batch_size)
        ...         # Implementation

        API/Client Usage:
        >>> # List all available generators
        >>> generators = GeneratorRegistry.list_all()
        >>>
        >>> # Get form schema
        >>> schema = GeneratorRegistry.get_schema("random-nucleotide")
        >>>
        >>> # Create from config dict
        >>> config_dict = {"masking_strategy": {"num_mutations": 2}}
        >>> generator = GeneratorRegistry.create("random-nucleotide", config_dict)

        Direct Usage:
        >>> # Call generator class directly
        >>> from proto_language.language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
        >>> config = RandomNucleotideGeneratorConfig()
        >>> generator = RandomNucleotideGenerator(config)
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
        category: Literal["autoregressive", "mutation", "inverse_folding"],
        uses_gpu: bool = False,
        tools_called: List[str] = [],
        supported_sequence_types: List[str] = [],
    ):
        """
        Decorator to register a generator class.

        This is the generator-specific implementation of the abstract register()
        method from BaseRegistry.

        Args:
            key (str): Unique identifier (e.g., "random-nucleotide", "evo2")
            label (str): Readable external name (e.g., "Random Nucleotide Generator", "EVO2 Generator")
            config (type[BaseModel]): Pydantic model class for configuration validation
            description (str): Readable description
            category (Literal['autoregressive', 'mutation', 'inverse_folding']): "autoregressive" (left-to-right), "mutation" (bidirectional/masked),
                or "inverse_folding" (structure-conditioned)
            uses_gpu (bool): If True, generator requires GPU for computation
            tools_called (list[str]): List of tool keys this generator calls
            supported_sequence_types (list[str]): List of supported sequence types (e.g., ["dna", "protein"]).
                Empty list means supports all types.

        Returns:
            Decorator that registers the class and returns it unchanged

        Examples:
            >>> @generator(
            ...     key="random-nucleotide",
            ...     label="Random Nucleotide",
            ...     config=RandomNucleotideGeneratorConfig,
            ...     description="Random nucleotide mutations",
            ...     category="mutation",
            ...     uses_gpu=False,
            ...     supported_sequence_types=["dna", "rna"],
            ... )
            ... class RandomNucleotideGenerator(Generator):
            ...     def __init__(self, config: RandomNucleotideGeneratorConfig):
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
                uses_gpu=uses_gpu,
                tools_called=tools_called,
                supported_sequence_types=supported_sequence_types,
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
            key (str): Registered generator identifier (e.g., "random-nucleotide")
            config_dict (dict[str, Any]): Configuration as plain dict (from JSON/client)

        Returns:
            Generator: Configured Generator instance ready to use

        Raises:
            ValueError: If key is not registered
            ValidationError: If config_dict has invalid values

        Examples:
            >>> # From API endpoint receiving JSON
            >>> generator = GeneratorRegistry.create(
            ...     key="random-nucleotide",
            ...     config_dict={"masking_strategy": {"num_mutations": 2}}
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
    def get_key(cls, generator: Generator) -> str:
        """Get registry key for a generator instance."""
        for key, spec in cls._registry.items():
            if isinstance(generator, spec.generator_class):
                return key
        raise ValueError(f"Generator '{generator.__class__.__name__}' is not registered")

    @classmethod
    def list_all(cls) -> List[GeneratorSpec]:
        """List all registered generators as Pydantic models."""
        return list(cls._registry.values())


# Alias for simpler decorator syntax: @generator(...) instead of @generator(...)
generator = GeneratorRegistry.register
