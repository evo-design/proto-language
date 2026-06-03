"""Decorator-based registration for generator classes with metadata and JSON schema export."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pydantic
from pydantic import BaseModel, Field

from proto_language.core import Generator, GeneratorInputType
from proto_language.utils.base import BaseRegistry, BaseSpec
from proto_language.utils.serialization import format_pydantic_error

if TYPE_CHECKING:
    from proto_language.utils.docs_api import ComponentDoc, ConfigModelDoc

GeneratorCategory = Literal["autoregressive", "mutation", "inverse_folding", "gradient"]

INPUT_TYPE_TO_CATEGORY: dict[GeneratorInputType, GeneratorCategory] = {
    GeneratorInputType.STARTING_SEQUENCE: "mutation",
    GeneratorInputType.PROMPT: "autoregressive",
    GeneratorInputType.STRUCTURE: "inverse_folding",
    GeneratorInputType.LOGITS: "gradient",
}


class GeneratorSpec(BaseSpec):
    """Specification for a registered generator.

    Extends BaseSpec with generator-specific metadata for discovery and schema generation.

    Attributes:
        key (str): Unique kebab-case registry identifier.
        label (str): Human-readable display name.
        description (str): Short description of the generator.
        uses_gpu (bool): Whether this component requires GPU resources.
        config_model (type[BaseModel]): Pydantic model class for the component configuration.
        category (GeneratorCategory): Generator category bucket — one of
            ``"autoregressive"``, ``"mutation"``, ``"inverse_folding"``, ``"gradient"``.
            Derived from the generator subclass's ``input_type`` classvar (see
            ``INPUT_TYPE_TO_CATEGORY``).
        input_type (GeneratorInputType): Typed declaration of what kind of starting input
            this generator consumes (``prompt`` / ``starting_sequence`` / ``structure`` /
            ``logits``).
        allows_empty_starting_sequence (bool): Whether a starting-sequence generator
            can initialize a length-only target segment itself.
        tools_called (list[str]): List of external tool keys this generator invokes.
        supported_sequence_types (list[str]): Sequence types this generator can produce (e.g. 'protein', 'dna').
        generator_class (type[Generator]): Generator subclass implementing the generation logic.
    """

    category: GeneratorCategory = Field(
        title="Generator Category",
        description="How the generator produces sequences: autoregressive, mutation, inverse_folding, or gradient.",
    )
    input_type: GeneratorInputType = Field(
        title="Input Type",
        description="Kind of starting input the generator consumes: prompt, starting_sequence, structure, or logits.",
    )
    allows_empty_starting_sequence: bool = Field(
        default=False,
        title="Allows Empty Start",
        description="Whether the generator can initialize a length-only segment without a starting sequence",
    )
    tools_called: list[str] = Field(
        title="Tools Called",
        description="Tool keys this generator calls (helps locate relevant tool documentation)",
    )
    supported_sequence_types: list[str] = Field(
        title="Supported Sequence Types",
        description="Sequence types this generator can produce (e.g. ['dna', 'protein']); empty list means all types.",
    )

    # Private field - excluded from serialization
    generator_class: type[Generator] = Field(
        exclude=True,
        title="Generator Class",
        description="Generator subclass implementing the generation logic (excluded from serialization)",
    )


class GeneratorRegistry(BaseRegistry[GeneratorSpec]):
    """Registry for generator discovery and schema generation.

    Inherits common registry functionality from BaseRegistry and adds
    generator-specific metadata (input_type, category, uses_gpu).

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
        ...     label="Random Nucleotide",
        ...     config=RandomNucleotideGeneratorConfig,
        ...     description="Random nucleotide mutations",
        ...     uses_gpu=False,
        ... )
        ... class RandomNucleotideGenerator(Generator):
        ...     input_type = GeneratorInputType.STARTING_SEQUENCE
        ...
        ...     def __init__(self, config: RandomNucleotideGeneratorConfig):
        ...         super().__init__()
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
        >>> from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
        >>> config = RandomNucleotideGeneratorConfig()
        >>> generator = RandomNucleotideGenerator(config)
    """

    # Each registry subclass must have its own _registry dict
    _registry: ClassVar[dict[str, GeneratorSpec]] = {}

    @classmethod
    def register(  # type: ignore[override]
        cls,
        key: str,
        label: str,
        config: type[BaseModel],
        description: str,
        uses_gpu: bool = False,
        tools_called: list[str] | None = None,
        supported_sequence_types: list[str] | None = None,
    ) -> Callable[[type[Generator]], type[Generator]]:
        """Decorator to register a generator class.

        The generator's ``category`` is derived from its ``input_type`` classvar via :data:`INPUT_TYPE_TO_CATEGORY`; the class must declare an ``input_type`` classvar.

        Args:
            key (str): Unique identifier (e.g., "random-nucleotide", "evo2")
            label (str): Readable external name (e.g., "Random Nucleotide Generator", "EVO2 Generator")
            config (type[BaseModel]): Pydantic model class for configuration validation
            description (str): Readable description
            uses_gpu (bool): If True, generator requires GPU for computation
            tools_called (list[str] | None): List of tool keys this generator calls
            supported_sequence_types (list[str] | None): List of supported sequence types (e.g., ["dna", "protein"]).
                Empty list means supports all types.

        Returns:
            Callable[[type[Generator]], type[Generator]]: Decorator that registers the class and returns it unchanged

        Raises:
            TypeError: If the decorated class does not declare a concrete ``input_type`` classvar.

        Examples:
            >>> @generator(
            ...     key="random-nucleotide",
            ...     label="Random Nucleotide",
            ...     config=RandomNucleotideGeneratorConfig,
            ...     description="Random nucleotide mutations",
            ...     uses_gpu=False,
            ...     supported_sequence_types=["dna", "rna"],
            ... )
            ... class RandomNucleotideGenerator(Generator):
            ...     input_type = GeneratorInputType.STARTING_SEQUENCE
            ...
            ...     def __init__(self, config: RandomNucleotideGeneratorConfig):
            ...         super().__init__()
            ...         # Implementation
            ...         pass
        """
        if supported_sequence_types is None:
            supported_sequence_types = []
        if tools_called is None:
            tools_called = []

        def decorator(generator_class: type[Generator]) -> type[Generator]:
            cls._check_duplicate(key, generator_class.__name__)

            input_type = getattr(generator_class, "input_type", None)
            if not isinstance(input_type, GeneratorInputType):
                raise TypeError(
                    f"Generator class {generator_class.__name__!r} must declare an ``input_type`` "
                    f"classvar set to a GeneratorInputType member (e.g. "
                    f"``input_type = GeneratorInputType.STARTING_SEQUENCE``)."
                )

            cls._registry[key] = GeneratorSpec(
                key=key,
                label=label,
                description=description,
                config_model=config,
                generator_class=generator_class,
                category=INPUT_TYPE_TO_CATEGORY[input_type],
                input_type=input_type,
                allows_empty_starting_sequence=generator_class.allows_empty_starting_sequence,
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
        config_dict: dict[str, Any],
    ) -> Generator:
        """Factory method to create Generator instance from JSON-compatible config.

        It:
        1. Retrieves the registered generator specification
        2. Validates config_dict using Pydantic (catches errors early)
        3. Creates a Generator instance with validated config

        Args:
            key (str): Registered generator identifier (e.g., "random-nucleotide")
            config_dict (dict[str, Any]): Configuration as a plain dict.

        Returns:
            Generator: Configured Generator instance ready to use

        Raises:
            ValueError: If key is not registered or if ``config_dict`` fails
                Pydantic validation. ValidationError is reformatted as
                ``generator '<key>' config invalid — <field>: <msg>; ...``.

        Examples:
            >>> # From API endpoint receiving JSON
            >>> generator = GeneratorRegistry.create(
            ...     key="random-nucleotide", config_dict={"masking_strategy": {"num_mutations": 2}}
            ... )
            >>> generator.assign(segment)
            >>> generator.sample()
        """
        spec = cls.get(key)

        try:
            validated_config = spec.config_model(**config_dict)
        except pydantic.ValidationError as e:
            raise ValueError(format_pydantic_error(e, f"generator {key!r} config invalid")) from e

        return spec.generator_class(validated_config)  # type: ignore[call-arg]

    @classmethod
    def find_key(cls, generator: Generator) -> str | None:
        """Get registry key for a generator instance, or ``None`` if not registered."""
        generator_class = type(generator)
        for key, spec in cls._registry.items():
            if spec.generator_class == generator_class:
                return key
        return None

    @classmethod
    def get_key(cls, generator: Generator) -> str:
        """Get registry key for a generator instance. Raises ``ValueError`` if not registered."""
        key = cls.find_key(generator)
        if key is None:
            raise ValueError(f"Generator '{generator.__class__.__name__}' is not registered")
        return key

    @classmethod
    def list_all(cls) -> list[GeneratorSpec]:
        """List all registered generators as Pydantic models."""
        return list(cls._registry.values())

    @classmethod
    def get_docs(cls, identifier: str) -> "ComponentDoc":
        """Return a ``ComponentDoc`` for the generator resolved from ``identifier``."""
        from proto_language.utils.docs_api import ComponentDoc, get_generator_doc

        doc: ComponentDoc = get_generator_doc(identifier)
        return doc

    @classmethod
    def get_config_doc(cls, identifier: str) -> "ConfigModelDoc":
        """Return a ``ConfigModelDoc`` for the generator's config model."""
        from proto_language.utils.docs_api import ConfigModelDoc, get_config_doc, resolve_key

        spec = cls.get(resolve_key("generator", identifier))
        doc: ConfigModelDoc = get_config_doc(spec.config_model)
        return doc


# Alias for simpler decorator syntax: @generator(...) instead of @generator(...)
generator = GeneratorRegistry.register
