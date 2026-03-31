"""
proto_language/language/constraint/constraint_registry.py

Provides a decorator-based API for registering constraint functions and
a factory method for creating Constraint instances.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Constraint, Segment


class ConstraintSpec(BaseSpec):
    """Specification for a registered constraint."""

    tools_called: List[str] = Field(description="List of tool keys this constraint calls (e.g., ['esmfold-prediction', 'prodigal-prediction']). Helps agent find relevant tool documentation.")
    category: Optional[str] = Field(default=None, description="Optional category for organization (e.g., 'protein_structure', 'sequence_composition'). Not required for custom constraints.")
    supported_sequence_types: List[str] = Field(description="List of supported sequence types (e.g., ['dna', 'protein']). Must be non-empty.")
    num_input_sequences_per_tuple: Optional[int] = Field(default=None, description="Number of Sequence objects required in each tuple of input_sequences. If None, any number is allowed.")

    # Private field - excluded from serialization
    function: SkipJsonSchema[Callable] = Field(exclude=True)


class ConstraintRegistry(BaseRegistry[ConstraintSpec]):
    """
    Registry for constraint discovery and API/client integration.

    All constraint functions use a standardized signature:
        (input_sequences: List[Tuple[Sequence, ...]], config) -> List[float]

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
        ...     input_sequences: List[Tuple[Sequence, ...]],
        ...     config: GCContentConfig
        ... ) -> List[float]:
        ...     return [calculate_penalty(seq_tuple[0], config) for seq_tuple in input_sequences]

        API/Client Usage:
        >>> # List all available constraints
        >>> constraints = ConstraintRegistry.list_all()
        >>>
        >>> # Get form schema
        >>> schema = ConstraintRegistry.get_schema("gc-content")
        >>>
        >>> # Create from user input
        >>> constraint = ConstraintRegistry.create(
        ...     key="gc-content",
        ...     segments=[segment],
        ...     config_dict={"min_gc": 40, "max_gc": 60}
        ... )

        Direct Library Usage (no registry needed):
        >>> # Users can bypass registry entirely
        >>> constraint = Constraint(
        ...     inputs=[segment],
        ...     function=gc_content_constraint,
        ...     function_config=GCContentConfig(min_gc=40, max_gc=60)
        ... )
    """

    # Each registry subclass must have its own _registry dict
    _registry: Dict[str, ConstraintSpec] = {}

    @classmethod
    def register(
        cls,
        key: str,
        label: str,
        config: Type[BaseModel],
        description: str,
        uses_gpu: bool = False,
        tools_called: List[str] = [],
        category: Optional[str] = None,
        supported_sequence_types: List[str] = [],
        num_input_sequences_per_tuple: Optional[int] = None,
    ):
        """
        Decorator to register a constraint function.

        All constraint functions must use the standardized signature:
            (input_sequences: List[Tuple[Sequence, ...]], config) -> List[float]

        Args:
            key (str): Unique identifier (e.g., "gc-content", "protein-length")
            label (str): Readable external name (e.g., "GC Content Range", "Protein Length")
            config (type[BaseModel]): Pydantic model class for configuration validation
            description (str): Readable description
            uses_gpu (bool): If True, constraint requires GPU for computation (e.g., ESMFold, Boltz).
            tools_called (list[str]): List of tool keys this constraint calls (helps agent find relevant documentation).
            category (str | None): Optional category for organization (e.g., 'protein_structure', 'sequence_composition').
            supported_sequence_types (list[str]): List of supported sequence types (e.g., ["dna", "protein"]).
            num_input_sequences_per_tuple (int | None): Number of Sequence objects required in each tuple of input_sequences. If None, any number is allowed.

        Returns:
            Decorator that registers the function and returns it unchanged

        Examples:
            >>> @constraint(
            ...     key="gc-content",
            ...     label="GC Content Range",
            ...     config=GCContentConfig,
            ...     description="GC content within range",
            ...     uses_gpu=False,
            ...     supported_sequence_types=["dna", "rna"],
            ... )
            ... def gc_content_constraint(
            ...     input_sequences: List[Tuple[Sequence, ...]],
            ...     config: GCContentConfig
            ... ) -> List[float]:
            ...     return [calculate_penalty(seq_tuple[0], config) for seq_tuple in input_sequences]
        """
        def decorator(func: Callable):
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, func.__name__)

            # Validate supported_sequence_types is non-empty
            if not supported_sequence_types:
                raise ValueError(f"supported_sequence_types must be non-empty for constraint '{key}'")

            # Store metadata as function attributes for Constraint class to use
            func._constraint_config_class = config
            func._constraint_supported_sequence_types = supported_sequence_types
            func._constraint_num_input_sequences_per_tuple = num_input_sequences_per_tuple

            cls._registry[key] = ConstraintSpec(
                key=key,
                label=label,
                config_model=config,
                description=description,
                function=func,
                uses_gpu=uses_gpu,
                tools_called=tools_called,
                category=category,
                supported_sequence_types=supported_sequence_types,
                num_input_sequences_per_tuple=num_input_sequences_per_tuple,
            )
            return func
        return decorator

    @classmethod
    def create(
        cls,
        key: str,
        segments: List[Segment],
        config_dict: Dict[str, Any],
        label: Optional[str] = None,
        threshold: Optional[float] = None,
        weight: Optional[float] = None,
    ) -> Constraint:
        """
        Factory method to create Constraint instance from JSON-compatible config.

        This is the primary integration point with API/client layers. It:
        1. Retrieves the registered constraint specification
        2. Validates config_dict using Pydantic (catches errors early)
        3. Creates a Constraint instance with validated config

        Args:
            key (str): Registered constraint identifier (e.g., "gc-content")
            segments (list[Segment]): List of Segment objects to evaluate
            config_dict (dict[str, Any]): Configuration as plain dict (from JSON/client)
            label (str | None): Optional label for metadata tracking
            threshold (float | None): Optional threshold for filtering. If provided,
                constraint acts as a filter: scores <= threshold are accepted (True),
                scores > threshold are rejected (False). If None, returns raw float
                scores for optimization.
            weight (float | None): Optional weight to scale constraint scores.
                Defaults to 1.0 if not provided.

        Returns:
            Constraint: Configured Constraint instance ready to evaluate

        Raises:
            ValueError: If key is not registered
            ValidationError: If config_dict has invalid values

        Examples:
            >>> # Scoring mode (default)
            >>> constraint = ConstraintRegistry.create(
            ...     key="gc-content",
            ...     segments=[dna_segment],
            ...     config_dict={"min_gc": 40, "max_gc": 60},
            ...     label="promoter_gc"
            ... )
            >>> scores = constraint.evaluate()  # Returns List[float]
            >>>
            >>> # Filtering mode (with threshold)
            >>> filter_constraint = ConstraintRegistry.create(
            ...     key="gc-content",
            ...     segments=[dna_segment],
            ...     config_dict={"min_gc": 40, "max_gc": 60},
            ...     threshold=0.5
            ... )
            >>> passed = filter_constraint.evaluate()  # Returns List[bool]
        """
        spec = cls.get(key)

        # Validate config with Pydantic (raises ValidationError if invalid)
        validated_config = spec.config_model(**config_dict)

        # Create Constraint with validated Pydantic model
        return Constraint(
            inputs=segments,
            function=spec.function,
            function_config=validated_config,
            label=label,
            threshold=threshold,
            weight=weight,
        )

    @classmethod
    def get_key(cls, constraint: Constraint) -> str:
        """Get registry key for a constraint instance."""
        for key, spec in cls._registry.items():
            if spec.function == constraint.function:
                return key
        raise ValueError(f"Constraint '{constraint.function.__name__}' is not registered")

    @classmethod
    def list_all(cls) -> List[ConstraintSpec]:
        """List all registered constraints as Pydantic models."""
        return list(cls._registry.values())


# Alias for simpler decorator syntax: @constraint(...) instead of @constraint(...)
constraint = ConstraintRegistry.register
