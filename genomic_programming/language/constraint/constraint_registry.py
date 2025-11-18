"""
Constraint registry for managing constraint functions.

Provides a decorator-based API for registering constraint functions and
a factory method for creating Constraint instances.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Literal, Optional, Type, get_type_hints, get_origin, get_args
import inspect

from pydantic import BaseModel, Field

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Constraint, Segment

from pydantic.json_schema import SkipJsonSchema

class ConstraintSpec(BaseSpec):
    """Specification for a registered constraint."""

    mode: Literal["score", "filter"] = Field(description="'score' returns float scores between 0.0 and 1.0, 'filter' returns bool accept/reject (runs first and skips subsequent scoring constraints for rejected candidates)")
    batched: bool = Field(description="True if the constraint processes an iterable of sequences rather than a single sequence")
    concatenate: bool = Field(description="Whether to concatenate segments")
    gpu_required: bool = Field(description="Whether constraint requires GPU")

    # Private field - excluded from serialization
    function: SkipJsonSchema[Callable] = Field(exclude=True)


class ConstraintRegistry(BaseRegistry[ConstraintSpec]):
    """
    Registry for constraint discovery and API/client integration.

    Inherits common registry functionality from BaseRegistry and adds
    constraint-specific features like batched/concatenate flags.

    Public Methods:
    - register(): Decorator to register constraint functions
    - list_all(): List constraints with metadata (batched, concatenate, gpu_required)
    - create(): Factory to create Constraint instances from config dicts
    - get(): Get constraint spec by key (inherited)
    - get_schema(): Get JSON schema for constraint configuration (inherited)
    - count(): Get number of registered constraints (inherited)

    Examples:
        Registration (in constraint files):
        >>> @ConstraintRegistry.register(
        ...     key="gc-content",
        ...     config=GCContentConfig,
        ...     description="Enforce GC content within range",
        ...     batched=False,
        ...     concatenate=True
        ... )
        ... def gc_content_constraint(sequence: Sequence, config: GCContentConfig) -> float:
        ...     return calculate_penalty(sequence, config)

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
        >>> # batched and concatenate are automatically read from function attributes
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
        mode: Literal["score", "filter"] = "score",
        batched: bool = False,
        concatenate: bool = True,
        gpu_required: bool = False,
    ):
        """
        Decorator to register a constraint function.

        This is the constraint-specific implementation of the abstract register()
        method from BaseRegistry. It adds batched, concatenate, and mode flags.

        Args:
            key: Unique identifier (e.g., "gc-content", "protein-length")
            label: Readable external name (e.g., "GC Content Range", "Protein Length")
            config: Pydantic model class for configuration validation
            description: Readable description
            batched: If True, function processes List[Sequence] → List[float|bool].
                       If False, function processes Sequence → float|bool.
            concatenate: If True, concatenate multiple segments before evaluation.
                        If False, pass segments as tuple (for disjoint evaluation).
            gpu_required: If True, constraint requires GPU for computation (e.g., ESMFold, Boltz).
            mode: "score" returns float penalties (0.0-1.0), "filter" returns bool accept/reject.

        Returns:
            Decorator that registers the function and returns it unchanged

        Examples:
            >>> @ConstraintRegistry.register(
            ...     key="gc-content",
            ...     label="GC Content Range",
            ...     config=GCContentConfig,
            ...     description="GC content within range",
            ...     mode="score",
            ...     batched=False,
            ...     concatenate=True,
            ...     gpu_required=False,
            ... )
            ... def gc_content_constraint(sequence: Sequence, config: GCContentConfig) -> float:
            ...     return calculate_penalty(sequence, config.min_gc, config.max_gc)
        """
        def decorator(func: Callable):
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, func.__name__)

            # Validate return type annotation matches mode
            cls._validate_return_type(func, mode, batched)

            # Store metadata as function attributes
            func._constraint_mode = mode
            func._constraint_batched = batched
            func._constraint_concatenate = concatenate
            func._constraint_gpu_required = gpu_required
            func._constraint_config_class = config
            
            cls._registry[key] = ConstraintSpec(
                key=key,
                label=label,
                config_model=config,
                description=description,
                function=func,
                batched=batched,
                concatenate=concatenate,
                gpu_required=gpu_required,
                mode=mode,
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
    ) -> Constraint:
        """
        Factory method to create Constraint instance from JSON-compatible config.
        
        This is the primary integration point with API/client layers. It:
        1. Retrieves the registered constraint specification
        2. Validates config_dict using Pydantic (catches errors early)
        3. Creates a Constraint instance with validated config
        
        Args:
            key: Registered constraint identifier (e.g., "gc-content")
            segments: List of Segment objects to evaluate
            config_dict: Configuration as plain dict (from JSON/client)
            label: Optional label for metadata tracking
            
        Returns:
            Configured Constraint instance ready to evaluate
            
        Raises:
            ValueError: If key is not registered
            pydantic.ValidationError: If config_dict has invalid values
            
        Examples:
            >>> # From API endpoint receiving JSON
            >>> constraint = ConstraintRegistry.create(
            ...     key="gc-content",
            ...     segments=[dna_segment],
            ...     config_dict={"min_gc": 40, "max_gc": 60},
            ...     label="promoter_gc"
            ... )
            >>> scores = constraint.evaluate()
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
        )

    @classmethod
    def list_all(cls) -> List[ConstraintSpec]:
        """
        List all registered constraints as Pydantic models.

        Returns list of ConstraintSpec models that FastAPI automatically serializes to JSON.
        Each spec includes key, label, description, config_model (serialized as JSON Schema),
        batched, concatenate, and gpu_required flags.

        Returns:
            List of ConstraintSpec Pydantic models

        Examples:
            >>> constraints = ConstraintRegistry.list_all()
            >>> for spec in constraints:
            ...     print(f"{spec.label} ({spec.key})")
            ...     print(f"  Processes many: {spec.batched}")
            ...     schema = spec.config_model.model_json_schema()
            ...     print(f"  Parameters: {list(schema['properties'].keys())}")
        """
        return list(cls._registry.values())

    @staticmethod
    def _validate_return_type(func: Callable, mode: str, batched: bool) -> None:
        """Validate constraint function signature matches expected types."""
        hints = get_type_hints(func)
        params = list(inspect.signature(func).parameters.keys())
        
        if len(params) != 2:
            raise TypeError(f"Function '{func.__name__}' must have exactly 2 parameters (sequences, config), found {len(params)}")
        
        # Validate input parameter if annotated
        if batched and params[0] in hints and get_origin(hints[params[0]]) not in (list, List):
            raise TypeError(f"Function '{func.__name__}' with batched=True must accept List as first parameter")
        
        # Validate return type if annotated
        if 'return' in hints:
            return_type = hints['return']
            expected = bool if mode == "filter" else float
            origin = get_origin(return_type)
            
            if batched and origin in (list, List):
                args = get_args(return_type)
                if args and args[0] != expected:
                    raise TypeError(f"Function '{func.__name__}' mode='{mode}' must return List[{'bool' if mode == 'filter' else 'float'}]")
            elif not batched and return_type != expected:
                raise TypeError(f"Function '{func.__name__}' mode='{mode}' must return {'bool' if mode == 'filter' else 'float'}")
