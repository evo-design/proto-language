"""
Constraint registry for managing constraint functions.

Provides a decorator-based API for registering constraint functions and
a factory method for creating Constraint instances.
"""

from typing import Dict, Type, Callable, List, Optional, Any
from dataclasses import dataclass

from pydantic import BaseModel

from ..base import Constraint, Segment, BaseRegistry, BaseSpec


@dataclass
class ConstraintSpec(BaseSpec):
    """Specification for a registered constraint."""
    function: Callable
    vectorized: bool = False
    concatenate: bool = True
    gpu_required: bool = False


class ConstraintRegistry(BaseRegistry[ConstraintSpec]):
    """
    Registry for constraint discovery and API/client integration.
    
    Inherits common registry functionality from BaseRegistry and adds
    constraint-specific features like vectorized/concatenate flags.
    
    Key Methods:
    - register(): Decorator to register constraint functions
    - create(): Factory to create Constraint instances from config dicts
    - list_all(): List constraints with schemas (includes vectorized/concatenate)
    - get_schema(): Get JSON schema for a constraint (inherited)
    - get_defaults(): Get default config values (inherited)
    - ensure_loaded(): Verify all constraints loaded (inherited)
    
    Examples:
        Registration (in constraint files):
        >>> @ConstraintRegistry.register(
        ...     key="gc-content",
        ...     config=GCContentConfig,
        ...     description="Enforce GC content within range",
        ...     vectorized=False,
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
        >>> constraint = Constraint(
        ...     inputs=[segment],
        ...     scoring_function=gc_content_constraint,
        ...     scoring_function_config=GCContentConfig(min_gc=40, max_gc=60)
        ... )
    """
    
    # Each registry subclass must have its own _registry dict
    _registry: Dict[str, ConstraintSpec] = {}
    
    @classmethod
    def register(
        cls,
        key: str,
        config: Type[BaseModel],
        description: str,
        vectorized: bool = False,
        concatenate: bool = True,
        gpu_required: bool = False,
    ):
        """
        Decorator to register a constraint function.
        
        This is the constraint-specific implementation of the abstract register()
        method from BaseRegistry. It adds vectorized and concatenate flags.
        
        Args:
            key: Unique identifier (e.g., "gc-content", "protein-length")
            config: Pydantic model class for configuration validation
            description: Human-readable description for UI display
            vectorized: If True, function processes List[Sequence] → List[float].
                       If False, function processes Sequence → float.
            concatenate: If True, concatenate multiple segments before evaluation.
                        If False, pass segments as tuple (for disjoint evaluation).
            gpu_required: If True, constraint requires GPU for computation (e.g., ESMFold, Boltz).
        
        Returns:
            Decorator that registers the function and returns it unchanged
        
        Examples:
            >>> @ConstraintRegistry.register(
            ...     key="gc-content",
            ...     config=GCContentConfig,
            ...     description="GC content within range",
            ...     vectorized=False
            ... )
            ... def gc_content_constraint(sequence: Sequence, config: GCContentConfig) -> float:
            ...     return calculate_penalty(sequence, config.min_gc, config.max_gc)
        """
        def decorator(func: Callable):
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, func.__name__)
            
            cls._registry[key] = ConstraintSpec(
                function=func,
                config_model=config,
                description=description,
                vectorized=vectorized,
                concatenate=concatenate,
                gpu_required=gpu_required,
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
            scoring_function=spec.function,
            scoring_function_config=validated_config,
            vectorized=spec.vectorized,
            concatenate=spec.concatenate,
            label=label,
        )
    
    @classmethod
    def list_all(cls) -> Dict[str, dict]:
        """
        List all registered constraints with metadata and schemas.
        
        Overrides BaseRegistry.list_all() to include constraint-specific fields
        (vectorized and concatenate flags).
        
        Returns:
            Dict mapping constraint keys to specifications:
            {
                "constraint-key": {
                    "description": "Human-readable description",
                    "vectorized": bool,
                    "concatenate": bool,
                    "config_schema": {...}  # JSON Schema
                }
            }
        
        Examples:
            >>> constraints = ConstraintRegistry.list_all()
            >>> for key, info in constraints.items():
            ...     print(f"{key}: {info['description']}")
            ...     print(f"  Vectorized: {info['vectorized']}")
            ...     print(f"  Config params: {list(info['config_schema']['properties'].keys())}")
        """
        return {
            key: {
                "description": spec.description,
                "vectorized": spec.vectorized,
                "concatenate": spec.concatenate,
                "gpu_required": spec.gpu_required,
                "config_schema": spec.config_model.model_json_schema(),
            }
            for key, spec in cls._registry.items()
        }

# Convenience alias for cleaner decorator syntax
constraint = ConstraintRegistry.register

