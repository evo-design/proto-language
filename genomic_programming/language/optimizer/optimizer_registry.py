"""
Optimizer registry for managing optimizer discovery and schema generation.

Provides a decorator-based API for registering optimizer classes with metadata and
automatic schema generation for API/client integration.
"""

from typing import Dict, List, Type

from pydantic import BaseModel, Field

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Optimizer


class OptimizerSpec(BaseSpec):
    """
    Specification for a registered optimizer.

    Extends BaseSpec with optimizer-specific metadata for discovery and schema generation.
    """

    # Private field - excluded from serialization
    optimizer_class: Type[Optimizer] = Field(exclude=True)

class OptimizerRegistry(BaseRegistry[OptimizerSpec]):
    """
    Registry for optimizer discovery and schema generation.

    Inherits common registry functionality from BaseRegistry and adds
    optimizer-specific metadata.

    Public Methods:
    - register(): Decorator to register optimizer classes
    - list_all(): List optimizers with metadata and schemas
    - get(): Get optimizer spec by key (inherited)
    - get_schema(): Get JSON schema for optimizer configuration (inherited)
    - count(): Get number of registered optimizers (inherited)

    Examples:
        Registration (in optimizer files):
        >>> @OptimizerRegistry.register(
        ...     key="mcmc",
        ...     config=MCMCOptimizerConfig,
        ...     description="Metropolis-Hastings MCMC optimization",
        ... )
        ... class MCMCOptimizer(Optimizer):
        ...     def __init__(self, constructs, generators, constraints, config: MCMCOptimizerConfig):
        ...         super().__init__(
        ...             constructs=constructs,
        ...             generators=generators,
        ...             constraints=constraints,
        ...             batch_size=config.batch_size
        ...         )
        ...         # Implementation

        API/Client Usage:
        >>> # List all available optimizers
        >>> optimizers = OptimizerRegistry.list_all()
        >>>
        >>> # Get form schema
        >>> schema = OptimizerRegistry.get_schema("mcmc")

        Direct Usage:
        >>> # Call optimizer class directly
        >>> from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
        >>> config = MCMCOptimizerConfig(batch_size=5, num_steps=100)
        >>> optimizer = MCMCOptimizer(
        ...     constructs=constructs,
        ...     generators=generators,
        ...     constraints=constraints,
        ...     config=config
        ... )
    """

    # Each registry subclass must have its own _registry dict
    _registry: Dict[str, OptimizerSpec] = {}

    @classmethod
    def register(
        cls,
        key: str,
        label: str,
        config: Type[BaseModel],
        description: str,
    ):
        """
        Decorator to register an optimizer class.

        This is the optimizer-specific implementation of the abstract register()
        method from BaseRegistry.

        Args:
            key: Unique identifier (e.g., "mcmc", "beam-search")
            label: Readable external name (e.g., "MCMC Optimizer", "Beam Search")
            config: Pydantic model class for configuration validation
            description: Readable description

        Returns:
            Decorator that registers the class and returns it unchanged

        Examples:
            >>> @OptimizerRegistry.register(
            ...     key="mcmc",
            ...     label="MCMC Optimizer",
            ...     config=MCMCOptimizerConfig,
            ...     description="Metropolis-Hastings MCMC optimization",
            ... )
            ... class MCMCOptimizer(Optimizer):
            ...     def __init__(self, constructs, generators, constraints, config: MCMCOptimizerConfig):
            ...         # Implementation
            ...         pass
        """
        def decorator(optimizer_class: Type[Optimizer]):
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, optimizer_class.__name__)

            cls._registry[key] = OptimizerSpec(
                key=key,
                label=label,
                description=description,
                config_model=config,
                optimizer_class=optimizer_class,
            )
            return optimizer_class
        return decorator

    @classmethod
    def list_all(cls) -> List[OptimizerSpec]:
        """
        List all registered optimizers as Pydantic models.

        Returns list of OptimizerSpec models that FastAPI automatically serializes to JSON.
        Each spec includes key, label, description, and config_model (serialized as JSON Schema).

        Returns:
            List of OptimizerSpec Pydantic models

        Examples:
            >>> optimizers = OptimizerRegistry.list_all()
            >>> for spec in optimizers:
            ...     print(f"{spec.label} ({spec.key})")
            ...     schema = spec.config_model.model_json_schema()
            ...     print(f"  Parameters: {list(schema['properties'].keys())}")
        """
        return list(cls._registry.values())
