"""
Optimizer registry for managing optimizer discovery and schema generation.

Provides a decorator-based API for registering optimizer classes with metadata and
automatic schema generation for API/client integration.
"""

from typing import Dict, Type
from dataclasses import dataclass

from pydantic import BaseModel

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Optimizer


@dataclass
class OptimizerSpec(BaseSpec):
    """
    Specification for a registered optimizer.

    Extends BaseSpec with optimizer-specific metadata for discovery and schema generation.
    """
    optimizer_class: Type[Optimizer]  # The optimizer class

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
        config: Type[BaseModel],
        description: str,
    ):
        """
        Decorator to register an optimizer class.

        This is the optimizer-specific implementation of the abstract register()
        method from BaseRegistry.

        Args:
            key: Unique identifier (e.g., "mcmc", "beam-search")
            config: Pydantic model class for configuration validation
            description: Human-readable description for UI display

        Returns:
            Decorator that registers the class and returns it unchanged

        Examples:
            >>> @OptimizerRegistry.register(
            ...     key="mcmc",
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
                optimizer_class=optimizer_class,
                config_model=config,
                description=description,
            )
            return optimizer_class
        return decorator

    @classmethod
    def list_all(cls) -> Dict[str, dict]:
        """
        List all registered optimizers with metadata and schemas.

        Returns:
            Dict mapping optimizer keys to specifications:
            {
                "optimizer-key": {
                    "description": "Human-readable description",
                    "config_schema": {...}  # JSON Schema
                }
            }

        Examples:
            >>> optimizers = OptimizerRegistry.list_all()
            >>> for key, info in optimizers.items():
            ...     print(f"{key}: {info['description']}")
        """
        return {
            key: {
                "description": spec.description,
                "config_schema": spec.config_model.model_json_schema(),
            }
            for key, spec in cls._registry.items()
        }
