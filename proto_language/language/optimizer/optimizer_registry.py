"""Provides a decorator-based API for registering optimizer classes with metadata and.

automatic schema generation for API/client integration.
"""

from collections.abc import Callable
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from proto_language.base_registry import BaseRegistry, BaseSpec
from proto_language.language.core import Optimizer


class OptimizerSpec(BaseSpec):
    """Specification for a registered optimizer.

    Extends BaseSpec with optimizer-specific metadata for discovery and schema generation.

    Attributes:
        key (str): Unique kebab-case registry identifier.
        label (str): Human-readable display name.
        description (str): Short description shown in the client UI.
        uses_gpu (bool): Whether this component requires GPU resources.
        config_model (type[BaseModel]): Pydantic model class for the component configuration.
        targets_single_segment (bool): Whether this optimizer operates on a single segment at a time.
        compatible_generators (list[str] | None): Generator keys this optimizer accepts. None means all unclaimed generators.
        required_constraint_mode (Literal["discrete", "gradient"] | None): If set, all constraints must support this mode.
        optimizer_class (type[Optimizer]): Optimizer subclass implementing the optimization logic.
    """

    targets_single_segment: bool = Field(
        default=False,
        description="Whether this optimizer requires a target_segment parameter",
    )
    compatible_generators: list[str] | None = Field(
        default=None,
        description="Generator keys this optimizer accepts. None means all unclaimed generators.",
    )
    required_constraint_mode: Literal["discrete", "gradient"] | None = Field(
        default=None,
        description="If set, all constraints must support this mode. "
        "'gradient' accepts mode='gradient' or 'dual'. 'discrete' accepts mode='discrete' or 'dual'.",
    )

    # Private field - excluded from serialization
    optimizer_class: type[Optimizer] = Field(exclude=True)


class OptimizerRegistry(BaseRegistry[OptimizerSpec]):
    """Registry for optimizer discovery and schema generation.

    Inherits common registry functionality from BaseRegistry and adds
    optimizer-specific metadata.

    Public Methods:
    - register(): Decorator to register optimizer classes
    - list_all(): List optimizers with metadata and schemas
    - get(): Get optimizer spec by key (inherited)
    - get_schema(): Get JSON schema for optimizer configuration (inherited)
    - count(): Get number of registered optimizers (inherited)

    Examples:
        Registration:
        >>> @optimizer(
        ...     key="mcmc",
        ...     config=MCMCOptimizerConfig,
        ...     description="Metropolis-Hastings MCMC optimization",
        ... )
        ... class MCMCOptimizer(Optimizer):
        ...     def __init__(self, constructs, generators, constraints, config: MCMCOptimizerConfig):
        ...         self.config = config
        ...         super().__init__(
        ...             constructs=constructs,
        ...             generators=generators,
        ...             constraints=constraints,
        ...             num_results=config.num_results,
        ...             seed=config.seed,
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
        >>> config = MCMCOptimizerConfig(num_results=5, num_steps=100)
        >>> optimizer = MCMCOptimizer(
        ...     constructs=constructs, generators=generators, constraints=constraints, config=config
        ... )
    """

    # Each registry subclass must have its own _registry dict
    _registry: ClassVar[dict[str, OptimizerSpec]] = {}

    @classmethod
    def register(  # type: ignore[override]
        cls,
        key: str,
        label: str,
        config: type[BaseModel],
        description: str,
        uses_gpu: bool = False,
        targets_single_segment: bool = False,
        compatible_generators: list[str] | None = None,
        required_constraint_mode: Literal["discrete", "gradient"] | None = None,
    ) -> Callable[[type[Optimizer]], type[Optimizer]]:
        """Decorator to register an optimizer class.

        This is the optimizer-specific implementation of the abstract register()
        method from BaseRegistry.

        Args:
            key (str): Unique identifier (e.g., "mcmc", "beam-search")
            label (str): Readable external name (e.g., "MCMC Optimizer", "Beam Search")
            config (type[BaseModel]): Pydantic model class for configuration validation
            description (str): Readable description
            uses_gpu (bool): If True, optimizer requires GPU for computation
            targets_single_segment (bool): If True, optimizer operates on a single target segment
            compatible_generators (list[str] | None): Generator keys this optimizer accepts.
                None means all unclaimed generators.
            required_constraint_mode (Literal["discrete", "gradient"] | None): If set, all
                constraints must support this mode. Enforced at optimizer construction.

        Returns:
            Callable[[type[Optimizer]], type[Optimizer]]: Decorator that registers the class and returns it unchanged

        Examples:
            >>> @optimizer(
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

        def decorator(optimizer_class: type[Optimizer]) -> type[Optimizer]:
            # Prevent duplicate registration using base class helper
            cls._check_duplicate(key, optimizer_class.__name__)

            cls._registry[key] = OptimizerSpec(
                key=key,
                label=label,
                description=description,
                config_model=config,
                optimizer_class=optimizer_class,
                uses_gpu=uses_gpu,
                targets_single_segment=targets_single_segment,
                compatible_generators=compatible_generators,
                required_constraint_mode=required_constraint_mode,
            )
            return optimizer_class

        return decorator

    @classmethod
    def find_key(cls, optimizer: Optimizer) -> str | None:
        """Get registry key for an optimizer instance, or ``None`` if not registered."""
        optimizer_class = type(optimizer)
        for key, spec in cls._registry.items():
            if spec.optimizer_class == optimizer_class:
                return key
        return None

    @classmethod
    def get_key(cls, optimizer: Optimizer) -> str:
        """Get registry key for an optimizer instance. Raises ``ValueError`` if not registered."""
        key = cls.find_key(optimizer)
        if key is None:
            raise ValueError(f"Optimizer '{type(optimizer).__name__}' is not registered")
        return key

    @classmethod
    def list_all(cls) -> list[OptimizerSpec]:
        """List all registered optimizers as Pydantic models."""
        return list(cls._registry.values())


# Alias for simpler decorator syntax: @optimizer(...) instead of @optimizer(...)
optimizer = OptimizerRegistry.register
