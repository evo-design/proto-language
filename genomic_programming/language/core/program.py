from __future__ import annotations
from typing import Callable, List, Optional, Type

from pydantic import BaseModel

from proto_language.utils import use_cloud_gpu

from .construct import Construct
from .constraint import Constraint
from .generator import Generator
from .optimizer import Optimizer
from proto_language.language.optimizer.cloud_optimizer_proxy import CloudOptimizerProxy


class Program:
    """
    Programs represent user-defined biological designs.

    This class is a user-friendly wrapper around optimizers like MCMC.
    It provides a simplified interface for setting up and
    running sequence optimization workflows with constructs, generators, and constraints.

    Examples:
        Basic MCMC optimization program:
        >>> from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
        >>> config = MCMCOptimizerConfig(
        ...     num_steps=100,
        ...     max_temperature=1.0,
        ...     min_temperature=0.001
        ... )
        >>> program = Program(
        ...     optimizer_type=MCMCOptimizer,
        ...     optimizer_config=config,
        ...     constructs=[construct1, construct2],
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, length_constraint],
        ...     constraint_weights=[1.0, 0.5]
        ... )
        >>> program.run()  # Execute optimization
        >>> final_sequences = program.constructs
    """

    def __init__(
        self,
        optimizer_type: Type[Optimizer],
        optimizer_config: BaseModel,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: bool | List[str] = True,
    ) -> None:
        """
        Initialize a Program with an optimizer class and its dependencies.

        Args:
            optimizer_type: The Optimizer class to use (e.g., MCMCOptimizer).
            optimizer_config: Pydantic config object for the optimizer (e.g., MCMCOptimizerConfig).
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            custom_logging: Optional custom logging function for tracking optimization progress.
            clear_tool_cache: (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If optimizer_type is not a valid Optimizer subclass.
        """
        # Store constructor arguments for validation
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights = constraint_weights
        self.custom_logging = custom_logging
        self.clear_tool_cache = clear_tool_cache

        # Validate before instantiation to catch errors early
        self._validate_program()

        # Lazy initialization - optimizer will be created on first run()
        self.optimizer: Optional[Optimizer] = None

    @property
    def energy_scores(self) -> List[float]:
        """
        Get energy scores from the underlying optimizer.

        Returns:
            List of energy scores where lower values indicate better solutions.
        
        Raises:
            RuntimeError: If optimizer hasn't been initialized yet (run() not called).
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call run() first.")
        return self.optimizer.energy_scores

    def _validate_program(self) -> None:
        """
        Validate that the inputs and configuration are properly set up.

        Raises:
            ValueError: If optimizer_type is not a class or not a subclass of Optimizer.
        """
        # Ensure optimizer_type is a class
        if not isinstance(self.optimizer_type, type):
            raise ValueError(
                f"optimizer_type must be a class, got {type(self.optimizer_type)}. "
            )

        # Ensure optimizer_type is a subclass of Optimizer
        if not issubclass(self.optimizer_type, Optimizer):
            raise ValueError(
                f"optimizer_type must be a subclass of Optimizer, "
                f"got {self.optimizer_type}. "
                f"Available options include MCMCOptimizer and BeamSearchOptimizer."
            )

    def _initialize_optimizer(self) -> None:
        """Initialize the optimizer (lazy initialization)."""
        # Create the Optimizer with optional custom_logging
        optimizer_kwargs = {
            "constructs": self.constructs,
            "generators": self.generators,
            "constraints": self.constraints,
            "config": self.optimizer_config,
            "constraint_weights": self.constraint_weights,
            "clear_tool_cache": self.clear_tool_cache,
        }
        
        if self.custom_logging is not None:
            optimizer_kwargs["custom_logging"] = self.custom_logging

        # Create the optimizer instance
        actual_optimizer = self.optimizer_type(**optimizer_kwargs)

        # Wrap in cloud proxy if GPU execution should use cloud
        if use_cloud_gpu() and self.optimizer_type.__name__ in CloudOptimizerProxy._SERVICE_REGISTRY:
            self.optimizer = CloudOptimizerProxy(
                wrapped_optimizer=actual_optimizer,
                optimizer_type=self.optimizer_type,
            )
        else:
            self.optimizer = actual_optimizer

    def run(self) -> None:
        """
        Execute the sequence optimization process.

        Prints initial and final sequence states and energies for monitoring progress.
        The actual optimization is performed by the underlying Optimizer.
        """
        # Initialize optimizer on first run (lazy initialization)
        if self.optimizer is None:
            self._initialize_optimizer()

        # Calculate initial energy scores
        self.optimizer.score_energy()

        # Print initial sequences and energies for all batch elements
        print("Optimization started. Initial constructs for all batch elements:")
        num_seqs = len(self.constructs[0].joined_sequences)
        for seq_idx in range(num_seqs):
            energy = self.energy_scores[seq_idx]
            print(f"  [{seq_idx}] Energy: {energy:.4f}")
            for construct_idx, construct in enumerate(self.constructs):
                seq = construct.joined_sequences[seq_idx]
                seq_preview = seq[:80] + ('...' if len(seq) > 80 else '')
                print(f"    Construct {construct_idx}: {seq_preview}")

        # Run optimization
        self.optimizer.run()

        # Print final sequences and energies for all batch elements
        print("Optimization complete. Final constructs for all batch elements:")
        num_seqs = len(self.constructs[0].joined_sequences)
        for seq_idx in range(num_seqs):
            energy = self.energy_scores[seq_idx]
            print(f"  [{seq_idx}] Energy: {energy:.4f}")
            for construct_idx, construct in enumerate(self.constructs):
                seq = construct.joined_sequences[seq_idx]
                seq_preview = seq[:80] + ('...' if len(seq) > 80 else '')
                print(f"    Construct {construct_idx}: {seq_preview}")

        # Clean up model caches
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up cached models to free GPU memory."""
        from proto_language.tools.language_models.evo2.evo2 import clear_evo2_cache
        from proto_language.tools.language_models.esm3.esm3 import clear_esm3_cache
        from proto_language.tools.language_models.esm2.esm2 import clear_esm2_cache

        clear_evo2_cache()
        clear_esm3_cache()
        clear_esm2_cache()
