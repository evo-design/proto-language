from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from .construct import Construct
from .constraint import Constraint
from .generator import Generator
from .optimizer import Optimizer


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
        ...     temperature=1.0,
        ...     temperature_min=0.001
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

        # Validate before instantiation to catch errors early
        self._validate_program()

        # Create the Optimizer
        self.optimizer = optimizer_type(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            config=optimizer_config,
            constraint_weights=constraint_weights,
        )

    @property
    def energy_scores(self) -> List[float]:
        """
        Get energy scores from the underlying optimizer.

        Returns:
            List of energy scores where lower values indicate better solutions.
        """
        return self.optimizer.energy_scores
    
    @property
    def time_step(self) -> int:
        """
        Get current time step from the underlying optimizer.

        Returns:
            Current iteration step number (latest time_step from history, or 0 if no history).
        """
        if self.optimizer.history:
            return self.optimizer.history[-1]["time_step"]
        return 0

    @property
    def history(self) -> List[Dict[str, Any]]:
        """
        Get optimization history from the underlying optimizer.

        Returns:
            List of history entries, each containing time_step, energy_scores, and constructs.
        """
        return self.optimizer.history

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

    def run(self) -> None:
        """
        Execute the sequence optimization process and stores the optimization history in self.history.

        Prints initial and final sequence states and energies for monitoring progress.
        The actual optimization is performed by the underlying Optimizer.
        """

        # Calculate initial energy scores
        self.optimizer.score_energy()

        # Print initial sequences and energies for all batch elements
        print("Initial constructs for all batch elements:")
        if self.energy_scores:
            # For BeamSearchOptimizer: one energy per construct
            # For other optimizers: one energy per batch element across all constructs
            if len(self.energy_scores) == len(self.constructs):
                for construct_idx, construct in enumerate(self.constructs):
                    print(f"  Construct {construct_idx}:")
                    energy = self.energy_scores[construct_idx]
                    for batch_idx, batch_sequence in enumerate(construct.batch_sequences):
                        sequence = batch_sequence.sequence
                        print(f"    Batch {batch_idx}: {sequence} (energy: {energy})")
            else:
                global_batch_idx = 0
                for construct_idx, construct in enumerate(self.constructs):
                    print(f"  Construct {construct_idx}:")
                    for batch_idx, batch_sequence in enumerate(construct.batch_sequences):
                        sequence = batch_sequence.sequence
                        if global_batch_idx < len(self.energy_scores):
                            energy = self.energy_scores[global_batch_idx]
                        else:
                            energy = float('inf')  # Fallback if index out of range
                        print(f"    Batch {batch_idx}: {sequence} (energy: {energy})")
                        global_batch_idx += 1
        else:
            print("  No energy scores available yet")

        # Run optimization
        self.optimizer.sample()

        # Print final sequences and energies for all batch elements
        print("Final constructs for all batch elements:")
        if self.energy_scores:
            # For BeamSearchOptimizer: one energy per construct
            # For other optimizers: one energy per batch element across all constructs
            if len(self.energy_scores) == len(self.constructs):
                for construct_idx, construct in enumerate(self.constructs):
                    print(f"  Construct {construct_idx}:")
                    energy = self.energy_scores[construct_idx]
                    for batch_idx, batch_sequence in enumerate(construct.batch_sequences):
                        sequence = batch_sequence.sequence
                        print(f"    Batch {batch_idx}: {sequence} (energy: {energy})")
            else:
                global_batch_idx = 0
                for construct_idx, construct in enumerate(self.constructs):
                    print(f"  Construct {construct_idx}:")
                    for batch_idx, batch_sequence in enumerate(construct.batch_sequences):
                        sequence = batch_sequence.sequence
                        if global_batch_idx < len(self.energy_scores):
                            energy = self.energy_scores[global_batch_idx]
                        else:
                            energy = float('inf')  # Fallback if index out of range
                        print(f"    Batch {batch_idx}: {sequence} (energy: {energy})")
                        global_batch_idx += 1
        else:
            print("  No energy scores available after sampling")