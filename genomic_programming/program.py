from typing import Any, Dict, List, Optional, Tuple, Type

from .base import Construct, Constraint, Generator, IterativeGenerator


class Program:
    """
    Programs represent user-defined biological designs.

    This class is a user-friendly wrapper around iterative generators like MCMC and
    Sequential generators. It provides a simplified interface for setting up and
    running sequence optimization workflows with constructs, generators, and constraints.

    Examples:
        Basic MCMC optimization program:
        >>> from proto_language.generator import MCMCGenerator
        >>> program = Program(
        ...     iterative_generator_type=MCMCGenerator,
        ...     constructs=[construct1, construct2],
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, length_constraint],
        ...     constraint_weights=[1.0, 0.5],
        ...     num_steps=100,
        ...     temperature=1.0,
        ...     temperature_min=0.001
        ... )
        >>> program.run()  # Execute optimization
        >>> final_sequences = program.constructs
    """

    def __init__(
        self,
        iterative_generator_type: Type[IterativeGenerator],
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Program with an iterative generator class and its dependencies.

        Args:
            iterative_generator_type: The IterativeGenerator class to use (e.g., MCMCGenerator, SequentialGenerator).
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            **kwargs: Additional keyword arguments passed to the IterativeGenerator.

        Raises:
            ValueError: If iterative_generator_type is not a valid IterativeGenerator subclass.
        """
        # Store constructor arguments for validation
        self.iterative_generator_type = iterative_generator_type
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights = constraint_weights
        self.kwargs = kwargs
        
        # Validate before instantiation to catch errors early
        self._validate_program()

        # Create the IterativeGenerator
        # All generators take constructs (plural)
        self.ebm = iterative_generator_type(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
            **kwargs,
        )

    @property
    def energy_scores(self) -> List[float]:
        """Get energy scores from the underlying generator."""
        return self.ebm.energy_scores
    
    @property  
    def time_step(self) -> int:
        """Get current time step from the underlying generator."""
        return self.ebm.current_step

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get optimization history from the underlying generator."""
        return self.ebm.history

    def _validate_program(self) -> None:
        """
        Validate that the inputs and configuration are properly set up.

        Raises:
            ValueError: If iterative_generator_type is not a class or not a subclass
                of IterativeGenerator.
        """
        # Ensure iterative_generator_type is a class
        if not isinstance(self.iterative_generator_type, type):
            raise ValueError(
                f"iterative_generator_type must be a class, got {type(self.iterative_generator_type)}. "
            )

        # Ensure iterative_generator_type is a subclass of IterativeGenerator
        if not issubclass(self.iterative_generator_type, IterativeGenerator):
            raise ValueError(
                f"iterative_generator_type must be a subclass of IterativeGenerator, "
                f"got {self.iterative_generator_type}. "
                f"Available options include MCMCGenerator, SequentialGenerator, BeamSearchGenerator."
            )

    def run(self) -> None:
        """
        Execute the sequence optimization process and stores the optimization history in self.history.

        Prints initial and final sequence states and energies for monitoring progress.
        The actual optimization is performed by the underlying IterativeGenerator.
        """

        # Calculate initial energy scores
        self.ebm.score_energy()

        # Print initial sequences and energies for all batch elements
        print("Initial constructs for all batch elements:")
        if self.energy_scores:
            # For BeamSearchGenerator: one energy per construct
            # For other generators: one energy per batch element across all constructs
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

        # Run iterative generation
        self.ebm.sample()

        # Print final sequences and energies for all batch elements
        print("Final constructs for all batch elements:")
        if self.energy_scores:
            # For BeamSearchGenerator: one energy per construct
            # For other generators: one energy per batch element across all constructs
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