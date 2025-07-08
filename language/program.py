from typing import Any, List, Optional, Tuple, Type

from .base import Construct, Constraint, Generator, IterativeGenerator


class Program:
    """
    Programs represent user-defined biological designs.

    This class is a user-friendly wrapper around iterative generators like MCMC and
    Sequential generators. It provides a simplified interface for setting up and
    running sequence optimization workflows with constructs, generators, and constraints.

    Examples:
        Basic MCMC optimization program:
        >>> from language.generator import MCMCGenerator
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
        >>> history = program.run()  # Execute optimization
        >>> final_sequences = program.user_sequences
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
        self.ebm = iterative_generator_type(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
            **kwargs,
        )

    @property
    def user_sequences(self) -> Tuple[Construct, ...]:
        """
        Get all user-facing construct objects.

        Returns:
            Tuple of Construct objects representing the current optimized state.
        """
        return self.ebm.user_sequences

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
                f"Available options include MCMCGenerator, SequentialGenerator."
            )

    def run(self) -> List[Tuple[Construct, ...]]:
        """
        Execute the sequence optimization process and return the optimization history.

        Prints initial and final sequence states and energies for monitoring progress.
        The actual optimization is performed by the underlying IterativeGenerator.

        Returns:
            List of user_sequences snapshots taken at tracked intervals during optimization.
            Each element represents the state at a specific step.
        """

        # TODO: Update how we print batched sequences. Currently, we print the first sequence in the batch.
        # Get initial state for printing
        initial_sequences = [
            construct.batch_sequences[0].sequence for construct in self.user_sequences
        ]
        initial_energies = [
            construct.batch_sequences[0]._metadata["energy_score"]
            for construct in self.user_sequences
        ]

        print(f"Initial sequences: {initial_sequences}")
        print(f"Initial energies: {initial_energies}")

        # Run iterative generation
        sequence_history = self.ebm.sample()

        # Get the final sequences (user_sequences property will have updated automatically)
        final_sequences = [
            construct.batch_sequences[0].sequence for construct in self.user_sequences
        ]
        final_energies = [
            construct.batch_sequences[0]._metadata["energy_score"]
            for construct in self.user_sequences
        ]

        print(f"Final sequences: {final_sequences}")
        print(f"Final energies: {final_energies}")

        return sequence_history
