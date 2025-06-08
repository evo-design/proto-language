from typing import Any, List, Optional, Tuple, Type

from .base import ProgramIterativeGenerator, ProgramSequence, ProgramConstraint, ProgramGenerator, BatchedProgramSequence


class Program:
    """
    High-level interface for constraint-driven sequence optimization.

    This class provides a user-friendly wrapper around iterative generators like MCMC and sequential samplers. 
    A program represents a user's biological design and handles the complexity of generator/constraint integration 
    while exposing simple methods for running optimization and accessing results.

    Key features:
    - Automatic validation of generator/constraint compatibility
    - Clean user-facing sequence access with rich metadata
    - Progress tracking and energy monitoring
    - Support for complex multi-part sequence designs

    The Program class bridges the gap between the low-level generator infrastructure
    and high-level user needs, making sequence optimization accessible while maintaining
    full flexibility of the underlying system.

    Examples:
        Basic sequence optimization:
        >>> from language.generator import ProgramMCMCGenerator
        >>> from language.constraint import gc_content_constraint
        >>> 
        >>> program = Program(
        ...     ebm_class=ProgramMCMCGenerator,
        ...     generators=[mutation_generator],
        ...     constraints=[gc_constraint, length_constraint],
        ...     sequence_order=((batch1,),),
        ...     num_steps=1000
        ... )
        >>> history = program.run()
        >>> final_sequences = program.user_sequences
        
        Multi-part design with complex constraints:
        >>> from language.generator import ProgramSequentialGenerator
        >>> 
        >>> program = Program(
        ...     ebm_class=ProgramSequentialGenerator,
        ...     generators=[evo2_generator, esm2_generator],
        ...     constraints=[structure_constraint, binding_constraint],
        ...     sequence_order=((batch1, batch2), (batch3,)),
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> history = program.run()
    """
    
    def __init__(
        self,
        ebm_class: Type[ProgramIterativeGenerator],
        generators: List[ProgramGenerator],
        constraints: List[ProgramConstraint],
        sequence_order: Tuple[Tuple[BatchedProgramSequence]],
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Program with generators, constraints, and optimization strategy.

        Args:
            ebm_class: Class of iterative generator to use (e.g., ProgramMCMCGenerator,
                      ProgramSequentialGenerator). Must be a subclass of ProgramIterativeGenerator.
            generators: List of sequence generators that propose changes. These must
                       already be registered with initialized sequences.
            constraints: List of constraint functions that evaluate sequence quality.
                        Their inputs must match the generator outputs.
            sequence_order: Nested tuple structure defining sequence concatenation for output.
                           Each inner tuple contains BatchedProgramSequence objects that will
                           be concatenated together to form a single output sequence.
                           Example: ((batch1, batch2), (batch3,)) produces two output sequences:
                           one from concatenating batch1+batch2, and one from batch3 alone.
            **kwargs: Additional configuration passed to the iterative generator,
                     such as num_steps, temperature, constraint_weights, etc.

        Raises:
            ValueError: If configuration is invalid, generators aren't registered,
                       or constraints don't match generator outputs.

        Note:
            The sequence_order must contain exactly the same BatchedProgramSequence
            objects as produced by the generators, ensuring proper data flow.
        """
        # Initialize using the class
        self.ebm = ebm_class(generators=generators, constraints=constraints, sequence_order=sequence_order, **kwargs)
        self.sequence_order = sequence_order
        self.generators = generators
        
        # Register the EBM (it will handle its own ordering internally)
        self.ebm.register()
        self._validate_init()
    
    @property
    def user_sequences(self) -> Tuple[ProgramSequence]:
        """
        Access the current optimized sequences with metadata and concatenation.
        
        This property provides the main interface for accessing optimization results.
        It delegates to the underlying iterative generator's user_sequences property
        to provide clean, user-friendly sequence objects with rich metadata.
        
        Each returned ProgramSequence includes:
        - Concatenated sequence string according to sequence_order
        - energy_score: Current energy from constraint evaluation
        - time_step: Current optimization step number
        - Additional metadata from constraint evaluations
        
        Returns:
            Tuple of ProgramSequence objects representing the current best sequences.
            One sequence per group defined in sequence_order, with full metadata.

        Examples:
            Accessing optimization results:
            >>> program = Program(...)
            >>> history = program.run()
            >>> final_sequences = program.user_sequences
            >>> 
            >>> # Access the best sequence
            >>> best_seq = final_sequences[0]
            >>> print(f"Best energy: {best_seq._metadata['energy_score']}")
            >>> print(f"Final sequence: {best_seq.sequence}")
            >>> print(f"Optimization steps: {best_seq._metadata['time_step']}")
        """
        return self.ebm.user_sequences
    
    def _validate_init(self) -> None:
        """
        Validate that the inputs and configuration are properly set up.
        
        This method ensures that:
        - The EBM is a proper ProgramIterativeGenerator instance
        - All generators have been registered and initialized
        - All constraints are properly connected to generator outputs
        - The sequence_order contains the same BatchedProgramSequence objects as generator outputs
        - The configuration is internally consistent
        
        Raises:
            ValueError: If any validation checks fail, with specific error messages
                       indicating the problem.
        """
        if not isinstance(self.ebm, ProgramIterativeGenerator):
            raise ValueError("ebm must be a ProgramIterativeGenerator")
        if not self.ebm.generators:
            raise ValueError("ebm must have generators")
        if not self.ebm.constraints:
            raise ValueError("ebm must have constraints")
        
        # Collect all _generator_outputs variable IDs
        variable_ids = set()
        for generator in self.ebm.generators:
            if not generator._is_initialized:
                raise ValueError("Not all generators have been registered.")
            generator_outputs = generator.get_generator_outputs()
            for sequence_batch in generator_outputs:
                variable_ids.add(id(sequence_batch))
        
        # Verify all constraint inputs are tied to generator _generator_outputs
        for constraint in self.ebm.constraints:
            for input_ in constraint.inputs:
                if id(input_) not in variable_ids:
                    raise ValueError("Found a constraint not tied to a given generator.")
        
        # Validate that all BatchedProgramSequence objects in sequence_order exist in generator outputs
        all_generator_outputs = {id(seq) for gen in self.generators for seq in gen.get_generator_outputs()}
        all_sequence_order_ids = {id(seq) for group in self.sequence_order for seq in group}
        
        if all_sequence_order_ids != all_generator_outputs:
            raise ValueError("sequence_order must contain exactly the same BatchedProgramSequence objects as generator outputs")

    def run(self) -> List[Tuple[ProgramSequence]]:
        """
        Execute the sequence optimization process and return the optimization history.

        This method runs the iterative optimization algorithm (MCMC, sequential, etc.)
        while tracking progress and providing user-friendly output. It handles all
        the complexity of the optimization loop while providing clean progress reporting.

        The method:
        1. Reports initial sequences and energies
        2. Executes the optimization algorithm
        3. Reports final results
        4. Returns the complete optimization history

        Returns:
            List of user_sequences snapshots taken at tracked intervals during optimization.
            Each element represents the state at a specific step, allowing analysis of
            convergence and optimization dynamics.

        Examples:
            Running optimization with progress tracking:
            >>> program = Program(...)
            >>> history = program.run()
            >>> print(f"Optimization took {len(history)} tracked steps")
            >>> 
            >>> # Analyze convergence
            >>> final_step = history[-1]
            >>> energies = [seq._metadata['energy_score'] for seq in final_step]
            >>> print(f"Final energy: {min(energies)}")
            >>> 
            >>> # Track energy over time
            >>> energy_history = []
            >>> for step in history:
            ...     step_energies = [seq._metadata['energy_score'] for seq in step]
            ...     energy_history.append(min(step_energies))
        """

        # Get initial state for printing
        initial_sequences = [seq.sequence for seq in self.user_sequences]
        initial_energies = [seq._metadata.get("energy_score", "N/A") for seq in self.user_sequences]

        print(f"Initial sequences: {initial_sequences}")
        print(f"Initial energies: {initial_energies}")

        # Run iterative generation
        sequence_history = self.ebm.sample()

        # Get the final sequences (user_sequences property will have updated automatically)
        final_sequences = [seq.sequence for seq in self.user_sequences]
        final_energies = [seq._metadata.get("energy_score", "N/A") for seq in self.user_sequences]

        print(f"Final sequences: {final_sequences}")
        print(f"Final energies: {final_energies}")

        return sequence_history

