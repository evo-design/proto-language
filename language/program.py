from typing import Any, Dict, List, Optional, Tuple, Type

from .base import ProgramIterativeGenerator, ProgramSequence, ProgramConstraint, ProgramGenerator


class Program:
    """
    Small wrapper class that samples from an EBM-typed generator. Can keep track of state across
    multiple calls to `sample()`.

    TODO(@brianhie): Decide if this class is even needed.
    """
    def __init__(
        self,
        ebm_class: Type["ProgramIterativeGenerator"],
        constraints: List[ProgramConstraint],
        generators: List[ProgramGenerator],
        **kwargs: Any,
    ) -> None:
        # Initialize using the class
        self.ebm = ebm_class(generators=generators, constraints=constraints, **kwargs)
    
    def register(self) -> None:
        """
        Register the generators and constraints with the EBM.
        """
        self.ebm.register()

    def _validate_ebm(self) -> None:
        """
        Validates that the EBM is properly configured with initialized generators
        and constraints that are tied to generator outputs.
        """
        if not isinstance(self.ebm, ProgramIterativeGenerator):
            raise ValueError("ebm must be a ProgramIterativeGenerator")
        if not self.ebm.generators:
            raise ValueError("ebm must have generators")
        if not self.ebm.constraints:
            raise ValueError("ebm must have constraints")
        
        # Collect all generator output variable IDs
        variable_ids = set()
        for generator in self.ebm.generators:
            if not generator._is_initialized:
                raise ValueError("Not all generators have been registered.")
            outputs = generator.get_outputs()
            for output in outputs:
                variable_ids.add(id(output))
        
        # Verify all constraint inputs are tied to generator outputs
        for constraint in self.ebm.constraints:
            for input_ in constraint.inputs:
                if id(input_) not in variable_ids:
                    raise ValueError("Found a constraint not tied to a given generator.")

    def run(self) -> List[Tuple[ProgramSequence]]:
        """
        Run MCMC on an EBM generator while keeping track of state.

        Returns:
            List[Tuple[ProgramSequence]]: A list of ProgramSequence tuple outputs at tracked steps with metadata stored in the ProgramSequence objects.
        """
        self._validate_ebm()

        # Get initial state for printing
        initial_sequence = tuple(output.sequence for output in self.ebm.get_outputs())
        initial_energy = self.ebm.score_energy()

        print(f"Initial sequence: {initial_sequence}")
        print(f"Initial energy: {initial_energy:.4f}")

        # Run MCMC
        sequence_history = self.ebm.sample()
        # Get the final sequence.
        final_sequences = tuple(output.sequence for output in self.ebm.get_outputs())

        print(f"Final sequence: {final_sequences}")
        print(f"Final energy: {self.ebm.score_energy():.4f}")

        return sequence_history

