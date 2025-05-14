from abc import ABC, abstractmethod
import random
from typing import Any, List, Dict, Optional, Tuple
import copy

from .base import (
    ProgramGenerator,
    ProgramIterativeGenerator,
    ProgramSequence,
    ProgramConstraint,
)
from .sequence import ProgramDNASequence, ProgramRNASequence, ProgramProteinSequence


class UniformMutationGenerator(ProgramGenerator):
    """
    A uniform proposal over DNA, RNA, or protein sequences.

    Initializes with a random sequence and samples a point mutation on each call to `sample()`.
    """
    def __init__(
        self,
        sequence_length: int,
        sequence_type: str = 'dna',
    ) -> None:
        """
        Initializes the uniform proposal.

        Args:
            sequence_length (int): The length of the random sequence.
            sequence_type (str): The type of sequence ('dna', 'rna', or 'protein').

        Raises:
            ValueError: If the provided sequence type is not supported.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_type = sequence_type.lower()

        if self.sequence_type == 'dna':
            self.vocab = 'ACGT'
            self.sequence_class = ProgramDNASequence
        elif self.sequence_type == 'rna':
            self.vocab = 'ACGU'
            self.sequence_class = ProgramRNASequence
        elif self.sequence_type == 'protein':
            self.vocab = 'ACDEFGHIKLMNPQRSTVWY'
            self.sequence_class = ProgramProteinSequence
        else:
            raise ValueError(
                f'Sequence type must "dna", "rna", or "protein", found {self.sequence_type}'
            )

    def register(
        self,
        outputs: Optional[Tuple[ProgramSequence]] = None,
    ) -> Tuple[ProgramSequence]:
        """
        Initialize a random sequence.

        outputs (Optional[Tuple[ProgramSequence]]): Optional initialization of output
                                                   variables.
        Returns:
            Tuple[ProgramSequence]: Output sequence variables. These variables get updated
                                    in-place throughout generation.
        """
        self._is_initialized = True

        if outputs is None:
            random_sequence = ''.join(random.choices(self.vocab, k=self.sequence_length))
            self.outputs = (self.sequence_class(sequence=random_sequence),)
        else:
            if len(outputs) != 1:
                raise ValueError('Provided outputs must have one entry')
            if not isinstance(outputs[0], ProgramSequence):
                raise ValueError('Must provide a ProgramSequence')
            self.outputs = outputs
        
        return self.outputs

    def sample(self) -> None:
        """
        Introduces a mutation at a random position in the sequence.
        """
        if not self._is_initialized:
            self.register()

        mutated_index = random.randint(0, self.sequence_length - 1)
        current_sequence = self.outputs[0].sequence
        current_char = current_sequence[mutated_index]
        
        # Make sure the mutated character is different from the current one
        possible_mutations = [c for c in self.vocab if c != current_char]
        mutated_char = random.choice(possible_mutations)
        
        self.outputs[0].sequence = (
            current_sequence[:mutated_index] +
            mutated_char +
            current_sequence[mutated_index + 1:]
        )


class Evo2Generator(ProgramGenerator):
    """
    TODO(@brianhie): Implement this.
    """
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramDNASequence(self, 0), )
        return self.outputs
    
    def sample(self) -> None:
        pass
        

class SemanticMiningGenerator(ProgramGenerator):
    """
    TODO(@brianhie): Implement this.
    """
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramDNASequence(self, 0), )
        return self.outputs
    
    def sample(self) -> None:
        pass


class BindCraftGenerator(ProgramGenerator):
    """
    TODO(@brianhie): Implement this.
    """
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramProteinSequence(self, 0), )
        return self.outputs
    
    def sample(self) -> None:
        pass


class ProgramMCMCGenerator(ProgramIterativeGenerator):
    """
    Metropolis-Hastings MCMC loop using generators to implement the proposal distribution and
    constraints to define the remainder of the energy function.
    """
    def __init__(
        self,
        generators: List[ProgramGenerator],
        constraints: List[ProgramConstraint],
        **hyperparameters: Any,
    ) -> None:
        """
        Initializes the MCMC generator. Wraps a list of generators and constraints.

        Args:
            generators (List[ProgramGenerator]): A list of generators. These must be registered.
            constraints (List[ProgramConstraint]): A list of constraints. The inputs to these
                                                   constraints must be the same objects produced by
                                                   generator registration.
            constraint_weights (Optional[List[float]]): Weights to use for each constraint. Must be the
                                                        same length as the `constraints` list.
            num_steps (Optional[int]): The number of steps to run on each call of `sample()`.
            temperature (Optional[float]): The Metropolis-Hastings sampling temperature.
            track_step_size (Optional[int]): The number of steps between tracking the sequence and energy.
            **hyperparameters (Any): Keyword arguments representing the
                                     configuration and hyperparameters for the
                                     specific generator implementation.

        Raises:
            ValueError: Throws an error if there are mismatches between constraints/weights or if there
                        are problems in the configuration of generators and constraints.
        """
        super().__init__(**hyperparameters)
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights: List[float] = hyperparameters.get(
            "constraint_weights",
            [1.] * len(constraints),
        )
        self.num_steps: int = hyperparameters.get("num_steps", 1)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        self.track_step_size: int = hyperparameters.get("track_step_size", 10)

        if len(self.constraints) != len(self.constraint_weights):
            raise ValueError("Constraint weights must match number of constraints.")

        # Generators must already be registered, since their variables are hooked up to constraints.
        variable_ids = set()
        for generator in self.generators:
            if not generator._is_initialized:
                raise ValueError("Not all generators have been registered.")
            outputs = generator.get_outputs()
            for output in outputs:
                variable_ids.add(id(output))

        # All constraint inputs must be the same as generator outputs.
        for constraint in self.constraints:
            for input_ in constraint.inputs:
                if id(input_) not in variable_ids:
                    raise ValueError("Found a constraint not tied to a given generator.")

    def register(self) -> Tuple[ProgramSequence]:
        """
        Because this generator wraps a list of pre-registered sub-generators, simply
        return an in-order tuple of these generators' outputs.

        Returns:
            Tuple[ProgramSequence]: Output sequence variables. These variables get updated
                                    in-place throughout generation.
        """
        self._is_initialized = True

        self.outputs = []
        for generator in self.generators:
            self.outputs += list(generator.get_outputs())
        self.outputs = tuple(self.outputs)

        return self.outputs

    def sample(self) -> List[Tuple[ProgramSequence]]:
        """
        Runs the MCMC sampling loop to update sequences in-place.
        
        Performs Metropolis-Hastings sampling by proposing changes from a randomly 
        selected generator and accepting or rejecting based on the energy ratio.
        
        The temperature parameter controls the acceptance probability:
        - Higher temperature (>1.0): More likely to accept worse solutions (more exploration)
        - Lower temperature (<1.0): Less likely to accept worse solutions (more exploitation)
        - Temperature = 1.0: Standard Metropolis-Hastings behavior
        
        Returns:
            List[Tuple[ProgramSequence]]: A list of ProgramSequence tuple outputs at tracked steps with metadata stored in the ProgramSequence objects.
        """
        # Calculate and store initial energy score
        old_energy = self.score_energy()
        for output in self.outputs:
            output._metadata['energy_score'] = old_energy
            output._metadata['num_step'] = 0

        # Initialize history tracking
        sequence_snapshot = tuple(copy.deepcopy(output) for output in self.outputs)
        sequence_history = [sequence_snapshot]

        # Execute one MCMC optimization step
        for step in range(1, self.num_steps + 1):
            # 1. Pick a generator.
            generator = random.choice(self.generators)
            # Track old sequences x(t).
            old_seqs = [s.sequence for s in generator.get_outputs()]

            # 2. Sample x' from generator.
            generator.sample()
            # Evaluate new energy for x'.
            new_energy = self.score_energy()

            # 3. Compute acceptance probability g(x') / g(x(t)) with temperature.
            alpha = (new_energy / (old_energy + 1e-12)) ** (self.temperature)
            alpha = min(1.0, alpha)

            # 4. Accept/reject according to random number [0.0, 1.0).
            if random.random() > alpha:
                old_energy = new_energy
            else:
                for seq_obj, old in zip(generator.get_outputs(), old_seqs):
                    seq_obj.sequence = old
            
            # Track sequence and energy periodically
            if step % self.track_step_size == 0:
                # Store the energy score
                for output in self.outputs:
                    output._metadata['energy_score'] = old_energy
                    output._metadata['num_step'] = step
                # Create deep copies of the sequence objects
                sequence_snapshot = tuple(copy.deepcopy(output) for output in self.outputs)
                sequence_history.append(sequence_snapshot)

        # Return a dictionary with the tracked state information
        return sequence_history