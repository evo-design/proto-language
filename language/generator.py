from abc import ABC, abstractmethod
import random
from typing import Any, List, Dict, Optional, Tuple

from .base import ProgramGenerator, ProgramSequence, ProgramConstraint
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
            self.outputs = ( self.sequence_class(self, 0, random_sequence), )
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


class ProgramMCMCGenerator(ProgramGenerator):
    def __init__(
        self,
        generators: List[ProgramGenerator],
        constraints: List[ProgramConstraint],
        **hyperparameters: Any,
    ) -> None:
        super().__init__(**hyperparameters)
        self.generators = generators
        self.constraints = constraints
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        self.num_steps: int = hyperparameters.get("num_steps", 1000)
        self.num_samples: int = hyperparameters.get("num_samples", 100)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramDNASequence(self, 0), )
        return self.outputs
    
    def sample(self) -> None:
        pass


class Evo2Generator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramDNASequence(self, 0), )
        return self.outputs
    
    def sample(self) -> None:
        pass
        

class SemanticMiningGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramDNASequence(self, 0), )
        return self.outputs
    
    def sample(self) -> None:
        pass


class BindCraftGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramProteinSequence(self, 0), )
        return self.outputs
    
    def sample(self) -> None:
        pass
