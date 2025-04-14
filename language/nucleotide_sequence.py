# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

# Define nucleotide types
NUCLEOTIDE_TYPES = ["A", "C", "G", "T"]


class NucleotideSequenceSegmentFactory(ABC):    
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get(self) -> str:
        pass

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def num_mutation_candidates(self) -> int:
        pass


class ConstantNucleotideSequence(NucleotideSequenceSegmentFactory):    
    def __init__(self, sequence: str) -> None:
        super().__init__()
        self.sequence = sequence
        
    def get(self) -> str:
        return self.sequence

    def mutate(self) -> None:
        pass

    def num_mutation_candidates(self) -> int:
        return 0


class FixedLengthNucleotideSequence(NucleotideSequenceSegmentFactory):    
    def __init__(self, initial_sequence: Union[str, int]) -> None:

        super().__init__()
        
        self.sequence = (
            initial_sequence
            if isinstance(initial_sequence, str)
            else random_nucleotide_sequence(length=initial_sequence)
        )

    def get(self) -> str:
        return self.sequence

    def mutate(self) -> None:
        self.sequence = substitute_one_nucleotide(self.sequence)

    def num_mutation_candidates(self) -> int:
        return len(self.sequence)


class VariableLengthNucleotideSequence(NucleotideSequenceSegmentFactory):    
    def __init__(
        self,
        initial_sequence: Union[str, int],
        mutation_operation_probabilities: List[float] = [
            3., # Substitution weight.
            1., # Deletion weight.
            1., # Insertion weight.
        ],
    ) -> None:

        super().__init__()
        
        self.sequence = (
            initial_sequence
            if isinstance(initial_sequence, str)
            else random_nucleotide_sequence(length=initial_sequence)
        )
        
        self.mutation_operation_probabilities = np.array(mutation_operation_probabilities)
        self.mutation_operation_probabilities /= self.mutation_operation_probabilities.sum()

    def get(self) -> str:
        return self.sequence

    def mutate(self) -> None:
        mutation_operation = np.random.choice(
            [
                self._mutate_substitution,
                self._mutate_deletion,
                self._mutate_insertion,
            ],
            p=self.mutation_operation_probabilities,
        )
        mutation_operation()

    def _mutate_substitution(self) -> None:
        self.sequence = substitute_one_nucleotide(self.sequence)

    def _mutate_deletion(self) -> None:
        if len(self.sequence) > 1:  # edge case
            self.sequence = delete_one_nucleotide(self.sequence)

    def _mutate_insertion(self) -> None:
        self.sequence = insert_one_nucleotide(self.sequence)

    def num_mutation_candidates(self) -> int:
        return len(self.sequence)


# Helper functions for nucleotide sequence operations

def substitute_one_nucleotide(sequence: str, corpus: List[str]) -> str:
    sequence = list(sequence)
    index = np.random.choice(len(sequence))
    sequence[index] = np.random.choice(corpus)
    return "".join(sequence)


def random_nucleotide_sequence(length: int) -> str:
    return "".join([np.random.choice(NUCLEOTIDE_TYPES) for _ in range(length)])


def delete_one_nucleotide(sequence: str) -> str:
    index = np.random.choice(len(sequence))
    return sequence[:index] + sequence[index + 1 :]


def insert_one_nucleotide(sequence: str) -> str:
    n = len(sequence)
    index = np.random.randint(0, n) if n > 0 else 0
    insertion = np.random.choice(NUCLEOTIDE_TYPES)
    return sequence[:index] + insertion + sequence[index:]