"""
Defines the base classes for the high-level program.

ProgramSequence: The sequence variables.

ProgramConstraint: The constraint scoring functions.

ProgramGenerator: The generative models and samplers.

ProgramIterativeGenerator: A special generative model that implements iterative generation with an energy function
"""
from abc import ABC, abstractmethod
import collections
from dataclasses import dataclass
from typing import (
    Callable, List, Tuple, Dict, Any, Set, Optional,
)


class ProgramSequence:
    """
    Base class for the program sequence variables. A variable is defined by its
    generator and its index into the generator's output.
    """
    VALID_SEQUENCE_TYPES = {'dna', 'rna', 'protein'}

    def __init__(
        self,
        sequence: Optional[str] = None,
        sequence_type: Optional[str] = None,
        valid_chars: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the ProgramSequence object.

        A ProgramSequence can be initialized to contain a sequence, or it can be empty
        and filled in later (e.g., by executing a generator).

        Args:
            sequence (Optional[str]): The value of the sequence string.
            sequence_type (Optional[str]): Type of sequence. Must be either 'dna', 'rna', or 'protein'
            valid_chars (Optional[Set[str]]): A set of valid characters.
            metadata (Optional[Dict[str, Any]]): Metadata for the sequence.

        Raises:
            ValueError: If sequence_type is provided but not one of the valid types.
        """
        if sequence_type and sequence_type not in self.VALID_SEQUENCE_TYPES:
                raise ValueError(f"sequence_type must be one of {self.VALID_SEQUENCE_TYPES}, got {sequence_type}")

        self._sequence: Optional[str] = sequence
        self._sequence_type: Optional[str] = sequence_type
        self._valid_chars: Optional[Set[str]] = valid_chars
        self._metadata: Dict[str, Any] = metadata if metadata is not None else {'sequence': sequence}

    def _validate_sequence(self, sequence: str) -> None:
        """
        Checks if the sequence consists of valid characters.

        Args:
            sequence (str): The sequence to validate.
            valid_chars (Set[str]): A set of valid characters.
        """
        if self._valid_chars is None:
            return

        invalid_chars = set(sequence) - self._valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid characters found: {', '.join(invalid_chars)}. "
                             f"Valid characters are: {', '.join(sorted(self._valid_chars))}")

    @property
    def sequence(self) -> str:
        """
        Get the sequence string.

        Returns:
            str: The sequence string.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, new_sequence: str) -> None:
        """
        Sets the sequence string.

        Args:
            new_sequence (str): Value of the new sequence.
        """
        self._validate_sequence(new_sequence)

        self._sequence = new_sequence

    def __len__(self) -> int:
        """
        Returns the length of the sequence.

        Returns:
            int: The length of the sequence string.
        """
        if self._sequence is None:
            return 0
        return len(self._sequence)

    def __str__(self) -> str:
        """
        Get the sequence string by calling `str()` on the object.

        Returns:
            str: The sequence string.
        """
        if self._sequence is None:
            return ""
        return self._sequence


class ProgramConstraint(ABC):
    """
    Base class for constraints or scoring functions applied to sequences.
    """
    def __init__(
        self,
        inputs: ProgramSequence | List[ProgramSequence],
        scoring_function: Callable[[List[ProgramSequence], Dict[str, Any]], float],
        **kwargs: Any,
    ) -> None:
        """
        Initializes the constraint, potentially with configuration parameters.

        Args:
            inputs (ProgramSequence | List[ProgramSequence]): The input variables.
            scoring_function (Callable[[List[ProgramSequence], Dict[str, Any]], float]):
                The scoring function to call on the inputs. 
                - List[ProgramSequence]: The sequences to score
                - Dict[str, Any]: Config parameters from kwargs
            **kwargs (Any): Arbitrary keyword arguments for configuration.
        """
        self.inputs: List[ProgramSequence] = [inputs] if isinstance(inputs, ProgramSequence) else inputs
        self.scoring_function: Callable[[List[ProgramSequence], Dict[str, Any]], float] = scoring_function
        self.config: Dict[str, Any] = kwargs

    def evaluate(self) -> float:
        """
        Evaluates the constraint using `self.inputs`.

        Returns:
            float: A score representing how well the sequences satisfy the
                   constraint. Implementations should aim for a score in the
                   interval [0.0, 1.0].
        """
        return self.scoring_function(self.inputs, self.config)


class ProgramGenerator(ABC):
    """
    Abstract base class for program generation algorithms (samplers).

    Defines the interface for initializing and sampling sequences.
    Subclasses implement specific generation strategies (e.g., MCMC, autoregressive decoding).

    Subclasses must implement both initialize() and sample().
    """
    def __init__(
        self,
        **hyperparameters: Any,
    ) -> None:
        """
        Initializes the generator with specific hyperparameters.

        Args:
            **hyperparameters (Any): Keyword arguments representing the
                                     configuration and hyperparameters for the
                                     specific generator implementation.
        """
        self.hyperparameters: Dict[str, Any] = hyperparameters
        self._is_initialized: bool = False
        self.outputs: Optional[Tuple[ProgramSequence]] = None

    @abstractmethod
    def register(
        self,
        *args: Any,
        outputs: Optional[Tuple[ProgramSequence]] = None,
        **kwargs: Any,
    ) -> Tuple[ProgramSequence]:
        """
        Create the output sequence variables and return them to the user.

        Args:
            *args (Any): Any positional arguments for registration.
            outputs (Optional[Tuple[ProgramSequence]]): Allow the user to manually specify
                                                        the sequence variables.

        Returns:
            Tuple[ProgramSequence]: Output sequence variables. These variables get updated
                                    in-place throughout generation.
        """
        self._is_initialized = True
        self.outputs: Tuple[ProgramSequence] = outputs
        raise NotImplementedError("Subclasses must implement the register method.")

    def __len__(self) -> int:
        """
        Returns the number of outputs of the generator.

        Returns:
            int: The number of outputs of the generator

        Raises:
            RuntimeError: If called before initialize().
        """
        if not self._is_initialized or self.outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call initialize() first."
            )
        return len(self.outputs)

    def get_outputs(self) -> Tuple[ProgramSequence]:
        """
        Access the output sequence variables.

        Returns:
            Tuple[ProgramSequence]: Output sequence variables.

        Raises:
            RuntimeError: If called before initialize().
        """
        if not self._is_initialized or self.outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call initialize() first."
            )
        return tuple(self.outputs)

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> None:
        """
        Generates and returns a list of ProgramSequence instances based on the
        generator's internal state and hyperparameters.

        Raises:
            RuntimeError: If called before initialize().
        """
        if not self._is_initialized or self.outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call initialize() first."
            )
        raise NotImplementedError("Subclasses must implement the sample method.")


class ProgramIterativeGenerator(ProgramGenerator):
    """
    Special generative model that implements iterative generation with an energy function
    as a (weighted) combination of constraint functions.
    """
    def _check_constraint_attributes(self) -> None:
        """
        Class must have a list of constraints.
        """
        if not hasattr(self, 'constraints'):
            raise ValueError("ProgramIterativeGenerator objects must have constraints.")

        for constraint in self.constraints:
            if not isinstance(constraint, ProgramConstraint):
                raise ValueError(f"Found type {type(constraint)}, expected a ProgramConstraint")

        if not hasattr(self, 'constraint_weights'):
            self.constraint_weights = [1.] * len(self.constraints)

    def score_energy(self) -> float:
        """
        Multiply the constraints to produce the energy function.

        Returns:
            float: The value of the energy function.
        """
        self._check_constraint_attributes()

        assert len(self.constraints) == len(self.constraint_weights)

        energy = 1.
        for constraint, weight in zip(self.constraints, self.constraint_weights):
            energy *= weight * constraint.evaluate()
        return energy

    def score_energy_additive(self) -> float:
        """
        Add the constraints to produce the energy function.

        Returns:
            float: The value of the energy function.
        """
        self._check_constraint_attributes()

        assert len(self.constraints) == len(self.constraint_weights)

        energy = 0.
        for constraint, weight in zip(self.constraints, self.constraint_weights):
            energy += weight * constraint.evaluate()
        return energy

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> List[Tuple[ProgramSequence]]:
        """
        Generates sequences based on the generator's internal state and hyperparameters.
        Unlike the base ProgramGenerator.sample() which returns None, this method returns
        a list of ProgramSequence tuple outputs at tracked steps with metadata stored in the ProgramSequence objects.

        Returns:
            List[Tuple[ProgramSequence]]: A list of ProgramSequence tuple outputs at tracked steps with metadata stored in the ProgramSequence objects.

        Raises:
            RuntimeError: If called before initialize().
        """
        if not self._is_initialized or self.outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call initialize() first."
            )
        raise NotImplementedError("Subclasses must implement the sample method.")
