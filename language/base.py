"""
Defines the base classes for the high-level program.

ProgramSequence: The sequence variables.

ProgramConstraint: The constraint scoring functions.

ProgramGenerator: The generative models and samplers.
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
    def __init__(
        self,
        generator: "ProgramGenerator",
        generator_output_idx: int,
        sequence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_chars: Optional[Set[str]] = None,
    ) -> None:
        """
        Initializes the ProgramSequence object.

        A ProgramSequence can be initialized to contain a sequence, or it can be empty
        and filled in later (e.g., by executing a generator).

        Args:
            generator (ProgramGenerator): The generator that updates `sequence.`
            generator_output_idx (int): The index into the generator's output list.
            sequence (Optional[str]): The value of the sequence string.
            valid_chars (Optional[Set[str]]): A set of valid characters that the sequence
                                              can take on.
        """
        self.generator: ProgramGenerator = generator
        self.generator_output_idx: int = generator_output_idx
        self._sequence: Optional[str] = sequence
        self._metadata: Dict[str, Any] = metadata if metadata is not None else {'sequence': sequence}
        self._valid_chars: Optional[Set[str]] = valid_chars

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

    def __hash__(self) -> int:
        """
        Computes hash based on the sequence string.

        Returns:
            int: The hash value.
        """
        if self._sequence is None:
            return hash("")
        return hash(self._sequence)


class ProgramConstraint(ABC):
    """
    Base class for constraints or scoring functions applied to sequences.
    """
    def __init__(
        self,
        inputs: ProgramSequence | List[ProgramSequence],
        scoring_function: Callable[[ProgramSequence | List[ProgramSequence]], float],
        **kwargs: Any,
    ) -> None:
        """
        Initializes the constraint, potentially with configuration parameters.

        Args:
            inputs (ProgramSequence | List[ProgramSequence]): The input variables.
            scoring_function (Callable[[ProgramSequence | List[ProgramSequence]], float]):
                The scoring function to call on the inputs.
            **kwargs (Any): Arbitrary keyword arguments for configuration.
        """
        self.inputs: List[ProgramSequence] = inputs if isinstance(inputs, list) else [inputs]
        self.scoring_function: Callable[[ProgramSequence | List[ProgramSequence]], float] = scoring_function
        self.config: Dict[str, Any] = kwargs

    def evaluate(self) -> float:
        """
        Evaluates the constraint using `self.inputs`.

        Returns:
            float: A score representing how well the sequences satisfy the
                   constraint. Implementations should aim for a score in the
                   interval [0.0, 1.0].
        """
        return self.scoring_function(self.inputs)


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
