"""
Generator base class for the proto-language.

Provides the abstract interface for sequence generation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple

from .segment import Segment


class Generator(ABC):
    """
    Generator base class that creates/modifies sequences during optimization.

    Subclasses must implement assign() to assign sequences to generators and
    sample() to propose modified sequences.
    """

    def __init__(
        self,
        batch_size: int = 1,
        **hyperparameters: Any,
    ) -> None:
        """
        Initialize the generator with configuration parameters.

        Args:
            batch_size: Number of sequence variants to generate simultaneously.
            **hyperparameters: Additional configuration specific to the generator type.
                These are stored and can be accessed by subclasses.
        """
        self.batch_size: int = batch_size
        self.hyperparameters: Dict[str, Any] = hyperparameters
        self._is_initialized: bool = False
        self.iteration_count: int = 0

        # Support both single output (common) and multiple outputs (less common)
        self._generator_output: Optional[Segment] = None
        self._generator_outputs: Optional[Tuple[Segment, ...]] = None

    def _validate_generator(self) -> None:
        """
        Validate that the generator has been assigned and outputs are properly initialized.

        Raises:
            RuntimeError: If generator hasn't been assigned or outputs aren't initialized.
        """
        if not self._is_initialized:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been assigned. "
                "Call assign() first."
            )

        if self._generator_output is None and self._generator_outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} must initialize either "
                "_generator_output (for single output) or _generator_outputs (for multiple outputs) "
                "in the assign() method."
            )

    @abstractmethod
    def assign(
        self, assigned_segments: Segment | Iterable[Segment]
    ) -> None:
        """
        Assign Segment objects to the generator and initialize the generator.
        The generator will modify these Segment objects internally during sampling.

        Args:
            assigned_segments: Either a single Segment or an iterable of Segment objects.

        Raises:
            NotImplementedError: If not implemented by subclass.

        Note:
            This method must set self._is_initialized = True and initialize either:
            - self._generator_output: Segment (for single output - most common)
            - self._generator_outputs: Iterable[Segment] (for multiple outputs)

            Most generators should use _generator_output for simplicity.
            Use _generator_outputs only when the generator manages multiple distinct outputs.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the assign method."
        )

    @abstractmethod
    def sample(self, **kwargs: Any) -> None:
        """
        Sample new sequences by modifying generator outputs in-place.

        Args:
            **kwargs: Keyword arguments for sampling (subclass-specific).

        Raises:
            RuntimeError: If called before assign() has been called.
            NotImplementedError: If not implemented by subclass.

        Note:
            This method should modify sequences in-place for efficiency.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    def get_generator_outputs(self) -> Tuple[Segment, ...]:
        """
        Access the internal _generator_output or _generator_outputs as a tuple.

        Returns:
            Tuple of Segment objects that are modified during sampling.
            For single outputs, returns a tuple with one element.
            For multiple outputs, returns the tuple directly.

        Raises:
            RuntimeError: If called before assign() or if neither output is initialized.
        """
        # Ensure generator is properly initialized
        self._validate_generator()

        # Return the appropriate outputs as a tuple
        return self._generator_outputs or (self._generator_output,)

    def __len__(self) -> int:
        """
        Get the number of Segments managed by this generator.

        Returns:
            Number of Segment objects (1 for single output, N for multiple outputs).

        Raises:
            RuntimeError: If called before assign() has been called.
        """
        return len(self.get_generator_outputs())

    def get_iteration_count(self) -> int:
        """
        Get the number of times sample() has been invoked since the last reset.

        Returns:
            Current iteration count as an integer.
        """
        return self.iteration_count

    def reset_iteration_count(self) -> None:
        """
        Reset the internal iteration counter to zero.
        """
        self.iteration_count = 0

    def _increment_iteration_count(self) -> None:
        """
        Protected helper to increment the iteration counter by one.

        Subclasses should call this at the end of their sample() implementations
        if they want to use the shared iteration counter and any schedulers
        that depend on it.
        """
        self.iteration_count += 1

