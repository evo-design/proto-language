"""
IterativeGenerator base class for the proto-language.

Specialized generator for iterative optimization with energy-based evaluation.
"""

import copy
import warnings
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .constraint import Constraint
from .construct import Construct
from .generator import Generator
from .segment import Segment


class IterativeGenerator(Generator):
    """
    Specialized generator for iterative optimization with energy-based evaluation.

    Extends Generator to support iterative algorithms like MCMC that require
    energy evaluation and state tracking. The class works with multiple
    sub-generators and constraints.
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        batch_size: int = 1,
        **hyperparameters: Any,
    ) -> None:
        """
        Initialize the IterativeGenerator.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            batch_size: Number of sequence variants to generate simultaneously.
            **hyperparameters: Additional configuration parameters.
        """
        super().__init__(batch_size=batch_size, **hyperparameters)
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights = constraint_weights or [1.0] * len(constraints)
        self.history: List[Dict[str, Any]] = []  # Each entry: {"time_step": int, "energy_scores": List[float], "constructs": List[Construct]}
        self.energy_scores: List[float] = []  # Each index corresponds to a batch element, empty until first score_energy() call

        # Any batch_size specified during sub-generator construction is overwritten
        for i, gen in enumerate(self.generators):
            if gen.batch_size != self.batch_size:
                # Only warn if the generator was explicitly initialized with a non-default batch_size
                if gen.batch_size != 1:
                    warnings.warn(
                        f"Generator {i} ({gen.__class__.__name__}) was initialized with batch_size={gen.batch_size}, "
                        f"but IterativeGenerator is overwriting it to batch_size={self.batch_size}. "
                        f"To avoid this warning, do not specify batch_size when creating sub-generators for IterativeGenerator.",
                        UserWarning,
                        stacklevel=2
                    )
                # Expand the generator's assigned segment(s) to match the new batch_size
                for segment in gen.get_generator_outputs():
                    if segment.batch_size != self.batch_size:
                        segment.create_batch(self.batch_size)
            gen.batch_size = self.batch_size

        # Set self._generator_outputs to be a flat tuple of all Segment objects from all sub-generators
        self._generator_outputs = tuple(
            seq for gen in self.generators for seq in gen.get_generator_outputs()
        )  # Unused
        self._is_initialized = True

    def _validate_generator(self) -> None:
        """
        Validate that constructs, generators, constraints are properly configured.
        Must be called in final subclass __init__ to ensure all attributes are set.

        Raises:
            RuntimeError: If called before assign() has been called.
            ValueError: If any validation checks fail.
        """
        # Ensure basic generator validation
        super()._validate_generator()

        # Ensure constructs, generators, and constraints are non-empty lists
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        if not self.generators:
            raise ValueError("Generators list cannot be empty")
        if not self.constraints:
            raise ValueError("Constraints list cannot be empty")

        # Ensure constraint_weights are positive and finite
        invalid_weights = [
            w for w in self.constraint_weights if w <= 0 or not np.isfinite(w)
        ]
        if invalid_weights:
            raise ValueError(
                f"Constraint weights must be positive and finite. Found invalid weights: {invalid_weights}"
            )

        # Ensure constraint count matches weight count
        if len(self.constraints) != len(self.constraint_weights):
            raise ValueError(
                f"Constraint count ({len(self.constraints)}) must match weight count ({len(self.constraint_weights)})"
            )

        # Ensure types for all constructs, generators, and constraints are correct
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise ValueError(
                    f"Construct {i} has type {type(construct)}, expected Construct"
                )

        for i, generator in enumerate(self.generators):
            if not isinstance(generator, Generator):
                raise ValueError(
                    f"Generator {i} has type {type(generator)}, expected Generator"
                )

        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise ValueError(
                    f"Constraint {i} has type {type(constraint)}, expected Constraint"
                )

        # Ensure all generators are assigned construct segments
        if not all(generator._is_initialized for generator in self.generators):
            raise ValueError("Not all generators have been initialized.")

        # Ensure all construct segments are assigned to a generator
        unassigned_segments = [
            segment
            for construct in self.constructs
            for segment in construct.segments
            if not segment._is_assigned
        ]
        if unassigned_segments:
            raise ValueError(
                f"Found {len(unassigned_segments)} construct segments not assigned to any generator."
            )

        # Ensure all constraints have at least one generator-assigned input Segment
        generator_segment_ids = set(
            id(segment)
            for generator in self.generators
            for segment in generator.get_generator_outputs()
        )
        for i, constraint in enumerate(self.constraints):
            if not constraint.inputs:
                raise ValueError(f"Constraint {i} has no inputs assigned")
            if not any(id(inp) in generator_segment_ids for inp in constraint.inputs):
                raise ValueError(f"Constraint {i} has no generator-connected inputs")

    def _replicate_best_sequence(self, best_idx: int) -> None:
        """
        Copy the best sequence to all positions within each Segment.

        This helper method ensures that when a proposal is accepted, the sequence
        with the best energy is propagated to all positions within each batch.
        This is essential for maintaining consistency in constructs access.

        Args:
            best_idx: Index of the best sequence to propagate across all batches.

        Raises:
            RuntimeError: If called before assign() has been called.
            ValueError: If any batch has fewer sequences than best_idx.

        Note:
            This method modifies sequences in-place for all generator outputs.
        """
        # Get generator outputs (works with both single and multiple outputs)
        generator_outputs = self.get_generator_outputs()

        for segment in generator_outputs:
            if len(segment.batch_sequences) <= best_idx:
                raise ValueError(
                    f"Segment has only {len(segment.batch_sequences)} sequences, "
                    f"cannot propagate best sequence at index {best_idx}"
                )

            best_sequence = segment.batch_sequences[best_idx]
            # Propagate the best sequence to all positions in this Segment
            # TODO: Check if this propagation makes sense for top_k implemenetation
            for sequence in segment.batch_sequences:
                sequence.sequence = best_sequence.sequence
                sequence._metadata = best_sequence._metadata.copy()

    @abstractmethod
    def sample(self, **kwargs: Any) -> None:
        """
        Run one or more steps of iterative generation.

        Subclasses should implement this method to run the generation process.
        Implementations should modify generator outputs in-place and may store
        snapshots of constructs in `self.history`.

        Args:
            **kwargs: Keyword arguments for sampling (subclass-specific).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    def assign(
        self, assigned_segments: Segment | Iterable[Segment]
    ) -> None:
        """
        IterativeGenerator doesn't support manual assignment.

        Raises:
            NotImplementedError: Always, as IterativeGenerator auto-initializes from pre-assigned generators.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} auto-initializes from pre-assigned generators. "
            "Manual assignment is not supported. Ensure all sub-generators are assigned before "
            "creating the IterativeGenerator."
        )

    def score_energy(self, operation: str = "add") -> None:
        """
        Compute energy scores by combining constraint evaluation scores
        Energy scores are stored in self.energy_scores.

        The energy function is computed as a weighted combination of all
        constraint scores. Lower energy values indicate better solutions.

        Args:
            operation: How to combine constraint scores across constraints:
                - 'add': Sum weighted constraint scores (default)
                - 'multiply': Multiply weighted constraint scores

        Raises:
            ValueError: If generator is not properly initialized or operation is not 'add' or 'multiply'.
            RuntimeError: If called before assign() has been called.

        Note:
            Energy computation uses current sequence values, so it reflects
            the most recent state after any sampling operations. The computed
            energy scores are accessible via self.energy_scores.
        """
        # Ensure generator is properly initialized
        self._validate_generator()

        # Get weighted scores from all constraints: shape (n_constraints, n_samples)
        constraint_scores = np.array(
            [
                np.array(constraint.evaluate()) * weight
                for constraint, weight in zip(self.constraints, self.constraint_weights)
            ]
        )

        # Combine across constraints for each sample
        if operation == "multiply":
            energies = np.prod(constraint_scores, axis=0)
        elif operation == "add":
            energies = np.sum(constraint_scores, axis=0)
        else:
            raise ValueError(f"Operation must be 'multiply' or 'add', got {operation}")

        energies_list = energies.tolist()
        self.energy_scores = energies_list
    
    def append_snapshot_to_history(self, step: int = 0) -> None:
        """Save snapshot of current construct state and energy scores to history."""
        # Store as structured history entry with separate metadata
        history_entry = {
            "time_step": step,
            "energy_scores": self.energy_scores.copy(),
            "constructs": copy.deepcopy(self.constructs)
        }
        
        self.history.append(history_entry)

