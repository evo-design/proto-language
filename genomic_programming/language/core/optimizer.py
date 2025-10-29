"""
Optimizer base class for the biological programming language.

Base class for iterative optimization algorithms that coordinate multiple
generators and constraints to search for optimal biological sequences.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import copy
import math

from .constraint import Constraint
from .construct import Construct
from .generator import Generator
from ...tools.tool_cache import ToolCache, _program_tool_cache


class Optimizer(ABC):
    """
    Base class for optimization algorithms.

    Coordinates multiple generators and constraints to search for optimal
    biological sequences through iterative optimization. Unlike generators
    which modify sequences directly, optimizers orchestrate the search process
    by coordinating generators, evaluating constraints, and making decisions
    about which sequences to keep.
    """

    @abstractmethod
    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        num_candidates: int,
        num_selected: int,
        constraint_weights: Optional[List[float]] = None,
        clear_tool_cache: bool | List[str] = True,
    ) -> None:
        """
        Initialize the Optimizer with dual-pool semantics.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            num_candidates: Number of candidate proposals to generate per iteration.
            num_selected: Number of sequences to select and maintain as results.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            clear_tool_cache: (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
        """
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights = constraint_weights or [1.0] * len(constraints)
        self.num_candidates = num_candidates
        self.num_selected = num_selected
        self.clear_tool_cache = clear_tool_cache
        self.energy_scores: List[float] = []  # Each index corresponds to a candidate, empty until first score_energy() call
        self.history: List[Dict[str, Any]] = []

        # Default values for progress tracking (can be overridden by subclasses)
        self.num_steps: int = 1
        self.track_step_size: int = 1

        # Create program-scoped tool cache
        self.tool_cache = ToolCache()
        _program_tool_cache.set(self.tool_cache)

        self._initialize_sequence_pools()
        self._validate_optimizer()

    @property
    def segments(self):
        """All segments from all constructs being optimized."""
        return tuple(seg for construct in self.constructs for seg in construct.segments)

    @abstractmethod
    def run(self) -> None:
        """
        Subclasses should implement this method to run the optimization process.
        Implementations should modify generator outputs in-place.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    def _save_progress_snapshot(self, time_step: int) -> None:
        """
        Save current optimization state to history.

        Default implementation saves time_step, energy_scores, and constructs.
        Subclasses can override to add optimizer-specific metadata.

        Args:
            time_step: Current step/round/segment index
        """
        self.history.append({
            "time_step": time_step,
            "energy_scores": self.energy_scores[:self.num_selected].copy(),
            "constructs": copy.deepcopy(self.constructs)
        })

    def score_energy(self, operation: str = "add", verbose: bool = False) -> None:
        """
        Compute energy scores by combining all constraint evaluation scores.
        Evaluates the current state of all Sequence objects stored in segments.candidate_sequences.

        Args:
            operation: How to combine constraint scores across constraints:
                - 'add': Sum weighted constraint scores (default)
                - 'multiply': Multiply weighted constraint scores
            verbose: If True, print detailed energy score calculations for each constraint

        Raises:
            ValueError: If optimizer is not properly initialized or operation is not 'add' or 'multiply'.
        """
        # Ensure generator is properly initialized
        self._validate_optimizer()

        if verbose:
            print("\n" + "="*60)
            print("Energy Scoring")
            print("="*60)
            if operation == "add":
                print("Formula: energy = Σ(weight_i x constraint_score_i)")
            else:
                print("Formula: energy = Π(weight_i x constraint_score_i)")
            print()

        # Get weighted scores from all constraints: list of lists (n_constraints, n_samples)
        weighted_scores = []
        for idx, (constraint, weight) in enumerate(zip(self.constraints, self.constraint_weights)):
            # Evaluate constraint and get raw scores
            raw_scores = constraint.evaluate()
            
            if verbose:
                print(f"Constraint {idx+1}: {constraint.__class__.__name__} (weight={weight})")
                for i, score in enumerate(raw_scores):
                    print(f"  Candidate {i}: {score:.4f}")
                print()
            
            # Apply weight
            weighted = [score * weight for score in raw_scores]
            weighted_scores.append(weighted)

        # Combine across constraints for each sample
        if operation == "add":
            self.energy_scores = [sum(scores) for scores in zip(*weighted_scores)]
        elif operation == "multiply":
            from math import prod
            self.energy_scores = [prod(scores) for scores in zip(*weighted_scores)]
        else:
            raise ValueError(f"Operation must be 'multiply' or 'add', got {operation}")
        
        if verbose:
            print(f"Final Energy Scores:")
            for i, score in enumerate(self.energy_scores):
                print(f"  Candidate {i}: {score:.4f}")
            print("="*60)
            print()

        # After evaluating all constraints, optionally clear the cache.
        if self.clear_tool_cache:
            if isinstance(self.clear_tool_cache, bool):
                self.tool_cache.clear()
            elif isinstance(self.clear_tool_cache, list):
                for tool in self.clear_tool_cache:
                    self.tool_cache.clear(tool)
            else:
                raise ValueError(f"Invalid type of clear_tool_cache: {type(self.clear_tool_cache)}")

    def _validate_optimizer(self) -> None:
        """
        Validate that constructs, generators, constraints are properly configured.
        Must be called in final subclass __init__ to ensure all attributes are set.

        Raises:
            ValueError: If user inputs are invalid (empty lists, invalid weights, etc.).
            TypeError: If objects have incorrect types.
            RuntimeError: If optimizer state is invalid (unassigned segments, etc.).
        """

        # Ensure constructs, generators, and constraints are non-empty lists
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        if not self.generators:
            raise ValueError("Generators list cannot be empty")
        if not self.constraints:
            raise ValueError("Constraints list cannot be empty")

        # Ensure constraint_weights are positive and finite
        invalid_weights = [
            w for w in self.constraint_weights if w <= 0 or not math.isfinite(w)
        ]
        if invalid_weights:
            raise ValueError(f"Constraint weights must be positive and finite. Found invalid weights: {invalid_weights}")

        # Ensure constraint count matches weight count
        if len(self.constraints) != len(self.constraint_weights):
            raise ValueError(f"Constraint count ({len(self.constraints)}) must match weight count ({len(self.constraint_weights)})")

        # Ensure all constructs have correct type and have segments
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise TypeError(f"Construct {i} has type {type(construct)}, expected Construct")
            if not construct.segments:
                raise ValueError(f"Construct {i} has no segments")

        # Ensure all generators have correct type and have assigned segments
        for i, generator in enumerate(self.generators):
            if not isinstance(generator, Generator):
                raise TypeError(f"Generator {i} has type {type(generator)}, expected Generator")
            if not generator._assigned_segment:
                raise RuntimeError(f"Generator {i} ({generator.__class__.__name__}) has no segment assigned")

        # Ensure all constraints have correct type and have input segments
        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise TypeError(f"Constraint {i} has type {type(constraint)}, expected Constraint")
            if not constraint.inputs:
                raise RuntimeError(f"Constraint {i} has no input segment(s) assigned")

        # Ensure all segments are assigned
        unassigned_segments = [segment for segment in self.segments if not segment._is_assigned]
        if unassigned_segments:
            raise RuntimeError(f"Found {len(unassigned_segments)} non-constant segments not assigned to any generator.")

    def _initialize_sequence_pools(self) -> None:
        """Initialize the sequence pools for all segments.
        
        Creates independent copies of the initial sequence for both pools.
        Using deepcopy ensures mutations don't affect other pool members.
        """
        for segment in self.segments:
            # Create independent copies, not references to the same object
            segment.selected_sequences = [copy.deepcopy(segment.selected_sequences[0]) for _ in range(self.num_selected)]
            segment.candidate_sequences = [copy.deepcopy(segment.candidate_sequences[0]) for _ in range(self.num_candidates)]
