"""
Optimizer base class for the biological programming language.

Base class for iterative optimization algorithms that coordinate multiple
generators and constraints to search for optimal biological sequences.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional

from proto_language.tools.tool_cache import ToolCache, _program_tool_cache

from .constraint import Constraint
from .construct import Construct
from .generator import Generator


class Optimizer(ABC):
    """
    Base class for optimization algorithms.

    Coordinates multiple generators and constraints to search for optimal
    biological sequences through iterative optimization. Unlike generators
    which modify sequences directly, optimizers orchestrate the search process
    by coordinating generators, evaluating constraints, and making decisions
    about which sequences to keep.

    Filter Constraints:
        Constraints with a threshold parameter act as binary filters that accept or reject
        candidates before scoring. Rejected candidates receive infinite penalty scores
        and skip all subsequent constraint evaluations, improving performance when
        constraints are computationally expensive.

        Filter evaluation order:
        1. All filter constraints (those with threshold set) are evaluated first
        2. Candidates must pass ALL filters (AND logic)
        3. Only accepted candidates are evaluated by scoring constraints
        4. Rejected candidates receive filter_penalty score (default: inf)
    """

    @abstractmethod
    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        num_candidates: int,
        num_selected: int,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
        custom_logging: Optional[Callable] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the Optimizer with dual-pool semantics.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            num_candidates: Number of candidate proposals to generate per iteration.
            num_selected: Number of sequences to select and maintain as results.
            clear_tool_cache: (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
            custom_logging: Optional callback called after each iteration with
                signature ``(step: int, segments: tuple) -> None``.
            verbose: Whether to print detailed progress information. Default: False.
        """
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.num_candidates = num_candidates
        self.num_selected = num_selected
        self.clear_tool_cache = clear_tool_cache
        self.custom_logging = custom_logging
        self.verbose = verbose
        self.energy_scores: List[float] = [float("inf")] * num_candidates  # Initialized to inf (unscored)
        self.history: List[Dict[str, Any]] = []
        self._initial_state: Optional[Dict] = None  # Captured on first run() for restart

        # Default value for progress tracking (can be overridden by subclasses)
        self.num_steps: int = 1

        # Create program-scoped tool cache
        self.tool_cache = ToolCache()
        _program_tool_cache.set(self.tool_cache)

        self._initialize_sequence_pools()
        self._validate_optimizer()

    @property
    def segments(self):
        """All segments from all constructs being optimized."""
        return tuple(seg for construct in self.constructs for seg in construct.segments)

    @property
    def constraint_weights(self) -> List[float]:
        """Get all constraint weights."""
        return [constraint.weight for constraint in self.constraints]

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

        Raises:
            RuntimeError: If energy_scores length doesn't match num_selected.
        """
        if len(self.energy_scores) != self.num_selected:
            raise RuntimeError(
                f"energy_scores has length {len(self.energy_scores)}, expected {self.num_selected}. "
                f"Ensure energy_scores is truncated to num_selected after selection."
            )
        self.history.append({
            "time_step": time_step,
            "energy_scores": self.energy_scores.copy(),
            "constructs": copy.deepcopy(self.constructs)
        })

    def score_energy(
        self,
        operation: Literal["add", "multiply"] = "add",
        filter_penalty: float = float("inf"),
    ) -> None:
        """
        Compute energy scores by combining all constraint evaluation scores on the candidate sequences.

        Filter constraints are evaluated first. Rejected candidates skip subsequent
        constraint evaluations for performance.

        Evaluation order:
            1. Filter constraints (with threshold) evaluated first
            2. Candidates must pass ALL filters (AND logic)
            3. Scoring constraints (without threshold) only evaluate candidates that passed all filters
            4. Rejected candidates receive filter_penalty without further evaluation

        Args:
            operation: How to combine scores: 'add' (sum) or 'multiply' (product)
            filter_penalty: Score for rejected candidates (default: inf)

        Raises:
            ValueError: If optimizer is not properly initialized or operation is not 'add' or 'multiply'.
        """
        self._validate_optimizer()

        num_sequences = (
            len(self.segments[0].candidate_sequences) if self.segments else 0
        )
        passed = [True] * num_sequences

        # Separate constraints into filters and scoring constraints
        filters = [c for c in self.constraints if c.threshold is not None]
        scorers = [c for c in self.constraints if c.threshold is None]

        if self.verbose:
            op = "Σ" if operation == "add" else "Π"
            print(
                f"\n{'='*60}\n"
                f"Energy Scoring: {len(filters)} filters, {len(scorers)} scoring\n"
                f"Formula: energy = {op}(weight_i x constraint_score_i)\n"
            )

        # Pass 1: Evaluate all filter constraints first to skip expensive scoring on rejected candidates
        for idx, constraint in enumerate(filters):
            if self.verbose:
                print(f"Filter {idx+1}: {constraint.label}")
            results = constraint.evaluate(mask=passed, verbose=self.verbose)
            passed = [p and r for p, r in zip(passed, results)]

        # Pass 2: Score passing candidates (skip rejected candidates for performance)
        all_scores = []
        for idx, constraint in enumerate(scorers):
            if self.verbose:
                print(f"Constraint {idx+1}: {constraint.label}")
            all_scores.append(constraint.evaluate(mask=passed, verbose=self.verbose))

        # Aggregate scores across all scoring constraints into a single energy score per candidate.
        # NaN propagates through sum/prod operations, resuling in NaN if any constraint is unevaluated.
        if operation == "add":
            self.energy_scores = [sum(s[i] for s in all_scores) for i in range(num_sequences)]
        elif operation == "multiply":
            self.energy_scores = [math.prod(s[i] for s in all_scores) for i in range(num_sequences)]
        else:
            raise ValueError(f"Operation must be 'add' or 'multiply'")

        # Check for inconsistent state
        assert len(self.energy_scores) == len(passed), \
            ("Inconsistent state: energy scores should have the same length as passed mask")

        # NaN signals "not evaluated" and propagates through arithmetic, making bugs visible
        for i, (score, did_pass) in enumerate(zip(self.energy_scores, passed)):
            if did_pass and math.isnan(score):
                raise RuntimeError(f"Inconsistent state: candidate {i} passed all filters but has NaN score.")

        # Apply filter_penalty to rejected candidates
        self.energy_scores = [
            score if passed[i] else filter_penalty
            for (i, score) in enumerate(self.energy_scores)
        ]

        if self.verbose:
            print("Final Energy Scores:")
            for i, score in enumerate(self.energy_scores):
                print(f"  Candidate {i}: {score:.4f}{' [REJECTED]' if not passed[i] else ''}")

        self._clear_tool_cache()

    def _clear_tool_cache(self) -> None:
        """
        Clear tool cache based on configuration.

        Config Behavior:
        - int: Clear cache if size (in bytes) exceeds this threshold.
        - bool (True): Clear entire cache.
        - List[str]: Clear specific tools in list.
        """
        if not self.clear_tool_cache:
            return

        # Case 1: Byte threshold (int).
        if isinstance(self.clear_tool_cache, int) and not isinstance(self.clear_tool_cache, bool):
            threshold_bytes = self.clear_tool_cache

            if self.tool_cache.current_size > threshold_bytes:
                self.tool_cache.prune(threshold_bytes)

        # Case 2: Clear all (bool).
        elif isinstance(self.clear_tool_cache, bool):
            self.tool_cache.clear()

        # Case 3: Clear all for specific tools (List[str]).
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

        # Check for duplicate generator instances
        seen_gen_ids: set[int] = set()
        for generator in self.generators:
            gen_id = id(generator)
            if gen_id in seen_gen_ids:
                raise ValueError(
                    f"Generator '{generator.__class__.__name__}' instance appears multiple times "
                    "in the generators list. Each generator instance can only be used once."
                )
            seen_gen_ids.add(gen_id)

        # Check for duplicate constraint instances
        seen_con_ids: set[int] = set()
        for constraint in self.constraints:
            con_id = id(constraint)
            if con_id in seen_con_ids:
                raise ValueError(
                    f"Constraint '{constraint.label}' instance appears multiple times "
                    "in the constraints list. Each constraint instance can only be used once."
                )
            seen_con_ids.add(con_id)

        # Build set of segments that have an active generator in THIS optimizer
        # Validate generators: all generators must have assigned segments
        assigned_segments = set()
        for gen in self.generators:
            if gen._assigned_segment is None:
                raise RuntimeError(f"Generator '{gen.__class__.__name__}' has no segment assigned. All generators in an optimizer must have an assigned segment.")
            assigned_segments.add(gen._assigned_segment)

        # Validate constraints don't reference empty segments without generators
        for constraint in self.constraints:
            for segment in constraint.inputs:
                has_generator = segment in assigned_segments
                if not segment.populated_sequences and not has_generator:
                    raise RuntimeError(
                        f"Constraint '{constraint.label}' references segment '{segment.label or 'unlabeled'}' "
                        "which has no populated sequence and no generator assigned. "
                        "Segments must have sequences or generators before constraints can be evaluated."
                    )

    def _initialize_sequence_pools(self) -> None:
        """Initialize sequence pools from previous optimizer or original sequence.

        Behavior:
        - Uses previous optimizer's selected_sequences if available (sorted best-first)
        - Falls back to candidate_sequences, then original_sequence
        - Pads with copies of best sequence if fewer than num_selected available
        - Candidates are all initialized from best sequence (will be mutated by generators)
        """
        for segment in self.segments:
            # Source: previous optimizer's results or original sequence
            # TODO: The best sequence logic here is incorrect. Should we move initialize_sequence_pools to the program level?
            source = segment.selected_sequences or segment.candidate_sequences or [segment.original_sequence]
            best_seq = source[0]

            # Selected pool: up to num_selected from source, pad with best
            segment.selected_sequences = [
                copy.deepcopy(source[i] if i < len(source) else best_seq)
                for i in range(self.num_selected)
            ]

            # Candidate pool: all copies of best (will be mutated by generators)
            segment.candidate_sequences = [
                copy.deepcopy(best_seq) for _ in range(self.num_candidates)
            ]

    def _prepare_run(self) -> None:
        """Call at start of run(). Captures state on first run, restores on subsequent."""
        if self._initial_state is None:
            self._capture_initial_state()
        else:
            self._restore_initial_state()

    def _capture_initial_state(self) -> None:
        """Capture current segment and optimizer state."""
        self._initial_state = {
            'segments': [
                {
                    'selected': copy.deepcopy(seg.selected_sequences),
                    'candidates': copy.deepcopy(seg.candidate_sequences),
                }
                for seg in self.segments
            ],
            'energy_scores': self.energy_scores.copy(),
        }

    def _restore_initial_state(self) -> None:
        """Restore to captured state. Override for optimizer-specific state."""
        for i, seg in enumerate(self.segments):
            state = self._initial_state['segments'][i]
            seg.selected_sequences = copy.deepcopy(state['selected'])
            seg.candidate_sequences = copy.deepcopy(state['candidates'])
        self.energy_scores = self._initial_state['energy_scores'].copy()
        self.history = []
