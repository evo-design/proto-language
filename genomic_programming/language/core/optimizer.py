"""
Optimizer base class for the biological programming language.

Base class for iterative optimization algorithms that coordinate multiple
generators and constraints to search for optimal biological sequences.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from proto_language.language.core.constraint import Constraint
from typing import Any, Dict, List, Optional, Literal
import copy
import math

from .constraint import Constraint
from .construct import Construct
from .generator import Generator
from proto_language.tools.tool_cache import ToolCache, _program_tool_cache


class Optimizer(ABC):
    """
    Base class for optimization algorithms.

    Coordinates multiple generators and constraints to search for optimal
    biological sequences through iterative optimization. Unlike generators
    which modify sequences directly, optimizers orchestrate the search process
    by coordinating generators, evaluating constraints, and making decisions
    about which sequences to keep.

    Filter Constraints:
        Constraints with mode="filter" act as binary filters that accept or reject
        candidates before scoring. Rejected candidates receive infinite penalty scores
        and skip all subsequent constraint evaluations, improving performance when
        constraints are computationally expensive.

        Filter evaluation order:
        1. All filter constraints are evaluated first
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

    def score_energy(self, operation: Literal["add", "multiply"] = "add", verbose: bool = False, filter_penalty: float = float('inf')) -> None:
        """
        Compute energy scores by combining all constraint evaluation scores.
        Evaluates the current state of all Sequence objects stored in segments.candidate_sequences.

        Filter constraints (mode="filter") use short-circuit evaluation: once a candidate
        is rejected by any filter, it skips all subsequent filter and scoring evaluations.

        Evaluation order:
            1. Filter constraints evaluated sequentially in order
            2. Each filter only evaluates candidates not yet rejected
            3. Scoring constraints only evaluate candidates that passed all filters
            4. Rejected candidates receive filter_penalty without further evaluation

        Args:
            operation: How to combine constraint scores across constraints:
                - 'add': Sum weighted constraint scores (default)
                - 'multiply': Multiply weighted constraint scores
            verbose: If True, print detailed energy score calculations for each constraint
            filter_penalty: Score assigned to rejected candidates (default: inf)

        Raises:
            ValueError: If optimizer is not properly initialized or operation is not 'add' or 'multiply'.
        """
        self._validate_optimizer()
        
        num_sequences = len(self.segments[0].candidate_sequences) if self.segments else 0
        passed = [True] * num_sequences
        
        if verbose:
            print(f"\n{'='*60}\nEnergy Scoring: {sum(1 for c in self.constraints if c.mode == 'filter')} filters, {sum(1 for c in self.constraints if c.mode != 'filter')} scoring constraints\nFormula: energy = {'Σ' if operation == 'add' else 'Π'}(weight_i x constraint_score_i)\n")
        
        # Evaluate constraints, skipping subsequent constraint evaluations for sequences that have been rejected by filters
        weighted_scores = []
        for idx, constraint in enumerate(self.constraints):
            # Only evaluate sequences that have passed all previous filters
            results = constraint.evaluate(mask=passed)
            
            if constraint.mode == "filter":
                passed = self._apply_filter_constraint(results, passed, constraint, idx, verbose)
            else:
                full_scores = self._apply_scoring_constraint(results, passed, num_sequences, filter_penalty, self.constraint_weights[idx], constraint, idx, verbose)
                weighted_scores.append(full_scores)
        
        # Combine scores across all scoring constraints
        if weighted_scores:
            if operation == "add":
                self.energy_scores = [sum(scores) for scores in zip(*weighted_scores)]
            elif operation == "multiply":
                self.energy_scores = [math.prod(scores) for scores in zip(*weighted_scores)]
        else:
            # No scoring constraints, just use penalty for rejected, 0.0 for accepted
            self.energy_scores = [filter_penalty if not acc else 0.0 for acc in passed]
        
        if verbose:
            print("Final Energy Scores:")
            for i, score in enumerate(self.energy_scores):
                print(f"  Candidate {i}: {score:.4f}{' [REJECTED]' if not passed[i] else ''}")
        
        self._clear_tool_cache()

    def _apply_filter_constraint(
        self, 
        results: List[bool], 
        passed: List[bool], 
        constraint: Constraint, 
        filter_idx: int, 
        verbose: bool
    ) -> List[bool]:
        """
        Apply filter constraint results to update passed status.
        
        Constraints evaluate only candidates that passed previous filters (sparse evaluation).
        This method maps sparse results back to the full candidate array (dense).
        
        Example:
            passed = [True, False, True, False]  # candidates 0 and 2 passed previous filters
            results = [True, False]              # sparse: only evaluated candidates 0 and 2
            returns: [True, False, False, False] # dense: candidate 0 passed, 2 rejected
        """
        passed_before = passed.copy() if verbose else None
        
        sparse_idx = 0
        for i in range(len(passed)):
            if passed[i]:
                passed[i] = results[sparse_idx]
                sparse_idx += 1
        
        if verbose:
            print(f"Filter {filter_idx+1}: {constraint.label}")
            for i in range(len(passed)):
                print(f"  Candidate {i}: {('PASS' if passed[i] else 'REJECT') if passed_before[i] else '[SKIPPED - already rejected]'}")
        
        return passed

    def _apply_scoring_constraint(
        self, 
        results: List[float], 
        passed: List[bool], 
        num_sequences: int, 
        filter_penalty: float, 
        weight: float,
        constraint: Constraint, 
        scoring_idx: int, 
        verbose: bool
    ) -> List[float]:
        """
        Expand sparse scoring results to full array with penalties for rejected candidates.
        
        Scoring constraints evaluate only candidates that passed all filters (sparse evaluation).
        This method expands sparse scores to full array, applying weights to passed candidates
        and assigning filter_penalty to rejected ones.
        
        Example:
            results = [0.8, 0.3]                    # sparse: scores for passed candidates only
            passed = [True, False, True, False]     # candidates 0 and 2 passed filters
            weight = 2.0, filter_penalty = inf
            returns: [1.6, inf, 0.6, inf]           # dense: weighted scores + penalties
        """
        full_scores = []
        sparse_idx = 0
        
        for i in range(num_sequences):
            if passed[i]:
                full_scores.append(results[sparse_idx] * weight)
                sparse_idx += 1
            else:
                full_scores.append(filter_penalty)
        
        if verbose:
            print(f"Constraint {scoring_idx+1}: {constraint.label} (weight={weight})")
            for i, score in enumerate(full_scores):
                print(f"  Candidate {i}: {f'{score:.4f}' if passed[i] else '[REJECTED by filter]'}")
        
        return full_scores

    def _clear_tool_cache(self) -> None:
        """Clear tool cache based on clear_tool_cache configuration."""
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

        Creates independent copies of sequences for both pools. When running multiple
        sequential optimizers, preserves the best sequences from the previous optimizer


        Behavior:
        - If previous optimizer produced sequences: uses up to num_selected best sequences
        - If fewer available than needed: pads with copies of the best sequence
        - If no previous sequences: uses original_sequence for all
        """
        for segment in self.segments:
            if segment.selected_sequences:
                # Take up to num_selected sequences from previous optimizer (already sorted best-first)
                start_sequences = segment.selected_sequences[:self.num_selected]

                # If previous optimizer produced fewer sequences, pad with copies of the best
                if len(start_sequences) < self.num_selected:
                    best_seq = start_sequences[0]
                    start_sequences.extend([copy.deepcopy(best_seq)
                                           for _ in range(self.num_selected - len(start_sequences))])
            elif segment.candidate_sequences:
                # Fallback to candidates if selected pool is empty (shouldn't normally happen)
                start_sequences = segment.candidate_sequences[:self.num_selected]
                if len(start_sequences) < self.num_selected:
                    start_sequences.extend([copy.deepcopy(start_sequences[0])
                                           for _ in range(self.num_selected - len(start_sequences))])
            else:
                # First optimizer - use original_sequence
                start_sequences = [segment.original_sequence] * self.num_selected

            # Create independent copies for selected pool
            segment.selected_sequences = [copy.deepcopy(seq) for seq in start_sequences]

            # Candidates initialized from best sequence (will be mutated by generators)
            segment.candidate_sequences = [copy.deepcopy(start_sequences[0]) for _ in range(self.num_candidates)]
