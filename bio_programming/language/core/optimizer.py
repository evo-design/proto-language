"""
Optimizer base class for the biological programming language.

Base class for iterative optimization algorithms that coordinate multiple
generators and constraints to search for optimal biological sequences.
"""

from __future__ import annotations

import copy
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd
from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache

from proto_language.utils.export import (
    build_batch_results,
    build_candidate_results,
    export_tables,
    flatten_table,
    to_fasta,
)

logger = logging.getLogger(__name__)

from .constraint import Constraint
from .construct import Construct
from .generator import Generator
from .sequence import Sequence


class Optimizer(ABC):
    """
    Base class for optimization algorithms.

    Coordinates multiple generators and constraints to search for optimal
    biological sequences through iterative optimization. Unlike generators
    which modify sequences directly, optimizers orchestrate the search process
    by coordinating generators, evaluating constraints, and making decisions
    about which sequences to keep.

    Pool Initialization:
        ``_initialize_sequence_pools()`` is called during ``__init__()`` and by
        ``Program.run_stage()`` before each subsequent optimizer. It reads from
        ``selected_sequences`` (from previous optimizer) or ``original_sequence``
        and initializes both pools by cycling through source to preserve diversity.

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

    _require_non_empty_constraints: bool = True

    @abstractmethod
    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        num_results: int | None,
        tracking_interval: int,
        track_candidates: bool,
        verbose: bool,
        candidates_per_result: int = 1,
        num_candidates: int | None = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
        custom_logging: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the Optimizer with dual-pool semantics.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            num_results: Number of sequences to select and maintain as results.
                May be None to defer resolution to Program(num_results=N).
            tracking_interval: Save history snapshot and log progress every N steps.
                Step 0 (initial) and the final step are always saved.
            track_candidates: Include per-candidate results in history snapshots.
            verbose: Whether to print detailed progress information.
            candidates_per_result: Number of candidate proposals per result sequence.
                Used to compute num_candidates when deferred.
            num_candidates: Number of candidate proposals to generate per iteration.
                Computed as ``num_results * candidates_per_result`` when None.
            clear_tool_cache: (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
            custom_logging: Optional callback with signature ``(step: int, segments: tuple) -> None``.
                Called at tracked steps only (governed by ``tracking_interval``).
        """
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.num_results = num_results
        self.tracking_interval = tracking_interval
        self.track_candidates = track_candidates
        self.verbose = verbose
        self._candidates_per_result = candidates_per_result
        self.num_candidates = num_candidates
        self.clear_tool_cache = clear_tool_cache
        self.custom_logging = custom_logging
        self.energy_scores: List[float] = []
        self.history: List[Dict[str, Any]] = []
        self._initial_state: Optional[Dict] = None  # Captured on first run() for restart
        self._labels_deduplicated: bool = False

        # Per-candidate tracking (set by score_energy / optimizer-specific logic)
        self._candidate_outcomes: list[str] = []
        self._candidate_energy_scores: list[float] = []

        # Default value for progress tracking (can be overridden by subclasses)
        self.num_steps: int = 1

        # Create program-scoped tool cache
        self.tool_cache = ToolCache()
        _program_tool_cache.set(self.tool_cache)

        self._validate_optimizer()

        if self.num_results is not None:
            self._resolve_num_results(self.num_results)

        logger.debug(f"Optimizer initialized: {self.__class__.__name__}, candidates={num_candidates}, results={num_results}")

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

    def score_energy(
        self,
        operation: Literal["add", "multiply"] = "add",
        filter_penalty: float = float("inf"),
    ) -> None:
        """
        Compute energy scores by combining all constraint evaluation scores on the candidate sequences.

        Filter constraints are evaluated first. Rejected candidates skip subsequent
        constraint evaluations for performance. Sets ``_candidate_outcomes`` with
        "accepted" for passing candidates or the rejecting constraint's label.

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
            logger.info(
                f"Energy Scoring: {len(filters)} filters, {len(scorers)} scoring | "
                f"Formula: energy = {op}(weight_i x constraint_score_i)"
            )

        # Pass 1: Evaluate filter constraints first to skip expensive scoring on rejected candidates.
        self._candidate_outcomes = ["accepted"] * num_sequences
        for idx, constraint in enumerate(filters):
            if self.verbose:
                logger.info(f"Filter {idx+1}: {constraint.label}")
            results = constraint.evaluate(mask=passed, verbose=self.verbose)
            for i, (p, r) in enumerate(zip(passed, results)):
                if p and not r:
                    self._candidate_outcomes[i] = constraint.label
            passed = [p and r for p, r in zip(passed, results)]

        # Pass 2: Score passing candidates (skip rejected candidates for performance)
        all_scores = []
        for idx, constraint in enumerate(scorers):
            if self.verbose:
                logger.info(f"Constraint {idx+1}: {constraint.label}")
            all_scores.append(constraint.evaluate(mask=passed, verbose=self.verbose))

        # Warn if no scoring constraints exist (all are filters)
        if not all_scores:
            identity = "0.0" if operation == "add" else "1.0"
            logger.warning(
                f"All constraints are filters (have threshold set). "
                f"Passing candidates will receive energy score {identity} since there are no scoring constraints."
            )

        # Aggregate scores across all scoring constraints into a single energy score per candidate.
        # NaN propagates through sum/prod operations, resulting in NaN if any constraint is unevaluated.
        if operation == "add":
            self.energy_scores = [sum(s[i] for s in all_scores) for i in range(num_sequences)]
        elif operation == "multiply":
            self.energy_scores = [math.prod(s[i] for s in all_scores) for i in range(num_sequences)]
        else:
            raise ValueError(f"Operation must be 'add' or 'multiply'")

        # Check for inconsistent state
        assert len(self.energy_scores) == num_sequences, \
            ("Inconsistent state: energy scores should have the same length as candidates")

        # NaN signals "not evaluated" and propagates through arithmetic, making bugs visible
        for i, score in enumerate(self.energy_scores):
            if self._candidate_outcomes[i] == "accepted" and math.isnan(score):
                raise RuntimeError(f"Inconsistent state: candidate {i} passed all filters but has NaN score.")

        # Apply filter_penalty to rejected candidates
        self.energy_scores = [
            score if self._candidate_outcomes[i] == "accepted" else filter_penalty
            for i, score in enumerate(self.energy_scores)
        ]

        if self.verbose:
            logger.info("Final Energy Scores:")
            for i, score in enumerate(self.energy_scores):
                outcome = self._candidate_outcomes[i]
                if outcome == "accepted":
                    logger.info(f"  Candidate {i}: {score:.4f} [ACCEPTED]")
                else:
                    logger.info(f"  Candidate {i}: {score:.4f} [REJECTED by {outcome}]")

        # Snapshot candidate energies before optimizers truncate/swap energy_scores
        self._candidate_energy_scores = list(self.energy_scores)

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
        Validate optimizer configuration before execution.

        Checks:
            1. Non-empty lists: Constructs, generators, and constraints must be provided.
            2. Type validation: All objects must have correct types.
            3. Structure validation: Constructs have segments, generators/constraints are assigned.
            4. No duplicate instances: Each generator/constraint instance can only appear once.
            5. Unique constraint labels: Required for metadata namespacing.
            6. Valid constraint inputs: Constraints can only reference populated segments.

        Raises:
            ValueError: If user inputs are invalid (empty lists, duplicates, etc.).
            TypeError: If objects have incorrect types.
            RuntimeError: If state is invalid (unassigned segments, invalid references).
        """
        # 1. Non-empty lists
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        if not self.generators:
            raise ValueError("Generators list cannot be empty")
        if self._require_non_empty_constraints and not self.constraints:
            raise ValueError("Constraints list cannot be empty")

        # 2. Type validation
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise TypeError(f"Construct {i} has type {type(construct)}, expected Construct")
        for i, generator in enumerate(self.generators):
            if not isinstance(generator, Generator):
                raise TypeError(f"Generator {i} has type {type(generator)}, expected Generator")
        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise TypeError(f"Constraint {i} has type {type(constraint)}, expected Constraint")

        # 3. Structure validation
        for i, construct in enumerate(self.constructs):
            if not construct.segments:
                raise ValueError(f"Construct {i} has no segments")

        assigned_segments: set = set()
        for i, gen in enumerate(self.generators):
            if not gen._assigned_segment:
                raise RuntimeError(f"Generator {i} ({gen.__class__.__name__}) has no segment assigned")
            assigned_segments.add(gen._assigned_segment)

        for i, con in enumerate(self.constraints):
            if not con.inputs:
                raise RuntimeError(f"Constraint {i} has no input segment(s) assigned")

        # 4. No duplicate instances
        seen_gen_ids: set[int] = set()
        for gen in self.generators:
            if id(gen) in seen_gen_ids:
                raise ValueError(f"Generator '{gen.__class__.__name__}' appears multiple times. Each instance can only be used once.")
            seen_gen_ids.add(id(gen))

        seen_con_ids: set[int] = set()
        for con in self.constraints:
            if id(con) in seen_con_ids:
                raise ValueError(f"Constraint '{con.label}' appears multiple times. Each instance can only be used once.")
            seen_con_ids.add(id(con))

        # 5. Ensure unique constraint labels per segment (required for metadata namespacing)
        self._deduplicate_constraint_labels()

        # 6. Valid constraint inputs
        # Constraints can only reference segments that have sequences or a generator assigned.
        for constraint in self.constraints:
            for segment in constraint.inputs:
                if not segment.populated_sequences and segment not in assigned_segments:
                    raise RuntimeError(
                        f"Constraint '{constraint.label}' references segment '{segment.label or 'unlabeled'}' "
                        "which has no populated sequence and no generator assigned."
                    )

    def _deduplicate_constraint_labels(self) -> None:
        """Ensure unique constraint labels per segment for metadata namespacing.

        Only runs once to prevent label accumulation on repeated validation
        calls (e.g. constraint_1_1_1...). Extracted as a standalone method so
        subclasses with custom ``_validate_optimizer()`` can call it directly.
        """
        if not self._labels_deduplicated:
            segment_label_counts: Dict[tuple, int] = {}  # (base_label, segment_id) -> count
            for constraint in self.constraints:
                # Capture label before any renaming so multi-segment constraints
                # use a stable key across all their segments.
                base_label = constraint.label
                # Track unique segments for this constraint to handle same segment appearing multiple times in inputs (e.g. symmetric proteins)
                seen_segments_for_constraint: set[int] = set()
                for segment in constraint.inputs:
                    seg_id = id(segment)
                    if seg_id in seen_segments_for_constraint:
                        continue  # Skip duplicate segments within same constraint
                    seen_segments_for_constraint.add(seg_id)

                    key = (base_label, seg_id)
                    if key in segment_label_counts:
                        # Collision detected, append counter to this constraint label
                        segment_label_counts[key] += 1
                        constraint.label = f"{base_label}_{segment_label_counts[key]}"
                    else:
                        segment_label_counts[key] = 0
            self._labels_deduplicated = True

    def _validate_target_segment(self, target_segment) -> None:
        """Validate target_segment is in constructs."""
        if target_segment not in self.segments:
            raise ValueError(
                f"target_segment '{target_segment.label or 'unlabeled'}' "
                "is not in any of the provided constructs"
            )

    def _initialize_sequence_pools(self) -> None:
        """Initialize sequence pools from previous optimizer's results or original sequence.

        Source priority:
        1. ``segment.selected_sequences`` (if populated) - from previous optimizer
        2. ``[segment.original_sequence]`` (if first optimizer) - falls back to original

        Both ``selected_sequences`` and ``candidate_sequences`` are initialized by cycling
        through source to preserve diversity when pool sizes differ.

        Example: source=[A,B,C], num_results=5 → [A,B,C,A,B]
        """
        # Determine source length from first segment (all segments have same length)
        source_len = len(self.segments[0].selected_sequences or [self.segments[0].original_sequence])

        # Log truncation or expansion with optimizer name for context
        optimizer_name = self.__class__.__name__
        if source_len > self.num_results:
            logger.info(
                f"Handoff to {optimizer_name}: Truncating {source_len} sequences from result of previous optimizer to {self.num_results} "
                f"sequences as starting sequences for current optimizer (keeping first {self.num_results})"
            )
        elif source_len < self.num_results:
            logger.warning(
                f"Handoff to {optimizer_name}: Expanding sequences from {source_len} sequences from previous optimizer to {self.num_results} "
                f"sequences by cycling through the existing {source_len} sequences and duplicating until {self.num_results} starting sequences for this optimizer are populated."
            )
        else:
            logger.info(f"Handoff to {optimizer_name}: Starting sequences for current optimizer are populated by {source_len} sequences from previous optimizer.")

        for segment in self.segments:
            # Source: previous optimizer's results or original sequence
            source = segment.selected_sequences or [segment.original_sequence]

            # Selected pool: cycle through source to preserve diversity
            segment.selected_sequences = [
                copy.deepcopy(source[i % len(source)])
                for i in range(self.num_results)
            ]

            # Candidate pool: cycle through source to preserve diversity
            segment.candidate_sequences = [
                copy.deepcopy(source[i % len(source)])
                for i in range(self.num_candidates)
            ]

    def _resolve_num_results(self, num_results: int) -> None:
        """Resolve num_results and initialize sequence pools.

        Called in two cases:
        1. During __init__ when config.num_results is set directly.
        2. By Program.__init__ to flow program-level num_results to optimizers
           whose config.num_results was left as None.
        """
        if num_results < 1:
            raise ValueError(f"num_results must be >= 1, got {num_results}")
        self.num_results = num_results
        if hasattr(self, "config"):
            self.config.num_results = num_results
        if self.num_candidates is None:
            self.num_candidates = num_results * self._candidates_per_result
        if self.num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {self.num_candidates}")
        self.energy_scores = [float("inf")] * self.num_candidates
        self._initialize_sequence_pools()

    def _prepare_run(self) -> None:
        """Call at start of run(). Validates state, captures on first run, restores on subsequent."""
        if self.num_results is None:
            raise RuntimeError("num_results must be set. Set it via the optimizer config or use Program(num_results=...).")
        if self._initial_state is None:
            self._capture_initial_state()
        else:
            self._restore_initial_state()

    def _capture_initial_state(self) -> None:
        """Capture current segment and optimizer state via serialization."""
        self._initial_state = {
            'segments': [
                {
                    'selected': [seq.to_dict() for seq in seg.selected_sequences],
                    'candidates': [seq.to_dict() for seq in seg.candidate_sequences],
                }
                for seg in self.segments
            ],
            'energy_scores': self.energy_scores.copy(),
        }

    def _restore_initial_state(self) -> None:
        """Restore to captured state via deserialization."""
        for i, seg in enumerate(self.segments):
            state = self._initial_state['segments'][i]
            seg.selected_sequences = [Sequence.from_dict(s) for s in state['selected']]
            seg.candidate_sequences = [Sequence.from_dict(s) for s in state['candidates']]
        self.energy_scores = self._initial_state['energy_scores'].copy()
        self._candidate_outcomes = []
        self._candidate_energy_scores = []
        self._labels_deduplicated = False
        self.history = []

    def _save_progress_snapshot(self, time_step: int) -> None:
        """Save current optimization state to history.

        Validates internal consistency: all segments have the same number of
        ``selected_sequences`` and ``energy_scores`` matches that count.
        Allows partial snapshots (e.g. TopK mid-run with fewer than k selected).
        """
        expected_len = len(self.segments[0].selected_sequences)
        for segment in self.segments:
            if len(segment.selected_sequences) != expected_len:
                raise RuntimeError(f"selected_sequences length mismatch: segment '{segment.label or 'unlabeled'}' has {len(segment.selected_sequences)}, expected {expected_len}")
        if len(self.energy_scores) != expected_len:
            raise RuntimeError(f"energy_scores has length {len(self.energy_scores)}, expected {expected_len} (matching selected_sequences)")

        result = build_batch_results(self.constructs, self.energy_scores)
        result["time_step"] = time_step

        if self.track_candidates and self._candidate_outcomes:
            result["candidate_results"] = build_candidate_results(self.constructs, self._candidate_outcomes, self._candidate_energy_scores)

        self.history.append(result)

    # =========================================================================
    # Export
    # =========================================================================

    def export(
        self,
        path: Path | str = "./results",
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        table: Literal[
            "sequences", "constraints", "constructs", "optimization"
        ] | None = None,
        segments: set[str] | None = None,
        batch_indices: set[int] | None = None,
        constraints: set[str] | None = None,
        include_candidates: bool = False,
    ) -> Path:
        """Export results to files.

        Without *table*: writes all 4 tables (sequences, constraints,
        constructs, optimization).  csv/tsv/json produce a directory with one
        file per table; xlsx produces a single workbook with 4 sheets.

        With *table*: writes a single file to *path*.

        Args:
            path: Output directory (all tables) or file path (single table / xlsx).
            format: ``"csv"`` | ``"tsv"`` | ``"json"`` | ``"xlsx"``.
            table: Single table name, or None for all.
            segments: Only include these segment labels.
            batch_indices: Only include these batch indices.
            constraints: Only include these constraint labels (constraints table only).
            include_candidates: Include candidate rows (optimization table only).
        """
        batch_results = build_batch_results(self.constructs, self.energy_scores)
        filters = dict(
            segments=segments,
            batch_indices=batch_indices,
            constraints=constraints,
            include_candidates=include_candidates,
        )
        return export_tables(
            lambda t: flatten_table(t, batch_results, self.history, **filters),
            path, format, table,
        )

    def to_dataframe(
        self,
        table: Literal["sequences", "constraints", "constructs", "optimization"] = "sequences",
        segments: set[str] | None = None,
        constraints: set[str] | None = None,
        batch_indices: set[int] | None = None,
        include_candidates: bool = False,
    ) -> pd.DataFrame:
        """Get a result table as a pandas DataFrame.

        Accepts the same filter arguments as :meth:`export`.
        """
        return pd.DataFrame(flatten_table(
            table,
            build_batch_results(self.constructs, self.energy_scores),
            self.history,
            segments=segments,
            batch_indices=batch_indices,
            constraints=constraints,
            include_candidates=include_candidates,
        ))

    def to_fasta(
        self,
        path: Path | str | None = None,
        segments: set[str] | None = None,
        batch_indices: set[int] | None = None,
        header_format: str = "{construct}_{segment}_batch{batch_idx}",
    ) -> str:
        """Export sequences in FASTA format.

        Args:
            path: Output file path. If None, returns string only.
            header_format: Format string for headers. Available fields:
                construct, segment, batch_idx, energy_score, sequence_type.

        Returns:
            FASTA-formatted string.
        """
        return to_fasta(
            build_batch_results(self.constructs, self.energy_scores),
            segments=segments,
            batch_indices=batch_indices,
            header_format=header_format,
            output=Path(path) if path else None,
        )
