"""proto_language/language/core/program.py"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from proto_language.utils.export import (
    build_results,
    export_tables,
    flatten_table,
    to_fasta,
)

from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class Program:
    """
    Programs represent user-defined biological designs.

    This class supports sequential execution of multiple optimizers, where each
    optimizer builds on the results of the previous one. All optimizers must
    share the same construct objects to ensure state persistence.

    Optimizer Handoff Contract:
        When running multiple optimizers sequentially, the Program ensures proper
        state transfer between stages:

        **After each optimizer completes:**

        Optimizers are responsible for their own sorting. TopK keeps
        ``result_sequences`` sorted by energy throughout its run. Other
        optimizers' natural ordering is preserved as-is.

        **Before the next optimizer runs:**

        1. ``_initialize_sequence_pools()`` reads from ``result_sequences``
           (or ``original_sequence`` if first optimizer)
        2. Both ``result_sequences`` and ``proposal_sequences`` are initialized by
           cycling through source to preserve diversity when pool sizes differ
           (e.g., source=[A,B,C], num_results=5 -> [A,B,C,A,B])

        **Optimizer-specific behavior:**

        - **TopK**: Clears ``result_sequences`` and repopulates dynamically during run
          (always sorted by energy)
        - **MCMC**: Uses ``result_sequences`` as parallel trajectories, overwrites
          ``proposal_sequences`` each step via ``_populate_proposal_sequences()``
        - **CyclingOptimizer**: Works directly on ``proposal_sequences``
        - **BeamSearch**: Ignores previous state entirely, starts fresh from configured prompt

    Examples:
        Sequential optimization with TopK followed by MCMC:
        >>> from proto_language.language.optimizer import (
        ...     TopKOptimizer, TopKOptimizerConfig,
        ...     MCMCOptimizer, MCMCOptimizerConfig
        ... )
        >>>
        >>> # First optimizer: broad exploration with TopK
        >>> optimizer_1 = TopKOptimizer(
        ...     constructs=[construct],
        ...     generators=[broad_mutation_gen],
        ...     constraints=[gc_constraint_1],
        ...     config=TopKOptimizerConfig(num_samples=100, num_results=3),
        ... )
        >>>
        >>> # Second optimizer: fine-tuning with MCMC
        >>> optimizer_2 = MCMCOptimizer(
        ...     constructs=[construct],
        ...     generators=[fine_mutation_gen],
        ...     constraints=[gc_constraint_2],
        ...     config=MCMCOptimizerConfig(num_steps=100),
        ... )
        >>>
        >>> # Create program with sequential optimizers
        >>> program = Program(optimizers=[optimizer_1, optimizer_2], num_results=3)
        >>> program.run()
        >>>
        >>> # Access results from final optimizer
        >>> final_sequences = program.constructs[0].joined_sequences
        >>> final_energies = program.energy_scores
        >>>
        >>> # Access history from each optimizer
        >>> topk_history = program.optimizers[0].history
        >>> mcmc_history = program.optimizers[1].history
    """

    def __init__(
        self,
        optimizers: List[Optimizer],
        num_results: int,
        verbose: bool = False,
        compute: ToolPool | None = None,
    ) -> None:
        """
        Initialize a Program with a list of optimizers to run sequentially.

        Args:
            optimizers (list[Optimizer]): List of Optimizer objects to run in sequence. Each optimizer
                       builds on the results of the previous one. All optimizers must
                       share the same construct objects (by identity).
            num_results (int): Number of result sequences to produce by default. Flows down to any
                optimizer whose config field is not set. Optimizer-level config
                always takes priority.
            verbose (bool): If True, print detailed energy score calculations for each constraint
                     for all optimizers.
            compute (ToolPool | None): Compute backend: 'local' or 'cloud'.

        Raises:
            ValueError: If optimizers list is empty or if optimizers don't share
                       the same construct objects.
        """
        if not optimizers:
            raise ValueError("optimizers list cannot be empty")

        if compute is None:
            from proto_tools.tools.tool_registry import ToolRegistry
            from proto_tools.utils.device import number_of_available_gpus
            from proto_tools.utils.tool_pool import ToolPool

            has_backend = getattr(ToolRegistry, '_dispatch_configured', False)
            has_gpus = number_of_available_gpus() > 0

            if has_backend and has_gpus:
                compute = ToolPool()
            elif has_backend:
                from contextlib import nullcontext
                logger.debug(
                    "No local GPUs; external dispatch configured — "
                    "GPU tools will route via _try_dispatch."
                )
                compute = nullcontext()
            elif has_gpus:
                compute = ToolPool()
            else:
                from contextlib import nullcontext

                logger.warning(
                    "No GPUs detected and no external dispatch configured. "
                    "Tools will execute inline without parallelism."
                )
                compute = nullcontext()

        self.compute = compute

        self.optimizers = optimizers
        self.num_results = num_results

        # Flow num_results to optimizers that don't have it set. Optimizers can optionally override
        for opt in self.optimizers:
            if opt.num_results is None:
                opt._resolve_num_results(self.num_results)
            elif opt.num_results != self.num_results:
                logger.warning(f"{opt.__class__.__name__} num_results={opt.num_results} Overrides program num_results={self.num_results}")

        # If top level verbosity is true, force verbosity in all optimizers.
        self.verbose = verbose
        if self.verbose:
            for optimizer in self.optimizers:
                optimizer.verbose = self.verbose

        # Extract constructs from first optimizer
        self.constructs = optimizers[0].constructs

        # Auto-label constructs and tag segments with their construct label for metadata tracking
        for i, construct in enumerate(self.constructs):
            if construct.label is None:
                construct.label = f"construct_{i}"
            for segment in construct.segments:
                segment.construct_label = construct.label

        self.current_stage = 0
        self._stage_results: List[Dict] = []
        self._validate_program()
        logger.debug(f"Program initialized: optimizers={len(self.optimizers)}, constructs={len(self.constructs)}")

    @property
    def energy_scores(self) -> List[float]:
        """
        Get energy scores from the final optimizer.

        Returns:
            list[float]: List of energy scores where lower values indicate better solutions.

        Raises:
            RuntimeError: If run() hasn't been called yet.
        """
        if not hasattr(self.optimizers[-1], "energy_scores"):
            raise RuntimeError("Optimization not complete. Call run() first.")
        return self.optimizers[-1].energy_scores

    def _validate_program(self) -> None:
        """
        Validate program configuration before execution.

        Checks:
            1. Resolved num_results: All optimizers must have num_results set.
            2. Construct identity: All optimizers must share the same construct objects
               (by identity, not equality) to ensure state persists across stages.
            3. Unique labels: Construct labels must be unique for unambiguous referencing.
            4. Segment population: Every segment must have either an input sequence or
               a generator assigned in at least one optimizer.
            5. Segment uniqueness: Segments cannot be reused across constructs.
            6. Instance isolation: Generators and constraints cannot be reused across
               optimizers to prevent shared mutable state bugs.

        Raises:
            ValueError: If any validation check fails.
        """
        # 1. Validate all optimizers have resolved num_results.
        for i, opt in enumerate(self.optimizers):
            if opt.num_results is None:
                raise ValueError(f"Optimizer {i} ({opt.__class__.__name__}) has no num_results. Set it via the optimizer's config or pass num_results to Program().")

        reference_constructs = self.optimizers[0].constructs

        # 2. Validate construct identity across optimizers
        # All optimizers must reference the exact same construct objects so results
        # from one stage automatically propagate to subsequent stages.
        for i, optimizer in enumerate(self.optimizers[1:], start=1):
            if len(optimizer.constructs) != len(reference_constructs):
                raise ValueError(
                    f"Optimizer {i} has {len(optimizer.constructs)} constructs, "
                    f"but optimizer 0 has {len(reference_constructs)}. "
                    "All optimizers must share the same construct objects."
                )
            for j, (construct, ref_construct) in enumerate(zip(optimizer.constructs, reference_constructs)):
                if construct is not ref_construct:
                    raise ValueError(
                        f"Optimizer {i} construct {j} is not the same object as "
                        f"optimizer 0 construct {j}. All optimizers must share the "
                        "same construct objects (by identity) for state persistence."
                    )

        # 3. Validate unique construct labels
        construct_labels = [c.label for c in self.constructs]
        if len(construct_labels) != len(set(construct_labels)):
            duplicates = [l for l in construct_labels if construct_labels.count(l) > 1]
            raise ValueError(f"Construct labels must be unique. Duplicates: {set(duplicates)}")

        # 4. Validate no segment reuse across constructs
        seen_segments: dict[int, str] = {}
        for construct in self.constructs:
            for segment in construct.segments:
                prev_construct = seen_segments.get(id(segment))
                if prev_construct is not None:
                    raise ValueError(
                        f"Segment '{segment.label}' is used in multiple constructs: "
                        f"'{prev_construct}' and '{construct.label}'. "
                        "Each segment instance can only belong to one construct."
                    )
                seen_segments[id(segment)] = construct.label

        # 5. Validate all segments will be populated
        # A segment must have either an input sequence or a generator in some optimizer.
        generator_segments = {
            gen._assigned_segment
            for opt in self.optimizers
            for gen in opt.generators
            if gen._assigned_segment
        }
        for construct in self.constructs:
            for segment in construct.segments:
                if segment not in generator_segments and not segment.populated_sequences:
                    raise ValueError(
                        f"Segment '{segment.label or 'unlabeled'}' is never populated. "
                        "It has no input sequence and no generator assigned in any optimizer."
                    )

        # 6. Validate no generator/constraint reuse across optimizers
        # Each optimizer needs its own instances to avoid shared mutable state issues.
        if len(self.optimizers) > 1:
            seen_generators: Dict[int, int] = {}
            seen_constraints: Dict[int, int] = {}

            for opt_idx, optimizer in enumerate(self.optimizers):
                for gen in optimizer.generators:
                    prev_idx = seen_generators.get(id(gen))
                    if prev_idx is not None:
                        raise ValueError(
                            f"Generator '{gen.__class__.__name__}' reused across optimizer {prev_idx} and {opt_idx}. Each generator instance can only be used once."
                        )
                    seen_generators[id(gen)] = opt_idx

                for con in optimizer.constraints:
                    prev_idx = seen_constraints.get(id(con))
                    if prev_idx is not None:
                        raise ValueError(
                            f"Constraint '{con.label}' reused across optimizer {prev_idx} and {opt_idx}. Each constraint instance can only be used once."
                        )
                    seen_constraints[id(con)] = opt_idx

    def _log_stage_results(self, stage_index: int, results: list) -> None:
        """Log results for a completed optimization stage."""
        logger.debug(f"Final state for optimizer {stage_index + 1}:")
        for result in results:
            energy = result['energy_score']
            energy_str = f"{energy:.4f}" if energy is not None else "None"
            logger.debug(f"  [{result['result_idx']}] energy={energy_str}")
            for construct in result["constructs"]:
                seqs = [seg["sequence"] for seg in construct["segments"]]
                logger.debug(f"    {construct['label']}: {' | '.join(seqs)}")

    def run(self) -> None:
        """
        Execute the sequence optimization process for all optimizers sequentially.

        Each optimizer builds on the results of the previous one. State automatically
        persists between optimizers through the shared construct objects.
        """
        # Reset stage tracking for fresh run
        self.current_stage = 0
        self._stage_results = []

        # Restore initial state on re-run (first optimizer captures pre-pipeline state)
        if self.optimizers[0]._initial_state is not None:
            self.optimizers[0]._restore_initial_state()
            # Clear stale initial states from subsequent optimizers so they
            # recapture fresh state from this run (not stale first-run state).
            for opt in self.optimizers[1:]:
                opt._initial_state = None

        # Run all stages sequentially
        logger.debug(f"Program.run: starting {len(self.optimizers)} optimization stages")
        with self._enter_compute():
            for optimizer_stage_idx in range(len(self.optimizers)):
                self.run_stage(optimizer_stage_idx)

    def run_stage(self, stage_index: int) -> None:
        """
        Execute a specific optimization stage.

        Allows running optimizers one at a time with inspection of results between
        stages. Each stage builds on results from previous stages through shared
        construct objects. Results are accessible via `get_stage_results()`.

        You can re-run any previously completed stage, which resets the pipeline
        to that point and invalidates all subsequent stages.

        Args:
            stage_index (int): Zero-based index of the optimizer stage to run.

        Raises:
            IndexError: If stage_index is out of range.
            RuntimeError: If attempting to skip forward (can only run current or previous stages).

        Example:
            >>> program = Program(optimizers=[opt1, opt2], num_results=3)
            >>> program.run_stage(0)  # Run first optimizer
            >>> results = program.get_stage_results(0)  # Access results
            >>> program.run_stage(1)  # Run second optimizer
        """
        with self._enter_compute():
            self._validate_program()
            if stage_index < 0 or stage_index >= len(self.optimizers):
                raise IndexError(f"Stage index {stage_index} out of range (0-{len(self.optimizers)-1}).")
            if stage_index > self.current_stage:
                raise RuntimeError(f"Cannot skip to stage {stage_index}. Current stage is {self.current_stage}.")

            # Re-running a previous stage: restore state from that stage's optimizer
            if stage_index < self.current_stage:
                # Use the target stage's initial state (captured before it ran)
                self.optimizers[stage_index]._restore_initial_state()
                self._stage_results = self._stage_results[:stage_index]
                # Clear stale initial states from subsequent optimizers
                for opt in self.optimizers[stage_index + 1:]:
                    opt._initial_state = None

            optimizer = self.optimizers[stage_index]
            logger.debug(f"Program.run_stage: stage={stage_index}, optimizer={optimizer.__class__.__name__}")
            optimizer._initialize_sequence_pools()

            # Clear stale constraint metadata from previous stages
            self._clear_sequence_metadata()

            optimizer.run()

            stage_result = self.extract_results(optimizer.energy_scores)

            if self.verbose:
                self._log_stage_results(stage_index, stage_result["results"])

            self._stage_results.append(stage_result)
            self.current_stage = stage_index + 1

    @contextmanager
    def _enter_compute(self):
        """Enter compute context if not already active, otherwise no-op.

        Checks _active_pool ContextVar to avoid double-entry when
        run_stage() is called from run() (which already entered the context).
        """
        from proto_tools.utils.tool_pool import _active_pool

        if _active_pool.get() is not None:
            yield
        else:
            with self.compute:
                yield

    def get_stage_results(self, stage_index: int) -> Dict[str, Any]:
        """Get results from a specific optimization stage."""
        if stage_index < 0 or stage_index >= len(self._stage_results):
            raise IndexError(f"Stage {stage_index} not available. Only {len(self._stage_results)} stage(s) run.")
        return self._stage_results[stage_index]

    def _clear_sequence_metadata(self) -> None:
        """Clear constraint results from all sequences in all segments.

        Called at the start of each optimization stage to prevent stale constraints
        from previous stages persisting into new stages.
        """
        for construct in self.constructs:
            for segment in construct.segments:
                for seq in segment.result_sequences:
                    seq._constraints_metadata = {}
                for seq in segment.proposal_sequences:
                    seq._constraints_metadata = {}

    def extract_results(self, energy_scores: List[float]) -> Dict[str, Any]:
        """Extract results from constructs."""
        return build_results(self.constructs, energy_scores)

    def serialize_state(self) -> Dict:
        """
        Serialize minimal program state for persistence between stages.

        Only stores what's needed to reconstruct Sequence objects:
        sequence, sequence_type, valid_chars. Constraint metadata is
        excluded since it will be re-evaluated in subsequent stages.
        """
        segment_states = []
        for construct in self.constructs:
            for segment in construct.segments:
                segment_state = {
                    "result_sequences": [
                        {
                            "sequence": seq.sequence,
                            "sequence_type": seq.sequence_type,
                            "valid_chars": list(seq.valid_chars) if seq.valid_chars else None,
                        }
                        for seq in segment.result_sequences
                    ],
                }
                segment_states.append(segment_state)

        return {"segments": segment_states}

    def restore_state(self, state: Dict, stage_index: int = None) -> None:
        """
        Restore program state from serialized data.

        Args:
            state (dict): Dictionary returned by serialize_state()
            stage_index (int): Optional stage index to set current_stage to (for resuming from a specific stage)

        Raises:
            ValueError: If state doesn't match program structure
        """
        from .sequence import Sequence

        all_segments = [seg for construct in self.constructs for seg in construct.segments]

        if len(all_segments) != len(state["segments"]):
            raise ValueError(
                f"State mismatch: program has {len(all_segments)} segments "
                f"but state has {len(state['segments'])} segments"
            )

        for segment, segment_state in zip(all_segments, state["segments"]):
            segment.result_sequences = [
                Sequence(
                    sequence=seq_data["sequence"],
                    sequence_type=seq_data["sequence_type"],
                    valid_chars=set(seq_data["valid_chars"]) if seq_data.get("valid_chars") else None,
                )
                for seq_data in segment_state["result_sequences"]
            ]

        # Update current_stage if specified (for resuming multi-stage optimization)
        if stage_index is not None:
            self.current_stage = stage_index

    # =========================================================================
    # Export
    # =========================================================================

    def _collect_history(self, stage: Optional[int] = None) -> List[Dict[str, Any]]:
        if stage is not None:
            if stage < 0 or stage >= len(self.optimizers):
                raise IndexError(
                    f"Stage {stage} out of range "
                    f"(program has {len(self.optimizers)} optimizers)"
                )
            return list(self.optimizers[stage].history)
        history = []
        for stage_idx, optimizer in enumerate(self.optimizers):
            for entry in optimizer.history:
                annotated = dict(entry)
                annotated["stage"] = stage_idx
                history.append(annotated)
        return history

    def _results_for_stage(self, stage: int | None = None) -> dict[str, Any]:
        """Return results for the given stage (or current final state)."""
        if stage is not None:
            return self.get_stage_results(stage)
        return self.extract_results(self.energy_scores)

    def export(
        self,
        path: Path | str = "./results",
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        table: Literal["sequences", "constraints", "constructs", "optimization"] | None = None,
        stage: int | None = None,
        segments: set[str] | None = None,
        result_indices: set[int] | None = None,
        constraints: set[str] | None = None,
        include_proposals: bool = False,
    ) -> Path:
        """Export results to files.

        Without *table*: writes all 4 tables (sequences, constraints,
        constructs, optimization).  csv/tsv/json produce a directory with one
        file per table; xlsx produces a single workbook with 4 sheets.

        With *table*: writes a single file to *path*.

        Args:
            path (Path | str): Output directory (all tables) or file path (single table / xlsx).
            format (Literal['csv', 'tsv', 'json', 'xlsx']): ``"csv"`` | ``"tsv"`` | ``"json"`` | ``"xlsx"``.
            table (Literal['sequences', 'constraints', 'constructs', 'optimization'] | None): Single table name, or None for all.
            stage (int | None): Filter to this optimizer stage index.
            segments (set[str] | None): Only include these segment labels.
            result_indices (set[int] | None): Only include these result indices.
            constraints (set[str] | None): Only include these constraint labels (constraints table only).
            include_proposals (bool): Include proposal rows (optimization table only).
        """
        results = self._results_for_stage(stage)
        history = self._collect_history(stage)
        filters = dict(
            segments=segments,
            result_indices=result_indices,
            constraints=constraints,
            include_proposals=include_proposals,
        )
        return export_tables(
            lambda t: flatten_table(t, results, history, **filters),
            path, format, table,
        )

    def to_dataframe(
        self,
        table: Literal[
            "sequences", "constraints", "constructs", "optimization"
        ] = "sequences",
        stage: int | None = None,
        segments: set[str] | None = None,
        constraints: set[str] | None = None,
        result_indices: set[int] | None = None,
        include_proposals: bool = False,
    ) -> pd.DataFrame:
        """Get a result table as a pandas DataFrame.

        Accepts the same filter arguments as :meth:`export`.

        Args:
            table (Literal['sequences', 'constraints', 'constructs', 'optimization']): Output format: 'wide' for one column per metric, 'long' for melted rows.
            stage (int | None): Zero-based optimizer stage index to export from.
            segments (set[str] | None): Subset of segment IDs to include, or None for all.
            constraints (set[str] | None): Subset of constraint keys to include, or None for all.
            result_indices (set[int] | None): Indices of specific results to include, or None for all.
            include_proposals (bool): Whether to include proposal sequences alongside accepted results.
        """
        return pd.DataFrame(flatten_table(
            table,
            self._results_for_stage(stage),
            self._collect_history(stage),
            segments=segments,
            result_indices=result_indices,
            constraints=constraints,
            include_proposals=include_proposals,
        ))

    def to_fasta(
        self,
        path: Path | str | None = None,
        stage: int | None = None,
        segments: set[str] | None = None,
        result_indices: set[int] | None = None,
        header_format: str = "{construct}_{segment}_result{result_idx}",
    ) -> str:
        """Export sequences in FASTA format.

        Args:
            path (Path | str | None): Output file path. If None, returns string only.
            header_format (str): Format string for headers. Available fields:
                construct, segment, result_idx, energy_score, sequence_type.
            stage (int | None): Zero-based optimizer stage index to export from.
            segments (set[str] | None): Subset of segment IDs to include, or None for all.
            result_indices (set[int] | None): Indices of specific results to include, or None for all.

        Returns:
            str: FASTA-formatted string.
        """
        return to_fasta(
            self._results_for_stage(stage),
            segments=segments,
            result_indices=result_indices,
            header_format=header_format,
            output=Path(path) if path else None,
        )
