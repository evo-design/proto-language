from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

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

        1. Program sorts ``selected_sequences`` by ``energy_scores`` (best/lowest first)
        2. Sorting is skipped with a warning if all scores are inf (e.g., CyclingOptimizer
           without constraints) - sequences remain in their original order

        **Before the next optimizer runs:**

        1. ``_initialize_sequence_pools()`` reads from sorted ``selected_sequences``
           (or ``original_sequence`` if first optimizer)
        2. Both ``selected_sequences`` and ``candidate_sequences`` are initialized by
           cycling through source to preserve diversity when pool sizes differ
           (e.g., source=[A,B,C], num_selected=5 -> [A,B,C,A,B])

        **Optimizer-specific behavior:**

        - **TopK**: Clears ``selected_sequences`` and repopulates dynamically during run
        - **MCMC**: Uses ``selected_sequences`` as parallel trajectories, overwrites
          ``candidate_sequences`` each step via ``_populate_candidate_sequences()``
        - **CyclingOptimizer**: Works directly on ``candidate_sequences``
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
        ...     config=TopKOptimizerConfig(num_samples=100, k=3),
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
        >>> program = Program(optimizers=[optimizer_1, optimizer_2])
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
        verbose: bool = False,
    ) -> None:
        """
        Initialize a Program with a list of optimizers to run sequentially.

        Args:
            optimizers: List of Optimizer objects to run in sequence. Each optimizer
                       builds on the results of the previous one. All optimizers must
                       share the same construct objects (by identity).
            verbose: If True, print detailed energy score calculations for each constraint
                     for all optimizers.

        Raises:
            ValueError: If optimizers list is empty or if optimizers don't share
                       the same construct objects.
        """
        if not optimizers:
            raise ValueError("optimizers list cannot be empty")

        self.optimizers = optimizers

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
            List of energy scores where lower values indicate better solutions.

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
            1. Construct identity: All optimizers must share the same construct objects
               (by identity, not equality) to ensure state persists across stages.
            2. Unique labels: Construct labels must be unique for unambiguous referencing.
            3. Segment population: Every segment must have either an input sequence or
               a generator assigned in at least one optimizer.
            4. Instance isolation: Generators and constraints cannot be reused across
               optimizers to prevent shared mutable state bugs.

        Raises:
            ValueError: If any validation check fails.
        """
        reference_constructs = self.optimizers[0].constructs

        # 1. Validate construct identity across optimizers
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

        # 2. Validate unique construct labels
        construct_labels = [c.label for c in self.constructs]
        if len(construct_labels) != len(set(construct_labels)):
            duplicates = [l for l in construct_labels if construct_labels.count(l) > 1]
            raise ValueError(f"Construct labels must be unique. Duplicates: {set(duplicates)}")

        # 3. Validate no segment reuse across constructs
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

        # 4. Validate all segments will be populated
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

        # 5. Validate no generator/constraint reuse across optimizers
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

    def _log_stage_results(self, stage_index: int, batch_results: list) -> None:
        """Log results for a completed optimization stage."""
        logger.debug(f"Final state for optimizer {stage_index + 1}:")
        for result in batch_results:
            energy = result['energy_score']
            energy_str = f"{energy:.4f}" if energy is not None else "None"
            logger.debug(f"  [{result['batch_idx']}] energy={energy_str}")
            for construct in result["constructs"]:
                seqs = [seg["sequence"] for seg in construct["segments"]]
                logger.debug(f"    {construct['label']}: {' | '.join(seqs)}")

    def _sort_sequences_by_energy(self, optimizer: Optimizer) -> None:
        """Sort selected_sequences by energy_scores (best/lowest first). NaN/inf sort to end."""
        if all(math.isinf(s) or math.isnan(s) for s in optimizer.energy_scores):
            logger.warning(f"Skipping sort for {optimizer.__class__.__name__}: all energy_scores are inf/NaN")
            return

        # Sort indices by energy (lowest/best first), NaN/inf sort to end
        def sort_key(i):
            e = optimizer.energy_scores[i]
            return (math.isnan(e) or math.isinf(e), e)

        indices = sorted(range(optimizer.num_selected), key=sort_key)
        for segment in optimizer.segments:
            segment.selected_sequences = [segment.selected_sequences[i] for i in indices]
        optimizer.energy_scores = [optimizer.energy_scores[i] for i in indices]

        logger.debug(f"Sorted {optimizer.__class__.__name__} results: best={optimizer.energy_scores[0]:.4f}, worst={optimizer.energy_scores[-1]:.4f}")

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
            stage_index: Zero-based index of the optimizer stage to run.

        Raises:
            IndexError: If stage_index is out of range.
            RuntimeError: If attempting to skip forward (can only run current or previous stages).

        Example:
            >>> program = Program(optimizers=[opt1, opt2])
            >>> program.run_stage(0)  # Run first optimizer
            >>> results = program.get_stage_results(0)  # Access results
            >>> program.run_stage(1)  # Run second optimizer
        """
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

        # Capture results before sort so batch_idx matches history snapshots
        stage_result = self.extract_batch_results(optimizer.energy_scores)

        # Sort for next-stage handoff (next optimizer gets best sequences first)
        self._sort_sequences_by_energy(optimizer)

        if self.verbose:
            self._log_stage_results(stage_index, stage_result["batch_results"])

        self._stage_results.append(stage_result)
        self.current_stage = stage_index + 1

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
                for seq in segment.selected_sequences:
                    seq._constraints_metadata = {}
                for seq in segment.candidate_sequences:
                    seq._constraints_metadata = {}

    def extract_batch_results(self, energy_scores: List[float]) -> Dict[str, Any]:
        """Extract batch results from constructs."""
        from proto_language.utils.export import build_batch_results
        return build_batch_results(self.constructs, energy_scores)

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
                    "selected_sequences": [
                        {
                            "sequence": seq.sequence,
                            "sequence_type": seq.sequence_type,
                            "valid_chars": list(seq.valid_chars) if seq.valid_chars else None,
                        }
                        for seq in segment.selected_sequences
                    ],
                }
                segment_states.append(segment_state)

        return {"segments": segment_states}

    def restore_state(self, state: Dict, stage_index: int = None) -> None:
        """
        Restore program state from serialized data.

        Args:
            state: Dictionary returned by serialize_state()
            stage_index: Optional stage index to set current_stage to (for resuming from a specific stage)

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
            segment.selected_sequences = [
                Sequence(
                    sequence=seq_data["sequence"],
                    sequence_type=seq_data["sequence_type"],
                    valid_chars=set(seq_data["valid_chars"]) if seq_data.get("valid_chars") else None,
                )
                for seq_data in segment_state["selected_sequences"]
            ]

        # Update current_stage if specified (for resuming multi-stage optimization)
        if stage_index is not None:
            self.current_stage = stage_index

    # =========================================================================
    # Export Methods
    # =========================================================================

    def _get_batch_results(self, stage: Optional[int] = None) -> Dict[str, Any]:
        """Get batch results, optionally for a specific optimization stage."""
        if stage is not None:
            return self.get_stage_results(stage)
        return self.extract_batch_results(self.energy_scores)

    def _collect_history(
        self, stage: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Collect history from optimizers.

        Args:
            stage: If set, only include history from this optimizer stage index.
        """
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

    def export_results(
        self,
        path: Path | str = "./results",
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        stage: Optional[int] = None,
        segments: Optional[Set[str]] = None,
        batch_indices: Optional[Set[int]] = None,
    ) -> Path:
        """Export all result tables as a multi-file bundle.

        For csv/tsv/json: creates a directory with 4 files
        (sequences, constraints, constructs, optimization).
        For xlsx: creates one workbook with 4 sheets.

        Args:
            path: Output directory (csv/tsv/json) or file path (xlsx).
            format: Output format.
            stage: Optional optimization stage index. Filters both results
                and history to this stage.
            segments: If set, only include these segment labels.
            batch_indices: If set, only include these batch indices.

        Returns:
            Path where results were saved.
        """
        from proto_language.utils.export import (
            flatten_constraints,
            flatten_constructs,
            flatten_optimization,
            flatten_sequences,
            to_xlsx_workbook,
            write_export,
        )

        batch_results = self._get_batch_results(stage)
        history = self._collect_history(stage)
        f = {"segments": segments, "batch_indices": batch_indices}

        tables = {
            "sequences": flatten_sequences(batch_results, **f),
            "constraints": flatten_constraints(batch_results, **f),
            "constructs": flatten_constructs(batch_results, **f),
            "optimization": flatten_optimization(history, **f),
        }

        path = Path(path)
        if format == "xlsx":
            to_xlsx_workbook(tables, path)
        else:
            path.mkdir(parents=True, exist_ok=True)
            for name, rows in tables.items():
                write_export(rows, format, path / f"{name}.{format}")
        return path

    def export_sequences(
        self,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        path: Optional[Path] = None,
        stage: Optional[int] = None,
        segments: Optional[Set[str]] = None,
        batch_indices: Optional[Set[int]] = None,
    ) -> str | Path:
        """Export sequences table (one row per batch x construct x segment).

        Args:
            format: Output format.
            path: Output path. If None, returns string (not supported for xlsx).
            stage: Optional optimization stage index.
            segments: If set, only include these segment labels.
            batch_indices: If set, only include these batch indices.

        Returns:
            String content or Path where file was saved.
        """
        from proto_language.utils.export import flatten_sequences, write_export

        rows = flatten_sequences(
            self._get_batch_results(stage),
            segments=segments,
            batch_indices=batch_indices,
        )
        return write_export(rows, format, Path(path) if path else None)

    def export_constraints(
        self,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        path: Optional[Path] = None,
        stage: Optional[int] = None,
        segments: Optional[Set[str]] = None,
        constraints: Optional[Set[str]] = None,
        batch_indices: Optional[Set[int]] = None,
    ) -> str | Path:
        """Export constraints table (one row per batch x construct x segment x constraint).

        Args:
            format: Output format.
            path: Output path. If None, returns string (not supported for xlsx).
            stage: Optional optimization stage index.
            segments: If set, only include these segment labels.
            constraints: If set, only include these constraint labels.
            batch_indices: If set, only include these batch indices.

        Returns:
            String content or Path where file was saved.
        """
        from proto_language.utils.export import flatten_constraints, write_export

        rows = flatten_constraints(
            self._get_batch_results(stage),
            segments=segments,
            constraints=constraints,
            batch_indices=batch_indices,
        )
        return write_export(rows, format, Path(path) if path else None)

    def export_constructs(
        self,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        path: Optional[Path] = None,
        stage: Optional[int] = None,
        segments: Optional[Set[str]] = None,
        batch_indices: Optional[Set[int]] = None,
    ) -> str | Path:
        """Export constructs table (one row per batch x construct).

        Args:
            format: Output format.
            path: Output path. If None, returns string (not supported for xlsx).
            stage: Optional optimization stage index.
            segments: If set, only include these segment labels in columns.
            batch_indices: If set, only include these batch indices.

        Returns:
            String content or Path where file was saved.
        """
        from proto_language.utils.export import flatten_constructs, write_export

        rows = flatten_constructs(
            self._get_batch_results(stage),
            segments=segments,
            batch_indices=batch_indices,
        )
        return write_export(rows, format, Path(path) if path else None)

    def export_fasta(
        self,
        stage: Optional[int] = None,
        segments: Optional[Set[str]] = None,
        batch_indices: Optional[Set[int]] = None,
        header_format: str = "{construct}_{segment}_batch{batch_idx}",
        path: Optional[Path] = None,
    ) -> str | Path:
        """Export sequences in FASTA format for bioinformatics pipelines.

        Args:
            stage: Optional optimization stage index.
            segments: If set, only include these segment labels.
            batch_indices: If set, only include these batch indices.
            header_format: Python format string for FASTA headers. Available
                fields: construct, segment, batch_idx, energy_score, sequence_type.
            path: Output path. If None, returns string.

        Returns:
            FASTA string or Path where file was saved.
        """
        from proto_language.utils.export import to_fasta

        batch_results = self._get_batch_results(stage)
        result = to_fasta(
            batch_results,
            segments=segments,
            batch_indices=batch_indices,
            header_format=header_format,
            output=Path(path) if path else None,
        )
        if path:
            return Path(path)
        return result

    def export_optimization(
        self,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        path: Optional[Path] = None,
        stage: Optional[int] = None,
        segments: Optional[Set[str]] = None,
        batch_indices: Optional[Set[int]] = None,
    ) -> str | Path:
        """Export optimization trajectory (one row per timepoint x batch).

        Args:
            format: Output format.
            path: Output path. If None, returns string (not supported for xlsx).
            stage: If set, only include history from this optimizer stage.
            segments: If set, only include these segment labels.
            batch_indices: If set, only include these batch indices.

        Returns:
            String content or Path where file was saved.
        """
        from proto_language.utils.export import flatten_optimization, write_export

        rows = flatten_optimization(
            self._collect_history(stage),
            segments=segments,
            batch_indices=batch_indices,
        )
        return write_export(rows, format, Path(path) if path else None)

    def to_dataframe(
        self,
        table: Literal[
            "sequences", "constraints", "constructs", "optimization"
        ] = "sequences",
        stage: Optional[int] = None,
        segments: Optional[Set[str]] = None,
        constraints: Optional[Set[str]] = None,
        batch_indices: Optional[Set[int]] = None,
    ) -> "pd.DataFrame":
        """Export a result table as a pandas DataFrame.

        Args:
            table: Which table to export.
            stage: Optional optimization stage index.
            segments: If set, only include these segment labels.
            constraints: If set, only include these constraint labels
                (only applies to "constraints" table).
            batch_indices: If set, only include these batch indices.

        Returns:
            pandas DataFrame.

        Raises:
            ImportError: If pandas is not installed.
            ValueError: If table name is invalid.
        """
        from proto_language.utils.export import (
            flatten_constraints,
            flatten_constructs,
            flatten_optimization,
            flatten_sequences,
            to_dataframe,
        )

        batch_results = self._get_batch_results(stage)
        f = {"segments": segments, "batch_indices": batch_indices}

        if table == "sequences":
            rows = flatten_sequences(batch_results, **f)
        elif table == "constraints":
            rows = flatten_constraints(
                batch_results, constraints=constraints, **f
            )
        elif table == "constructs":
            rows = flatten_constructs(batch_results, **f)
        elif table == "optimization":
            rows = flatten_optimization(self._collect_history(stage), **f)
        else:
            raise ValueError(
                f"Unknown table '{table}'. "
                f"Choose from: sequences, constraints, constructs, optimization"
            )
        return to_dataframe(rows)
