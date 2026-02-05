from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from proto_language.utils.helpers import filter_inf_nan_scores

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

        # Run all stages sequentially
        logger.debug(f"Program.run: starting {len(self.optimizers)} optimization stages")
        for optimizer_stage_idx in range(len(self.optimizers)):
            self.run_stage(optimizer_stage_idx)

        self.cleanup()

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

        # Sort results by energy (best first) for consistent handoff to next optimizer
        self._sort_sequences_by_energy(optimizer)

        stage_result = self.extract_batch_results(optimizer.energy_scores)
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
        """Clear constraint metadata from all sequences in all segments.

        Called at the start of each optimization stage to prevent stale metadata
        from previous stages persisting into new stages.
        """
        for construct in self.constructs:
            for segment in construct.segments:
                for seq in segment.selected_sequences:
                    seq._metadata["constraints"] = {}
                for seq in segment.candidate_sequences:
                    seq._metadata["constraints"] = {}

    def extract_batch_results(self, energy_scores: List[float]) -> Dict[str, Any]:
        """
        Extract batch results from constructs after optimization.

        Returns a structured format with nested constructs and segments, each containing
        their constraint metadata. This format is easy to parse and supports CSV export
        at different granularities (program, construct, or segment level).

        Note:
            Infinite/NaN energy scores (from filter rejection) are converted to None
            for JSON serialization compatibility. Use optimizer.energy_scores directly
            if you need the raw values.

        Args:
            energy_scores: List of energy scores (one per batch)

        Returns:
            Dictionary containing:
                - batch_results: List of structured batch results with nested constructs/segments
                - best_batch_idx: Index of the batch with lowest energy

        Example output format:
            {
                "batch_results": [{
                    "batch_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [{
                        "label": "construct_0",
                        "type": "dna",
                        "segments": [{
                            "label": "promoter",
                            "sequence": "ATCG",
                            "constraints": {
                                "gc_content_constraint": {
                                    "score": 0.5,
                                    "weight": 1.0,
                                    "weighted_score": 0.5,
                                    "data": {
                                        "gc_content": 50.0
                                    }
                                }
                            }
                        }]
                    }]
                }],
                "best_batch_idx": 0
            }
        """
        if not self.constructs or not self.constructs[0].segments:
            return {"batch_results": [], "best_batch_idx": 0}

        num_selected = len(self.constructs[0].segments[0].selected_sequences)
        batch_results = []

        for batch_idx in range(num_selected):
            structured_constructs = []

            for construct in self.constructs:
                structured_segments = []

                for seg_idx, segment in enumerate(construct.segments):
                    seq = segment.selected_sequences[batch_idx]
                    segment_label = segment.label or f"segment_{seg_idx}"

                    structured_segments.append({
                        "label": segment_label,
                        "sequence": seq.sequence,
                        "constraints": seq._metadata["constraints"],
                    })

                structured_constructs.append({
                    "label": construct.label,
                    "type": construct.sequence_type,
                    "segments": structured_segments,
                })

            batch_results.append({
                "batch_idx": batch_idx,
                "energy_score": filter_inf_nan_scores(energy_scores[batch_idx]),
                "constructs": structured_constructs,
            })

        # For best_idx calculation, treat None (was inf/nan) as infinity
        def get_score(i: int) -> float:
            score = batch_results[i]["energy_score"]
            return float('inf') if score is None else score

        best_idx = min(range(len(batch_results)), key=get_score) if batch_results else 0
        return {"batch_results": batch_results, "best_batch_idx": best_idx}

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

    def cleanup(self) -> None:
        """Clean up cached models to free GPU memory."""
        from proto_language.tools.language_models.esm2 import clear_esm2_cache
        from proto_language.tools.language_models.esm3 import clear_esm3_cache
        from proto_language.tools.language_models.evo2 import clear_evo2_cache

        clear_evo2_cache()
        clear_esm3_cache()
        clear_esm2_cache()

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_segment(
        self,
        construct: str,
        segment: str,
        batch_idx: int = 0,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        style: Literal["wide", "long"] = "wide",
        path: Optional[Path] = None,
        stage: Optional[int] = None,
    ) -> Path:
        """
        Export all constraint metadata for a specific segment.

        Args:
            construct: Construct label
            segment: Segment label
            batch_idx: Batch index (default 0)
            format: Output format - "csv", "tsv", "json", or "xlsx"
            style: "wide" (single row with constraint.metric columns) or
                   "long" (one row per constraint)
            path: Output path. Defaults to ./{segment}_{batch_idx}.{format}
            stage: Optional optimization stage index. If None, uses current state.

        Returns:
            Path where file was saved
        """
        from proto_language.utils.export import flatten_segment_metadata, write_export

        batch_results = self.get_stage_results(stage) if stage is not None else self.extract_batch_results(self.energy_scores)
        rows = flatten_segment_metadata(batch_results, construct, segment, batch_idx, style)

        if path is None:
            path = Path(f"./{segment}_{batch_idx}.{format}")
        else:
            path = Path(path)

        write_export(rows, format, path)
        return path

    def export_construct(
        self,
        construct: str,
        batch_idx: int = 0,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        style: Literal["wide", "long"] = "wide",
        path: Optional[Path] = None,
        stage: Optional[int] = None,
    ) -> Path:
        """
        Export metadata for all segments in a construct.

        Args:
            construct: Construct label
            batch_idx: Batch index (default 0)
            format: Output format - "csv", "tsv", "json", or "xlsx"
            style: "wide" (one row per segment) or
                   "long" (one row per segment × constraint)
            path: Output path. Defaults to ./{construct}_{batch_idx}.{format}
            stage: Optional optimization stage index. If None, uses current state.

        Returns:
            Path where file was saved
        """
        from proto_language.utils.export import (
            flatten_construct_metadata,
            write_export,
        )

        batch_results = self.get_stage_results(stage) if stage is not None else self.extract_batch_results(self.energy_scores)
        rows = flatten_construct_metadata(batch_results, construct, batch_idx, style)

        if path is None:
            path = Path(f"./{construct}_{batch_idx}.{format}")
        else:
            path = Path(path)

        write_export(rows, format, path)
        return path

    def export_program(
        self,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        style: Literal["wide", "long"] = "wide",
        path: Optional[Path] = None,
        stage: Optional[int] = None,
    ) -> Path:
        """
        Export metadata for all segments across all batches.

        Args:
            format: Output format - "csv", "tsv", "json", or "xlsx"
            style: "wide" (one row per batch) or
                   "long" (one row per batch x construct x segment)
            path: Output path. Defaults to ./program_results.{format}
            stage: Optional optimization stage index. If None, uses current state.

        Returns:
            Path where file was saved
        """
        from proto_language.utils.export import flatten_program_metadata, write_export

        batch_results = self.get_stage_results(stage) if stage is not None else self.extract_batch_results(self.energy_scores)
        rows = flatten_program_metadata(batch_results, style)

        if path is None:
            path = Path(f"./program_results.{format}")
        else:
            path = Path(path)

        write_export(rows, format, path)
        return path

    def export_batch_history(
        self,
        batch_idx: int = 0,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        style: Literal["wide", "long"] = "wide",
        path: Optional[Path] = None,
    ) -> Path:
        """
        Export metadata for a single batch across optimization history.

        Args:
            batch_idx: Batch index to track (default 0)
            format: Output format - "csv", "tsv", "json", or "xlsx"
            style: "wide" (one row per timepoint) or
                   "long" (one row per timepoint x construct x segment)
            path: Output path. Defaults to ./batch_{batch_idx}_history.{format}

        Returns:
            Path where file was saved

        Note:
            This uses the optimizer history, which is only available after
            running the program and before clearing the optimizer state.
        """
        from proto_language.utils.export import flatten_batch_over_time, write_export

        # Collect history from all optimizers
        history = []
        for optimizer in self.optimizers:
            history.extend(optimizer.history)

        rows = flatten_batch_over_time(history, batch_idx, style)

        if path is None:
            path = Path(f"./batch_{batch_idx}_history.{format}")
        else:
            path = Path(path)

        write_export(rows, format, path)
        return path
