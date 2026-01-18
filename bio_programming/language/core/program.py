from __future__ import annotations

from typing import Any, Dict, List

from .optimizer import Optimizer


class Program:
    """
    Programs represent user-defined biological designs.

    This class supports sequential execution of multiple optimizers, where each
    optimizer builds on the results of the previous one. All optimizers must
    share the same construct objects to ensure state persistence.

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
        self.current_stage = 0
        self._stage_results: List[Dict] = []
        self._validate_program()

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
        Validate program configuration.

        Validates:
        1. All optimizers share the same construct objects (by identity)
        2. No dangling segments (segments with no input sequence and no generator assigned
           in any optimizer)

        Raises:
            ValueError: If optimizers don't share identical construct objects (by identity) or if any segment is never populated.
        """
        reference_constructs = self.optimizers[0].constructs

        for i, optimizer in enumerate(self.optimizers[1:], start=1):
            if len(optimizer.constructs) != len(reference_constructs):
                raise ValueError(
                    f"Optimizer {i} has {len(optimizer.constructs)} constructs, "
                    f"but optimizer 0 has {len(reference_constructs)} constructs. "
                    "All optimizers must share the same construct objects."
                )

            for j, (construct, ref_construct) in enumerate(
                zip(optimizer.constructs, reference_constructs)
            ):
                if construct is not ref_construct:
                    raise ValueError(
                        f"Optimizer {i} construct {j} is not the same object as "
                        f"optimizer 0 construct {j}. All optimizers must share the "
                        "same construct objects (by identity) to ensure state "
                        "persistence between sequential optimizations."
                    )

        # Collect all segments assigned to generators across ALL optimizers
        all_generator_segments = set()
        for optimizer in self.optimizers:
            for gen in optimizer.generators:
                if gen._assigned_segment:
                    all_generator_segments.add(gen._assigned_segment)

        # Validate no dangling segments (no input sequence AND no generator ever assigned)
        all_segments = [seg for construct in self.constructs for seg in construct.segments]
        for segment in all_segments:
            has_generator = segment in all_generator_segments
            if not has_generator and not segment.populated_sequences:
                raise ValueError(
                    f"Segment '{segment.label or 'unlabeled'}' is never populated. "
                    "It has no input sequence and no generator is assigned to it in any optimizer."
                )

        # Validate no generator/constraint reuse across optimizers
        self._validate_no_instance_reuse()

    def _validate_no_instance_reuse(self) -> None:
        """
        Validate that no generator or constraint instance is reused across optimizers.

        Each optimizer must have its own generator and constraint instances to avoid
        shared mutable state issues between optimization stages.

        Raises:
            ValueError: If a generator or constraint instance appears in multiple optimizers.
        """
        # No reuse possible with single optimizer
        if len(self.optimizers) <= 1:
            return
        # Track which optimizer each generator instance belongs to
        seen_generators: Dict[int, int] = {}  # id(generator) -> optimizer_index
        for opt_idx, optimizer in enumerate(self.optimizers):
            for generator in optimizer.generators:
                gen_id = id(generator)
                if gen_id in seen_generators:
                    raise ValueError(
                        f"Generator '{generator.__class__.__name__}' instance is reused "
                        f"across optimizer {seen_generators[gen_id]} and optimizer {opt_idx}. "
                        "Each optimizer must have its own generator instances to avoid "
                        "shared state issues. Create a new generator instance for each optimizer."
                    )
                seen_generators[gen_id] = opt_idx

        # Track which optimizer each constraint instance belongs to
        seen_constraints: Dict[int, int] = {}  # id(constraint) -> optimizer_index
        for opt_idx, optimizer in enumerate(self.optimizers):
            for constraint in optimizer.constraints:
                con_id = id(constraint)
                if con_id in seen_constraints:
                    raise ValueError(
                        f"Constraint '{constraint.label}' instance is reused "
                        f"across optimizer {seen_constraints[con_id]} and optimizer {opt_idx}. "
                        "Each optimizer must have its own constraint instances to avoid "
                        "shared state issues. Create a new constraint instance for each optimizer."
                    )
                seen_constraints[con_id] = opt_idx

    def _print_stage_results(self, stage_index: int, batch_results: list) -> None:
        """Print results for a completed optimization stage."""
        print(f"\nFinal state for optimizer {stage_index + 1}:")
        for result in batch_results:
            print(f"  [{result['batch_idx']}] energy={result['energy_score']:.4f}")
            for i, segments in enumerate(result["constructs"]):
                print(f"    Construct {i}: {' | '.join(segments)}")

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
        for stage_idx in range(len(self.optimizers)):
            self.run_stage(stage_idx)

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
        optimizer._initialize_sequence_pools()
        optimizer.run()

        stage_result = self.extract_batch_results(optimizer.energy_scores)
        if self.verbose:
            self._print_stage_results(stage_index, stage_result["batch_results"])

        self._stage_results.append(stage_result)
        self.current_stage = stage_index + 1

    def get_stage_results(self, stage_index: int) -> Dict[str, Any]:
        """Get results from a specific optimization stage."""
        if stage_index < 0 or stage_index >= len(self._stage_results):
            raise IndexError(f"Stage {stage_index} not available. Only {len(self._stage_results)} stage(s) run.")
        return self._stage_results[stage_index]

    def extract_batch_results(self, energy_scores: List[float]) -> Dict[str, Any]:
        """
        Extract batch results from constructs after optimization.

        Returns segment-level sequences (not joined) with metadata. This is the
        canonical format used by both the core layer and API.

        Note:
            Infinite/NaN energy scores (from filter rejection) are converted to None
            for JSON serialization compatibility. Use optimizer.energy_scores directly
            if you need the raw values.

        Args:
            energy_scores: List of energy scores (one per batch)

        Returns:
            Dictionary containing:
                - batch_results: List of batch result dicts with segment-level sequences
                - best_batch_idx: Index of the batch with lowest energy
        """
        import math
        from .sequence import Sequence
        from proto_language.utils import propagate_metadata

        def filter_inf_energy(score: float) -> float | None:
            """Convert inf/nan to None for JSON compatibility."""
            if math.isinf(score) or math.isnan(score):
                return None
            return score

        if not self.constructs or not self.constructs[0].segments:
            return {"batch_results": [], "best_batch_idx": 0}

        num_selected = len(self.constructs[0].segments[0].selected_sequences)
        batch_results = []

        for batch_idx in range(num_selected):
            construct_sequences = []
            batch_metadata: Dict[str, Any] = {}

            for construct_idx, construct in enumerate(self.constructs):
                selected_seqs = [seg.selected_sequences[batch_idx] for seg in construct.segments]
                construct_sequences.append([s.sequence for s in selected_seqs])

                joined = Sequence.from_sequences(selected_seqs, merge_metadata=True)
                propagate_metadata(joined._metadata, batch_metadata, f"construct_{construct_idx}")

            batch_results.append({
                "batch_idx": batch_idx,
                "constructs": construct_sequences,
                "energy_score": filter_inf_energy(energy_scores[batch_idx]),
                "metadata": batch_metadata
            })

        # For best_idx calculation, treat None (was inf/nan) as infinity
        def get_score(i: int) -> float:
            score = batch_results[i]["energy_score"]
            return float('inf') if score is None else score

        best_idx = min(range(len(batch_results)), key=get_score) if batch_results else 0
        return {"batch_results": batch_results, "best_batch_idx": best_idx}

    def serialize_state(self) -> Dict:
        """
        Serialize the current program state for persistence between stages.

        Returns:
            Dictionary containing current_stage, segments with sequences/metadata,
            and energy_scores from last completed optimizer.
        """
        segment_states = []
        for construct in self.constructs:
            for segment in construct.segments:
                segment_state = {
                    "selected_sequences": [
                        {
                            "sequence": seq.sequence,
                            "metadata": seq._metadata,
                        }
                        for seq in segment.selected_sequences
                    ],
                }
                segment_states.append(segment_state)

        energy_scores = []
        if self.current_stage > 0:
            last_optimizer = self.optimizers[self.current_stage - 1]
            energy_scores = last_optimizer.energy_scores

        return {
            "current_stage": self.current_stage,
            "segments": segment_states,
            "energy_scores": energy_scores,
        }

    def restore_state(self, state: Dict) -> None:
        """
        Restore program state from serialized data.

        Args:
            state: Dictionary returned by serialize_state()

        Raises:
            ValueError: If state doesn't match program structure
        """
        from .sequence import Sequence

        self.current_stage = state["current_stage"]

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
                    sequence_type=segment.sequence_type,
                    metadata=seq_data["metadata"],
                )
                for seq_data in segment_state["selected_sequences"]
            ]

    def cleanup(self) -> None:
        """Clean up cached models to free GPU memory."""
        from proto_language.tools.language_models.esm2.esm2 import clear_esm2_cache
        from proto_language.tools.language_models.esm3.esm3 import clear_esm3_cache
        from proto_language.tools.language_models.evo2.evo2 import clear_evo2_cache

        clear_evo2_cache()
        clear_esm3_cache()
        clear_esm2_cache()
