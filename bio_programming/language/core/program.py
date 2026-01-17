from __future__ import annotations

from typing import Dict, List

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
        self.stage_results: List[Dict] = []
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

    def run(self) -> None:
        """
        Execute the sequence optimization process for all optimizers sequentially.

        Each optimizer builds on the results of the previous one. State automatically
        persists between optimizers through the shared construct objects.

        Also prints final state after all optimizers complete.
        """
        # On re-run, restore segments from opt1's initial state (the original state)
        if self.optimizers[0]._initial_state is not None:
            self.optimizers[0]._restore_initial_state()

        for optimizer_idx, optimizer in enumerate(self.optimizers):
            optimizer._initialize_sequence_pools()
            optimizer._initial_state = None  # Force re-capture for this pipeline run

            # Run this optimizer
            optimizer.run()

            # TODO: fix base program and optimizer classes for optimizers that don't compute energy scores
            # Print final state for this optimizer
            print(f"\nFinal state for optimizer {optimizer_idx + 1}:")
            num_seqs = len(self.constructs[0].joined_sequences)
            for seq_idx in range(num_seqs):
                # Only print energy if scores were computed (e.g., when constraints exist)
                if optimizer.energy_scores and seq_idx < len(optimizer.energy_scores):
                    energy = optimizer.energy_scores[seq_idx]
                    print(f"  [{seq_idx}] Energy: {energy:.4f}")
                else:
                    print(f"  [{seq_idx}]")
                for construct_idx, construct in enumerate(self.constructs):
                    seq = construct.joined_sequences[seq_idx]
                    print(f"    Construct {construct_idx}: {seq}")

        # Clean up model caches
        self.cleanup()

    def run_stage(self, stage_index: int) -> Dict:
        """
        Execute a specific optimization stage.

        Allows running optimizers one at a time with inspection of results between
        stages. Each stage builds on results from previous stages through shared
        construct objects.

        Args:
            stage_index: Zero-based index of the optimizer stage to run.

        Returns:
            Dictionary containing:
                - best_sequence: The best sequence from this stage
                - best_energy: The lowest energy score from this stage
                - all_sequences: All selected sequences from this stage
                - all_energies: All energy scores from this stage

        Raises:
            IndexError: If stage_index is out of range.
            RuntimeError: If attempting to skip stages (must run sequentially).

        Example:
            >>> program = Program(optimizers=[opt1, opt2])
            >>> results = program.run_stage(0)  # Run first optimizer
            >>> if results["best_energy"] < threshold:
            ...     program.run_stage(1)  # Run second optimizer
        """
        if stage_index < 0 or stage_index >= len(self.optimizers):
            raise IndexError(
                f"Stage index {stage_index} out of range. "
                f"Program has {len(self.optimizers)} stages (0-{len(self.optimizers)-1})."
            )

        if stage_index != self.current_stage:
            raise RuntimeError(
                f"Cannot run stage {stage_index}. Must run stages sequentially. "
                f"Current stage is {self.current_stage}."
            )

        optimizer = self.optimizers[stage_index]

        # On re-run of first stage, restore segments from opt1's initial state
        if stage_index == 0 and optimizer._initial_state is not None:
            optimizer._restore_initial_state()

        optimizer._initialize_sequence_pools()
        optimizer._initial_state = None  # Force re-capture for this pipeline run

        optimizer.run()

        print(f"\nFinal state for optimizer {stage_index + 1}:")
        num_seqs = len(self.constructs[0].joined_sequences)
        for seq_idx in range(num_seqs):
            # Only print energy if scores were computed (e.g., when constraints exist)
            if optimizer.energy_scores and seq_idx < len(optimizer.energy_scores):
                energy = optimizer.energy_scores[seq_idx]
                print(f"  [{seq_idx}] Energy: {energy:.4f}")
            else:
                print(f"  [{seq_idx}]")
            for construct_idx, construct in enumerate(self.constructs):
                seq = construct.joined_sequences[seq_idx]
                print(f"    Construct {construct_idx}: {seq}")

        # Capture results for this stage
        all_sequences = [seq.sequence for seq in self.constructs[0].joined_sequences]
        all_energies = (
            list(optimizer.energy_scores)
            if optimizer.energy_scores
            else [0.0] * len(all_sequences)
        )
        best_idx = (
            min(range(len(all_energies)), key=lambda i: all_energies[i])
            if all_energies
            else 0
        )

        results = {
            "best_sequence": all_sequences[best_idx],
            "best_energy": all_energies[best_idx],
            "all_sequences": all_sequences,
            "all_energies": all_energies,
        }

        self.stage_results.append(results)

        self.current_stage = stage_index + 1

        return results

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
                    "segment_id": id(segment),
                    "selected_sequences": [
                        {
                            "sequence": seq.sequence,
                            "metadata": seq._metadata,
                        }
                        for seq in segment.selected_sequences
                    ],
                    "original_sequence": {
                        "sequence": segment.original_sequence.sequence,
                        "metadata": segment.original_sequence._metadata,
                    },
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
