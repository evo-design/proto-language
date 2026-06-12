"""Top-level orchestrator that runs Optimizer stages over shared Constructs.

A ``Program`` owns the design's ``Construct`` objects and composes one or more
``Optimizer`` stages that run sequentially. ``run()`` walks the stages in order; each
stage drives the propose-score-refine loop (the optimizer asks its ``Generator`` for
proposal ``Sequence`` s, every ``Constraint`` scores them, and survivors become the
``Segment`` 's result ``Sequence`` s). Because all optimizers share the same construct
objects by identity, each stage's results flow into the next without manual handoff.
A program-level ``seed`` owns run determinism and ``num_results`` flows down to any
optimizer that does not set its own. After the run, ``constructs`` hold the final
sequences, ``energy_scores`` reports the final-stage energies (lower is better), and
``export()`` / ``to_dataframe()`` / ``to_fasta()`` emit results.

Examples:
    Build a single-stage program and inspect its final joined sequence:
    >>> from proto_language.constraint import gc_content_constraint
    >>> from proto_language.core import Constraint, Construct, Program, Segment
    >>> from proto_language.generator import (
    ...     RandomNucleotideGenerator,
    ...     RandomNucleotideGeneratorConfig,
    ... )
    >>> from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
    >>> seg = Segment(length=20, sequence_type="dna")
    >>> construct = Construct([seg])
    >>> gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    >>> gen.assign(seg)
    >>> gc = Constraint(
    ...     inputs=[seg],
    ...     function=gc_content_constraint,
    ...     function_config={"min_gc": 80, "max_gc": 90},
    ... )
    >>> optimizer = MCMCOptimizer(
    ...     constructs=[construct],
    ...     generators=[gen],
    ...     constraints=[gc],
    ...     config=MCMCOptimizerConfig(num_results=1, proposals_per_result=20, num_steps=10),
    ... )
    >>> program = Program(optimizers=[optimizer], num_results=1)
    >>> program.run()
    >>> program.constructs[0].joined_sequences[0]  # the optimized DNA Sequence

    Write a results folder (tables + FASTA + assets/) to disk:
    >>> program.export(path="run_out")  # PosixPath('run_out')
"""

import logging
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from proto_tools.utils.tool_pool import ToolPool

from proto_language.core.generator import GeneratorInputType
from proto_language.core.optimizer import Optimizer, derive_seeds
from proto_language.utils.io import (
    build_results,
    flatten_table,
    to_fasta,
    write_results_folder,
)

logger = logging.getLogger(__name__)


class Program:
    """Programs represent user-defined biological designs.

    This class supports sequential execution of multiple optimizers, where each
    optimizer builds on the results of the previous one. All optimizers must
    share the same construct objects to ensure state persistence.

    Optimizer Handoff Contract:
        When running multiple optimizers sequentially, the Program ensures proper
        state transfer between stages:

        **After each optimizer completes:**

        Optimizers are responsible for their own sorting. Rejection Sampling keeps
        ``result_sequences`` sorted by energy throughout its run. Other
        optimizers' natural ordering is preserved as-is.

        **Before the next optimizer runs:**

        1. ``_initialize_sequence_pools()`` reads from ``result_sequences``
           (or ``original_sequence`` if first optimizer)
        2. Both ``result_sequences`` and ``proposal_sequences`` are initialized by
           cycling through source to preserve diversity when pool sizes differ
           (e.g., source=[A,B,C], num_results=5 -> [A,B,C,A,B])

        **Optimizer-specific behavior:**

        - **Rejection Sampling**: Clears ``result_sequences`` and repopulates dynamically
          during run (always sorted by energy)
        - **MCMC**: Uses ``result_sequences`` as parallel trajectories, overwrites
          ``proposal_sequences`` each step via ``_populate_proposal_sequences()``
        - **CyclingOptimizer**: Works directly on ``proposal_sequences``
        - **BeamSearch**: Ignores previous state entirely, starts fresh from configured prompt

    Examples:
        Sequential optimization with Rejection Sampling followed by MCMC:
        >>> from proto_language.optimizer import (
        ...     RejectionSamplingOptimizer,
        ...     RejectionSamplingOptimizerConfig,
        ...     MCMCOptimizer,
        ...     MCMCOptimizerConfig,
        ... )
        >>>
        >>> # First optimizer: broad exploration with Rejection Sampling
        >>> optimizer_1 = RejectionSamplingOptimizer(
        ...     constructs=[construct],
        ...     generators=[broad_mutation_gen],
        ...     constraints=[gc_constraint_1],
        ...     config=RejectionSamplingOptimizerConfig(num_samples=100, num_results=3),
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
        >>> rs_history = program.optimizers[0].history
        >>> mcmc_history = program.optimizers[1].history
    """

    def __init__(
        self,
        optimizers: list[Optimizer],
        num_results: int,
        verbose: bool = False,
        compute: ToolPool | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a Program with a list of optimizers to run sequentially.

        Args:
            optimizers (list[Optimizer]): List of Optimizer objects to run in sequence. Each optimizer
                       builds on the results of the previous one. All optimizers must
                       share the same construct objects (by identity).
            num_results (int): Number of result sequences to produce by default. Flows down to any
                optimizer whose config field is not set. Optimizer-level config
                always takes priority.
            verbose (bool): If True, print detailed energy score calculations for each constraint
                     for all optimizers.
            compute (ToolPool | None): Context manager for tool execution. If None,
                auto-detects: nullcontext when external dispatch is configured (cloud SDK
                backend or a deployment dispatch backend), else ToolPool()
                (symmetric across GPU and CPU-only hosts).
            seed (int | None): Random seed for fully reproducible optimization. When set,
                derives unique optimizer config seeds, overriding optimizer-level
                seeds. Same seed + same input = same output.

        Raises:
            ValueError: If optimizers list is empty or if optimizers don't share
                       the same construct objects.
        """
        if not optimizers:
            raise ValueError(
                "Program requires at least one Optimizer (got empty list); pass optimizers=[opt1, opt2, ...] to chain stages"
            )

        if compute is None:
            from contextlib import nullcontext

            from proto_tools.tools.tool_registry import ToolRegistry
            from proto_tools.utils.tool_pool import ToolPool

            # A local ToolPool bypasses cloud dispatch, so skip it when external dispatch is configured.
            has_backend = ToolRegistry.dispatch_backend_configured()
            if has_backend:
                logger.debug("External dispatch configured; GPU tools will route to the hosted service.")
                compute = nullcontext()
            else:
                # Symmetric across GPU and CPU-only hosts.
                compute = ToolPool()

        self.compute = compute

        if seed is not None and seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        self.optimizers = optimizers
        self.num_results = num_results

        # Flow num_results to optimizers that don't have it set. Optimizers can optionally override
        for opt in self.optimizers:
            if opt.num_results is None:
                opt._resolve_num_results(self.num_results)
            elif opt.num_results != self.num_results:
                logger.warning(
                    f"{opt.__class__.__name__} num_results={opt.num_results} Overrides program num_results={self.num_results}"
                )

        # Flow seed to optimizers: program seed overrides optimizer-level seeds
        self.seed = seed
        if self.seed is not None:
            for opt, derived in zip(self.optimizers, derive_seeds(self.seed, len(self.optimizers)), strict=True):
                if opt.seed is not None:
                    logger.warning(f"{opt.__class__.__name__} seed={opt.seed} overridden by program seed={self.seed}")
                opt.seed = derived

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
        self._stage_results: list[dict[str, Any]] = []
        self._validate_program()
        logger.debug(f"Program initialized: optimizers={len(self.optimizers)}, constructs={len(self.constructs)}")

    @property
    def energy_scores(self) -> list[float]:
        """Get energy scores from the final optimizer.

        Returns:
            list[float]: List of energy scores where lower values indicate better solutions.

        Raises:
            RuntimeError: If run() hasn't been called yet.
        """
        if not hasattr(self.optimizers[-1], "energy_scores"):
            raise RuntimeError("Optimization not complete. Call run() first.")
        return self.optimizers[-1].energy_scores

    def _validate_program(self) -> None:
        """Validate program configuration before execution.

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
                raise ValueError(
                    f"Optimizer {i} ({opt.__class__.__name__}) has no num_results. Set it via the optimizer's config or pass num_results to Program()."
                )

        reference_constructs = self.optimizers[0].constructs

        # 2. Validate construct identity: all optimizers must share the same construct
        # objects so each stage's results propagate to the next.
        for i, optimizer in enumerate(self.optimizers[1:], start=1):
            if len(optimizer.constructs) != len(reference_constructs):
                raise ValueError(
                    f"Optimizer {i} has {len(optimizer.constructs)} constructs, "
                    f"but optimizer 0 has {len(reference_constructs)}. "
                    "All optimizers must share the same construct objects."
                )
            for j, (construct, ref_construct) in enumerate(
                zip(optimizer.constructs, reference_constructs, strict=False)
            ):
                if construct is not ref_construct:
                    raise ValueError(
                        f"Optimizer {i} construct {j} is not the same object as "
                        f"optimizer 0 construct {j}. All optimizers must share the "
                        "same construct objects (by identity) for state persistence."
                    )

        # 3. Validate unique construct labels
        construct_labels = [c.label for c in self.constructs]
        if len(construct_labels) != len(set(construct_labels)):
            duplicates = [label for label in construct_labels if construct_labels.count(label) > 1]
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
                seen_segments[id(segment)] = construct.label or "unlabeled"

        # 5. Validate all segments will be populated
        # A segment must have either an input sequence or a generator in some optimizer.
        generator_segments = {
            segment for opt in self.optimizers for gen in opt.generators if gen.is_assigned for segment in gen.segments
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
            seen_generators: dict[int, int] = {}
            seen_constraints: dict[int, int] = {}

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

        # 7. Validate each generator's input_type contract is satisfiable in stage order.
        self._validate_generator_inputs()

    def _validate_generator_inputs(self) -> None:
        """Walk optimizer stages in order; raise if any generator's declared input is unavailable at runtime.

        Per-``input_type`` checks:
            - ``STARTING_SEQUENCE``: target segment must already be populated (initial input or prior-stage output),
              unless the generator declares it can initialize a length-only target.
            - ``PROMPT``: ``generator.config.prompts`` must be non-empty, unless the stage supplies prompts at
              runtime (``CyclingOptimizer`` via pipeline/conditioning_fn; ``BeamSearchOptimizer`` via ``config.prompt``).
            - ``STRUCTURE``: ``generator.config.structure_inputs`` must be non-empty, or the stage must be a
              ``CyclingOptimizer`` (named pipeline or conditioning_fn supplies structures).

        Note:
            ``LOGITS`` is not checked here — ``PositionWeightGenerator`` raises a clear runtime error
            (``"Proposal on segment 'X' has no logits."``) if wired outside a ``GradientOptimizer``.
            Stale-logits/structure invalidation is also enforced at runtime in ``Generator.sample()``.

        Raises:
            ValueError: If any generator's declared ``input_type`` is not satisfied at its stage.
        """
        from proto_language.optimizer.beam_search_optimizer import BeamSearchOptimizer
        from proto_language.optimizer.cycling_optimizer import CyclingOptimizer

        # Segments with a usable starting sequence going into each stage; grows as stages write to them.
        populated_segments: set[int] = {
            id(segment)
            for construct in self.constructs
            for segment in construct.segments
            if segment.populated_sequences
        }

        for stage_idx, optimizer in enumerate(self.optimizers):
            # Cycling/BeamSearch supply input dynamically; skip the config-source check for those input_types.
            under_cycling = isinstance(optimizer, CyclingOptimizer)
            supplies_prompts = under_cycling or isinstance(optimizer, BeamSearchOptimizer)

            for generator in optimizer.generators:
                if not generator.is_assigned:
                    continue

                kind = generator.input_type
                target_segments = list(generator.segments)
                config = getattr(generator, "config", None)

                # (1) PROMPT: autoregressive generators need non-empty config.prompts (unless stage-supplied).
                if kind == GeneratorInputType.PROMPT and not supplies_prompts:
                    prompts = getattr(config, "prompts", None)
                    if not prompts:
                        raise ValueError(
                            f"Stage {stage_idx} autoregressive generator {generator.__class__.__name__} "
                            f"requires non-empty prompts on its config (got {prompts!r})."
                        )

                # (2) STRUCTURE: inverse-folding generators need config.structure_inputs (unless under Cycling).
                if kind == GeneratorInputType.STRUCTURE and not under_cycling:
                    structure_inputs = getattr(config, "structure_inputs", None)
                    if not structure_inputs:
                        raise ValueError(
                            f"Stage {stage_idx} inverse folding generator {generator.__class__.__name__} "
                            f"requires structure_inputs on its config, or wrap the stage in a CyclingOptimizer "
                            f"(or, if it generates its own structures rather than consuming them, it should not "
                            f"declare input_type=STRUCTURE)."
                        )

                # Only the primary is consumed by ``_sample()``; tied segments are mirrored from it.
                primary = target_segments[0]
                primary_label = primary.label or "unlabeled"

                # (3) STARTING_SEQUENCE: mutation generators need a sequence to mutate on the primary segment.
                if (
                    kind == GeneratorInputType.STARTING_SEQUENCE
                    and id(primary) not in populated_segments
                    and not generator.allows_empty_starting_sequence
                ):
                    raise ValueError(
                        f"Stage {stage_idx} mutation generator {generator.__class__.__name__} "
                        f"targets segment {primary_label!r} but no starting sequence is available. "
                        f"Set segment.input_sequence, or place a prior optimizer stage that writes to it."
                    )

                # (4) Mark target segments as populated so downstream STARTING_SEQUENCE checks see them.
                for target in target_segments:
                    populated_segments.add(id(target))

    def _log_stage_results(self, stage_index: int, results: list[dict[str, Any]]) -> None:
        """Log results for a completed optimization stage."""
        logger.debug(f"Final state for optimizer {stage_index + 1}:")
        for result in results:
            energy = result["energy_score"]
            energy_str = f"{energy:.4f}" if energy is not None else "None"
            logger.debug(f"  [{result['result_idx']}] energy={energy_str}")
            for construct in result["constructs"]:
                seqs = [seg["sequence"] for seg in construct["segments"]]
                logger.debug(f"    {construct['label']}: {' | '.join(seqs)}")

    def run(self) -> None:
        """Execute the sequence optimization process for all optimizers sequentially.

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

        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        logger.info(f"Running program: {len(self.optimizers)} stage(s), num_results={self.num_results}{seed_str}")
        with self._enter_compute(), self._log_duration("Program"):
            for optimizer_stage_idx in range(len(self.optimizers)):
                self.run_stage(optimizer_stage_idx)

    def run_stage(self, stage_index: int) -> None:
        """Execute a specific optimization stage.

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
                raise IndexError(f"Stage index {stage_index} out of range (0-{len(self.optimizers) - 1}).")
            if stage_index > self.current_stage:
                raise RuntimeError(f"Cannot skip to stage {stage_index}. Current stage is {self.current_stage}.")

            # Re-running a previous stage: restore state from that stage's optimizer
            if stage_index < self.current_stage:
                # Use the target stage's initial state (captured before it ran)
                self.optimizers[stage_index]._restore_initial_state()
                self._stage_results = self._stage_results[:stage_index]
                # Clear stale initial states from subsequent optimizers
                for opt in self.optimizers[stage_index + 1 :]:
                    opt._initial_state = None

            optimizer = self.optimizers[stage_index]
            stage_label = f"Stage {stage_index + 1}/{len(self.optimizers)}"
            logger.info(f"{stage_label}: {optimizer.__class__.__name__}")
            optimizer._initialize_sequence_pools()

            # Clear stale constraint metadata from previous stages
            self._clear_sequence_metadata()

            stage_start = time.perf_counter()
            optimizer.run()
            stage_elapsed = time.perf_counter() - stage_start

            stage_result = self.extract_results(optimizer.energy_scores)

            if self.verbose:
                self._log_stage_results(stage_index, stage_result["results"])

            finite = [s for s in optimizer.energy_scores if math.isfinite(s)]
            best_str = f"best_energy={min(finite):.4f}" if finite else "no proposals accepted"
            logger.info(f"{stage_label} complete in {stage_elapsed:.1f}s, {best_str}")

            self._stage_results.append(stage_result)
            self.current_stage = stage_index + 1

    @contextmanager
    def _enter_compute(self) -> Iterator[None]:
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

    @contextmanager
    def _log_duration(self, label: str) -> Iterator[None]:
        """Log elapsed wall time as INFO on context exit."""
        start = time.perf_counter()
        yield
        logger.info(f"{label} complete in {time.perf_counter() - start:.1f}s")

    def get_stage_results(self, stage_index: int) -> dict[str, Any]:
        """Get results from a specific optimization stage."""
        if stage_index < 0 or stage_index >= len(self._stage_results):
            raise IndexError(f"Stage {stage_index} not available. Only {len(self._stage_results)} stage(s) run.")
        return self._stage_results[stage_index]

    def _clear_sequence_metadata(self) -> None:
        """Clear constraint and generator metadata from every sequence.

        Run at each stage boundary so a sequence surviving into the next stage does
        not carry the prior stage's constraint results or generator provenance.
        """
        for construct in self.constructs:
            for segment in construct.segments:
                for seq in segment.result_sequences:
                    seq._constraints_metadata = {}
                    seq._generator_metadata = {}
                for seq in segment.proposal_sequences:
                    seq._constraints_metadata = {}
                    seq._generator_metadata = {}

    def extract_results(self, energy_scores: list[float]) -> dict[str, Any]:
        """Extract results from constructs."""
        return build_results(self.constructs, energy_scores)

    def serialize_state(self) -> dict[str, Any]:
        """Serialize program state for persistence between stages.

        Stores sequence identity plus optimizer handoff state. Constraint and
        generator metadata are excluded since they will be re-evaluated in
        subsequent stages.
        """
        segment_states = []
        for construct in self.constructs:
            for segment in construct.segments:
                segment_state = {
                    "result_sequences": [self._serialize_handoff_sequence(seq) for seq in segment.result_sequences],
                }
                segment_states.append(segment_state)

        return {"segments": segment_states}

    @staticmethod
    def _serialize_handoff_sequence(seq: Any) -> dict[str, Any]:
        """Serialize only the sequence fields needed across optimizer stages."""
        seq_data = {
            "sequence": seq.sequence,
            "sequence_type": seq.sequence_type,
            "valid_chars": sorted(seq.valid_chars) if seq.valid_chars else None,
        }
        # Preserve optimizer handoff state across hosted stage boundaries.
        if seq.logits is not None:
            seq_data["logits"] = seq.logits.tolist()
            seq_data["logits_shape"] = list(seq.logits.shape)
        if seq.structure is not None:
            seq_data["structure"] = seq.structure.model_dump(mode="json")
        return seq_data

    def restore_state(self, state: dict[str, Any], stage_index: int | None = None) -> None:
        """Restore program state from serialized data.

        Args:
            state (dict[str, Any]): Dictionary returned by serialize_state()
            stage_index (int | None): Optional stage index to set current_stage to (for resuming from a specific stage)

        Raises:
            ValueError: If state doesn't match program structure
        """
        from proto_language.core.sequence import Sequence

        all_segments = [seg for construct in self.constructs for seg in construct.segments]

        if len(all_segments) != len(state["segments"]):
            raise ValueError(
                f"State mismatch: program has {len(all_segments)} segments "
                f"but state has {len(state['segments'])} segments"
            )

        for segment, segment_state in zip(all_segments, state["segments"], strict=False):
            segment.result_sequences = [Sequence.from_dict(seq_data) for seq_data in segment_state["result_sequences"]]

        # Update current_stage if specified (for resuming multi-stage optimization)
        if stage_index is not None:
            self.current_stage = stage_index

    # =========================================================================
    # Export
    # =========================================================================

    def _collect_history(self, stage: int | None = None) -> list[dict[str, Any]]:
        if stage is not None:
            if stage < 0 or stage >= len(self.optimizers):
                raise IndexError(f"Stage {stage} out of range (program has {len(self.optimizers)} optimizers)")
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
        path: Path | str | None = None,
        *,
        format: Literal["csv", "tsv", "json", "xlsx"] = "csv",
        stage: int | None = None,
        segments: set[str] | None = None,
        result_indices: set[int] | None = None,
        constraints: set[str] | None = None,
        include_proposals: bool = False,
        project: str | None = None,
    ) -> Path:
        """Export results to *path* as a folder: 4 tables + FASTA + ``assets/``.

        When *path* is ``None``, names the folder per the unified convention
        (``{project}__{YYYY-MM-DD_HHMMSS}``) under CWD.

        Layout::

            <path>/
            ├── sequences.<fmt>        one row per (result, construct, segment)
            ├── constraints.<fmt>      one row per (result, construct, segment, constraint)
            ├── constructs.<fmt>       one row per (result, construct) — joined ``full_sequence``
            ├── optimization.<fmt>     one row per (timepoint, result) — from history
            ├── sequences.fasta
            ├── assets/
            │   ├── res{i}_con{c}_seg{s}_structure.{pdb|cif}    seq.structure
            │   ├── res{i}_con{c}_seg{s}_logits.npy             seq.logits via np.save
            │   └── *.csv              row-shaped nested metadata sidecars

        xlsx packs the four tables into a single ``<path>/results.xlsx`` workbook.

        Args:
            path (Path | str | None): Output directory; ``None`` uses the convention.
            format (Literal['csv', 'tsv', 'json', 'xlsx']): Table format.
            stage (int | None): Filter to this optimizer stage index.
            segments (set[str] | None): Only include these segment labels.
            result_indices (set[int] | None): Only include these result indices.
            constraints (set[str] | None): Only include these constraint labels (constraints table only).
            include_proposals (bool): Include proposal rows (optimization table only).
            project (str | None): Folder name source when *path* is ``None``.
        """
        return write_results_folder(
            results=self._results_for_stage(stage),
            history=self._collect_history(stage),
            path=path,
            format=format,
            include_proposals=include_proposals,
            segments=segments,
            result_indices=result_indices,
            constraints=constraints,
            project=project,
        )

    def to_dataframe(
        self,
        table: Literal["sequences", "constraints", "constructs", "optimization"] = "sequences",
        stage: int | None = None,
        segments: set[str] | None = None,
        constraints: set[str] | None = None,
        result_indices: set[int] | None = None,
        include_proposals: bool = False,
    ) -> pd.DataFrame:
        """Get a result table as a pandas DataFrame.

        Accepts the same filter arguments as :meth:`export`.

        Args:
            table (Literal['sequences', 'constraints', 'constructs', 'optimization']): Result
                table to return. Row grains: 'sequences' (result, construct, segment),
                'constraints' (+ constraint), 'constructs' (result, construct),
                'optimization' (timepoint, result).
            stage (int | None): Zero-based optimizer stage index to export from.
            segments (set[str] | None): Subset of segment labels to include, or None for all.
            constraints (set[str] | None): Subset of constraint labels to include, or None for all.
            result_indices (set[int] | None): Indices of specific results to include, or None for all.
            include_proposals (bool): Whether to include proposal sequences alongside accepted results.
        """
        return pd.DataFrame(
            flatten_table(
                table,
                self._results_for_stage(stage),
                self._collect_history(stage),
                segments=segments,
                result_indices=result_indices,
                constraints=constraints,
                include_proposals=include_proposals,
            )
        )

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
