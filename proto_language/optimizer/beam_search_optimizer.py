"""This optimizer splits a single long segment into beams of `beam_length` tokens and.

performs beam search, accumulating KV cache state across beams.
"""

import inspect
import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import model_validator

from proto_language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.optimizer.optimizer_registry import optimizer
from proto_language.utils.base import BaseOptimizerConfig, ConfigField

logger = logging.getLogger(__name__)


@dataclass
class BeamState:
    """State for a single beam during beam search.

    Attributes:
        running_sequence (str): Accumulated sequence (initial prompt + all generated tokens so far)
        kv_cache (Any | None): Opaque generator cache handle for this beam (None if KV caching disabled)
        beam_scores (list[float]): Per-beam energy scores for score aggregation
    """

    running_sequence: str
    kv_cache: Any | None = None
    beam_scores: list[float] = field(default_factory=list)


class BeamSearchOptimizerConfig(BaseOptimizerConfig):
    """Configuration object for BeamSearchOptimizer.

    This class defines configuration parameters for the beam search optimizer, which
    generates a single long segment by splitting it into beams of `beam_length` tokens
    and performing beam search at each beam boundary.

    Attributes:
        prompt (str): Initial prompt sequence to start beam search generation. All
            beams begin from this prompt and extend it autoregressively. For DNA,
            this might be ``"ATCG"``; for proteins, a short amino acid sequence.
            Must be non-empty.

        beam_length (int): Number of tokens to generate per iteration.
            The segment is split into ceil(segment.sequence_length / beam_length) beams.

        num_results (int | None): Number of result sequences (beam width) to maintain at each
            step. At each beam boundary, the top ``num_results`` sequences by
            energy score are selected to continue. Higher values explore more
            paths but increase computation. Must be at least 1.

        proposals_per_result (int): Number of proposal sequences to generate per
            result sequence (beam) at each beam search step. Total proposals per step is
            ``num_results x proposals_per_result``. Higher values increase
            diversity but also increase computation. Must be at least 1.

        score_by (Literal['mean', 'last']): How to aggregate beam scores when selecting beams.
            - ``"mean"``: Average scores across all beams - rewards consistent trajectory
            - ``"last"``: Use only the most recent beam's score - only cares about current state
            Default: ``"mean"``.

        prepend_prompt (bool): Whether to prepend the initial prompt to the generated
            sequences in the final output. If ``True``, full sequences include the
            prompt; if ``False``, only generated tokens are returned. Default: ``True``.

        use_kv_caching (bool): Whether to use key-value (KV) caching for faster
            sequential generation. KV caching stores intermediate model states to
            avoid recomputing prefix-context at each step. Significantly speeds
            up generation but requires more GPU memory. Only works with compatible
            generators (e.g., Evo2). Default: ``False``.

        max_resample_attempts (int): Maximum number of resampling attempts when
            beams produce invalid (inf/NaN) energy proposals. The optimizer will
            resample beams until each has ``proposals_per_result`` valid proposals,
            up to this many attempts. Higher values increase robustness but may
            slow down optimization with very restrictive constraints. Must be at
            least 1. Default: ``3``.

        verbose (bool): Whether to print detailed progress information including
            beam energies, result sequences, and generation statistics at each
            iteration. Default: ``False``.
        tracking_interval (int): Number of steps between progress snapshots.
        track_proposals (bool): Whether to record proposal sequences alongside accepted results.
    """

    # Required parameters
    prompt: str = ConfigField(
        title="Initial Prompt",
        description="Non-empty seed sequence that every beam begins from and extends (e.g. 'ATCG' for DNA).",
    )
    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Number of beams (top-K by energy) retained at each beam boundary. Overrides program-level count.",
    )
    proposals_per_result: int = ConfigField(
        ge=1,
        title="Proposals Per Result",
        description="Number of proposals to generate per result sequence at each beam step.",
    )

    # Generation parameters
    beam_length: int = ConfigField(
        ge=1,
        title="Tokens Per Step",
        description="Tokens per beam-search step before re-ranking; segment split into ceil(len/this) steps.",
    )
    score_by: Literal["mean", "last"] = ConfigField(
        default="mean",
        title="Score Aggregation",
        description="'mean' averages a beam's per-step energies across all steps; 'last' uses only the most recent.",
    )
    prepend_prompt: bool = ConfigField(
        default=True,
        title="Prepend Prompt",
        description="Whether to prepend the prompt to the generated sequence in the output.",
    )
    use_kv_caching: bool = ConfigField(
        default=False,
        title="Use KV Caching",
        description="Reuse cached KV state across beam steps to speed up generation; needs a KV-capable generator.",
    )

    # Advanced parameters
    max_resample_attempts: int = ConfigField(
        default=3,
        ge=1,
        title="Max Resample Attempts",
        description="Maximum number of times to resample beams with invalid (inf/NaN) energies before giving up.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "BeamSearchOptimizerConfig":
        """Validate beam search configuration."""
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        return self


@optimizer(
    key="beam-search",
    label="Beam Search Optimizer",
    config=BeamSearchOptimizerConfig,
    description="Beam search optimizer that generates a single segment with beam search at each boundary",
    targets_single_segment=True,
    compatible_generators=["evo1", "evo2", "progen2"],
)
class BeamSearchOptimizer(Optimizer):
    """Beam search optimizer for sequence generation.

    This optimizer implements beam search for sequence optimization where a single target
    segment is generated with beam search. The optimizer maintains K beams (running sequences)
    and generates K x N total proposals at each step by producing N variations per beam.
    After constraint evaluation on the FULL accumulated sequence, only the top K sequences by
    energy are retained for the next step.

    Attributes:
        target_segment: The target segment to generate with beam search.
        generator: Single autoregressive generator for sequence generation.
        prompt: Initial prompt sequence starting all beams.
        beam_length: Tokens per beam.
        num_results: Number of beams to maintain (K).
        proposals_per_result: Proposals generated per result sequence (N).
        score_by: Score aggregation method ('mean' or 'last').
        use_kv_caching: Whether KV caching is enabled.
        beams: Current beam states.

    Example:
        >>> from proto_language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> gen_config = Evo2GeneratorConfig(prompts="ATCG", prepend_prompt=True)
        >>> generator = Evo2Generator(config=gen_config)
        >>> segment = Segment(length=10000, sequence_type="dna")
        >>> construct = Construct([segment])
        >>> config = BeamSearchOptimizerConfig(prompt="ATCG", beam_length=2000, num_results=5, proposals_per_result=10)
        >>> beam_search = BeamSearchOptimizer(
        ...     target_segment=segment,
        ...     constructs=[construct],
        ...     generators=[generator],
        ...     constraints=[gc_constraint],
        ...     config=config,
        ... )
        >>> beam_search.run()
        >>> top_sequences = beam_search.target_segment.result_sequences
    """

    # Class attribute required by OptimizerRegistry
    config_class = BeamSearchOptimizerConfig
    config: BeamSearchOptimizerConfig

    def __init__(
        self,
        target_segment: Segment,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: BeamSearchOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the Beam Search Optimizer.

        Args:
            target_segment (Segment): The specific Segment to optimize with beam search. Must belong to one of the constructs.
            constructs (list[Construct]): List of Construct objects. The target_segment must belong to one of these constructs.
            generators (list[Generator]): List containing a single autoregressive Generator object (must have category="autoregressive").
            constraints (list[Constraint]): List of Constraint objects for evaluation (lower scores are better).
            config (BeamSearchOptimizerConfig): Configuration object containing algorithm parameters.
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked beams (governed by ``tracking_interval``).
            clear_tool_cache (int | bool | list[str]): (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
        """
        if len(generators) != 1:
            raise ValueError(
                f"BeamSearchOptimizer only supports one generator, but currently has {len(generators)} generators."
            )
        generator = generators[0]

        # Store config before super().__init__() so _resolve_num_results can access it
        self.config = config

        # Store config values required for validation
        self.target_segment: Segment = target_segment
        self.prompt: str = config.prompt
        self.beam_length: int = config.beam_length
        self.generator: Generator = generator
        self.use_kv_caching: bool = config.use_kv_caching

        # Base class init (calls _validate_optimizer)
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_results=config.num_results,
            proposals_per_result=config.proposals_per_result,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
            seed=config.seed,
        )

        self.prepend_prompt: bool = config.prepend_prompt
        self.score_by: str = config.score_by
        self.max_resample_attempts: int = config.max_resample_attempts
        self.batch_size: int = self.generator.batch_size

        if self.num_results is not None:
            self.beams: list[BeamState] = [BeamState(running_sequence=self.prompt) for _ in range(self.num_results)]
        else:
            self.beams = []

        # Calculate number of beams based on target segment
        self.num_beams = math.ceil(self.target_segment.sequence_length / self.beam_length)

        # Override base class num_steps for progress tracking
        self.num_steps = self.num_beams

        # Set up generator for beam search
        # Pre-allocate max_seqlen for the full optimization - vortex doesn't support
        self.generator.max_seqlen = len(self.prompt) + self.target_segment.sequence_length  # type: ignore[attr-defined]
        self.generator.store_kv_cache = self.use_kv_caching  # type: ignore[attr-defined]
        self.generator.cached_generation = True  # type: ignore[attr-defined]
        self.generator.batched = True  # type: ignore[attr-defined]

    def _validate_optimizer(self) -> None:
        """Validate beam search optimizer configuration.

        Extends base validation with beam-search-specific checks:
        target_segment membership, ``_sample()`` signature, KV caching interface,
        non-empty prompt, and beam_length bounds. Generator key compatibility is
        enforced centrally via ``OptimizerSpec.compatible_generators``.
        """
        super()._validate_optimizer()
        self._validate_target_segment(self.target_segment)
        self._validate_generator_sample_signature()

        # KV caching support (if enabled)
        if self.use_kv_caching and not hasattr(self.generator, "kv_caches"):
            raise ValueError(
                f"Generator '{self.generator.__class__.__name__}' does not support KV caching (missing kv_caches attribute). "
                f"Set use_kv_caching=False or use a generator that supports KV caching."
            )
        if self.use_kv_caching and not callable(getattr(self.generator, "release_kv_cache", None)):
            raise ValueError(
                f"Generator '{self.generator.__class__.__name__}' does not support KV caching "
                f"(missing release_kv_cache method). Set use_kv_caching=False or use a generator that supports KV caching."
            )

        # Prompt + beam_length
        if not self.prompt:
            raise ValueError("Prompt for BeamSearchOptimizer cannot be empty")
        if self.beam_length > self.target_segment.sequence_length:
            raise ValueError(
                f"beam_length={self.beam_length} cannot be greater than "
                f"target_segment length ({self.target_segment.sequence_length})"
            )

    def _validate_generator_sample_signature(self) -> None:
        """Require generator._sample() to accept ``max_new_tokens`` and ``old_kv_cache``."""
        required_params = ("max_new_tokens", "old_kv_cache")
        try:
            sig = inspect.signature(self.generator._sample)
        except (TypeError, ValueError):
            return
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return
        missing = [name for name in required_params if name not in params]
        if missing:
            raise ValueError(
                f"Generator '{self.generator.__class__.__name__}' _sample() missing required parameter(s) {missing}"
            )

    def _capture_initial_state(self) -> None:
        """Capture state and reset BeamSearch-specific state for fresh run."""
        super()._capture_initial_state()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        self.beams = [BeamState(running_sequence=self.prompt) for _ in range(self.num_results)]

    def _restore_initial_state(self) -> None:
        """Restore to captured state and reset BeamSearch-specific state."""
        super()._restore_initial_state()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        self.beams = [BeamState(running_sequence=self.prompt) for _ in range(self.num_results)]

    def run(self) -> None:
        """Run beam search within a single segment.

        For each beam:
        1. Use K accumulated prompts from previous beams
        2. Generate K x N proposals (N per beam)
        3. Score all proposals using FULL accumulated sequence
        4. Select top K proposals and update beam states for next beam
        """
        with self._kv_cache_worker_context():
            try:
                self._run_beam_search()
            finally:
                self._release_current_beam_caches()

    def _run_beam_search(self) -> None:
        """Run the beam search loop after execution context setup."""
        self._prepare_run()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing

        # BeamSearch always starts from its configured prompt.
        if any(seq.sequence for seg in self.segments for seq in seg.result_sequences):
            logger.warning(
                "BeamSearchOptimizer starts from its configured prompt and overwrites existing sequences/prompts"
            )

        # BeamSearch starts from empty prompt; no meaningful initial snapshot
        self.energy_scores = [float("inf")] * self.num_results

        n_filter = sum(1 for c in self.constraints if c.threshold is not None)
        n_score = len(self.constraints) - n_filter
        logger.info(
            f"BeamSearchOptimizer: {self.num_beams} beams x {self.beam_length} tokens, "
            f"width={self.num_results}, target_length={self.target_segment.sequence_length}, "
            f"score_by={self.score_by!r}, "
            f"{len(self.constraints)} constraints ({n_filter} filter, {n_score} scoring)"
        )
        logger.debug(
            f"BeamSearchOptimizer kv_caching={'enabled' if self.use_kv_caching else 'disabled'}, "
            f"proposals_per_beam={self._proposals_per_result}"
        )

        tokens_generated = 0
        for beam_num in range(1, self.num_beams + 1):
            remaining_tokens = self.target_segment.sequence_length - tokens_generated
            beam_tokens = min(self.beam_length, remaining_tokens)
            proposal_beams = self._generate_and_score_with_resampling(
                self.prepend_prompt and beam_num == 1,
                beam_tokens,
            )
            self._select_topk_beams(proposal_beams)

            if beam_num % self.tracking_interval == 0 or beam_num == self.num_beams:
                result_energies = [score for score in self.energy_scores if math.isfinite(score)]
                accepted_energies = [
                    score
                    for outcome, score in zip(self._proposal_outcomes, self._proposal_energy_scores, strict=True)
                    if outcome == "accepted" and math.isfinite(score)
                ]
                self._save_progress_snapshot(
                    time_step=beam_num,
                    optimizer_metadata={
                        "type": "beam-search",
                        "beam_width": self.num_results,
                        "num_beams": self.num_beams,
                        "beam_length": self.beam_length,
                        "target_length": self.target_segment.sequence_length,
                        "proposals_per_beam": self._proposals_per_result,
                        "score_by": self.score_by,
                        "best_energy": min(result_energies) if result_energies else None,
                        "mean_energy": float(np.mean(result_energies)) if result_energies else None,
                        "proposal_count": len(self._proposal_outcomes),
                        "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
                        "accepted_min_energy": min(accepted_energies) if accepted_energies else None,
                        "accepted_mean_energy": float(np.mean(accepted_energies)) if accepted_energies else None,
                        "accepted_max_energy": max(accepted_energies) if accepted_energies else None,
                    },
                )
                self._log_beamsearch_progress(beam_num, beam_tokens, tokens_generated + beam_tokens)

            tokens_generated += beam_tokens

        self.target_segment.result_sequences = [
            Sequence(
                sequence=beam.running_sequence if self.prepend_prompt else beam.running_sequence[len(self.prompt) :],
                sequence_type=self.target_segment.sequence_type,
            )
            for beam in self.beams
        ]

    def _generate_proposals_for_beam(
        self,
        beam_idx: int,
        prepend_prompt: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[BeamState]:
        """Generate proposal BeamStates for a single beam.

        Generates proposals in batches (sized by generator batch_size) to manage GPU memory.

        Args:
            beam_idx (int): Index of the beam to generate proposals for
            prepend_prompt (bool): Whether to prepend prompt to generated sequences
            max_new_tokens (int | None): Max newly generated tokens per proposal; ``None`` lets the generator derive it from the target segment.

        Returns:
            list[BeamState]: List of BeamState proposals (length=proposals_per_result)
        """
        beam = self.beams[beam_idx]

        if self.verbose:
            self._log_beam_generation_start(beam_idx, beam)

        proposals = []
        for batch_start in range(0, self._proposals_per_result, self.batch_size):
            batch_count = min(self.batch_size, self._proposals_per_result - batch_start)

            # The worker clones from old_kv_cache; it does not consume the handle,
            # so one beam cache can seed multiple proposal batches.
            prompts = [beam.running_sequence] * batch_count
            kv_cache = beam.kv_cache if self.use_kv_caching and beam.kv_cache is not None else None

            # Resize proposal pool to match batch for zip(strict=True) compatibility
            self.target_segment.proposal_sequences = [
                Sequence(sequence="", sequence_type=self.target_segment.sequence_type) for _ in range(batch_count)
            ]
            self._sync_proposal_pools(self.target_segment)

            if self.verbose and batch_start == 0:
                self._log_cache_state(kv_cache)

            # Generate proposals
            with self._cached_generation_context():
                self.generator.sample(
                    prompts=prompts,
                    prepend_prompt=prepend_prompt,
                    max_new_tokens=max_new_tokens,
                    old_kv_cache=kv_cache,
                )

            # Collect results from this batch
            if self.use_kv_caching:
                kv_caches = self.generator.kv_caches  # type: ignore[attr-defined]
                if len(kv_caches) != batch_count:
                    raise RuntimeError(
                        f"Generator returned {len(kv_caches)} KV cache handles for {batch_count} generated prompts."
                    )
            else:
                kv_caches = [None] * batch_count
            for i in range(batch_count):
                generated_seq = self.target_segment.proposal_sequences[i].sequence
                new_prompt = generated_seq if prepend_prompt else beam.running_sequence + generated_seq

                proposals.append(
                    BeamState(
                        running_sequence=new_prompt,
                        kv_cache=kv_caches[i],
                        beam_scores=beam.beam_scores.copy(),
                    )
                )

        return proposals

    def _generate_and_score_with_resampling(
        self, prepend_prompt: bool = False, max_new_tokens: int | None = None
    ) -> list[BeamState]:
        """Generate and score proposals, resampling beams until each has valid proposals.

        Args:
            prepend_prompt (bool): Whether to prepend prompt to generated sequences
            max_new_tokens (int | None): Max newly generated tokens per proposal; ``None`` lets the generator derive it from the target segment.

        Returns:
            list[BeamState]: List of all valid proposal BeamStates with scores populated

        Raises:
            RuntimeError: If unable to get enough valid proposals after max attempts
        """
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        # Track valid proposals per beam
        beam_proposals: dict[int, list[BeamState]] = {b: [] for b in range(self.num_results)}

        # Initial generation: Generate proposals for all beams
        all_proposals = []
        for beam_idx in range(self.num_results):
            proposals = self._generate_proposals_for_beam(beam_idx, prepend_prompt, max_new_tokens)
            all_proposals.extend(proposals)

        # Score all proposals on their FULL accumulated sequences
        self.target_segment.proposal_sequences = [
            Sequence(sequence=beam.running_sequence, sequence_type=self.target_segment.sequence_type)
            for beam in all_proposals
        ]
        self._sync_proposal_pools(self.target_segment)
        self.score_energy()

        # Collect valid proposals (those that passed filter constraints)
        for i, (proposal, score) in enumerate(zip(all_proposals, self.energy_scores, strict=False)):
            if self._proposal_outcomes[i] == "accepted":
                beam_idx = i // self._proposals_per_result
                proposal.beam_scores.append(score)
                beam_proposals[beam_idx].append(proposal)
            else:
                # Rejected proposals will not be scored again or continued.
                self._release_kv_cache(proposal.kv_cache)

        # Resample beams until each has proposals_per_result valid proposals
        for attempt in range(1, self.max_resample_attempts + 1):
            beams_to_resample = [
                b for b in range(self.num_results) if len(beam_proposals[b]) < self._proposals_per_result
            ]

            if not beams_to_resample:
                break  # All beams have enough valid proposals

            if self.verbose:
                counts = {b: len(beam_proposals[b]) for b in beams_to_resample}
                logger.info(f"Resampling {len(beams_to_resample)} beams (attempt {attempt}): counts={counts}")

            for beam_idx in beams_to_resample:
                proposals = self._generate_proposals_for_beam(beam_idx, prepend_prompt, max_new_tokens)

                # Score proposals on their FULL accumulated sequences
                self.target_segment.proposal_sequences = [
                    Sequence(sequence=beam.running_sequence, sequence_type=self.target_segment.sequence_type)
                    for beam in proposals
                ]
                self._sync_proposal_pools(self.target_segment)
                self.score_energy()

                for j, (proposal, score) in enumerate(zip(proposals, self.energy_scores, strict=False)):
                    if self._proposal_outcomes[j] == "accepted":
                        proposal.beam_scores.append(score)
                        beam_proposals[beam_idx].append(proposal)
                    else:
                        # Rejected resamples will not be scored again or continued.
                        self._release_kv_cache(proposal.kv_cache)

        # Verify each beam has at least proposals_per_result valid proposals
        insufficient_beams = [b for b in range(self.num_results) if len(beam_proposals[b]) < self._proposals_per_result]
        if insufficient_beams:
            counts = {b: len(beam_proposals[b]) for b in insufficient_beams}
            for proposals in beam_proposals.values():
                for proposal in proposals:
                    # The run is aborting, so no accepted proposal can continue.
                    self._release_kv_cache(proposal.kv_cache)
            raise RuntimeError(
                f"After {self.max_resample_attempts} attempts, {len(insufficient_beams)} beams could not produce "
                f"{self._proposals_per_result} valid proposals: {counts}. Constraints may be too restrictive."
            )

        # Flatten and sort by energy to get all valid proposals
        all_valid_proposals = []
        for beam_idx in range(self.num_results):
            # Sort by most recent score and take top proposals_per_result
            sorted_proposals = sorted(beam_proposals[beam_idx], key=lambda b: b.beam_scores[-1])
            all_valid_proposals.extend(sorted_proposals[: self._proposals_per_result])
            for proposal in sorted_proposals[self._proposals_per_result :]:
                # Extra valid proposals beyond the per-beam quota are dropped.
                self._release_kv_cache(proposal.kv_cache)

        return all_valid_proposals

    def _select_topk_beams(self, proposal_beams: list[BeamState]) -> None:
        """Select top num_results proposals and update state for the next beam step.

        1. Score each proposal beam (mean or last score, per ``score_by``).
        2. Sort by score and keep the top ``num_results`` beams.
        3. Update ``_proposal_outcomes``: result beams get "accepted",
           pruned beams get "Beam pruned".
        4. Write all proposal beams to ``proposal_sequences`` and result
           beams to ``result_sequences`` so ``_save_progress_snapshot``
           captures the current state.

        Args:
            proposal_beams (list[BeamState]): All valid proposal BeamStates from expansion.
        """
        # 1. Score each proposal beam
        scored_proposals = [(i, beam, self._get_aggregated_score(beam)) for i, beam in enumerate(proposal_beams)]

        # 2. Sort by score and keep top num_results
        sorted_proposals = sorted(scored_proposals, key=lambda x: x[2])
        result_indices = {orig_idx for orig_idx, _, _ in sorted_proposals[: self.num_results]}
        selected_beams = [beam for _, beam, _ in sorted_proposals[: self.num_results]]
        selected_beam_ids = {id(beam) for beam in selected_beams}
        for beam in proposal_beams:
            if id(beam) not in selected_beam_ids:
                # Globally pruned proposal branches will not continue.
                self._release_kv_cache(beam.kv_cache)
        for beam in self.beams:
            # Parent beams are replaced by selected child beams after this step.
            self._release_kv_cache(beam.kv_cache)

        self.beams = selected_beams
        self.energy_scores = [score for _, _, score in sorted_proposals[: self.num_results]]

        # 3. Update _proposal_outcomes and _proposal_energy_scores
        self._proposal_outcomes = ["Beam pruned"] * len(proposal_beams)
        self._proposal_energy_scores = [sc for _, _, sc in scored_proposals]
        for idx in result_indices:
            self._proposal_outcomes[idx] = "accepted"

        # 4. Write proposal_sequences and result_sequences for snapshot
        self.target_segment.proposal_sequences = [
            Sequence(
                sequence=beam.running_sequence if self.prepend_prompt else beam.running_sequence[len(self.prompt) :],
                sequence_type=self.target_segment.sequence_type,
            )
            for beam in proposal_beams
        ]
        self._sync_proposal_pools(self.target_segment)
        self.target_segment.result_sequences = [
            Sequence(
                sequence=beam.running_sequence if self.prepend_prompt else beam.running_sequence[len(self.prompt) :],
                sequence_type=self.target_segment.sequence_type,
            )
            for beam in self.beams
        ]

        if self.verbose:
            logger.info(f"Selected top {self.num_results} beams:")
            for i, beam, score in sorted_proposals[: self.num_results]:
                logger.info(f"  [{i}] score={score:.4f}, prompt_len={len(beam.running_sequence)}")

    def _get_aggregated_score(self, beam: BeamState) -> float:
        """Get aggregated score for a beam based on score_by setting."""
        if not beam.beam_scores:
            return float("inf")
        return float(np.mean(beam.beam_scores)) if self.score_by == "mean" else beam.beam_scores[-1]

    def _release_kv_cache(self, kv_cache: Any | None) -> None:
        if not self.use_kv_caching or kv_cache is None:
            return
        self.generator.release_kv_cache(kv_cache)  # type: ignore[attr-defined]

    def _release_current_beam_caches(self) -> None:
        for beam in self.beams:
            # The optimizer is exiting, so remaining beam handles are no longer usable.
            self._release_kv_cache(beam.kv_cache)
            beam.kv_cache = None

    @contextmanager
    def _kv_cache_worker_context(self) -> Iterator[None]:
        """Keep worker-local cache handles alive across beam-search sample calls."""
        if not self.use_kv_caching:
            yield
            return

        from proto_tools.utils import ToolInstance

        with ToolInstance.persist():
            yield

    @contextmanager
    def _cached_generation_context(self) -> Iterator[None]:
        """Bypass ToolPool prompt partitioning while passing worker-local cache handles."""
        if not self.use_kv_caching:
            yield
            return

        # Evo2 KV-cache handles are worker-local, so cached continuations must
        # stay on the persistent worker that created the handle.
        from proto_tools.utils.tool_pool import _pool_executing

        token = _pool_executing.set(True)
        try:
            yield
        finally:
            _pool_executing.reset(token)

    ###########
    # LOGGING #
    ###########

    def _log_beamsearch_progress(self, beam_num: int, beam_tokens: int, tokens_generated: int) -> None:
        """Log beam progress as a multi-line INFO block."""
        logger.info(f"Beam {beam_num}/{self.num_beams} ({beam_tokens} tokens)")
        filter_summary = self._format_filter_summary()
        if filter_summary is not None:
            logger.info(f"  filters: {filter_summary}")
        for line in self._format_scoring_lines():
            logger.info(f"  {line}")
        logger.info(f"  energy:  {self._format_energy_summary()}")
        logger.info(f"  tokens_done={tokens_generated}/{self.target_segment.sequence_length}")

        logger.debug(f"Top {self.num_results} beams by {self.score_by} score:")
        for i, beam in enumerate(self.beams):
            agg_score = self._get_aggregated_score(beam)
            last_score = beam.beam_scores[-1] if beam.beam_scores else float("inf")
            logger.debug(
                f"  [{i}] agg={agg_score:.4f}, last={last_score:.4f}, "
                f"len={len(beam.running_sequence)}: '{beam.running_sequence}'"
            )

        if self.custom_logging:
            self.custom_logging(beam_num, self.segments)

    def _log_beam_generation_start(self, beam_idx: int, beam: BeamState) -> None:
        """Log the start of proposal generation for a beam."""
        logger.debug(f"[Beam {beam_idx}] Generating {self._proposals_per_result} proposals")
        logger.debug(f"  Prompt length: {len(beam.running_sequence)}")
        logger.debug(f"  Batch size: {self.batch_size}")

    def _log_cache_state(self, kv_cache: Any | None) -> None:
        """Log KV cache state for debugging."""
        if kv_cache is not None:
            logger.debug("  Cache: present")
        else:
            logger.debug("  Cache: None (first beam, will build cache)")
