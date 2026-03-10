"""
Beam Search Optimizer that performs beam search for sequence generation.

This optimizer splits a single long segment into beams of `beam_length` tokens and
performs beam search, accumulating KV cache state across beams.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
from pydantic import model_validator

logger = logging.getLogger(__name__)

from proto_language.base_config import BaseOptimizerConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.language.optimizer.optimizer_registry import optimizer


@dataclass
class BeamState:
    """State for a single beam during beam search.

    Attributes:
        running_sequence: Accumulated sequence (initial prompt + all generated tokens so far)
        kv_cache: KV cache state for this beam (None if KV caching disabled)
        beam_scores: Per-beam energy scores for score aggregation
    """

    running_sequence: str
    kv_cache: Optional[Dict] = None
    beam_scores: List[float] = field(default_factory=list)


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

        num_results (int): Number of result sequences (beam width) to maintain at each
            step. At each beam boundary, the top ``num_results`` sequences by
            energy score are selected to continue. Higher values explore more
            paths but increase computation. Must be at least 1.

        proposals_per_result (int): Number of proposal sequences to generate per
            result sequence (beam) at each beam search step. Total proposals per step is
            ``num_results x proposals_per_result``. Higher values increase
            diversity but also increase computation. Must be at least 1.

        score_by (str): How to aggregate beam scores when selecting beams.
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
            generators (e.g., Evo2). Default: ``True``.

        max_resample_attempts (int): Maximum number of resampling attempts when
            beams produce invalid (inf/NaN) energy proposals. The optimizer will
            resample beams until each has ``proposals_per_result`` valid proposals,
            up to this many attempts. Higher values increase robustness but may
            slow down optimization with very restrictive constraints. Must be at
            least 1. Default: ``3``.

        verbose (bool): Whether to print detailed progress information including
            beam energies, result sequences, and generation statistics at each
            iteration. Default: ``False``.
    """

    # Required parameters
    prompt: str = ConfigField(
        title="Prompt",
        description="The prompt to start the beam search (e.g. ATCG)"
    )
    num_results: Optional[int] = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Candidate designs (beam width) for this optimizer. Overrides program-level count.",
        advanced=True,
    )
    proposals_per_result: int = ConfigField(
        ge=1,
        title="Proposals Per Result",
        description="Number of proposals to generate per result sequence at each beam step.",
    )

    # Generation parameters
    beam_length: int = ConfigField(
        ge=1,
        title="Beam Length",
        description="Number of tokens to generate per beam.",
    )
    score_by: Literal["mean", "last"] = ConfigField(
        default="mean",
        title="Score By",
        description="How to aggregate beam scores: 'mean' (average all beams) or 'last' (use most recent).",
    )
    prepend_prompt: bool = ConfigField(
        default=True,
        title="Prepend Prompt",
        description="Whether to prepend the prompt to the generated sequence in the output.",
    )
    use_kv_caching: bool = ConfigField(
        default=True,
        title="KV Caching",
        description="Whether to use KV caching for generation. Enables faster sequential generation.",
    )

    # Advanced parameters
    max_resample_attempts: int = ConfigField(
        default=3,
        ge=1,
        title="Max Resample Attempts",
        description="Maximum number of times to resample beams with invalid (inf/NaN) energies before giving up.",
        advanced=True,
    )

    @model_validator(mode="after")
    def validate_config(self):
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
)
class BeamSearchOptimizer(Optimizer):
    """Beam search optimizer for sequence generation.

    This optimizer implements beam search for sequence optimization where a single target
    segment is generated with beam search. The optimizer maintains K beams (running sequences)
    and generates K x N total proposals at each step by producing N variations per beam.
    After constraint evaluation on the FULL accumulated sequence, only the top K sequences by
    energy are retained for the next step.

    Attributes:
        target_segment (Segment): The target segment to generate with beam search.
        generator (Generator): Single autoregressive generator for sequence generation.
        prompt (str): Initial prompt sequence starting all beams.
        beam_length (int): Tokens per beam.
        num_results (int): Number of beams to maintain (K).
        proposals_per_result (int): Proposals generated per result sequence (N).
        score_by (str): Score aggregation method ('mean' or 'last').
        use_kv_caching (bool): Whether KV caching is enabled.
        beams (List[BeamState]): Current beam states.

    Example:
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> gen_config = Evo2GeneratorConfig(prompts="ATCG", prepend_prompt=True)
        >>> generator = Evo2Generator(config=gen_config)
        >>> segment = Segment(length=10000, sequence_type="dna")
        >>> construct = Construct([segment])
        >>> config = BeamSearchOptimizerConfig(
        ...     prompt="ATCG",
        ...     beam_length=2000,
        ...     num_results=5,
        ...     proposals_per_result=10
        ... )
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

    def __init__(
        self,
        target_segment: Segment,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: BeamSearchOptimizerConfig,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
    ) -> None:
        """
        Initialize the Beam Search Optimizer.

        Args:
            target_segment: The specific Segment to optimize with beam search. Must belong to one of the constructs.
            constructs: List of Construct objects. The target_segment must belong to one of these constructs.
            generators: List containing a single autoregressive Generator object (must have category="autoregressive").
            constraints: List of Constraint objects for evaluation (lower scores are better).
            config: Configuration object containing algorithm parameters.
            custom_logging: Optional callback called at tracked beams (governed by ``tracking_interval``).
            clear_tool_cache: (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
        """
        if len(generators) != 1:
            raise ValueError(f"BeamSearchOptimizer only supports one generator, but currently has {len(generators)} generators.")
        generator = generators[0]
        generator.assign(target_segment)

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
        )

        self.prepend_prompt: bool = config.prepend_prompt
        self.score_by: str = config.score_by
        self.max_resample_attempts: int = config.max_resample_attempts
        self.batch_size: int = self.generator.batch_size

        if self.num_results is not None:
            self.beams: List[BeamState] = [BeamState(running_sequence=self.prompt) for _ in range(self.num_results)]
        else:
            self.beams: List[BeamState] = []

        # Calculate number of beams based on target segment
        self.num_beams = math.ceil(self.target_segment.sequence_length / self.beam_length)

        # Override base class num_steps for progress tracking
        self.num_steps = self.num_beams

        # Set up generator for beam search
        # Pre-allocate max_seqlen for the full optimization - vortex doesn't support
        self.generator.max_seqlen = len(self.prompt) + self.target_segment.sequence_length
        self.generator.store_kv_cache = self.use_kv_caching
        self.generator.cached_generation = True
        self.generator.batched = True

    def _validate_optimizer(self) -> None:
        """Validate beam search optimizer configuration.

        Extends base validation with beam-search-specific checks:
        target_segment membership, autoregressive generator, KV caching
        interface, non-empty prompt, and beam_length bounds.
        """
        super()._validate_optimizer()
        self._validate_target_segment(self.target_segment)

        # Generator must be autoregressive
        generator_spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self.generator))
        if generator_spec.category != "autoregressive":
            raise ValueError(
                f"BeamSearchOptimizer requires autoregressive generators. "
                f"The provided generator '{self.generator.__class__.__name__}' is not autoregressive."
            )

        # KV caching support (if enabled)
        if self.use_kv_caching:
            if not hasattr(self.generator, 'replicate_cache') or not callable(getattr(self.generator, 'replicate_cache', None)):
                raise ValueError(
                    f"Generator '{self.generator.__class__.__name__}' does not support KV caching (missing replicate_cache method). "
                    f"Set use_kv_caching=False or use a generator that supports KV caching."
                )
            if not hasattr(self.generator, 'kv_caches'):
                raise ValueError(
                    f"Generator '{self.generator.__class__.__name__}' does not support KV caching (missing kv_caches attribute). "
                    f"Set use_kv_caching=False or use a generator that supports KV caching."
                )

        # Prompt + beam_length
        if not self.prompt:
            raise ValueError("Prompt for BeamSearchOptimizer cannot be empty")
        if self.beam_length > self.target_segment.sequence_length:
            raise ValueError(
                f"beam_length={self.beam_length} cannot be greater than "
                f"target_segment length ({self.target_segment.sequence_length})"
            )

    def _capture_initial_state(self) -> None:
        """Capture state and reset BeamSearch-specific state for fresh run."""
        super()._capture_initial_state()
        self.beams = [BeamState(running_sequence=self.prompt) for _ in range(self.num_results)]

    def _restore_initial_state(self) -> None:
        """Restore to captured state and reset BeamSearch-specific state."""
        super()._restore_initial_state()
        self.beams = [BeamState(running_sequence=self.prompt) for _ in range(self.num_results)]

    def run(self) -> None:
        """
        Run beam search within a single segment.

        For each beam:
        1. Use K accumulated prompts from previous beams
        2. Generate K x N proposals (N per beam)
        3. Score all proposals using FULL accumulated sequence
        4. Select top K proposals and update beam states for next beam
        """
        self._prepare_run()

        # BeamSearch always starts from its configured prompt.
        if any(
            seq.sequence
            for seg in self.segments
            for seq in seg.result_sequences
        ):
            logger.warning("BeamSearchOptimizer starts from its configured prompt and overwrites existing sequences/prompts")

        # BeamSearch starts from empty prompt; no meaningful initial snapshot
        self.energy_scores = [float("inf")] * self.num_results

        if self.verbose:
            self._log_run_start()

        # Track tokens generated so far
        tokens_generated = 0

        for beam_num in range(1, self.num_beams + 1):
            # Calculate tokens to generate this beam (may be less for final beam)
            remaining_tokens = self.target_segment.sequence_length - tokens_generated
            beam_tokens = min(self.beam_length, remaining_tokens)

            prepend_prompt_to_first_beam = self.prepend_prompt and beam_num == 1

            # Generate and score proposals, resampling until all beams have valid proposals
            proposal_beams = self._generate_and_score_with_resampling(prepend_prompt_to_first_beam, beam_tokens)

            # Select top num_results proposals and update beam states
            self._select_topk_beams(proposal_beams)

            # Save per-beam snapshot (result_sequences set by _select_topk_beams)
            if beam_num % self.tracking_interval == 0 or beam_num == self.num_beams:
                self._save_progress_snapshot(time_step=beam_num)
                self._log_beamsearch_progress(beam_num, beam_tokens)

            tokens_generated += beam_tokens

        # Write final sequences to segment (same content as last _select_topk_beams
        # snapshot, so no additional snapshot needed)
        self.target_segment.result_sequences = [
            Sequence(
                sequence=beam.running_sequence if self.prepend_prompt else beam.running_sequence[len(self.prompt):],
                sequence_type=self.target_segment.sequence_type
            )
            for beam in self.beams
        ]

    def _generate_proposals_for_beam(
        self,
        beam_idx: int,
        prepend_prompt: bool = False,
        num_tokens: Optional[int] = None,
    ) -> List[BeamState]:
        """
        Generate proposal BeamStates for a single beam.

        Generates proposals in batches (sized by generator batch_size) to manage GPU memory.

        Args:
            beam_idx: Index of the beam to generate proposals for
            prepend_prompt: Whether to prepend prompt to generated sequences

        Returns:
            List of BeamState proposals (length=proposals_per_result)
        """
        beam = self.beams[beam_idx]

        if self.verbose:
            self._log_beam_generation_start(beam_idx, beam)

        proposals = []
        for batch_start in range(0, self._proposals_per_result, self.batch_size):
            batch_count = min(self.batch_size, self._proposals_per_result - batch_start)

            # Replicate prompt and KV cache for this batch
            prompts = [beam.running_sequence] * batch_count
            kv_cache = (
                self.generator.replicate_cache(beam.kv_cache, batch_count)
                if self.use_kv_caching and beam.kv_cache
                else None
            )

            # Resize proposal pool to match batch for zip(strict=True) compatibility
            self.target_segment.proposal_sequences = [
                Sequence(sequence="", sequence_type=self.target_segment.sequence_type)
                for _ in range(batch_count)
            ]
            self._sync_proposal_pools(self.target_segment)

            if self.verbose and batch_start == 0:
                self._log_cache_state(kv_cache)

            # Generate proposals
            self.generator.sample(
                prompts=prompts,
                prepend_prompt=prepend_prompt,
                num_tokens=num_tokens,
                old_kv_cache=kv_cache,
            )

            # Collect results from this batch
            kv_caches = self.generator.kv_caches if self.use_kv_caching else [None] * batch_count
            for i in range(batch_count):
                generated_seq = self.target_segment.proposal_sequences[i].sequence
                new_prompt = generated_seq if prepend_prompt else beam.running_sequence + generated_seq

                proposals.append(BeamState(
                    running_sequence=new_prompt,
                    kv_cache=kv_caches[i],
                    beam_scores=beam.beam_scores.copy(),
                ))

        return proposals

    def _generate_and_score_with_resampling(self, prepend_prompt: bool = False, num_tokens: Optional[int] = None) -> List[BeamState]:
        """
        Generate and score proposals, resampling beams until each has valid proposals.

        Args:
            prepend_prompt: Whether to prepend prompt to generated sequences

        Returns:
            List of all valid proposal BeamStates with scores populated

        Raises:
            RuntimeError: If unable to get enough valid proposals after max attempts
        """
        # Track valid proposals per beam
        beam_proposals: Dict[int, List[BeamState]] = {b: [] for b in range(self.num_results)}

        # Initial generation: Generate proposals for all beams
        all_proposals = []
        for beam_idx in range(self.num_results):
            proposals = self._generate_proposals_for_beam(beam_idx, prepend_prompt, num_tokens)
            all_proposals.extend(proposals)

        # Score all proposals on their FULL accumulated sequences
        self.target_segment.proposal_sequences = [
            Sequence(sequence=beam.running_sequence, sequence_type=self.target_segment.sequence_type)
            for beam in all_proposals
        ]
        self._sync_proposal_pools(self.target_segment)
        self.score_energy()

        # Collect valid proposals (those that passed filter constraints)
        for i, (proposal, score) in enumerate(zip(all_proposals, self.energy_scores)):
            if self._proposal_outcomes[i] == "accepted":
                beam_idx = i // self._proposals_per_result
                proposal.beam_scores.append(score)
                beam_proposals[beam_idx].append(proposal)

        # Resample beams until each has proposals_per_result valid proposals
        for attempt in range(1, self.max_resample_attempts + 1):
            beams_to_resample = [b for b in range(self.num_results)
                                if len(beam_proposals[b]) < self._proposals_per_result]

            if not beams_to_resample:
                break  # All beams have enough valid proposals

            if self.verbose:
                counts = {b: len(beam_proposals[b]) for b in beams_to_resample}
                logger.info(f"Resampling {len(beams_to_resample)} beams (attempt {attempt}): counts={counts}")

            for beam_idx in beams_to_resample:
                proposals = self._generate_proposals_for_beam(beam_idx, prepend_prompt, num_tokens)

                # Score proposals on their FULL accumulated sequences
                self.target_segment.proposal_sequences = [
                    Sequence(sequence=beam.running_sequence, sequence_type=self.target_segment.sequence_type)
                    for beam in proposals
                ]
                self._sync_proposal_pools(self.target_segment)
                self.score_energy()

                for j, (proposal, score) in enumerate(zip(proposals, self.energy_scores)):
                    if self._proposal_outcomes[j] == "accepted":
                        proposal.beam_scores.append(score)
                        beam_proposals[beam_idx].append(proposal)

        # Verify each beam has at least proposals_per_result valid proposals
        insufficient_beams = [b for b in range(self.num_results)
                             if len(beam_proposals[b]) < self._proposals_per_result]
        if insufficient_beams:
            counts = {b: len(beam_proposals[b]) for b in insufficient_beams}
            raise RuntimeError(
                f"After {self.max_resample_attempts} attempts, {len(insufficient_beams)} beams could not produce "
                f"{self._proposals_per_result} valid proposals: {counts}. Constraints may be too restrictive."
            )

        # Flatten and sort by energy to get all valid proposals
        all_valid_proposals = []
        for beam_idx in range(self.num_results):
            # Sort by most recent score and take top proposals_per_result
            sorted_proposals = sorted(
                beam_proposals[beam_idx], key=lambda b: b.beam_scores[-1]
            )[: self._proposals_per_result]
            all_valid_proposals.extend(sorted_proposals)

        return all_valid_proposals

    def _select_topk_beams(self, proposal_beams: List[BeamState]) -> None:
        """Select top num_results proposals and update state for the next beam step.

        1. Score each proposal beam (mean or last score, per ``score_by``).
        2. Sort by score and keep the top ``num_results`` beams.
        3. Update ``_proposal_outcomes`` — result beams get "accepted",
           pruned beams get "Beam pruned".
        4. Write all proposal beams to ``proposal_sequences`` and result
           beams to ``result_sequences`` so ``_save_progress_snapshot``
           captures the current state.

        Args:
            proposal_beams: All valid proposal BeamStates from expansion.
        """
        # 1. Score each proposal beam
        scored_proposals = [
            (i, beam, self._get_aggregated_score(beam))
            for i, beam in enumerate(proposal_beams)
        ]

        # 2. Sort by score and keep top num_results
        sorted_proposals = sorted(scored_proposals, key=lambda x: x[2])
        result_indices = {orig_idx for orig_idx, _, _ in sorted_proposals[:self.num_results]}
        self.beams = [beam for _, beam, _ in sorted_proposals[:self.num_results]]
        self.energy_scores = [score for _, _, score in sorted_proposals[:self.num_results]]

        # 3. Update _proposal_outcomes and _proposal_energy_scores
        self._proposal_outcomes = ["Beam pruned"] * len(proposal_beams)
        self._proposal_energy_scores = [sc for _, _, sc in scored_proposals]
        for idx in result_indices:
            self._proposal_outcomes[idx] = "accepted"

        # 4. Write proposal_sequences and result_sequences for snapshot
        self.target_segment.proposal_sequences = [
            Sequence(
                sequence=beam.running_sequence if self.prepend_prompt else beam.running_sequence[len(self.prompt):],
                sequence_type=self.target_segment.sequence_type,
            )
            for beam in proposal_beams
        ]
        self._sync_proposal_pools(self.target_segment)
        self.target_segment.result_sequences = [
            Sequence(
                sequence=beam.running_sequence if self.prepend_prompt else beam.running_sequence[len(self.prompt):],
                sequence_type=self.target_segment.sequence_type,
            )
            for beam in self.beams
        ]

        if self.verbose:
            logger.info(f"Selected top {self.num_results} beams:")
            for i, beam, score in sorted_proposals[:self.num_results]:
                logger.info(f"  [{i}] score={score:.4f}, prompt_len={len(beam.running_sequence)}")

    def _get_aggregated_score(self, beam: BeamState) -> float:
        """Get aggregated score for a beam based on score_by setting."""
        if not beam.beam_scores:
            return float('inf')
        return float(np.mean(beam.beam_scores)) if self.score_by == "mean" else beam.beam_scores[-1]

    ###########
    # LOGGING #
    ###########

    def _log_beamsearch_progress(self, beam_num: int, beam_tokens: int) -> None:
        """Log progress information for a beam during beam search."""
        if self.verbose:
            logger.info(f"Beam {beam_num}/{self.num_beams} ({beam_tokens} tokens)")
            logger.debug(f"Completed beam {beam_num}/{self.num_beams}")
            logger.debug(f"Top {self.num_results} beams by {self.score_by} score:")

            for i, beam in enumerate(self.beams):
                agg_score = self._get_aggregated_score(beam)
                last_score = beam.beam_scores[-1] if beam.beam_scores else float("inf")
                logger.debug(f"  [{i}] agg={agg_score:.4f}, last={last_score:.4f}, len={len(beam.running_sequence)}: '{beam.running_sequence}'")

        if self.custom_logging:
            self.custom_logging(beam_num, self.segments)

    def _log_beam_generation_start(self, beam_idx: int, beam: BeamState) -> None:
        """Log the start of proposal generation for a beam."""
        logger.debug(f"[Beam {beam_idx}] Generating {self._proposals_per_result} proposals")
        logger.debug(f"  Prompt length: {len(beam.running_sequence)}")
        logger.debug(f"  Batch size: {self.batch_size}")

    def _log_cache_state(self, kv_cache: Optional[Dict]) -> None:
        """Log KV cache state for debugging."""
        if kv_cache:
            kv = next(iter(kv_cache['mha'].key_value_memory_dict.values()))
            logger.debug(f"  Cache: KV shape={kv.shape}, seqlen_offset={kv_cache['mha'].seqlen_offset}")
        else:
            logger.debug("  Cache: None (first beam, will build cache)")

    def _log_run_start(self) -> None:
        """Log beam search configuration at the start of run()."""
        logger.debug(f"Processing segment with {self.num_beams} beams (beam_length={self.beam_length})")
        logger.debug(f"Total tokens to generate: {self.target_segment.sequence_length}")
        logger.debug(f"Beam width: {self.num_results}, Proposals per beam: {self._proposals_per_result}")
        logger.debug(f"Score by: {self.score_by}")
        logger.debug(f"KV caching: {'enabled' if self.use_kv_caching else 'disabled'}")
