"""
Beam Search Optimizer that performs beam search for sequence generation.

This optimizer splits a single long segment into beams of `beam_length` tokens and
performs beam search, accumulating KV cache state across beams.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry


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


class BeamSearchOptimizerConfig(BaseConfig):
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

        beam_width (int): Number of top sequences to maintain at each step (K in
            beam search terminology). At each beam, the top K sequences by energy
            score are selected to continue. Higher values explore more paths but
            increase computation. Must be at least 1.

        candidates_per_beam (int): Number of candidate sequences to generate per
            beam at each step (N in beam search terminology). Total candidates
            generated per beam is K x N. Higher values increase diversity but
            also increase computation. Must be at least 1.

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
            beams produce invalid (inf/NaN) energy candidates. The optimizer will
            resample beams until each has ``candidates_per_beam`` valid candidates,
            up to this many attempts. Higher values increase robustness but may
            slow down optimization with very restrictive constraints. Must be at
            least 1. Default: ``3``.

        batch_size (int): Optional batch size for generation. If not specified,
            generates all beam_width * candidates_per_beam candidates at once.
            Lower values reduce memory usage but increase generation time.
            Default: None (generates all candidates at once).

        verbose (bool): Whether to print detailed progress information including
            beam energies, selected sequences, and generation statistics at each
            iteration. Default: ``False``.
    """

    # Required parameters
    prompt: str = ConfigField(
        title="Prompt", 
        description="The prompt to start the beam search (e.g. ATCG)"
    )
    beam_width: int = ConfigField(
        ge=1, 
        title="Beam Width", 
        description="Number of top sequences to maintain (K)."
    )
    candidates_per_beam: int = ConfigField(
        ge=1,
        title="Candidates Per Beam",
        description="Number of candidates to generate per beam sequence (N).",
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
    batch_size: Optional[int] = ConfigField(
        default=None,
        ge=1,
        title="Batch Size",
        description="Optional batch size for generation. If None, generates all candidates at once.",
        advanced=True,
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
        hidden=True,
    )

    @model_validator(mode="after")
    def validate_config(self):
        """Validate beam search configuration."""
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if self.batch_size and self.batch_size > self.beam_width * self.candidates_per_beam:
            raise ValueError(f"batch_size={self.batch_size} exceeds total candidates ({self.beam_width * self.candidates_per_beam})")
        return self


@OptimizerRegistry.register(
    key="beam-search",
    label="Beam Search Optimizer",
    config=BeamSearchOptimizerConfig,
    description="Beam search optimizer that generates a single segment with beam search at each boundary",
)
class BeamSearchOptimizer(Optimizer):
    """Beam search optimizer for sequence generation.

    This optimizer implements beam search for sequence optimization where a single target
    segment is generated with beam search. The optimizer maintains K beams (running sequences)
    and generates K x N total candidates at each step by producing N variations per beam.
    After constraint evaluation on the FULL accumulated sequence, only the top K sequences by
    energy are retained for the next step.

    Attributes:
        target_segment (Segment): The target segment to generate with beam search.
        generator (Generator): Single autoregressive generator for sequence generation.
        prompt (str): Initial prompt sequence starting all beams.
        beam_length (int): Tokens per beam.
        beam_width (int): Number of beams to maintain (K).
        candidates_per_beam (int): Candidates generated per beam (N).
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
        ...     beam_width=5,
        ...     candidates_per_beam=10
        ... )
        >>> beam_search = BeamSearchOptimizer(
        ...     target_segment=segment,
        ...     constructs=[construct],
        ...     generators=[generator],
        ...     constraints=[gc_constraint],
        ...     config=config,
        ... )
        >>> beam_search.run()
        >>> top_sequences = beam_search.target_segment.selected_sequences
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
            custom_logging: Optional custom logging function called after each beam search step.
            clear_tool_cache: (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
        """
        if len(generators) != 1:
            raise ValueError(f"BeamSearchOptimizer only supports one generator, but currently has {len(generators)} generators.")
        generator = generators[0]
        generator.assign(target_segment)

        # Store config values required for validation
        self.target_segment: Segment = target_segment
        self.prompt: str = config.prompt
        self.beam_length: int = config.beam_length
        self.generator: Generator = generator

        # Base class init (calls _validate_optimizer)
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_candidates=config.beam_width * config.candidates_per_beam,
            num_selected=config.beam_width,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
        )

        self.prepend_prompt: bool = config.prepend_prompt
        self.beam_width: int = config.beam_width
        self.candidates_per_beam: int = config.candidates_per_beam
        self.score_by: str = config.score_by
        self.use_kv_caching: bool = config.use_kv_caching
        self.max_resample_attempts: int = config.max_resample_attempts
        self.batch_size: Optional[int] = config.batch_size

        # Initialize beam states
        self.beams: List[BeamState] = [BeamState(running_sequence=self.prompt) for _ in range(self.beam_width)]

        # Calculate number of beams based on target segment
        self.num_beams = math.ceil(self.target_segment.sequence_length / self.beam_length)

        # Set up generator for beam search
        self.generator.max_seqlen = len(self.prompt) + self.target_segment.sequence_length
        self.generator.store_kv_cache = self.use_kv_caching
        # Always use cached generation internally
        self.generator.cached_generation = True
        self.generator.batched = True

    def _validate_optimizer(self) -> None:
        """
        Beam Search processes a single target segment with beam search.

        Validation ensures:
        1. Constructs are valid and non-empty
        2. Constraints are valid and have input segments
        3. target_segment is in one of the constructs
        4. Generator is valid and autoregressive
        5. Prompt is not empty
        6. beam_length does not exceed target_segment length
        """
        # Validate constructs list is not empty and contains valid Constructs
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise TypeError(f"Construct {i} has type {type(construct)}, expected Construct")

        # Validate constraints
        if not self.constraints:
            raise ValueError("Constraints list cannot be empty")
        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise TypeError(f"Constraint {i} has type {type(constraint)}, expected Constraint")
            if not constraint.inputs:
                raise RuntimeError(f"Constraint {i} has no input segment(s) assigned")

        # Validate target_segment belongs to one of the constructs
        if self.target_segment not in self.segments:
            raise ValueError(f"target_segment '{self.target_segment.label or 'unlabeled'}' is not in any of the provided constructs")

        # Validate generator is valid and autoregressive
        if not isinstance(self.generator, Generator):
            raise TypeError(f"Generator has type {type(self.generator)}, expected Generator")
        generator_spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self.generator))
        if generator_spec.category != "autoregressive":
            raise ValueError(f"BeamSearchOptimizer requires autoregressive generators. The provided generator '{self.generator.__class__.__name__}' is not autoregressive.")
        
        # Validate prompt is not empty
        if not self.prompt:
            raise ValueError("Prompt for BeamSearchOptimizer cannot be empty")

        # Validate beam_length does not exceed target_segment length
        if self.beam_length > self.target_segment.sequence_length:
            raise ValueError(f"beam_length={self.beam_length} cannot be greater than target_segment length ({self.target_segment.sequence_length})")

    def _capture_initial_state(self) -> None:
        """Capture state and reset BeamSearch-specific state for fresh run."""
        super()._capture_initial_state()
        self.beams = [BeamState(running_sequence=self.prompt) for _ in range(self.beam_width)]

    def _restore_initial_state(self) -> None:
        """Restore to captured state and reset BeamSearch-specific state."""
        super()._restore_initial_state()
        self.beams = [BeamState(running_sequence=self.prompt) for _ in range(self.beam_width)]

    def run(self) -> None:
        """
        Run beam search within a single segment.

        For each beam:
        1. Use K accumulated prompts from previous beams
        2. Generate K x N candidates (N per beam)
        3. Score all candidates using FULL accumulated sequence
        4. Select top K candidates and update beam states for next beam
        """
        self._prepare_run()

        if self.verbose:
            self._log_run_start()

        # Track tokens generated so far
        tokens_generated = 0

        for beam_idx in range(self.num_beams):
            # Calculate tokens to generate this beam (may be less for final beam)
            remaining_tokens = self.target_segment.sequence_length - tokens_generated
            beam_tokens = min(self.beam_length, remaining_tokens)

            # Override generator's num_tokens for this beam
            self.generator.num_tokens = beam_tokens

            prepend_prompt_to_first_beam = self.prepend_prompt and beam_idx == 0

            if self.verbose:
                print(f"\n--- Beam {beam_idx + 1}/{self.num_beams} ({beam_tokens} tokens) ---")
            # Generate and score candidates, resampling until all beams have valid candidates
            candidate_beams = self._generate_and_score_with_resampling(prepend_prompt_to_first_beam)

            # Select top beam_width candidates and update beam states
            self._select_topk_beams(candidate_beams)

            tokens_generated += beam_tokens

            # Log progress
            if self.verbose:
                self._log_beamsearch_progress(beam_idx)

        # Write final sequences to segment
        self.target_segment.selected_sequences = [
            Sequence(
                sequence=beam.running_sequence if self.prepend_prompt else beam.running_sequence[len(self.prompt):],
                sequence_type=self.target_segment.sequence_type
            )
            for beam in self.beams
        ]

        # Save progress snapshot
        self._save_progress_snapshot(time_step=0)

    def _generate_candidates_for_beam(
        self,
        beam_idx: int,
        prepend_prompt: bool = False
    ) -> List[BeamState]:
        """
        Generate candidate BeamStates for a single beam.

        If batch_size is set, generates candidates in batches to manage GPU memory.

        Args:
            beam_idx: Index of the beam to generate candidates for
            prepend_prompt: Whether to prepend prompt to generated sequences

        Returns:
            List of BeamState candidates (length=candidates_per_beam)
        """
        beam = self.beams[beam_idx]
        batch_size = self.batch_size or self.candidates_per_beam

        if self.verbose:
            self._log_beam_generation_start(beam_idx, beam)

        candidates = []
        for batch_start in range(0, self.candidates_per_beam, batch_size):
            batch_count = min(batch_size, self.candidates_per_beam - batch_start)

            # Replicate prompt and KV cache for this batch
            prompts = [beam.running_sequence] * batch_count
            kv_cache = (
                self.generator.replicate_cache(beam.kv_cache, batch_count)
                if self.use_kv_caching and beam.kv_cache
                else None
            )

            if self.verbose and batch_start == 0:
                self._log_cache_state(kv_cache)

            # Generate candidates
            self.generator.sample(prompts=prompts, prepend_prompt=prepend_prompt, old_kv_cache=kv_cache)

            # Collect results from this batch
            kv_caches = self.generator.kv_caches if self.use_kv_caching else [None] * batch_count
            for i in range(batch_count):
                generated_seq = self.target_segment.candidate_sequences[i].sequence
                new_prompt = generated_seq if prepend_prompt else beam.running_sequence + generated_seq

                candidates.append(BeamState(
                    running_sequence=new_prompt,
                    kv_cache=kv_caches[i],
                    beam_scores=beam.beam_scores.copy(),
                ))

        return candidates

    def _generate_and_score_with_resampling(self, prepend_prompt: bool = False) -> List[BeamState]:
        """
        Generate and score candidates, resampling beams until each has valid candidates.

        Args:
            prepend_prompt: Whether to prepend prompt to generated sequences

        Returns:
            List of all valid candidate BeamStates with scores populated

        Raises:
            RuntimeError: If unable to get enough valid candidates after max attempts
        """
        # Track valid candidates per beam
        beam_candidates: Dict[int, List[BeamState]] = {b: [] for b in range(self.beam_width)}

        # Initial generation: Generate candidates for all beams
        all_candidates = []
        for beam_idx in range(self.beam_width):
            candidates = self._generate_candidates_for_beam(beam_idx, prepend_prompt)
            all_candidates.extend(candidates)

        # Score all candidates on their FULL accumulated sequences
        self.target_segment.candidate_sequences = [
            Sequence(sequence=beam.running_sequence, sequence_type=self.target_segment.sequence_type)
            for beam in all_candidates
        ]
        self.score_energy()

        # Collect valid candidates
        for i, (candidate, score) in enumerate(zip(all_candidates, self.energy_scores)):
            if not (math.isinf(score) or math.isnan(score)):
                beam_idx = i // self.candidates_per_beam
                candidate.beam_scores.append(score)
                beam_candidates[beam_idx].append(candidate)

        # Resample beams until each has candidates_per_beam valid candidates
        for attempt in range(1, self.max_resample_attempts + 1):
            beams_to_resample = [b for b in range(self.beam_width)
                                if len(beam_candidates[b]) < self.candidates_per_beam]

            if not beams_to_resample:
                break  # All beams have enough valid candidates

            if self.verbose:
                counts = {b: len(beam_candidates[b]) for b in beams_to_resample}
                print(f"  Resampling {len(beams_to_resample)} beams (attempt {attempt}): counts={counts}")

            for beam_idx in beams_to_resample:
                candidates = self._generate_candidates_for_beam(beam_idx, prepend_prompt)

                # Score candidates on their FULL accumulated sequences
                self.target_segment.candidate_sequences = [
                    Sequence(sequence=beam.running_sequence, sequence_type=self.target_segment.sequence_type)
                    for beam in candidates
                ]
                self.score_energy()

                for candidate, score in zip(candidates, self.energy_scores):
                    if not (math.isinf(score) or math.isnan(score)):
                        candidate.beam_scores.append(score)
                        beam_candidates[beam_idx].append(candidate)

        # Verify each beam has at least candidates_per_beam valid candidates
        insufficient_beams = [b for b in range(self.beam_width)
                             if len(beam_candidates[b]) < self.candidates_per_beam]
        if insufficient_beams:
            counts = {b: len(beam_candidates[b]) for b in insufficient_beams}
            raise RuntimeError(
                f"After {self.max_resample_attempts} attempts, {len(insufficient_beams)} beams could not produce "
                f"{self.candidates_per_beam} valid candidates: {counts}. Constraints may be too restrictive."
            )

        # Flatten and sort by energy to get all valid candidates
        all_valid_candidates = []
        for beam_idx in range(self.beam_width):
            # Sort by most recent score and take top candidates_per_beam
            sorted_candidates = sorted(
                beam_candidates[beam_idx], key=lambda b: b.beam_scores[-1]
            )[: self.candidates_per_beam]
            all_valid_candidates.extend(sorted_candidates)

        return all_valid_candidates

    def _select_topk_beams(self, candidate_beams: List[BeamState]) -> None:
        """
        Select top beam_width candidates and update beam states.

        Args:
            candidate_beams: All valid candidate BeamStates
        """
        # Score candidates using aggregated scores
        scored_candidates = [
            (beam, self._get_aggregated_score(beam))
            for beam in candidate_beams
        ]

        # Select top beam_width and update scores
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1])
        self.beams = [beam for beam, _ in sorted_candidates[:self.beam_width]]
        self.energy_scores = [score for _, score in sorted_candidates[:self.beam_width]]

        if self.verbose:
            print(f"Selected top {self.beam_width} beams:")
            for i, (beam, score) in enumerate(sorted_candidates[:self.beam_width]):
                print(f"  [{i}] score={score:.4f}, prompt_len={len(beam.running_sequence)}")

    def _get_aggregated_score(self, beam: BeamState) -> float:
        """Get aggregated score for a beam based on score_by setting."""
        if not beam.beam_scores:
            return 0.0
        return float(np.mean(beam.beam_scores)) if self.score_by == "mean" else beam.beam_scores[-1]

    ###########
    # LOGGING #
    ###########

    def _log_beamsearch_progress(self, beam_idx: int) -> None:
        """
        Log progress information for a beam during beam search.
        """
        print(f"Completed beam {beam_idx + 1}/{self.num_beams}")
        print(f"Top {self.beam_width} beams by {self.score_by} score:")

        for i, beam in enumerate(self.beams):
            agg_score = self._get_aggregated_score(beam)
            last_score = beam.beam_scores[-1] if beam.beam_scores else 0.0
            print(f"  [{i}] agg={agg_score:.4f}, last={last_score:.4f}, len={len(beam.running_sequence)}: '{beam.running_sequence}'")

        if self.custom_logging:
            self.custom_logging(beam_idx, self.segments)
        sys.stdout.flush()

    def _log_beam_generation_start(self, beam_idx: int, beam: BeamState) -> None:
        """Log the start of candidate generation for a beam."""
        print(f"\n[Beam {beam_idx}] Generating {self.candidates_per_beam} candidates")
        print(f"  Prompt length: {len(beam.running_sequence)}")
        if self.batch_size:
            print(f"  Batch size: {self.batch_size}")

    def _log_cache_state(self, kv_cache: Optional[Dict]) -> None:
        """Log KV cache state for debugging."""
        if kv_cache:
            kv = next(iter(kv_cache['mha'].key_value_memory_dict.values()))
            print(f"  Cache: KV shape={kv.shape}, seqlen_offset={kv_cache['mha'].seqlen_offset}")
        else:
            print("  Cache: None (first beam, will build cache)")

    def _log_run_start(self) -> None:
        """Log beam search configuration at the start of run()."""
        print(f"Processing segment with {self.num_beams} beams (beam_length={self.beam_length})")
        print(f"Total tokens to generate: {self.target_segment.sequence_length}")
        print(f"Beam width: {self.beam_width}, Candidates per beam: {self.candidates_per_beam}")
        print(f"Score by: {self.score_by}")
        print(f"KV caching: {'enabled' if self.use_kv_caching else 'disabled'}")
