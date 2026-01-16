"""
Multi-Segment Beam Search Optimizer that uses beam search across multiple segments in a Construct.
"""

from __future__ import annotations

import copy
import math
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry


class MultiSegmentBeamSearchOptimizerConfig(BaseConfig):
    """Configuration object for MultiSegmentBeamSearchOptimizer.

    This class defines configuration parameters for the multi-segment beam search optimizer,
    which explores sequence space by maintaining multiple candidate sequences (beams) and
    generating extensions for each beam at every segment.

    Attributes:
        prompt (str): Initial prompt sequence to start beam search generation. All
            beams begin from this prompt and extend it autoregressively. For DNA,
            this might be ``"ATCG"``; for proteins, a short amino acid sequence.
            Must be non-empty.

        beam_width (int): Number of top sequences to maintain at each step (K in
            beam search terminology). At each segment, the top K sequences by energy
            score are selected to continue. Higher values explore more paths but
            increase computation. Must be at least 1.

        candidates_per_beam (int): Number of candidate sequences to generate per
            beam at each step (N in beam search terminology). Total candidates
            generated per segment is K x N. Higher values increase diversity but
            also increase computation. Must be at least 1.

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
            least 1. Default: ``10``.

        verbose (bool): Whether to print detailed progress information including
            beam energies, selected sequences, and generation statistics at each
            segment. Default: ``False``.

    Note:
        Beam search explores K x N sequences per segment but only retains the top K
        by energy score. The total number of sequences evaluated grows linearly with
        the number of segments, not exponentially, due to pruning at each step.
    """

    # Required parameters
    prompt: str = ConfigField(
        title="Prompt", description="The prompt to start the beam search (e.g. ATCG)"
    )
    beam_width: int = ConfigField(
        ge=1, title="Beam Width", description="Number of top sequences to maintain (K)."
    )
    candidates_per_beam: int = ConfigField(
        ge=1,
        title="Candidates Per Beam",
        description="Number of candidates to generate per beam sequence (N).",
    )

    # Advanced parameters
    prepend_prompt: bool = ConfigField(
        default=True,
        title="Prepend Prompt",
        description="Whether to prepend the prompt to the generated sequence in the output.",
        advanced=True,
    )
    use_kv_caching: bool = ConfigField(
        default=True,
        title="KV Caching",
        description="Whether to use KV caching for generation. Enables faster sequential generation.",
        advanced=True,
    )
    max_resample_attempts: int = ConfigField(
        default=10,
        ge=1,
        title="Max Resample Attempts",
        description="Maximum number of times to resample beams with invalid (inf/NaN) energies before giving up.",
        advanced=True,
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
        hidden=True,
    )


@OptimizerRegistry.register(
    key="multi-segment-beam-search",
    label="Multi-Segment Beam Search Optimizer",
    config=MultiSegmentBeamSearchOptimizerConfig,
    description="Beam search optimizer that processes multiple segments sequentially with context accumulation",
)
class MultiSegmentBeamSearchOptimizer(Optimizer):
    """Beam search optimizer with context accumulation across multiple segments.

    This optimizer implements beam search for sequence optimization where segments
    in a target construct are treated as beams and generated sequentially. At each
    segment, the top K sequences from previous segments are extended with N new candidates
    each, and the best K sequences overall are selected to continue.

    The optimizer maintains K beams (running sequences) and generates K x N total
    candidates at each segment by producing N variations per beam. After constraint
    evaluation, only the top K sequences by energy are retained for the next segment.

    Attributes:
        target_construct (Construct): The target construct being optimized with beam search.
        generator (Generator): Single autoregressive generator for sequence generation.
        prompt (str): Initial prompt sequence starting all beams.
        beam_width (int): Number of beams to maintain (K).
        candidates_per_beam (int): Candidates generated per beam (N).
        use_kv_caching (bool): Whether KV caching is enabled.
        running_prompts (List[str]): Current accumulated sequences for each beam.
        top_beam_kv_caches (List[Optional[Dict]]): KV cache states for each beam.

    Example:
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> gen_config = Evo2GeneratorConfig(prompts="ATCG", prepend_prompt=True)
        >>> generator = Evo2Generator(config=gen_config)
        >>> construct = Construct([segment1, segment2, segment3])
        >>> config = MultiSegmentBeamSearchOptimizerConfig(
        ...     prompt="ATCG",
        ...     beam_width=5,
        ...     candidates_per_beam=10
        ... )
        >>> beam_search = MultiSegmentBeamSearchOptimizer(
        ...     target_construct=construct,
        ...     constructs=[construct],
        ...     generators=[generator],
        ...     constraints=[gc_constraint],
        ...     config=config,
        ... )
        >>> beam_search.run()
        >>> top_sequences = beam_search.target_construct.joined_sequences
    """

    # Class attribute required by OptimizerRegistry
    config_class = MultiSegmentBeamSearchOptimizerConfig

    def __init__(
        self,
        target_construct: Construct,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: MultiSegmentBeamSearchOptimizerConfig,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
    ) -> None:
        """
        Initialize the Multi-Segment Beam Search Optimizer.

        Args:
            target_construct: The target Construct to generate with beam search.
            constructs: List of Construct objects. The target_construct must be in this list.
            generators: List containing a single autoregressive Generator object (must have category="autoregressive").
            constraints: List of Constraint objects for evaluation (lower scores are better).
            config: Configuration object containing algorithm parameters (prompt, beam_width, candidates_per_beam, etc.).
            custom_logging: Optional custom logging function called after each segment.
            clear_tool_cache: (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
        """
        if len(generators) != 1:
            raise ValueError(f"MultiSegmentBeamSearchOptimizer only supports one generator, but currently has {len(generators)} generators.")
        self.generator = generators[0]
        # Assign generator to first segment of target construct for now (will be reassigned to each segment during run())
        self.generator.assign(target_construct.segments[0])
        self.target_construct: Construct = target_construct
        self.prompt = config.prompt

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
        self.use_kv_caching: bool = config.use_kv_caching
        self.max_resample_attempts: int = config.max_resample_attempts

        # Beam search state parameters (running prompts and corresponding KV caches)
        self.running_prompts: List[str] = [self.prompt] * self.beam_width
        self.top_beam_kv_caches: List[Optional[Dict]] = [None] * self.beam_width

        # IMPORTANT: set max_seqlen to the total target construct length
        total_segment_length = sum(segment.sequence_length for segment in self.target_construct.segments)
        self.generator.max_seqlen = len(self.prompt) + total_segment_length
        # Need to store kv caching as well if kv caching is enabled
        self.generator.store_kv_cache = self.use_kv_caching
        # Always use cached generation internally
        self.generator.cached_generation = True
        self.generator.batched = True

    def _validate_optimizer(self) -> None:
        """
        MultiSegmentBeamSearch processes ALL segments in the target construct sequentially with a single generator.

        Validation ensures:
        1. Constructs are valid and non-empty
        2. Constraints are valid and have input segments
        3. target_construct is in the constructs list and has segments
        4. Generator is valid and autoregressive
        5. Prompt is not empty
        """
        from proto_language.language.generator.generator_registry import GeneratorRegistry
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

        # Validate target_construct is in the constructs list
        if self.target_construct not in self.constructs:
            raise ValueError("target_construct is not in the constructs list")
        if not self.target_construct.segments:
            raise ValueError("target_construct has no segments")

        # Validate generator is valid and autoregressive
        if not isinstance(self.generator, Generator):
            raise TypeError(f"Generator has type {type(self.generator)}, expected Generator")
        generator_spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self.generator))
        if generator_spec.category != "autoregressive":
            raise ValueError(f"MultiSegmentBeamSearchOptimizer requires autoregressive generators. The provided generator '{self.generator.__class__.__name__}' is not autoregressive.")

        # Validate prompt is not empty
        if not self.prompt:
            raise ValueError("Prompt for MultiSegmentBeamSearchOptimizer cannot be empty")

    def _save_progress_snapshot(self, time_step: int) -> None:
        """
        Save snapshot with final optimization state.

        Args:
            time_step: Current step index (always 0 for single final snapshot)
        """
        self.history.append({
            "time_step": time_step,
            "segments_generated": len(self.target_construct.segments),
            "total_segments": len(self.target_construct.segments),
            "energy_scores": self.energy_scores[:self.beam_width].copy() if self.energy_scores else [],
            "constructs": copy.deepcopy(self.constructs)
        })

    def run(self) -> None:
        """
        Run beam search across all segments in target_construct

        For each segment in target_construct:
        1. Use K accumulated prompts from previous segments
        2. Replicate each prompt N times and generate KxN candidates
        3. Score all candidates with constraints (lower is better)
        4. Select top beam_width candidates and extend their prompts for next segment
        """
        if self.verbose:
            print(f"Processing {len(self.target_construct.segments)} segments with beam search")
            print(f"Beam width: {self.beam_width}, Candidates per beam: {self.candidates_per_beam}")
            print(f"KV caching: {'enabled' if self.use_kv_caching else 'disabled'}")

        # Beam search across each segment in target_construct
        for segment_idx, segment in enumerate(self.target_construct.segments):
            # 1. Assign generator to this segment
            self.generator.assign(segment)

            prepend_prompt_to_first_segment = self.prepend_prompt and segment_idx == 0

            # 2. Generate and score candidates, resampling until all beams have valid candidates
            all_kv_caches = self._generate_and_score_with_resampling(segment, prepend_prompt_to_first_segment)

            # 3. Select top beam_width candidates and update running prompts and corresponding KV caches
            top_idx = self._select_topk(segment, all_kv_caches, prepend_prompt_to_first_segment)

            # Log progress
            if self.verbose:
                self._log_beamsearch_progress(segment_idx, segment, top_idx)

        # Save progress snapshot once at the end
        self._save_progress_snapshot(time_step=0)

    def _generate_candidates_for_beam(
        self, segment: Segment, beam_idx: int, prepend_prompt: bool = False
    ) -> Tuple[List, List[Dict]]:
        """
        Generate candidates for a single beam.

        Args:
            segment: The current segment being processed
            beam_idx: Index of the beam to generate candidates for
            prepend_prompt: Whether to prepend prompt to generated sequences

        Returns:
            Tuple of (generated_sequences, kv_caches) for this beam (length=candidates_per_beam each)
        """
        prompt = self.running_prompts[beam_idx]
        replicated_prompts = [prompt] * self.candidates_per_beam

        # Replicate KV cache if enabled
        if self.use_kv_caching:
            cur_kv_cache = self.top_beam_kv_caches[beam_idx]
            replicated_kv_cache = self.generator.replicate_cache(cur_kv_cache, self.candidates_per_beam)
        else:
            replicated_kv_cache = None

        if self.verbose:
            print(f"\n[Beam {beam_idx}] Generating {self.candidates_per_beam} candidates")
            print(f"  Prompt: '{prompt[:50]}...' (len={len(prompt)})")
            if self.use_kv_caching and replicated_kv_cache is not None:
                kv = next(iter(replicated_kv_cache['mha'].key_value_memory_dict.values()))
                offset = replicated_kv_cache['mha'].seqlen_offset
                print("  Cache provided:")
                print(f"    KV shape: {kv.shape}")
                print(f"    KV device: {kv.device}")
                print(f"    seqlen_offset: {offset}")
            else:
                print("  Cache: None (first segment, will build cache)")
            print(f"  prepend_prompt: {prepend_prompt}")

        self.generator.sample(
            prompts=replicated_prompts,
            prepend_prompt=prepend_prompt,
            old_kv_cache=replicated_kv_cache,
        )

        # Collect generated sequences and KV caches
        generated_sequences = [copy.deepcopy(seq) for seq in segment.candidate_sequences[:self.candidates_per_beam]]
        
        if self.verbose:
            sample_seq = segment.candidate_sequences[0].sequence
            print(f"  Generated sample: '{sample_seq[:50]}...' (len={len(sample_seq)})")

        generated_kv_caches = self.generator.kv_caches if self.use_kv_caching else []

        return generated_sequences, generated_kv_caches

    def _score_energy_active_constraints(self) -> None:
        """
        Score energy using only active constraints with all input segments populated.

        Dynamically filters constraints to only evaluate those whose input segments
        all have non-empty candidate pools. This enables multi-segment constraints
        to work correctly as segments are generated sequentially by beam search.
        """
        # Filter to active constraints where all input segments have candidates
        active_constraints = [
            constraint for constraint in self.constraints
            if all(seg.num_candidates > 0 for seg in constraint.inputs)
        ]

        # If no active constraints, set all energy scores to 0 and return
        if not active_constraints:
            self.energy_scores = [0.0] * self.num_candidates
            return

        # Temporarily use filtered constraints for scoring
        orig_constraints = self.constraints
        self.constraints = active_constraints
        self.score_energy()

        # Restore original constraints and weights
        self.constraints = orig_constraints

    def _generate_and_score_with_resampling(self, segment: Segment, prepend_prompt: bool = False) -> List[Dict]:
        """
        Generate and score candidates, resampling beams until each has candidates_per_beam valid candidates.

        Accumulates valid candidates across resampling attempts and selects the best by energy.
        Always generates full batch size (candidates_per_beam) for each resample to maintain efficiency.

        Args:
            segment: Current segment being processed
            prepend_prompt: Whether to prepend prompt to generated sequences

        Returns:
            List of KV caches corresponding to final valid candidates in segment.candidate_sequences

        Raises:
            RuntimeError: If unable to get all beams to produce enough valid candidates after max attempts
        """
        # Track valid candidates per beam
        beam_candidates = {b: [] for b in range(self.beam_width)}  # beam_idx -> [(seq, energy, kv_cache), ...]
        
        # Initial generation: Generate candidates for all beams
        all_sequences = []
        all_kv_caches = []
        for beam_idx in range(self.beam_width):
            sequences, kv_caches = self._generate_candidates_for_beam(segment, beam_idx, prepend_prompt)
            all_sequences.extend(sequences)
            all_kv_caches.extend(kv_caches)

        segment.candidate_sequences = all_sequences
        self._score_energy_active_constraints()

        # Collect valid candidates from initial generation
        for i, energy in enumerate(self.energy_scores):
            if not (math.isinf(energy) or math.isnan(energy)):
                beam_idx = i // self.candidates_per_beam
                beam_candidates[beam_idx].append((
                    segment.candidate_sequences[i],
                    energy,
                    all_kv_caches[i] if self.use_kv_caching else None
                ))
        
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
                # Generate candidates for this single beam
                sequences, kv_caches = self._generate_candidates_for_beam(segment, beam_idx, prepend_prompt)
                segment.candidate_sequences = sequences
                self._score_energy_active_constraints()

                # Collect ALL valid candidates from this generation to maximize selection quality
                for i in range(self.candidates_per_beam):
                    energy = self.energy_scores[i]
                    if not (math.isinf(energy) or math.isnan(energy)):
                        beam_candidates[beam_idx].append((
                            segment.candidate_sequences[i],
                            energy,
                            kv_caches[i] if self.use_kv_caching else None
                        ))
        
        # Verify each beam has at least candidates_per_beam valid candidates
        insufficient_beams = [b for b in range(self.beam_width) 
                             if len(beam_candidates[b]) < self.candidates_per_beam]
        if insufficient_beams:
            counts = {b: len(beam_candidates[b]) for b in insufficient_beams}
            raise RuntimeError(f"After {self.max_resample_attempts} attempts, {len(insufficient_beams)} beams could not produce {self.candidates_per_beam} valid candidates: {counts}. Constraints may be too restrictive.")
        
        # Rebuild segment.candidate_sequences and energy_scores with exactly candidates_per_beam per beam
        # Layout: beam_0_candidates + beam_1_candidates + ... + beam_N_candidates
        segment.candidate_sequences = []
        self.energy_scores = []
        final_kv_caches = []

        for beam_idx in range(self.beam_width):
            # Sort this beam's candidates by energy (lower is better) and take top candidates_per_beam
            beam_cands = sorted(beam_candidates[beam_idx], key=lambda x: x[1])[:self.candidates_per_beam]
            
            for seq, energy, kv in beam_cands:
                # Deep copy only at final reconstruction to avoid unnecessary copies during collection
                segment.candidate_sequences.append(copy.deepcopy(seq))
                self.energy_scores.append(energy)
                final_kv_caches.append(kv)

        return final_kv_caches

    def _select_topk(self, segment: Segment, all_kv_caches: List[Dict], prepend_prompt: bool = False) -> List[int]:
        """
        Select top beam_width candidates by energy and update all beam search state.

        1. Identifies top beam_width candidates by energy (lower is better)
        2. Sets segment's selected_sequences and replicates them as candidates
        3. Updates running prompts by extending with new tokens from selected sequences
        4. Replicates energy_scores to match replicated candidate_sequences

        Args:
            segment: Current segment being processed
            all_kv_caches: Updated KV caches for all generated candidates. Empty if KV caching is disabled.
            prepend_prompt: Whether the prompt was prepended to generated sequences in this segment.
                            If True, selected_seq.sequence already contains the full prompt+generation,
                            so we replace running_prompts entirely. If False, we concatenate.

        Returns:
            List of indices for the top beam_width candidates
        """
        # 1. Get top beam_width candidates by energy
        top_idx = np.argsort(self.energy_scores)[:self.beam_width].tolist()

        # 2. Set selected sequences
        segment.selected_sequences = [segment.candidate_sequences[i] for i in top_idx]

        # 3. Replicate selected sequences as candidates (fresh objects to avoid metadata collision)
        # This is necessary if we want proper constraint evaluation for the next segment if constraints 
        # are applied across multiple segments. This is because constraints applied across multiple 
        # segments are concatenated across the candidate_sequences (batch) dimension.
        segment.candidate_sequences = [
            Sequence(
                sequence=selected_seq.sequence,
                sequence_type=segment.sequence_type,
                valid_chars=segment._valid_chars
            )
            for selected_seq in segment.selected_sequences
            for _ in range(self.candidates_per_beam)
        ]

        # 4. Update running prompts from top candidates (stored in segment.selected_sequences)
        # Candidates are generated sequentially per beam, so: beam_idx = candidate_idx // candidates_per_beam
        if prepend_prompt:
            # First segment with prepend_prompt=True: selected_seq.sequence already contains prompt+generation so replace running_prompts entirely (avoid duplication)
            self.running_prompts = [selected_seq.sequence for selected_seq in segment.selected_sequences]
        else:
            # Subsequent segments: selected_seq.sequence contains only new tokens, concatenate
            self.running_prompts = [
                self.running_prompts[idx // self.candidates_per_beam] + selected_seq.sequence
                for idx, selected_seq in zip(top_idx, segment.selected_sequences)
            ]

        # 5. Replicate energy scores to match replicated candidates
        selected_energies = [self.energy_scores[i] for i in top_idx]
        self.energy_scores = [
            energy for selected_energy in selected_energies
            for energy in [selected_energy] * self.candidates_per_beam
        ]

        # 6. Update top beam KV caches if KV caching is enabled
        if self.use_kv_caching:
            self.top_beam_kv_caches = [all_kv_caches[idx] for idx in top_idx]

        return top_idx

    def _log_beamsearch_progress(self, segment_idx: int, segment: Segment, top_idx: List[int]) -> None:
        """
        Log progress information for a segment during beam search.

        Args:
            segment_idx: Index of the current segment
            segment: The current segment being processed
            top_idx: Indices of the top beam_width selected candidates
        """
        num_prompts = self.beam_width * self.candidates_per_beam
        print(f"\n--- Segment {segment_idx + 1}/{len(self.target_construct.segments)} ---")
        print(f"Generated {segment.num_candidates} candidates using {num_prompts} prompts ({self.beam_width} beams x {self.candidates_per_beam} candidates per beam)")

        for i, sequence in enumerate(segment.candidate_sequences):
            print(f"  [{i}]: {sequence.sequence}")

        print(f"Evaluated {len(self.energy_scores)} candidates")
        best_energy = self.energy_scores[top_idx[0]]
        worst_energy = self.energy_scores[top_idx[-1]]
        print(f"Selected top {self.beam_width} sequences (energy range: {best_energy:.4f} - {worst_energy:.4f})")

        for rank, idx in enumerate(top_idx):
            seq = segment.candidate_sequences[idx]
            energy = self.energy_scores[idx]
            beam_idx = idx // self.candidates_per_beam
            seq_preview = seq.sequence[:50] + ('...' if len(seq.sequence) > 50 else '')
            print(f"  [{rank+1}] From beam {beam_idx}: '{seq_preview}' (energy: {energy:.4f})")

        if self.custom_logging:
            self.custom_logging(segment_idx, self.segments)
        sys.stdout.flush()
