"""
Beam Search Optimizer that uses the beam search algorithm to optimize a single Construct.
"""

from typing import List, Optional, Dict, Callable
import warnings
import copy
import sys
import numpy as np

from pydantic import Field

from ..core import Optimizer, Construct, Constraint, Generator, GeneratorType, Segment
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry


class BeamSearchOptimizerConfig(BaseConfig):
    """Configuration for BeamSearchOptimizer"""
    prepend_prompt: bool = Field(
        default=True,
        description="Whether to prepend the prompt to the generated sequence in the output"
    )
    beam_width: int = Field(
        ge=1,
        description="Number of top sequences to maintain (K)"
    )
    candidates_per_beam: int = Field(
        ge=1,
        description="Number of candidates to generate per beam sequence (N)"
    )
    use_kv_caching: bool = Field(
        default=True,
        description="Whether to use KV caching for generation. Enables faster sequential generation."
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print progress information"
    )


@OptimizerRegistry.register(
    key="beam-search",
    label="Beam Search Optimizer",
    config=BeamSearchOptimizerConfig,
    description="Beam search optimizer that processes segments sequentially with context accumulation",
)
class BeamSearchOptimizer(Optimizer):
    """
    Beam search optimizer with dual-pool design.

    This optimizer implements a beam search where:
    1. Segments are processed one at a time, in order
    2. For each segment, the top beam_width accumulated sequences (from all previous segments) 
       are used as prompts for generation
    3. The generator generates candidates_per_beam proposals per prompt (beam_width x candidates_per_beam total)
    4. Constraints evaluate all candidates (lower energy scores are better)
    5. Top beam_width candidates by energy are selected for the next segment

    Examples:
        Basic beam search with Evo2:
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>>
        >>> gen_config = Evo2GeneratorConfig(num_tokens=100)
        >>> generator = Evo2Generator(config=gen_config)
        >>>
        >>> construct = Construct([segment1, segment2, segment3])
        >>> config = BeamSearchOptimizerConfig(
        ...     beam_width=5,
        ...     candidates_per_beam=10,
        ... )
        >>> beam_search = BeamSearchOptimizer(
        ...     construct=construct,
        ...     generator=generator,
        ...     prompt="",
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     config=config,
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> beam_search.run()
        >>> top_sequences = beam_search.construct.joined_sequences  # Top beam_width full sequences
    """
    # Class attribute required by OptimizerRegistry
    config_class = BeamSearchOptimizerConfig

    def __init__(
        self,
        construct: Construct,
        generator: Generator,
        prompt: str,
        constraints: List[Constraint],
        config: BeamSearchOptimizerConfig,
        constraint_weights: Optional[List[float]] = None,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: bool | List[str] = True,
    ) -> None:
        """
        Initialize the Beam Search Optimizer.

        Args:
            construct: A single Construct object to optimize.
            generator: A single autoregressive Generator object (must have type=GeneratorType.AUTOREGRESSIVE).
            prompt: The DNA sequence prompt to start the beam search.
            constraints: List of Constraint objects for evaluation (lower scores are better).
            config: Configuration object containing algorithm parameters (beam_width, candidates_per_beam, etc.).
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            custom_logging: Optional custom logging function called after each segment.
            clear_tool_cache: (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.
        """
        # Beam Search only works with autoregressive generators with non-empty prompts
        if generator.type != GeneratorType.AUTOREGRESSIVE:
            raise ValueError(f"BeamSearchOptimizer requires autoregressive generators. The provided generator '{generator.__class__.__name__}' is not autoregressive.")
        
        if not prompt:
            raise ValueError("BeamSearchOptimizer requires a non-empty prompt to start beam search.")

        # Required for validation in base class. Each segment is assigned to the single generator for beam search.
        for segment in construct.segments:
            segment._is_assigned = True
            # BeamSearch overwrites segment.candidate_sequences during run()
            if any(seq.sequence for seq in segment.candidate_sequences):
                warnings.warn(f"BeamSearchOptimizer will overwrite {segment.num_candidates} existing candidate(s) in segment '{segment.label or 'unlabeled'}' during run()")
        
        # Required for validation in base class. Assign the generator to the first segment to pass validation
        generator.assign(construct.segments[0])
        
        super().__init__(
            constructs=[construct],
            generators=[generator],
            constraints=constraints,
            constraint_weights=constraint_weights,
            num_candidates=config.beam_width * config.candidates_per_beam,
            num_selected=config.beam_width,
            clear_tool_cache=clear_tool_cache,
        )
        self.construct: Construct = construct
        self.generator: Generator = generator
        self.prepend_prompt: bool = config.prepend_prompt
        self.beam_width: int = config.beam_width
        self.candidates_per_beam: int = config.candidates_per_beam
        self.use_kv_caching: bool = config.use_kv_caching
        self.verbose: bool = config.verbose
        self.custom_logging: Optional[Callable] = custom_logging
        
        # Beam search state parameters (running prompts and corresponding KV caches)
        self.running_prompts: List[str] = [prompt] * self.beam_width
        self.top_beam_kv_caches: List[Optional[Dict]] = [None] * self.beam_width
        
        # IMPORTANT: set max_seqlen to the total construct length!!
        self.generator.max_seqlen = len(prompt) + len(self.segments) * self.generator.num_tokens
        # Need to store kv caching as well if kv caching is enabled
        self.generator.store_kv_cache = self.use_kv_caching
        # Always use cached generation internally
        self.generator.cached_generation = True
        self.generator.batched = True

    def run(self) -> None:
        """
        Run beam search across all segments with context accumulation.

        For each segment:
        1. Use K accumulated prompts from previous segments
        2. Replicate each prompt N times and generate KxN candidates
        3. Score all candidates with constraints (lower is better)
        4. Select top beam_width candidates and extend their prompts for next segment
        """
        if self.verbose:
            print(f"Processing {len(self.construct.segments)} segments with beam search")
            print(f"Beam width: {self.beam_width}, Candidates per beam: {self.candidates_per_beam}")
            print(f"KV caching: {'enabled' if self.use_kv_caching else 'disabled'}")

        # Beam search across each segment
        for segment_idx, segment in enumerate(self.construct.segments):
            # 1. Assign generator to this segment
            self.generator.assign(segment)

            prepend_prompt_to_first_segment = self.prepend_prompt and segment_idx == 0

            # 2. Generate candidates in-place for each beam sequentially and accumulate all candidates and corresponding KV caches
            all_kv_caches = self._generate_candidates(segment, prepend_prompt_to_first_segment)

            # 3. Score all candidates with applicable constraints
            self._score_energy_active_constraints()

            # 4. Select top beam_width candidates and update running prompts and corresponding KV caches
            top_idx = self._select_topk(segment, all_kv_caches, prepend_prompt_to_first_segment)

            # Log progress
            if self.verbose:
                self._log_beamsearch_progress(segment_idx, segment, top_idx)


    def _generate_candidates(self, segment: Segment, prepend_prompt: bool = False) -> List[Dict]:
        """
        Generate candidates in-place for each beam sequentially and accumulate all candidates.
        
        For each beam in running_prompts:
        1. Replicate the prompt and KV cache
        2. Generate candidates_per_beam new sequences
        3. Store candidates and updated KV cache
        
        Finally, sets segment.candidate_sequences to all generated candidates.
        
        Args:
            segment: The current segment being processed
            is_first_segment: Whether this is the first segment being processed
        """
        all_candidates = []
        all_kv_caches = []
        
        for beam_idx, prompt in enumerate(self.running_prompts):
            # Replicate the current beam's prompt to sample candidates
            replicated_prompts = [prompt] * self.candidates_per_beam

            # Replicate the current beam's KV cache if KV caching is enabled
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
                    print(f"  Cache provided:")
                    print(f"    KV shape: {kv.shape}")
                    print(f"    KV device: {kv.device}")
                    print(f"    seqlen_offset: {offset}")
                else:
                    print(f"  Cache: None (first segment, will build cache)")
                print(f"  prepend_prompt: {prepend_prompt}")

            self.generator.sample(
                prompts=replicated_prompts,
                prepend_prompt=prepend_prompt,
                old_kv_cache=replicated_kv_cache,
            )

            # Store generated candidates
            all_candidates.extend([copy.deepcopy(seq) for seq in segment.candidate_sequences[:self.candidates_per_beam]])
            
            if self.verbose:
                sample_seq = segment.candidate_sequences[0].sequence
                print(f"  Generated sample: '{sample_seq[:50]}...' (len={len(sample_seq)})")

            # Store corresponding KV caches if KV caching is enabled
            if self.use_kv_caching:
                all_kv_caches.extend(self.generator.kv_caches)

        # In-place sampling requires manually setting the segment's candidate_sequences
        segment.candidate_sequences = all_candidates
        return all_kv_caches


    def _score_energy_active_constraints(self) -> None:
        """
        Score energy using only active constraints with all input segments populated.
        
        Dynamically filters constraints to only evaluate those whose input segments
        all have non-empty candidate pools. This enables multi-segment constraints
        to work correctly as segments are generated sequentially by beam search.
        """
        # Filter to active constraints where all input segments have candidates
        active_constraints = [
            (constraint, weight)
            for constraint, weight in zip(self.constraints, self.constraint_weights)
            if all(seg.num_candidates > 0 for seg in constraint.inputs)
        ]
        
        # If no active constraints, set all energy scores to 0 and return
        if not active_constraints:
            self.energy_scores = [0.0] * self.num_candidates
            return
        
        # Temporarily use filtered constraints for scoring
        orig_constraints, orig_weights = self.constraints, self.constraint_weights
        self.constraints, self.constraint_weights = zip(*active_constraints)
        self.score_energy(verbose=self.verbose)

        # Restore original constraints and weights
        self.constraints, self.constraint_weights = orig_constraints, orig_weights


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

        # 3. Replicate selected sequences as candidates
        # This is required for subsequent evaluation of constraints applied across multiple segments
        # since constraints applied across multiple segments concatenate across the candidate_sequences (batch) dimension.
        segment.candidate_sequences = [
            seq for selected_seq in segment.selected_sequences
            for seq in [selected_seq] * self.candidates_per_beam
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
        print(f"\n--- Segment {segment_idx + 1}/{len(self.construct.segments)} ---")
        print(f"Generated {segment.num_candidates} candidates using {num_prompts} prompts ({self.beam_width} beams x {self.candidates_per_beam} candidates per beam)")
        
        for i, sequence in enumerate(segment.candidate_sequences):
            seq_preview = sequence.sequence[:50] + ('...' if len(sequence.sequence) > 50 else '')
            print(f"  [{i}]: {seq_preview}")
        
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