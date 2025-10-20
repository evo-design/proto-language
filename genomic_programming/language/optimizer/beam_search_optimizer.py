"""
Beam Search Optimizer

Beam search optimizer that processes segments sequentially with context accumulation.
"""

from typing import List, Optional, Tuple
import numpy as np

from pydantic import Field

from ..core import Optimizer, Construct, Constraint, Generator, Segment
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry


class BeamSearchOptimizerConfig(BaseConfig):
    """Configuration for BeamSearchOptimizer"""
    beam_width: int = Field(
        ge=1,
        description="Number of top sequences to maintain (K)"
    )
    num_candidates: int = Field(
        ge=1,
        description="Number of candidates to generate per beam sequence (N)"
    )
    verbose: bool = Field(
        default=True,
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
    2. For each segment, the top K accumulated sequences (from all previous segments) 
       are used as prompts for generation
    3. The generator generates num_candidates proposals per prompt (K x N total)
    4. Constraints evaluate all candidates (lower energy scores are better)
    5. Top beam_width candidates by energy are selected for the next segment

    Examples:
        Basic beam search with Evo2:
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>>
        >>> gen_config = Evo2GeneratorConfig(n_tokens=100, prepend_prompt=False)
        >>> generator = Evo2Generator(config=gen_config)
        >>>
        >>> construct = Construct([segment1, segment2, segment3])
        >>> config = BeamSearchOptimizerConfig(
        ...     beam_width=5,
        ...     num_candidates=10,
        ... )
        >>> beam_search = BeamSearchOptimizer(
        ...     construct=construct,
        ...     generator=generator,
        ...     prompt="",
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     config=config,
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> beam_search.sample()
        >>> top_sequences = beam_search.construct.joined_sequences  # Top K full sequences
    """

    config_class = BeamSearchOptimizerConfig

    def __init__(
        self,
        construct: Construct,
        generator: Generator,
        prompt: str,
        constraints: List[Constraint],
        config: BeamSearchOptimizerConfig,
        constraint_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize the Beam Search Optimizer.

        Args:
            construct: A single Construct object to optimize.
            generator: A single autoregressive Generator object (must have autoregressive=True).
            prompt: The initial prompt to start the beam search from (typically empty string).
            constraints: List of Constraint objects for evaluation (lower scores are better).
            config: Configuration object containing algorithm parameters (beam_width, num_candidates, etc.).
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
        """
        if not generator.autoregressive:
            raise ValueError(f"BeamSearchOptimizer requires autoregressive generators. The provided generator '{generator.__class__.__name__}' is not autoregressive.")
        
        # Mark all segments as assigned and set generator before super().__init__()
        # This is needed for validation in base class
        for segment in construct.segments:
            segment._is_assigned = True
        generator._assigned_segment = construct.segments
        
        # Initialize with dual-pool design
        super().__init__(
            constructs=[construct],
            generators=[generator],
            constraints=constraints,
            constraint_weights=constraint_weights,
            num_candidates=config.beam_width * config.num_candidates,
            num_selected=config.beam_width,
        )
        self.construct = construct
        self.generator = generator
        self.beam_width = config.beam_width
        self.candidates_per_beam = config.num_candidates
        self.verbose = config.verbose

        # Initialize running prompts for each beam
        self.running_prompts = [prompt] * self.beam_width
        # Clear candidate sequence content - BeamSearch creates them during sample()
        for segment in self.segments:
            for seq in segment.candidate_sequences:
                if seq.sequence:
                    if self.verbose:
                        print(f"Warning: Clearing pre-populated sequence in segment '{segment.label or 'unlabeled'}'. "
                              f"BeamSearchOptimizer will generate candidates during sample().")
                seq.sequence = ""

    def sample(self) -> None:
        """
        Run beam search across all segments with context accumulation.

        For each segment:
        1. Use K accumulated prompts from previous segments
        2. Replicate each prompt N times and generate KxN candidates
        3. Score all candidates with constraints (lower is better)
        4. Select top K candidates and extend their prompts for next segment
        
        The final top K full sequences are available via construct.joined_sequences.
        """
        if self.verbose:
            print(f"Processing {len(self.construct.segments)} segments with beam search")
            print(f"Beam width: {self.beam_width}, Candidates per beam: {self.candidates_per_beam}")

        # Process each segment sequentially
        for segment_idx, segment in enumerate(self.construct.segments):
            if self.verbose:
                print(f"\n--- Processing Segment {segment_idx + 1}/{len(self.construct.segments)} ---")

            # 1. Assign generator to this segment
            self.generator._assigned_segment = segment

            # 2. Prepare prompts: replicate each running prompt candidates_per_beam times
            all_prompts, beam_indices = self._prepare_beam_prompts()

            if self.verbose:
                print(f"Using {len(all_prompts)} prompts ({self.beam_width} beams x {self.candidates_per_beam} candidates)")

            # 3. Generate candidates (writes to candidate_sequences)
            self.generator.sample(prompt_seqs=all_prompts)

            if self.verbose:
                print(f"Generated {segment.num_candidates} candidates")
                if segment.num_candidates <= 10:
                    for i, sequence in enumerate(segment.candidate_sequences):
                        seq_preview = sequence.sequence[:50] + ('...' if len(sequence.sequence) > 50 else '')
                        print(f"  [{i}]: {seq_preview}")

            # 4. Score all candidates with applicable constraints
            # Populate previous segments' candidate pools for multi-segment constraint evaluation
            self._populate_previous_segments_for_scoring(segment_idx)
            self._score_energy_filtered()

            if self.verbose:
                print(f"Evaluated {len(self.energy_scores)} candidates")

            # 5. Select top beam_width candidates by energy
            top_idx = np.argsort(self.energy_scores)[:self.beam_width].tolist()
            self._set_selected_sequences(segment, top_idx)

            # 6. Update running prompts by extending with new tokens from selected sequences
            self._update_running_prompts(segment, top_idx, beam_indices)

            if self.verbose:
                best_energy = self.energy_scores[top_idx[0]]
                worst_energy = self.energy_scores[top_idx[-1]]
                print(f"Selected top {self.beam_width} sequences (energy range: {best_energy:.4f} - {worst_energy:.4f})")

                # Show selected sequences
                for rank, idx in enumerate(top_idx):
                    seq = segment.candidate_sequences[idx]
                    energy = self.energy_scores[idx]
                    seq_preview = seq.sequence[:50] + ('...' if len(seq.sequence) > 50 else '')
                    print(f"  [{rank+1}] From beam {beam_indices[idx]}: '{seq_preview}' (energy: {energy:.4f})")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Beam search complete")
            print(f"Top {self.beam_width} full sequences available via construct.joined_sequences")
            print(f"{'='*60}")

    def _prepare_beam_prompts(self) -> Tuple[List[str], List[int]]:
        """
        Prepare prompts for beam search by replicating each beam's accumulated prompt N times.
        
        Returns:
            Tuple of (prompts, beam_indices) where:
                - prompts: List of KxN prompt strings for generation
                - beam_indices: List tracking which beam (0 to K-1) each candidate originated from
        """
        prompts = []
        beam_indices = []
        for beam_idx, prompt in enumerate(self.running_prompts):
            prompts.extend([prompt] * self.candidates_per_beam)
            beam_indices.extend([beam_idx] * self.candidates_per_beam)
        return prompts, beam_indices

    def _set_selected_sequences(self, segment: Segment, top_idx: List[int]) -> None:
        """Set segment's selected sequences to the top K candidates by energy."""
        segment.selected_sequences = [segment.candidate_sequences[i] for i in top_idx]

    def _update_running_prompts(self, segment: Segment, top_idx: List[int], beam_indices: List[int]) -> None:
        """
        Update running prompts by extending with newly generated tokens.
        
        For each selected candidate, extends its originating beam's prompt with new tokens.
        """
        self.running_prompts = [
            self.running_prompts[beam_indices[idx]] + segment.candidate_sequences[idx].sequence
            for idx in top_idx
        ]

    def _populate_previous_segments_for_scoring(self, current_segment_idx: int) -> None:
        """
        Populate previous segments' candidate pools with their selected sequences.
        
        Constraints evaluate candidate_sequences, but in beam search previous segments
        need to contain their K selected sequences (current beams), not stale KxN candidates.
        This replicates each selected sequence N times to match the current segment's pool size.
        
        Args:
            current_segment_idx: Index of segment currently being processed
        """
        for prev_segment in self.construct.segments[:current_segment_idx]:
            # Replicate each of K selected sequences N times to get KxN candidates
            prev_segment.candidate_sequences = [
                seq for selected_seq in prev_segment.selected_sequences
                for seq in [selected_seq] * self.candidates_per_beam
            ]
    
    def _score_energy_filtered(self) -> None:
        """
        Score energy using only constraints with all input segments populated.
        
        Dynamically filters constraints to only evaluate those whose input segments
        all have non-empty candidate pools. This enables multi-segment constraints
        to work correctly as segments are processed sequentially.
        """
        # Filter to constraints where all input segments have candidates
        applicable = [
            (constraint, weight)
            for constraint, weight in zip(self.constraints, self.constraint_weights)
            if all(seg.num_candidates > 0 for seg in constraint.inputs)
        ]
        
        if not applicable:
            self.energy_scores = [0.0] * (self.beam_width * self.candidates_per_beam)
            return
        
        if self.verbose and len(applicable) < len(self.constraints):
            print(f"  Applying {len(applicable)}/{len(self.constraints)} constraints (others waiting for segments)")
        
        # Temporarily use filtered constraints for scoring
        orig_constraints, orig_weights = self.constraints, self.constraint_weights
        self.constraints, self.constraint_weights = zip(*applicable)
        
        try:
            self.score_energy()
        finally:
            self.constraints, self.constraint_weights = orig_constraints, orig_weights