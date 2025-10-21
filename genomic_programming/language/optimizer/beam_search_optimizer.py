"""
Beam Search Optimizer that uses the beam search algorithm to optimize a single Construct.
"""

from typing import List, Optional, Tuple
import warnings
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
    candidates_per_beam: int = Field(
        ge=1,
        description="Number of candidates to generate per beam sequence (N)"
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
        >>> gen_config = Evo2GeneratorConfig(n_tokens=100, prepend_prompt=False)
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
    ) -> None:
        """
        Initialize the Beam Search Optimizer.

        Args:
            construct: A single Construct object to optimize.
            generator: A single autoregressive Generator object (must have autoregressive=True).
            prompt: The initial prompt to start the beam search from (typically empty string).
            constraints: List of Constraint objects for evaluation (lower scores are better).
            config: Configuration object containing algorithm parameters (beam_width, candidates_per_beam, etc.).
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
        """
        if not generator.autoregressive:
            raise ValueError(f"BeamSearchOptimizer requires autoregressive generators. The provided generator '{generator.__class__.__name__}' is not autoregressive.")
        
        # Required for validation in base class. Each segment is assigned to the single generator for beam search.
        for segment in construct.segments:
            segment._is_assigned = True
        
        super().__init__(
            constructs=[construct],
            generators=[generator],
            constraints=constraints,
            constraint_weights=constraint_weights,
            num_candidates=config.beam_width * config.candidates_per_beam,
            num_selected=config.beam_width,
        )
        self.construct = construct
        self.generator = generator
        self.beam_width = config.beam_width
        self.candidates_per_beam = config.candidates_per_beam
        self.verbose = config.verbose
        self.running_prompts = [prompt] * self.beam_width

        # Warn and clear candidate sequence content - BeamSearch overwrites during run()
        for segment in self.segments:
            if any(seq.sequence for seq in segment.candidate_sequences):
                warnings.warn(f"BeamSearchOptimizer will overwrite {segment.num_candidates} existing candidate(s) in segment '{segment.label or 'unlabeled'}' during run()")
            for seq in segment.candidate_sequences:
                seq.sequence = ""

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

        # Beam search across each segment
        for segment_idx, segment in enumerate(self.construct.segments):
            # 1. Assign generator to this segment
            self.generator._assigned_segment = segment

            # 2. Prepare prompts: replicate each running prompt candidates_per_beam times
            all_prompts, beam_indices = self._prepare_beam_prompts()

            # 3. Generate candidates (writes to candidate_sequences)
            self.generator.sample(prompt_seqs=all_prompts)

            # 4. Score all candidates with applicable constraints
            self._score_energy_active_constraints()

            # 5. Select top beam_width candidates and update state
            top_idx = self._select_topk(segment, beam_indices)

            # Log progress
            if self.verbose:
                self._log_beamsearch_progress(segment_idx, segment, len(all_prompts), top_idx, beam_indices)

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

    def _select_topk(self, segment: Segment, beam_indices: List[int]) -> List[int]:
        """
        Select top beam_width candidates by energy and update all beam search state.
        
        This performs three operations:
        1. Identifies top beam_width candidates by energy (lower is better)
        2. Sets segment's selected_sequences and replicates them as candidates
        3. Updates running prompts by extending with new tokens from selected sequences
        
        Args:
            segment: The current segment being processed
            beam_indices: List tracking which beam (0 to beam_width-1) each candidate originated from
            
        Returns:
            List of indices for the top beam_width candidates
        """
        # 1. Get top beam_width candidates by energy
        top_idx = np.argsort(self.energy_scores)[:self.beam_width].tolist()
        
        # 2. Set selected sequences
        segment.selected_sequences = [segment.candidate_sequences[i] for i in top_idx]

        # 3. Replicate selected sequences as candidates (for subsequent evaluation of constraints applied across multiple segments)
        segment.candidate_sequences = [
            seq for selected_seq in segment.selected_sequences
            for seq in [selected_seq] * self.candidates_per_beam
        ]
        
        # 4. Update running prompts from selected sequences
        self.running_prompts = [
            self.running_prompts[beam_indices[idx]] + selected_seq.sequence
            for idx, selected_seq in zip(top_idx, segment.selected_sequences)
        ]
        return top_idx
    
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
        self.score_energy()

        # Restore original constraints and weights
        self.constraints, self.constraint_weights = orig_constraints, orig_weights

    def _log_beamsearch_progress(
        self, 
        segment_idx: int, 
        segment: Segment, 
        num_prompts: int, 
        top_idx: List[int], 
        beam_indices: List[int]
    ) -> None:
        """
        Log progress information for a segment during beam search.
        
        Args:
            segment_idx: Index of the current segment
            segment: The current segment being processed
            num_prompts: Number of prompts used for generation
            top_idx: Indices of the top beam_width selected candidates
            beam_indices: List tracking which beam each candidate originated from
        """
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
            seq_preview = seq.sequence[:50] + ('...' if len(seq.sequence) > 50 else '')
            print(f"  [{rank+1}] From beam {beam_indices[idx]}: '{seq_preview}' (energy: {energy:.4f})")