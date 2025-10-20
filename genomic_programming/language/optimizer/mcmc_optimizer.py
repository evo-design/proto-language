"""
MCMC Optimizer

Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, final
import copy
import random
import sys

import numpy as np
from pydantic import Field, model_validator

from ..core import Optimizer, Construct, Generator, Constraint, Sequence, Segment
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0


class MCMCOptimizerConfig(BaseConfig):
    """Configuration for MCMCOptimizer"""
    num_selected: int = Field(
        default=1,
        ge=1,
        description="Number of sequences to maintain in the selected_sequences pool across iterations (the 'top-k'). "
                   "When num_selected=1 (default), behaves like standard single-chain MCMC. "
                   "When num_selected>1, maintains top-k sequences and generates num_candidates proposals per sequence each step."
    )
    num_candidates: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of candidate proposals to generate per sequence each step. "
                   "If None (default), automatically set to num_selected for balanced exploration. "
                   "Can be explicitly set for custom exploration strategies."
    )
    num_steps: int = Field(
        default=1,
        ge=1,
        description="Number of MCMC steps per sample() call"
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum temperature for annealing"
    )
    temperature_min: float = Field(
        default=0.0001,
        gt=0.0,
        description="Minimum temperature for annealing"
    )
    track_step_size: int = Field(
        default=1,
        ge=1,
        description="Interval for progress tracking"
    )
    verbose: bool = Field(
        default=True,
        description="Whether to print progress information"
    )

    @model_validator(mode='after')
    def validate_cross_field_constraints(self):
        """Validate cross-field constraints."""
        # Validate temperature_min < temperature for annealing
        if self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be less than temperature ({self.temperature}) for annealing to work properly"
            )

        # Validate num_selected <= num_candidates for diversity (only if num_candidates is set)
        if self.num_candidates is not None and self.num_selected > self.num_candidates:
            raise ValueError(
                f"num_selected ({self.num_selected}) cannot be greater than num_candidates ({self.num_candidates}). "
                f"This ensures enough proposal diversity."
            )

        return self


@OptimizerRegistry.register(
    key="mcmc",
    label="Metropolis-Hastings MCMC Optimizer",
    config=MCMCOptimizerConfig,
    description="Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization",
)
@final
class MCMCOptimizer(Optimizer):
    # Class attribute: Config class for this optimizer
    config_class = MCMCOptimizerConfig
    """
    Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization.

    This optimizer implements a Metropolis-Hastings sampling algorithm that uses
    multiple sub-generators as proposal distributions and constraints to define
    the energy function. It's designed for iterative sequence optimization where
    proposals are accepted or rejected based on energy improvements.

    The optimizer supports simulated annealing, multiple constraints with weights,
    and flexible sequence optimization for complex multi-part designs.

    Examples:
        Basic MCMC optimization (single chain):
        >>> constructs = [Construct([segment1, segment2])]
        >>> config = MCMCOptimizerConfig(
        ...     num_steps=100,
        ...     temperature=0.5,
        ...     temperature_min=0.001
        ... )
        >>> mcmc = MCMCOptimizer(
        ...     constructs=constructs,
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     config=config,
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> mcmc.sample()  # Uses default: num_selected=1, num_candidates=1
        >>> final_constructs = mcmc.constructs

        Top-k MCMC optimization (default num_candidates):
        >>> config = MCMCOptimizerConfig(
        ...     num_selected=5,  # Maintain top-5 sequences
        ...     # num_candidates defaults to 5 (same as num_selected)
        ...     num_steps=100,
        ...     temperature=1.0,
        ... )
        >>> mcmc_topk = MCMCOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],
        ...     constraints=[energy_constraint],
        ...     config=config
        ... )
        >>> # Each step generates 5 proposals per sequence (25 total proposals)
        >>> mcmc_topk.sample()

        Custom exploration strategy (explicit num_candidates):
        >>> config = MCMCOptimizerConfig(
        ...     num_selected=3,
        ...     num_candidates=20,  # Deep local search: 20 proposals per selected sequence
        ...     num_steps=50,
        ... )
        >>> mcmc_deep = MCMCOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],
        ...     constraints=[energy_constraint],
        ...     config=config
        ... )
        >>> # Each step generates 20 proposals per sequence (3 x 20 = 60 total proposals)
        >>> mcmc_deep.sample()
    """

    def __init__(
        self,
        constructs: List['Construct'],
        generators: List['Generator'],
        constraints: List['Constraint'],
        config: MCMCOptimizerConfig,
        constraint_weights: Optional[List[float]] = None,
        custom_logging: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the MCMC Optimizer with sub-generators and constraints.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            config: Configuration object containing algorithm parameters (temperature, num_steps, etc.).
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            custom_logging: Optional custom logging function called at tracked steps.

        Raises:
            ValueError: If any validation checks fail.
        """
        # mcmc_width is number of proposals per sequence in mcmc loop
        mcmc_width = config.num_candidates or config.num_selected
        # Base class expects total candidates for to store all proposals for all sequences
        total_candidates = config.num_selected * mcmc_width
        
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
            num_candidates=total_candidates,
            num_selected=config.num_selected,
        )
        
        # Store MCMC-specific interpretation (proposals per selected sequence)
        # Note: self.num_candidates from parent = total_candidates (num_selected * mcmc_width)
        self.mcmc_width = mcmc_width
        self.num_steps = config.num_steps
        self.temperature = config.temperature
        self.temperature_min = config.temperature_min
        self.track_step_size = config.track_step_size
        self.verbose = config.verbose
        self.custom_logging = custom_logging
        self.history: List[Dict[str, Any]] = []  # Each entry are deep copies: {"time_step": int, "energy_scores": List[float], "constructs": List[Construct]}

    def sample(self) -> None:
        """
        Execute Metropolis-Hastings MCMC sampling for sequence optimization.

        Runs the specified number of MCMC steps, where each step:
        1. Maintains top-k sequences in `selected_sequences` (`num_selected` number of sequences)
        2. Creates `candidate_sequences` by replicating each selected sequence `mcmc_width` times
        3. Generates proposals (mutates `candidate_sequences` in-place)
        4. Evaluates all proposals with Metropolis-Hastings MCMC acceptance criterion
        5. Moves top-k accepted candidates to `selected_sequences`

        Note:
            - Simulated annealing: T(step) = T_max * (T_min / T_max) ^ (step / num_steps)
            - Total proposals per step: num_selected x mcmc_width
            - Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        # Initialize: Score initial sequences (all identical at start, so all scores will be the same)
        self.score_energy()
        # Keep only num_selected scores to align with selected_sequences
        self.energy_scores = [self.energy_scores[0]] * self.num_selected

        if self.verbose:
            print(f"MCMC initialization:")
            print(f"  num_selected={self.num_selected}, mcmc_width={self.mcmc_width}")
            print(f"  Initial energies: {[f'{e:.4f}' for e in self.energy_scores]}")
            print()

        # Track initial state
        self.history.append({
            "time_step": 0,
            "energy_scores": self.energy_scores.copy(),
            "constructs": copy.deepcopy(self.constructs)
        })

        # MCMC loop
        for step in range(1, self.num_steps + 1):
            #1. Save state of selected sequences
            old_selected_sequences = self._save_sequence_state()

            # 2. Populate candidate_sequences by replicating each selected_sequence num_candidates times
            self._populate_candidate_sequences()

            # 3. Generate proposals (mutate candidate_sequences in-place) by randomly sampling a generator
            generator = random.choice(self.generators)
            generator.sample()

            # 4. Score the candidate_sequences
            self.score_energy() # energy_scores are now populated for all candidate_sequences energies

            # 5. Apply MCMC acceptance and move top-k to selected_sequences
            self._select_topk_with_mcmc_acceptance(step, old_selected_sequences)

            # Logging and history tracking
            if step % self.track_step_size == 0:
                if self.verbose:
                    self._log_topk_progress(step)
                self.history.append({
                    "time_step": step,
                    "energy_scores": self.energy_scores.copy(),
                    "constructs": copy.deepcopy(self.constructs)
                })

        # Track final state
        if self.num_steps % self.track_step_size != 0:
            self.history.append({
                "time_step": self.num_steps,
                "energy_scores": self.energy_scores.copy(),
                "constructs": copy.deepcopy(self.constructs)
            })

    def _populate_candidate_sequences(self) -> None:
        """Populate candidate_sequences by replicating each selected_sequence mcmc_width times.
        
        Updates candidate_sequences in-place.
        Layout: [sequence_0] * mcmc_width + [sequence_1] * mcmc_width + ...
        """
        for segment in self.segments:
            for selected_idx in range(self.num_selected):
                start_idx = selected_idx * self.mcmc_width
                for offset in range(self.mcmc_width):
                    segment.candidate_sequences[start_idx + offset] = copy.deepcopy(segment.selected_sequences[selected_idx])

    def _save_sequence_state(self) -> List[Tuple[Dict[int, Sequence], float]]:
        """Save state of selected sequences.

        Returns:
            List of tuples, one per selected sequence, each containing:
                - segments dict: {segment_id -> deepcopied Sequence object}
                - energy: float (aligned with selected_sequences)
        """
        sequence_state = []
        for selected_idx in range(self.num_selected):
            segments_dict = {}
            for segment in self.segments:
                seg_id = id(segment)
                segments_dict[seg_id] = copy.deepcopy(segment.selected_sequences[selected_idx])
            sequence_state.append((segments_dict, self.energy_scores[selected_idx]))
        return sequence_state

    def _select_topk_with_mcmc_acceptance(
        self,
        step: int,
        old_selected_sequences: List[Tuple[Dict[int, Sequence], float]]
    ) -> None:
        """Apply Metropolis-Hastings acceptance and move top-k to selected_sequences.

        For each proposal in candidate_sequences:
        1. Compute acceptance probability vs its old selected sequence
        2. If rejected, restore old selected sequence to that candidate position
        3. After all proposals evaluated, select top num_selected by energy
        4. Move selected candidates to selected_sequences
        
        Args:
            step: Current MCMC step for temperature annealing
            old_selected_sequences: Saved state of selected sequences before proposals
        """
        # Process each selected sequence's proposals
        for selected_idx in range(self.num_selected):
            old_segments_dict, old_selected_energy = old_selected_sequences[selected_idx]
            start_idx = selected_idx * self.mcmc_width
            end_idx = (selected_idx + 1) * self.mcmc_width

            for candidate_idx in range(start_idx, end_idx):
                proposal_energy = self.energy_scores[candidate_idx]
                alpha = self._compute_mcmc_acceptance_prob(old_selected_energy, proposal_energy, step)

                if random.random() >= alpha:
                    # Reject - restore old selected sequence to this candidate position
                    for segment in self.segments:
                        seg_id = id(segment)
                        segment.candidate_sequences[candidate_idx] = copy.deepcopy(old_segments_dict[seg_id])
                    self.energy_scores[candidate_idx] = old_selected_energy

        # Select top-k candidates by energy and move to selected_sequences
        top_k_idx = np.argsort(self.energy_scores)[:self.num_selected]
        for segment in self.segments:
            segment.selected_sequences = [copy.deepcopy(segment.candidate_sequences[idx]) for idx in top_k_idx]

        # Update energy_scores to match selected_sequences
        self.energy_scores = [self.energy_scores[idx] for idx in top_k_idx]

    def _calculate_temperature(self, step: int) -> float:
        """Calculate annealed temperature using exponential cooling schedule.
        
        Formula: T(step) = T_max * (T_min/T_max)^((step-1)/(num_steps-1))
        
        This ensures:
        - At step=1: T = T_max (start hot)
        - At step=num_steps: T = T_min (end cold)
        - Exponential decay between these bounds
        
        Args:
            step: Current MCMC step (1-indexed, range: 1 to num_steps)
            
        Returns:
            Temperature for the current step
            
        Note:
            For num_steps=1, returns T_max (no annealing).
            The (step-1) term ensures proper boundary conditions since steps are 1-indexed.
        """
        # Handle division by 0 for num_steps=1
        if self.num_steps == 1:
            return self.temperature
        return self.temperature * (self.temperature_min / self.temperature) ** ((step - 1) / (self.num_steps - 1))

    def _compute_mcmc_acceptance_prob(self, current_energy: float, proposed_energy: float, step: int) -> float:
        """Compute Metropolis-Hastings acceptance probability.
        
        Uses the standard MH acceptance criterion: alpha = min(1, exp(-(E_new - E_old) / T))
        where lower energy is better (constraint scores: 0.0 = perfect, higher = worse).
        
        This always accepts improvements (proposed_energy < current_energy) and accepts
        worse proposals with probability exp(-(ΔE / T)) where ΔE = proposed - current.
        
        Args:
            current_energy: Energy of current state (lower is better)
            proposed_energy: Energy of proposed state (lower is better)
            step: Current MCMC step for temperature annealing
            
        Returns:
            Acceptance probability in [0, 1]
        """
        temperature = self._calculate_temperature(step)
        log_acceptance_ratio = -(proposed_energy - current_energy) / temperature
        # Cap to prevent overflow in exp()
        log_acceptance_ratio = min(log_acceptance_ratio, MAX_EXP_ARG)
        return min(1.0, np.exp(log_acceptance_ratio))

    def _log_topk_progress(self, step: int) -> None:
        """Log optimization progress"""
        best_energy = min(self.energy_scores)
        mean_energy = np.mean(self.energy_scores)
        worst_energy = max(self.energy_scores)
        std_energy = np.std(self.energy_scores) if len(self.energy_scores) > 1 else 0.0
        current_temp = self._calculate_temperature(step)

        # Format output based on num_selected
        if self.num_selected == 1:
            print(
                f"Iteration {step:4d} | "
                f"energy: {best_energy:.6f}, "
                f"T: {current_temp:.4f}"
            )
        else:
            print(
                f"Iteration {step:4d} | "
                f"best: {best_energy:.6f}, "
                f"mean: {mean_energy:.6f}, "
                f"worst: {worst_energy:.6f}, "
                f"std: {std_energy:.6f}, "
                f"T: {current_temp:.4f}"
            )

        if self.custom_logging:
            self.custom_logging(step, self.segments)
        sys.stdout.flush()
        