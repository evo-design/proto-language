"""
Metropolis-Hastings MCMC Optimizer that uses multiple sub-generators as proposal distributions and constraints to define the energy function.
"""

from typing import Callable, Dict, List, Optional, Tuple, final
import copy
import random
import sys

import numpy as np
from pydantic import Field, model_validator

from ..core import Optimizer, Construct, Generator, GeneratorType, Constraint, Sequence
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0

class MCMCOptimizerConfig(BaseConfig):
    """Configuration for MCMCOptimizer"""
    # Required parameters
    num_selected: int = Field(
        ge=1,
        title="Num maintained candidates",
        description="Number of candidate sequences to optimize across iterations (the top-k). "
                   "When num_selected=1 (default), behaves like standard single-chain MCMC. "
                   "When num_selected>1, maintains top-k sequences and generates mcmc_width number of proposals per sequence each step."
    )
    mcmc_width: int = Field(
        ge=1,
        title="Num proposals per candidate",
        description="Number of generated proposals per candidate sequence each step, similar to `beam width` in beam search."
    )
    num_steps: int = Field(
        ge=1,
        title="Num steps",
        description="Number of MCMC steps to run."
    )

    # Optional parameters (have defaults)
    max_temperature: float = Field(
        default=1.0,
        gt=0.0,
        title="Max temperature",
        description="Maximum temperature for annealing"
    )
    min_temperature: float = Field(
        default=0.001,
        gt=0.0,
        title="Min temperature",
        description="Minimum temperature for annealing"
    )
    track_step_size: int = Field(
        default=1,
        ge=1,
        title="Track interval",
        description="Interval for progress tracking"
    )
    verbose: bool = Field(
        default=False,
        title="Verbose",
        description="Whether to print progress information."
    )

    @model_validator(mode='after')
    def validate_cross_field_constraints(self):
        """Validate cross-field constraints."""
        # Validate min_temperature < max_temperature for annealing
        if self.min_temperature >= self.max_temperature:
            raise ValueError(f"min_temperature ({self.min_temperature}) must be less than max_temperature ({self.max_temperature}) for annealing to work properly")

        return self


@OptimizerRegistry.register(
    key="mcmc",
    label="Metropolis-Hastings MCMC Optimizer",
    config=MCMCOptimizerConfig,
    description="Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization",
)
@final
class MCMCOptimizer(Optimizer):
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
        ...     max_temperature=0.5,
        ...     min_temperature=0.001
        ... )
        >>> mcmc = MCMCOptimizer(
        ...     constructs=constructs,
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     config=config,
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> mcmc.run()  # Uses default: num_selected=1, num_candidates=1
        >>> final_constructs = mcmc.constructs

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
        >>> mcmc_deep.run()
    """
    # Class attribute required by OptimizerRegistry
    config_class = MCMCOptimizerConfig

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: MCMCOptimizerConfig,
        constraint_weights: Optional[List[float]] = None,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: bool | List[str] = True,
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
            clear_tool_cache: (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If any validation checks fail.
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
            num_candidates=config.num_selected * config.mcmc_width,
            num_selected=config.num_selected,
            clear_tool_cache=clear_tool_cache,
        )
        
        # Store MCMC-specific interpretation (proposals per selected sequence)
        # Note: self.num_candidates from parent = total_candidates (num_selected * mcmc_width)
        self.mcmc_width: int = config.mcmc_width
        self.num_steps: int = config.num_steps
        self.max_temperature: float = config.max_temperature
        self.min_temperature: float = config.min_temperature
        self.track_step_size: int = config.track_step_size
        self.verbose: bool = config.verbose
        self.custom_logging: Optional[Callable] = custom_logging
        for generator in generators:
            if generator.type != GeneratorType.MUTATION:
                raise ValueError(f"MCMCOptimizer requires mutation generators. The provided generator '{generator.__class__.__name__}' is not a mutation generator.")

    def run(self) -> None:
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
        # Score candidate_sequences to populate energy_scores with candidate_sequences copies of inital energy score
        self.score_energy(verbose=self.verbose)

        if self.verbose:
            print(f"MCMC initialization:")
            print(f"  num_selected={self.num_selected}, mcmc_width={self.mcmc_width}")
            print(f"  Initial energy: {self.energy_scores[0]:.4f}")
            print()

        # Track initial state
        self._save_progress_snapshot(time_step=0)

        # MCMC loop
        for step in range(1, self.num_steps + 1):
            #1. Save state of selected_sequences to revert if rejected by Metropolis-Hastings acceptance criterion
            old_selected_sequences = self._save_sequence_state()

            # 2. Populate candidate_sequences by replicating each selected_sequence mcmc_width times
            self._populate_candidate_sequences()

            # 3. Generate proposals for candidate_sequences in-place by randomly sampling a generator
            generator = random.choice(self.generators)
            generator.sample()

            # 4. Score candidate_sequences
            self.score_energy(verbose=self.verbose)

            # 5. Metropolis-Hastings acceptance and update energy score, candidate_sequences, and selected_sequences state
            self._select_topk_with_mcmc_acceptance(step, old_selected_sequences)

            # Logging and history tracking
            if step % self.track_step_size == 0:
                self._save_progress_snapshot(time_step=step)
                if self.verbose:
                    self._log_mcmc_progress(step)

        # Track final state
        if self.num_steps % self.track_step_size != 0:
            self._save_progress_snapshot(time_step=self.num_steps)


    def _save_sequence_state(self) -> List[Tuple[Dict[int, Sequence], float]]:
        """Save state of selected sequences.

        Returns:
            List of tuples, one per selected sequence, each containing:
                - segments dict: {segment_id -> deepcopied Sequence object}
                - energy: float (from first num_selected entries of energy_scores after sorting)
        """
        sequence_state = []
        for selected_idx in range(self.num_selected):
            segments_dict = {}
            for segment in self.segments:
                seg_id = id(segment)
                segments_dict[seg_id] = copy.deepcopy(segment.selected_sequences[selected_idx])
            sequence_state.append((segments_dict, self.energy_scores[selected_idx]))
        return sequence_state


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


    def _select_topk_with_mcmc_acceptance(
        self,
        step: int,
        old_selected_sequences: List[Tuple[Dict[int, Sequence], float]]
    ) -> None:
        """Apply Metropolis-Hastings acceptance and sort candidates by energy in place.

        For each proposal in candidate_sequences:
        1. Compute Metropolis-Hastings acceptance probability
        2. If rejected, restore the old selected_sequence state
        3. Sort candidate_sequences and energy_scores by energy in place
        4. Copy top num_selected to selected_sequences
        
        Args:
            step: Current MCMC step for temperature annealing
            old_selected_sequences: Saved state of selected_sequences before proposals
        """
        # 1. Metropolis-Hastings acceptance for each selected sequence's proposals
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

        # 2. Sort candidate_sequences and energy_scores by energy in place
        sorted_idx = np.argsort(self.energy_scores)
        self.energy_scores = [self.energy_scores[idx] for idx in sorted_idx]
        for segment in self.segments:
            segment.candidate_sequences = [segment.candidate_sequences[idx] for idx in sorted_idx]

        # 3. Copy top num_selected to selected_sequences (copy by reference since _populate_candidate_sequences does deepcopy)
        for segment in self.segments:
            segment.selected_sequences = [segment.candidate_sequences[idx] for idx in range(self.num_selected)]


    def _compute_temperature(self, step: int) -> float:
        """Calculate annealed temperature: T(step) = T_max * (T_min/T_max)^((step-1)/(num_steps-1))
        
        Note:
        - At step=1: T = T_max (start hot), at step=num_steps: T = T_min (end cold)
        - Exponential decay between T_max and T_min
        - (step-1) ensures proper boundary conditions since steps are 1-indexed (range: 1 to num_steps)
        """
        if self.num_steps == 1:
            return self.max_temperature
        else:
            return self.max_temperature * (self.min_temperature / self.max_temperature) ** ((step - 1) / (self.num_steps - 1))


    def _compute_mcmc_acceptance_prob(self, current_energy: float, proposed_energy: float, step: int) -> float:
        """Compute Metropolis-Hastings acceptance probability: alpha = min(1, exp(-(E_new - E_old) / T))

        Note:
        - Always accepts improvements (proposed_energy < current_energy)
        - Accepts worse proposals with probability exp(-(ΔE / T)) where ΔE = proposed - current
        """
        temperature = self._compute_temperature(step)
        log_acceptance_ratio = -(proposed_energy - current_energy) / temperature
        # Cap to prevent overflow in exp()
        log_acceptance_ratio = min(log_acceptance_ratio, MAX_EXP_ARG)
        return min(1.0, np.exp(log_acceptance_ratio))


    def _log_mcmc_progress(self, step: int) -> None:
        """Log optimization progress"""
        # Use first num_selected energies (after sorting, these are the selected sequences)
        selected_energies = self.energy_scores[:self.num_selected]
        best_energy = min(selected_energies)
        mean_energy = np.mean(selected_energies)
        worst_energy = max(selected_energies)
        std_energy = np.std(selected_energies) if len(selected_energies) > 1 else 0.0
        current_temp = self._compute_temperature(step)

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
