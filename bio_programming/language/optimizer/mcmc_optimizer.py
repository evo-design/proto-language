"""
Metropolis-Hastings MCMC Optimizer that uses multiple sub-generators as proposal distributions and constraints to define the energy function.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, final
import math
import copy
import random
import sys

import numpy as np
from pydantic import model_validator


from proto_language.language.core import Optimizer, Construct, Generator, Constraint, Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0

class MCMCOptimizerConfig(BaseConfig):
    """Configuration object for MCMCOptimizer.

    This class defines configuration parameters for the Metropolis-Hastings MCMC
    optimizer, which explores sequence space through iterative mutation with
    probabilistic acceptance based on energy improvements.

    Attributes:
        num_selected (int): Number of candidate sequences to maintain and optimize
            across iterations (the top-K sequences by energy). When ``num_selected=1``
            (standard single-chain MCMC), only one sequence is optimized. When
            ``num_selected > 1``, maintains multiple sequences and generates proposals
            for each. Must be at least 1.

        mcmc_width (int): Number of proposals to generate per selected sequence at
            each step. Similar to beam width in beam search. Total proposals per step
            equals ``num_selected x mcmc_width``. Higher values increase exploration
            but also computation. Must be at least 1.

        num_steps (int): Number of MCMC steps to run. Each step generates proposals,
            evaluates them, and accepts/rejects based on Metropolis-Hastings criterion.
            More steps allow better exploration but increase runtime. Must be at least 1.

        max_temperature (float): Maximum temperature for simulated annealing at the
            start of optimization. Higher temperatures allow more exploration by
            accepting worse proposals with higher probability. Must be greater than 0.
            Default: 1.0.

        min_temperature (float): Minimum temperature for simulated annealing at the
            end of optimization. Lower temperatures make the algorithm more greedy,
            accepting only improvements. Must be greater than 0 and less than
            ``max_temperature``. Default: 0.001.

        track_step_size (int): Interval for saving progress snapshots to history.
            For example, ``track_step_size=10`` saves state every 10 steps. Must be
            at least 1. Default: 1.

        verbose (bool): Whether to print detailed progress information at each
            tracked step, including energy statistics and temperature. Default: ``False``.

    Note:
        Temperature annealing follows exponential decay:
            T(step) = T_max x (T_min / T_max)^(step / num_steps)
    """
    # Required parameters
    num_selected: int = ConfigField(
        ge=1,
        title="Num Candidates Maintained",
        description="Number of candidate sequences to optimize across iterations (the top-k).",  # "When num_selected=1 (default), behaves like standard single-chain MCMC. " "When num_selected>1, maintains top-k sequences and generates mcmc_width number of proposals per sequence each step.",
    )
    mcmc_width: int = ConfigField(
        ge=1,
        title="Num Proposals",
        description="Number of generated proposals per candidate sequence each step.",  # (Similar to `beam width`)
    )
    num_steps: int = ConfigField(
        ge=1, title="Num Steps", description="Number of MCMC steps to run."
    )

    # Advanced parameters
    max_temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Max Temperature",
        description="Maximum temperature for annealing",
        advanced=True,
    )
    min_temperature: float = ConfigField(
        default=0.001,
        gt=0.0,
        title="Min Temperature",
        description="Minimum temperature for annealing",
        advanced=True,
    )
    track_step_size: int = ConfigField(
        default=1,
        ge=1,
        title="Track Interval",
        description="Interval for progress tracking",
        advanced=True,
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
        advanced=True,
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
    label="MCMC Optimizer",
    config=MCMCOptimizerConfig,
    description="Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization",
)
@final
class MCMCOptimizer(Optimizer):
    """Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization.

    This optimizer implements Metropolis-Hastings sampling with simulated annealing
    to optimize sequences against constraint-based energy functions. It uses mutation
    generators as proposal distributions and accepts/rejects proposals based on energy
    changes and temperature.

    At each step, the optimizer generates ``num_selected x mcmc_width`` proposals by
    mutating each of the top-K sequences ``mcmc_width`` times. Proposals are accepted
    with probability min(1, exp(-ΔE/T)) where ΔE is the energy change and T is the
    annealed temperature. The top-K sequences by lowest energy are retained for
    the next step.

    Attributes:
        mcmc_width (int): Number of proposals per selected sequence.
        num_steps (int): Total number of MCMC steps to run.
        max_temperature (float): Starting temperature for annealing.
        min_temperature (float): Ending temperature for annealing.
        track_step_size (int): Interval for progress tracking.

    Example:
        >>> constructs = [Construct([segment1, segment2])]
        >>> config = MCMCOptimizerConfig(
        ...     num_selected=1,
        ...     mcmc_width=20,
        ...     num_steps=100,
        ...     max_temperature=0.5,
        ...     min_temperature=0.001
        ... )
        >>> mcmc = MCMCOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],
        ...     constraints=[gc_constraint],
        ...     config=config
        ... )
        >>> mcmc.run()
        >>> final_sequences = mcmc.constructs[0].joined_sequences

    Note:
        - Only supports mutation generators (``category="mutation"``)
        - Uses Metropolis-Hastings acceptance: always accepts improvements,
            accepts worse proposals with probability exp(-ΔE/T)
        - Simulated annealing: temperature decreases exponentially from
            ``max_temperature`` to ``min_temperature``
        - Lower energy scores are better (minimization objective)
    """
    # Class attribute required by OptimizerRegistry
    config_class = MCMCOptimizerConfig

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: MCMCOptimizerConfig,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
    ) -> None:
        """
        Initialize the MCMC Optimizer with sub-generators and constraints.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            config: Configuration object containing algorithm parameters (temperature, num_steps, etc.).
            custom_logging: Optional custom logging function called at tracked steps.
            clear_tool_cache: (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If any validation checks fail.
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_candidates=config.num_selected * config.mcmc_width,
            num_selected=config.num_selected,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
        )

        # Store MCMC-specific interpretation (proposals per selected sequence)
        # Note: self.num_candidates from parent = total_candidates (num_selected * mcmc_width)
        self.mcmc_width: int = config.mcmc_width
        self.num_steps: int = config.num_steps
        self.max_temperature: float = config.max_temperature
        self.min_temperature: float = config.min_temperature
        self.track_step_size: int = config.track_step_size

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
        self._prepare_run()

        # Score initial state if sequences are non-empty (skip for autoregressive generators like ProGen2)
        if any(seq.sequence for segment in self.segments for seq in segment.candidate_sequences):
            self.score_energy()
        else:
            self.energy_scores = [float('inf')] * self.num_candidates

        if self.verbose:
            print("MCMC initialization:")
            print(f"  num_selected={self.num_selected}, mcmc_width={self.mcmc_width}")
            print(f"  Initial energy: {self.energy_scores[0]:.4f}")
            print()

        # Track initial state
        self._save_progress_snapshot(time_step=0)

        # MCMC loop
        for step in range(1, self.num_steps + 1):
            # 1. Save state of selected_sequences to revert if rejected by Metropolis-Hastings acceptance criterion
            old_selected_sequences = self._save_sequence_state()

            # 2. Populate candidate_sequences by replicating each selected_sequence mcmc_width times
            self._populate_candidate_sequences()

            # 3. Generate proposals for candidate_sequences in-place by randomly sampling a generator
            generator = random.choice(self.generators)
            generator.sample()

            # 4. Score candidate_sequences
            self.score_energy()

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

                # Always reject inf or nan energies.
                if math.isnan(proposal_energy) or math.isinf(proposal_energy):
                    reject = True
                else:
                    alpha = self._compute_mcmc_acceptance_prob(old_selected_energy, proposal_energy, step)
                    reject = random.random() >= alpha

                if reject:
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
