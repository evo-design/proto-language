"""
Metropolis-Hastings MCMC Optimizer that uses multiple sub-generators as proposal distributions and constraints to define the energy function.
"""
from __future__ import annotations

import copy
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple, final

import numpy as np
from pydantic import model_validator

logger = logging.getLogger(__name__)


from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Sequence,
)
from proto_language.language.optimizer.optimizer_registry import optimizer

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
            for each.

        num_steps (int): Number of MCMC steps to run. Each step generates proposals,
            evaluates them, and accepts/rejects based on Metropolis-Hastings criterion.
            More steps allow better exploration but increase runtime.

        mcmc_width (int): Number of proposals to generate per selected sequence at
            each step. Total proposals per step equals ``num_selected x mcmc_width``.
            The best proposal (by energy) is selected, then MH acceptance is applied.
            Higher values increase exploration but also computation. Default: 1.

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
        - When ``num_selected=1`` (default), behaves like standard single-chain MCMC.
        - When ``num_selected > 1``, maintains ``num_selected`` sequence trajectories and
          generates ``mcmc_width`` (default: 1) proposals per trajectory each step.
        - Temperature annealing follows exponential decay:
          T(step) = T_max x (T_min / T_max)^(step / num_steps)
    """
    # Required parameters
    num_selected: int = ConfigField(
        ge=1,
        title="Num Candidates Maintained",
        description="Number of sequence trajectories to optimize across iterations (the top-k).",
    )
    num_steps: int = ConfigField(
        ge=1, title="Num Steps", description="Number of MCMC steps to run."
    )

    # Advanced parameters
    mcmc_width: int = ConfigField(
        default=1,
        ge=1,
        title="Num Proposals",
        description="Number of proposals per trajectory at each mcmc step",
        advanced=True,
    )
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


@optimizer(
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
    mutating each of the K sequences ``mcmc_width`` times. Each trajectory (batch index)
    is independent. For each trajectory, the best proposal (lowest energy) is selected,
    then MH acceptance is applied to decide whether to accept or reject that proposal.
    If rejected, the trajectory keeps its previous state.

    Attributes:
        num_selected (int): Number of sequence trajectories to optimize across iterations (the top-k).
        num_steps (int): Total number of MCMC steps to run.
        mcmc_width (int): Number of proposals per sequence trajectory.
        max_temperature (float): Starting temperature for annealing.
        min_temperature (float): Ending temperature for annealing.
        track_step_size (int): Interval for progress tracking.

    Example:
        >>> constructs = [Construct([segment1, segment2])]
        >>> config = MCMCOptimizerConfig(
        ...     num_selected=1,
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
        - When ``mcmc_width > 1``, generates multiple proposals per trajectory,
          selects the best one, then applies a single MH accept/reject decision
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
        1. Maintains `num_selected` independent trajectories in `selected_sequences`
        2. Creates `candidate_sequences` by replicating each selected sequence `mcmc_width` times
        3. Generates proposals (mutates `candidate_sequences` in-place)
        4. For each trajectory, independently apply MH acceptance and select the best accepted proposal

        Note:
            - Each trajectory (batch index) is independent with no cross-trajectory mixing.
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

        # Truncate to num_selected for initial snapshot (score_energy sets to num_candidates)
        self.energy_scores = self.energy_scores[:self.num_selected]

        if self.verbose:
            logger.info("MCMC initialization:")
            logger.info(f"  num_selected={self.num_selected}, mcmc_width={self.mcmc_width}")
            logger.info(f"  Initial energy: {self.energy_scores[0]:.4f}")

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
            passed_mask = self.score_energy()

            # 5. Metropolis-Hastings acceptance and update energy score, candidate_sequences, and selected_sequences state
            self._select_topk_with_mcmc_acceptance(step, old_selected_sequences, passed_mask)

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
                - energy: float (energy_scores[selected_batch_idx])
        """
        sequence_state = []
        for selected_batch_idx in range(self.num_selected):
            segments_dict = {}
            for segment in self.segments:
                seg_id = id(segment)
                segments_dict[seg_id] = copy.deepcopy(segment.selected_sequences[selected_batch_idx])
            sequence_state.append((segments_dict, self.energy_scores[selected_batch_idx]))
        return sequence_state

    def _populate_candidate_sequences(self) -> None:
        """Populate candidate_sequences by replicating each selected_sequence mcmc_width times.

        Updates candidate_sequences in-place.
        Layout: [sequence_0] * mcmc_width + [sequence_1] * mcmc_width + ...
        """
        for segment in self.segments:
            for selected_batch_idx in range(self.num_selected):
                start_idx = selected_batch_idx * self.mcmc_width
                for offset in range(self.mcmc_width):
                    segment.candidate_sequences[start_idx + offset] = copy.deepcopy(segment.selected_sequences[selected_batch_idx])

    def _select_topk_with_mcmc_acceptance(
        self,
        step: int,
        old_selected_sequences: List[Tuple[Dict[int, Sequence], float]],
        passed_mask: List[bool],
    ) -> None:
        """Apply Metropolis-Hastings acceptance independently per trajectory (batch index).

        For each batch index in selected_sequences:
        1. Find the best proposal by energy from its pool
        2. Apply MH acceptance to that single best proposal
        3. If rejected or no valid proposals, keep the old state

        Args:
            step: Current MCMC step for temperature annealing
            old_selected_sequences: Saved state of selected_sequences before proposals
        """
        for selected_batch_idx in range(self.num_selected):
            old_segments_dict, old_selected_energy = old_selected_sequences[selected_batch_idx]
            best_energy = float('inf')
            best_candidate_idx = None

            # Proposal pool for this trajectory: candidate_sequences[pool_start:pool_end]
            # Candidate sequences layout: [traj_0 x mcmc_width, traj_1 x mcmc_width, ...]
            proposal_pool_start_idx = selected_batch_idx * self.mcmc_width
            proposal_pool_end_idx = (selected_batch_idx + 1) * self.mcmc_width

            # Step 1: Find the best proposal by energy
            for candidate_idx in range(proposal_pool_start_idx, proposal_pool_end_idx):
                proposal_energy = self.energy_scores[candidate_idx]

                # Skip candidates that failed filter constraints
                if not passed_mask[candidate_idx]:
                    continue

                if proposal_energy < best_energy:
                    best_energy = proposal_energy
                    best_candidate_idx = candidate_idx

            # Step 2: Apply MH acceptance and update trajectory
            # Accept if: (1) valid proposal exists
            #            (2) passes MH criterion
            valid_proposals_exist = best_candidate_idx is not None
            alpha = self._compute_mcmc_alpha(old_selected_energy, best_energy, step)
            accepted = valid_proposals_exist and random.random() < alpha

            if accepted:
                for segment in self.segments:
                    segment.selected_sequences[selected_batch_idx] = copy.deepcopy(segment.candidate_sequences[best_candidate_idx])
            else:
                # Rejected or no valid proposals - restore old state
                best_energy = old_selected_energy
                for segment in self.segments:
                    segment.selected_sequences[selected_batch_idx] = copy.deepcopy(old_segments_dict[id(segment)])

            self.energy_scores[selected_batch_idx] = best_energy

        # Truncate to only keep selected energies (indices [num_selected:] are stale proposal energies)
        # score_energy() will resize back to num_candidates at the start of each step
        self.energy_scores = self.energy_scores[:self.num_selected]

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

    def _compute_mcmc_alpha(self, current_energy: float, proposed_energy: float, step: int) -> float:
        """Compute Metropolis-Hastings acceptance probability: alpha = min(1, exp(-(E_new - E_old) / T))

        Note:
        - Always accepts improvements (proposed_energy < current_energy)
        - Accepts worse proposals with probability exp(-(ΔE / T)) where ΔE = proposed - current

        Important: When mcmc_width > 1, this is applied to the BEST proposal from the pool,
        not a randomly selected one. This "best-of-N then MH" strategy is a heuristic that
        accelerates convergence but does not satisfy detailed balance for the true Boltzmann
        distribution. For mathematically rigorous MCMC sampling, use mcmc_width=1.
        """
        temperature = self._compute_temperature(step)
        log_acceptance_ratio = -(proposed_energy - current_energy) / temperature
        # Cap to prevent overflow in exp()
        log_acceptance_ratio = min(log_acceptance_ratio, MAX_EXP_ARG)
        return min(1.0, np.exp(log_acceptance_ratio))

    def _log_mcmc_progress(self, step: int) -> None:
        """Log optimization progress"""
        best_energy = min(self.energy_scores)
        mean_energy = np.mean(self.energy_scores)
        worst_energy = max(self.energy_scores)
        std_energy = np.std(self.energy_scores) if len(self.energy_scores) > 1 else 0.0
        current_temp = self._compute_temperature(step)

        # Format output based on num_selected
        if self.num_selected == 1:
            logger.debug(
                f"Iteration {step:4d} | "
                f"energy: {best_energy:.6f}, "
                f"T: {current_temp:.4f}"
            )
        else:
            logger.debug(
                f"Iteration {step:4d} | "
                f"best: {best_energy:.6f}, "
                f"mean: {mean_energy:.6f}, "
                f"worst: {worst_energy:.6f}, "
                f"std: {std_energy:.6f}, "
                f"T: {current_temp:.4f}"
            )

        if self.custom_logging:
            self.custom_logging(step, self.segments)
