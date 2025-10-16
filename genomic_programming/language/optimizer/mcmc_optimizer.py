"""
MCMC Optimizer

Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization.
"""

from typing import Any, Callable, Dict, List, Optional, final
import copy
import random
import sys

import numpy as np
from pydantic import Field

from ..core import Optimizer, Construct, Generator, Constraint, Sequence, Segment
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0


class MCMCOptimizerConfig(BaseConfig):
    """Configuration for MCMCOptimizer algorithm parameters only.

    Runtime objects (constructs, generators, constraints) should be passed
    separately to the optimizer's __init__ method, not in the config.
    """
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of sequences to maintain across iterations (the 'top-k'). "
                   "When batch_size=1 (default), behaves like standard single-chain MCMC. "
                   "When batch_size>1, maintains top-k sequences and generates "
                   "num_candidates proposals per sequence each step."
    )
    num_candidates: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of candidate proposals to generate per sequence each step. "
                   "If None (default), automatically set to batch_size for balanced exploration. "
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


@OptimizerRegistry.register(
    key="mcmc",
    config=MCMCOptimizerConfig,
    description="Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization",
)
@final
class MCMCOptimizer(Optimizer):
    # Class attribute: Config class for this optimizer
    config_class = MCMCOptimizerConfig
    """
    Metropolis-Hastings MCMC optimizer for constraint-driven sequence optimization.

    Batch Size & Candidates Semantics:
        - `batch_size` specifies how many sequences to maintain (default: 1)
        - `num_candidates` defaults to `batch_size` for balanced exploration
        - At each step: expand to (batch_size x num_candidates) proposals, then trim back to batch_size
        - Total proposals per step = batch_size x num_candidates

    This generator implements a Metropolis-Hastings sampling algorithm that uses
    multiple sub-generators as proposal distributions and constraints to define
    the energy function. It's designed for iterative sequence refinement where
    proposals are accepted or rejected based on energy improvements.

    The generator supports simulated annealing, multiple constraints with weights,
    and flexible sequence concatenation for complex multi-part designs.

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
        >>> mcmc.sample()  # Uses default: batch_size=1, num_candidates=1
        >>> final_constructs = mcmc.constructs

        Top-k MCMC optimization (default num_candidates):
        >>> config = MCMCOptimizerConfig(
        ...     batch_size=5,  # Maintain top-5 sequences
        ...     # num_candidates defaults to 5 (same as batch_size)
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
        ...     batch_size=3,
        ...     num_candidates=20,  # Deep local search: 20 proposals per parent
        ...     num_steps=50,
        ... )
        >>> mcmc_deep = MCMCOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],
        ...     constraints=[energy_constraint],
        ...     config=config
        ... )
        >>> # Each step generates 20 proposals per sequence (60 total proposals)
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
        # Pass batch_size to parent class
        # This makes:
        # 1. self.batch_size = batch_size (inherited from Optimizer)
        # 2. Sub-generators' batch_size gets overridden to match
        # 3. Segments maintain batch_size sequences throughout MCMC
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
            batch_size=config.batch_size,
        )

        # Store config
        self.config = config

        # MCMC-specific parameters
        # Default num_candidates to batch_size if not provided
        if config.num_candidates is None:
            self.num_candidates = config.batch_size
        else:
            self.num_candidates = config.num_candidates
        self.num_steps = config.num_steps
        self.temperature = config.temperature
        self.temperature_min = config.temperature_min
        self.track_step_size = config.track_step_size
        self.verbose = config.verbose
        self.custom_logging = custom_logging

        self._validate_generator()

    def _validate_generator(self) -> None:
        """
        Validate configuration for MCMCOptimizer.

        Raises:
            ValueError: If temperature parameters or batch_size are invalid.
        """
        super()._validate_optimizer()

        # Validate temperature parameters
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_min <= 0:
            raise ValueError(
                f"temperature_min must be positive, got {self.temperature_min}"
            )
        if self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be less than temperature ({self.temperature}) for annealing to work properly"
            )

        # Validate batch_size parameter
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

        # Validate num_candidates
        if self.num_candidates < 1:
            raise ValueError(f"num_candidates must be at least 1, got {self.num_candidates}")

        # Validate batch_size <= num_candidates for diversity
        if self.batch_size > self.num_candidates:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot be greater than num_candidates ({self.num_candidates}). "
                f"This ensures enough proposal diversity."
            )

    def sample(self) -> None:
        """
        Execute Metropolis-Hastings MCMC sampling for sequence optimization.

        Runs the specified number of MCMC steps, where each step:
        1. Maintains top-k sequences (batch_size=1 for standard MCMC)
        2. Generates num_candidates proposals per sequence (total: batch_size x num_candidates proposals)
        3. Evaluates all proposals and applies MCMC acceptance: accepted proposals + rejected (restored) sequences form candidate pool
        4. Selects top-k sequences by energy from all candidates
        5. Optionally logs progress and tracks state

        Algorithm:
        - When num_candidates=1 and batch_size=1: behaves as standard single-chain MCMC
        - When batch_size>1: maintains top-k sequences for diversity
        - For each proposal, applies Metropolis-Hastings acceptance criterion
        - Rejected proposals restore to saved state (deepcopied Sequence + energy)
        - After all proposals, selects top-k by energy for next iteration

        Note:
            - Simulated annealing: T(step) = T_max * (T_min / T_max) ^ (step / num_steps)
            - Total proposals per step: batch_size x num_candidates
            - Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        # Initialize: score the initial batch_size (top_k) sequences
        # These were already created by sub-generators via assign()
        self.score_energy()
        top_k_idx = np.arange(self.batch_size)  # Indices [0, 1, ..., batch_size-1]

        if self.verbose:
            print(f"MCMC initialization:")
            print(f"  batch_size={self.batch_size}, num_candidates={self.num_candidates}")
            print(f"  Initial energies: {[f'{e:.4f}' for e in self.energy_scores]}")
            print()

        self.append_snapshot_to_history(step=0)

        # Main MCMC loop
        for step in range(1, self.num_steps + 1):

            # Calculate temperature for simulated annealing
            cur_temp = self._calculate_temperature(step)

            # Save parent states before mutation (needed for rejection handling)
            parent_states = self._save_parent_states(top_k_idx)

            # Expand batch and replicate parents for proposal generation
            self._expand_batch_for_proposals(top_k_idx)

            # Generate proposals from each parent
            self._generate_proposals()

            # Apply MCMC acceptance and select top-k
            top_k_idx = self._select_topk_with_mcmc(top_k_idx, cur_temp, parent_states)

            # Trim segments to only keep top-k sequences (remove rejected proposals)
            self._trim_segments_to_topk(top_k_idx)

            # After trimming, update top_k_idx to reflect new indices (0 to top_k-1)
            top_k_idx = np.arange(self.batch_size)

            # Logging and history tracking
            if self._should_track_history(step):
                if self.verbose:
                    self._log_topk_progress(step)
                self.append_snapshot_to_history(step=step)

        self._track_final_state_if_needed()

    def _expand_batch_for_proposals(self, top_k_idx: np.ndarray) -> None:
        """Expand batch and replicate parent sequences for proposal generation.

        After trimming, segments contain only batch_size (top_k) sequences. This method:
        1. Expands batch to batch_size x num_candidates
        2. Replicates each parent to its designated block of positions:
           - Parent 0: positions [0, num_candidates)
           - Parent 1: positions [num_candidates, 2*num_candidates)
           - etc.

        Args:
            top_k_idx: Indices of parent sequences to replicate (should be [0, 1, ..., batch_size-1]).
        """
        expanded_batch_size = self.batch_size * self.num_candidates

        # Expand and replicate for each segment
        for segment in self.get_generator_outputs():
            new_batch = []
            for parent_idx in top_k_idx:
                source_seq = segment.batch_sequences[parent_idx]
                # Create num_candidates copies of this parent
                for _ in range(self.num_candidates):
                    new_batch.append(copy.deepcopy(source_seq))
            segment.batch_sequences = new_batch

        # Update generator batch sizes
        for gen in self.generators:
            gen.batch_size = expanded_batch_size

        # Expand energy_scores - replicate parent energies to match
        new_energy_scores = []
        for parent_idx in top_k_idx:
            parent_energy = self.energy_scores[parent_idx]
            for _ in range(self.num_candidates):
                new_energy_scores.append(parent_energy)
        self.energy_scores = new_energy_scores

    def _save_parent_states(self, top_k_idx: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Save complete parent states before mutation using deepcopy.

        This is critical for the MCMC rejection mechanism: if a proposal is rejected,
        we restore the parent's complete state (Sequence object + energy).

        Args:
            top_k_idx: Indices of parent sequences to save.

        Returns:
            Dict mapping parent_idx -> {
                'segments': {segment_id -> deepcopied Sequence object},
                'energy': float
            }
        """
        parent_states = {}
        for parent_idx in top_k_idx:
            parent_states[parent_idx] = {
                'segments': {},
                'energy': self.energy_scores[parent_idx]
            }
            for segment in self.get_generator_outputs():
                seg_id = id(segment)
                # Deepcopy captures complete Sequence state (sequence, metadata, all attributes)
                parent_states[parent_idx]['segments'][seg_id] = copy.deepcopy(
                    segment.batch_sequences[parent_idx]
                )
        return parent_states

    def _restore_parent_state(
        self,
        target_idx: int,
        parent_idx: int,
        parent_states: Dict[int, Dict[str, Any]]
    ) -> None:
        """Restore complete parent state after MCMC rejection.

        Replaces the rejected proposal at target_idx with the deepcopied parent
        Sequence object and restores its energy score.

        Args:
            target_idx: Batch index to restore to (the rejected proposal's position).
            parent_idx: Index of the parent in the saved states dict.
            parent_states: Dict from _save_parent_states() containing deepcopied
                Sequence objects and energy scores.
        """
        # Deepcopy again to ensure independent Sequence objects at each batch position
        for segment in self.get_generator_outputs():
            seg_id = id(segment)
            segment.batch_sequences[target_idx] = copy.deepcopy(
                parent_states[parent_idx]['segments'][seg_id]
            )

        # Restore energy score to match the restored sequence
        self.energy_scores[target_idx] = parent_states[parent_idx]['energy']

    def _generate_proposals(self) -> None:
        """Generate proposals by sampling from a random generator and scoring energies.

        Picks one generator randomly, calls its sample() method to mutate sequences,
        then evaluates all sequences using constraints to compute energy scores.
        """
        generator = random.choice(self.generators)
        generator.sample()
        self.score_energy()

    def _select_topk_with_mcmc(
        self,
        top_k_idx: np.ndarray,
        temperature: float,
        parent_states: Dict[int, Dict[str, Any]]
    ) -> np.ndarray:
        """Apply Metropolis-Hastings acceptance criterion and select top-k sequences.

        For each proposal:
        1. Compute acceptance probability: alpha = min(1, exp(-(E_proposal - E_parent) / T))
        2. Accept with probability alpha: keep proposal
        3. Reject with probability (1-alpha): restore parent state (sequence, metadata, energy)
        4. Add either accepted proposal or restored parent to candidates

        After all proposals are evaluated, select top batch_size candidates by energy.
        When batch_size=1, this behaves as standard single-chain MCMC.

        Args:
            top_k_idx: Indices of current parent sequences.
            temperature: Current temperature for acceptance calculation.
            parent_states: Saved parent states (sequences, metadata, energies).

        Returns:
            new_top_k_idx: Indices of selected top batch_size sequences for next iteration.
        """
        candidates = []

        for parent_pos, parent_idx in enumerate(top_k_idx):
            # Get parent energy from saved state (before proposals were generated)
            parent_energy = parent_states[parent_idx]['energy']
            start_idx = parent_pos * self.num_candidates
            end_idx = (parent_pos + 1) * self.num_candidates

            for proposal_idx in range(start_idx, end_idx):
                proposal_energy = self.energy_scores[proposal_idx]
                alpha = self._compute_acceptance_prob(parent_energy, proposal_energy, temperature)

                if random.random() < alpha:
                    # Accept proposal
                    candidates.append((proposal_energy, proposal_idx))
                else:
                    # Reject - restore complete parent state to this position
                    self._restore_parent_state(proposal_idx, parent_idx, parent_states)
                    # Use the saved parent energy for candidate list
                    restored_energy = parent_states[parent_idx]['energy']
                    candidates.append((restored_energy, proposal_idx))

        candidates.sort(key=lambda x: x[0])
        top_k_candidates = candidates[:self.batch_size]

        new_top_k_idx = np.array([idx for _, idx in top_k_candidates])

        return new_top_k_idx

    def _trim_segments_to_topk(self, top_k_idx: np.ndarray) -> None:
        """Trim all segments to only keep the top batch_size sequences.

        After MCMC selection, segments contain expanded batch of sequences
        (batch_size x num_candidates). This method trims them back to batch_size,
        keeping only the selected top sequences, making them ready for user inspection.

        Args:
            top_k_idx: Indices of the top batch_size sequences to keep (length = batch_size).
        """
        for segment in self.get_generator_outputs():
            segment.batch_sequences = [segment.batch_sequences[i] for i in top_k_idx]

        # Also trim energy_scores to match
        self.energy_scores = [self.energy_scores[i] for i in top_k_idx]

    def _log_topk_progress(self, step: int) -> None:
        """Log optimization progress.

        Prints current step, energy statistics, and temperature for monitoring.
        When batch_size=1, only best energy is meaningful (mean=best, std=0).

        Args:
            step: Current MCMC iteration number.
        """
        best_energy = min(self.energy_scores)
        mean_energy = np.mean(self.energy_scores)
        worst_energy = max(self.energy_scores)
        std_energy = np.std(self.energy_scores) if len(self.energy_scores) > 1 else 0.0
        current_temp = self._calculate_temperature(step)

        # Format output based on top_k
        if self.batch_size == 1:
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
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()

    # ==================== Helper Methods ====================

    def _calculate_temperature(self, step: int) -> float:
        """
        Calculate annealed temperature for given step.

        Uses exponential cooling schedule: T(step) = T_max * (T_min/T_max)^((step-1)/(num_steps-1))

        For num_steps=1, T(step) = T_max
        For num_steps>1, T(step) = T_max * (T_min/T_max)^((step-1)/(num_steps-1))
        For step=num_steps, T(step) = T_min

        Note: -1 is required in denominator because we loop from 1 to num_steps in the MCMC sampling loop
        """

        # Handle division by 0 for num_steps=1
        if self.num_steps == 1:
            return self.temperature
        return self.temperature * (self.temperature_min / self.temperature) ** ((step - 1) / (self.num_steps - 1))

    def _compute_acceptance_prob(self, current_energy: float, proposed_energy: float, temperature: float) -> float:
        """Compute Metropolis-Hastings acceptance probability."""
        energy_diff = -(proposed_energy - current_energy) / temperature
        energy_diff = min(energy_diff, MAX_EXP_ARG)
        return min(1.0, np.exp(energy_diff))

    def _should_track_history(self, step: int) -> bool:
        """Check if current step should be tracked in history."""
        return step % self.track_step_size == 0

    def _track_final_state_if_needed(self) -> None:
        """Save final state to history if it wasn't already tracked."""
        if self.num_steps % self.track_step_size != 0:
            self.append_snapshot_to_history(step=self.num_steps)
