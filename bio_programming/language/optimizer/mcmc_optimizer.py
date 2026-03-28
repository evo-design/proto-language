"""
proto_language/language/optimizer/mcmc_optimizer.py

Metropolis-Hastings MCMC Optimizer that uses multiple sub-generators as proposal distributions and constraints to define the energy function.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, final

import numpy as np
from pydantic import model_validator

logger = logging.getLogger(__name__)


from proto_language.base_config import BaseOptimizerConfig, ConfigField
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

class MCMCOptimizerConfig(BaseOptimizerConfig):
    """Configuration object for MCMCOptimizer.

    This class defines configuration parameters for the Metropolis-Hastings MCMC
    optimizer, which explores sequence space through iterative mutation with
    probabilistic acceptance based on energy improvements.

    Attributes:
        num_results (int | None): Number of result sequences to optimize in
            parallel. Each result sequence is an independent MCMC trajectory.
            When ``num_results=1`` (standard single-chain MCMC), only one
            sequence is optimized. Overrides program-level ``num_results`` if set.

        num_steps (int): Number of MCMC steps to run. Each step generates proposals,
            evaluates them, and accepts/rejects based on Metropolis-Hastings criterion.
            More steps allow better exploration but increase runtime.

        proposals_per_result (int): Number of proposals to generate per
            result sequence at each step. Total proposals per step equals
            ``num_results x proposals_per_result``.
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

        verbose (bool): Whether to print detailed progress information at each
            step, including energy statistics and temperature. Default: ``False``.
        tracking_interval (int): Number of steps between progress snapshots.
        track_proposals (bool): Whether to record proposal sequences alongside accepted results.

    Note:
        - When ``num_results=1`` (default), behaves like standard single-chain MCMC.
        - When ``num_results > 1``, maintains that many independent trajectories and
          generates ``proposals_per_result`` (default: 1) proposals per result sequence each step.
        - Temperature annealing follows exponential decay:
          T(step) = T_max x (T_min / T_max)^(step / num_steps)
    """
    # Required parameters
    num_steps: int = ConfigField(
        ge=1,
        title="Num Steps",
        description="Number of MCMC steps to run."
    )

    # Advanced parameters
    num_results: Optional[int] = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Candidate designs for this optimizer. Overrides program-level count.",
        advanced=True,
    )
    proposals_per_result: int = ConfigField(
        default=1,
        ge=1,
        title="Proposals Per Result",
        description="Number of proposals to generate per result sequence per MCMC step.",
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

    At each step, the optimizer generates ``num_results x proposals_per_result``
    proposals by mutating each of the K sequences ``proposals_per_result`` times.
    Each trajectory (result index) is independent. For each trajectory, the best proposal
    (lowest energy) is selected, then MH acceptance is applied to decide whether to
    accept or reject that proposal. If rejected, the trajectory keeps its previous state.

    Attributes:
        num_results: Number of result sequences to optimize in parallel.
        num_steps: Total number of MCMC steps to run.
        proposals_per_result: Number of proposals per result sequence.
        max_temperature: Starting temperature for annealing.
        min_temperature: Ending temperature for annealing.

    Example:
        >>> constructs = [Construct([segment1, segment2])]
        >>> config = MCMCOptimizerConfig(
        ...     num_results=1,
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
        - When ``proposals_per_result > 1``, generates multiple proposals per
          trajectory, selects the best one, then applies a single MH accept/reject decision
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
            constructs (list[Construct]): List of Construct objects to optimize.
            generators (list[Generator]): List of Generator objects for sequence modification.
            constraints (list[Constraint]): List of Constraint objects for evaluation.
            config (MCMCOptimizerConfig): Configuration object containing algorithm parameters (temperature, num_steps, etc.).
            custom_logging (Callable | None): Optional callback called at tracked steps (governed by ``tracking_interval``).
            clear_tool_cache (int | bool | list[str]): (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If any validation checks fail or num_results cannot be determined.
        """
        self.config = config

        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_results=config.num_results,
            proposals_per_result=config.proposals_per_result,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
        )

        self.num_steps: int = config.num_steps
        self.max_temperature: float = config.max_temperature
        self.min_temperature: float = config.min_temperature
        self._checkpoint_path: Optional[Path] = None
        self._checkpoint_interval_steps: int = 1
        self._resume_from_checkpoint: bool = False
        self._last_completed_step: int = 0

    def configure_checkpointing(
        self,
        checkpoint_path: str | Path,
        save_interval_steps: int = 1,
        resume: bool = True,
    ) -> None:
        """Enable optional MCMC checkpointing and resume support."""
        if save_interval_steps < 1:
            raise ValueError(
                f"save_interval_steps must be >= 1, got {save_interval_steps}"
            )
        self._checkpoint_path = Path(checkpoint_path)
        self._checkpoint_interval_steps = save_interval_steps
        self._resume_from_checkpoint = resume

    def checkpoint_now(self) -> None:
        """Persist the most recently completed MCMC step immediately."""
        if self._checkpoint_path is None:
            return
        self._save_checkpoint(self._last_completed_step)

    def _capture_rng_state(self) -> Dict[str, Any]:
        """Capture RNG state for deterministic resume."""
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        try:
            import torch  # type: ignore

            state["torch_cpu"] = torch.get_rng_state().cpu()
            if torch.cuda.is_available():
                state["torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            # Torch is optional; do not fail checkpointing if unavailable.
            pass
        return state

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        """Restore RNG state captured by _capture_rng_state."""
        if not state:
            return

        python_state = state.get("python")
        if python_state is not None:
            random.setstate(python_state)

        numpy_state = state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        try:
            import torch  # type: ignore

            torch_cpu_state = state.get("torch_cpu")
            if torch_cpu_state is not None:
                torch.set_rng_state(torch_cpu_state)

            torch_cuda_state = state.get("torch_cuda")
            if torch_cuda_state is not None and torch.cuda.is_available():
                if len(torch_cuda_state) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(torch_cuda_state)
                else:
                    logger.warning(
                        "Skipping CUDA RNG restore: checkpoint has %d devices, runtime has %d devices.",
                        len(torch_cuda_state),
                        torch.cuda.device_count(),
                    )
        except Exception:
            # Torch is optional; do not fail resume if unavailable.
            pass

    def _save_checkpoint(self, step: int) -> None:
        """Atomically persist MCMC state, including candidate-pool diagnostics."""
        if self._checkpoint_path is None:
            return

        checkpoint_state = {
            "version": 3,
            "optimizer": self.__class__.__name__,
            "step": step,
            "num_steps": self.num_steps,
            "num_results": self.num_results,
            "proposals_per_result": self._proposals_per_result,
            "energy_scores": self.energy_scores[: self.num_results],
            "result_sequences": [
                [seq.to_dict() for seq in segment.result_sequences]
                for segment in self.segments
            ],
            "proposal_sequences": [
                [seq.to_dict() for seq in segment.proposal_sequences]
                for segment in self.segments
            ],
            "proposal_energy_scores": list(self._proposal_energy_scores),
            "proposal_outcomes": list(self._proposal_outcomes),
            "rng_state": self._capture_rng_state(),
        }

        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._checkpoint_path.with_suffix(
            f"{self._checkpoint_path.suffix}.tmp"
        )
        with tmp_path.open("wb") as handle:
            pickle.dump(checkpoint_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(self._checkpoint_path)

    def _load_checkpoint(self) -> int:
        """Load and restore checkpoint state, returning the last completed step."""
        if self._checkpoint_path is None:
            return 0

        with self._checkpoint_path.open("rb") as handle:
            checkpoint_state = pickle.load(handle)

        if checkpoint_state.get("optimizer") != self.__class__.__name__:
            raise ValueError(
                f"Checkpoint optimizer mismatch: expected {self.__class__.__name__}, "
                f"got {checkpoint_state.get('optimizer')}"
            )

        checkpoint_num_results = checkpoint_state.get("num_results")
        if checkpoint_num_results != self.num_results:
            raise ValueError(
                f"Checkpoint num_results mismatch: expected {self.num_results}, "
                f"got {checkpoint_num_results}"
            )

        result_sequences = checkpoint_state.get("result_sequences")
        if result_sequences is None:
            # Backward compatibility with older checkpoints written before the
            # selected->result API rename.
            result_sequences = checkpoint_state.get("selected_sequences")

        if not isinstance(result_sequences, list) or len(result_sequences) != len(
            self.segments
        ):
            raise ValueError("Checkpoint result_sequences does not match segment count.")

        for segment, segment_state in zip(self.segments, result_sequences):
            if len(segment_state) != self.num_results:
                raise ValueError(
                    f"Checkpoint result sequence count mismatch for segment "
                    f"'{segment.label or 'unlabeled'}'."
                )
            segment.result_sequences = [Sequence.from_dict(s) for s in segment_state]

        proposal_sequences = checkpoint_state.get("proposal_sequences")
        if proposal_sequences is None:
            # Backward compatibility with older checkpoints.
            proposal_sequences = checkpoint_state.get("candidate_sequences")
        if (
            isinstance(proposal_sequences, list)
            and len(proposal_sequences) == len(self.segments)
            and all(
                isinstance(segment_state, list)
                and len(segment_state) == self.num_proposals
                for segment_state in proposal_sequences
            )
        ):
            for segment, segment_state in zip(self.segments, proposal_sequences):
                segment.proposal_sequences = [Sequence.from_dict(s) for s in segment_state]
        else:
            # Proposal pool is regenerated from result pool each iteration anyway.
            self._populate_proposal_sequences()

        energy_scores = checkpoint_state.get("energy_scores")
        if not isinstance(energy_scores, list) or len(energy_scores) != self.num_results:
            raise ValueError("Checkpoint energy_scores does not match num_results.")
        self.energy_scores = energy_scores

        proposal_energy_scores = checkpoint_state.get("proposal_energy_scores")
        if proposal_energy_scores is None:
            proposal_energy_scores = checkpoint_state.get("candidate_energy_scores")
        if (
            isinstance(proposal_energy_scores, list)
            and len(proposal_energy_scores) == self.num_proposals
        ):
            self._proposal_energy_scores = list(proposal_energy_scores)
        else:
            self._proposal_energy_scores = []

        proposal_outcomes = checkpoint_state.get("proposal_outcomes")
        if proposal_outcomes is None:
            proposal_outcomes = checkpoint_state.get("candidate_outcomes")
        if (
            isinstance(proposal_outcomes, list)
            and len(proposal_outcomes) == self.num_proposals
            and all(isinstance(outcome, str) for outcome in proposal_outcomes)
        ):
            self._proposal_outcomes = list(proposal_outcomes)
        else:
            self._proposal_outcomes = []

        checkpoint_num_steps = checkpoint_state.get("num_steps")
        if checkpoint_num_steps != self.num_steps:
            logger.warning(
                "Checkpoint num_steps=%s differs from current num_steps=%s; "
                "resuming to current target.",
                checkpoint_num_steps,
                self.num_steps,
            )

        self._restore_rng_state(checkpoint_state.get("rng_state", {}))
        step = int(checkpoint_state.get("step", 0))
        self._last_completed_step = max(0, min(step, self.num_steps))
        return self._last_completed_step

    def run(self) -> None:
        """
        Execute Metropolis-Hastings MCMC sampling for sequence optimization.

        Runs the specified number of MCMC steps, where each step:
        1. Maintains `num_results` independent trajectories in `result_sequences`
        2. Creates `proposal_sequences` by replicating each result sequence `proposals_per_result` times
        3. Generates proposals (mutates `proposal_sequences` in-place)
        4. For each trajectory, independently apply MH acceptance and select the best accepted proposal

        Note:
            - Each trajectory (result index) is independent with no cross-trajectory mixing.
            - Simulated annealing: T(step) = T_max * (T_min / T_max) ^ (step / num_steps)
            - Total proposals per step: num_results x proposals_per_result
            - Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        self._prepare_run()
        start_step = 0
        resumed_from_checkpoint = False

        if (
            self._checkpoint_path is not None
            and self._resume_from_checkpoint
            and self._checkpoint_path.exists()
        ):
            start_step = self._load_checkpoint()
            resumed_from_checkpoint = True
            logger.info(
                "Loaded MCMC checkpoint at step %d from %s",
                start_step,
                self._checkpoint_path,
            )
        else:
            # Score initial state if sequences are non-empty (skip for autoregressive generators like ProGen2)
            if any(seq.sequence for segment in self.segments for seq in segment.proposal_sequences):
                self.score_energy()
            else:
                self.energy_scores = [float('inf')] * self.num_proposals

            # Truncate to num_results for initial snapshot (score_energy sets to num_proposals)
            self.energy_scores = self.energy_scores[:self.num_results]
            self._last_completed_step = 0

            if self.verbose:
                logger.info("MCMC initialization:")
                logger.info(f"  num_results={self.num_results}, proposals_per_result={self._proposals_per_result}")
                logger.info(f"  Initial energy: {self.energy_scores[0]:.4f}")

            # Track initial state
            self._save_progress_snapshot(time_step=0)
            if self._checkpoint_path is not None:
                self._save_checkpoint(step=0)

        if resumed_from_checkpoint and start_step >= self.num_steps:
            logger.info(
                "Checkpoint already reached target step (%d); skipping optimization run.",
                self.num_steps,
            )
            return

        # MCMC loop
        for step in range(start_step + 1, self.num_steps + 1):
            # 1. Save state of result_sequences to revert if rejected by Metropolis-Hastings acceptance criterion
            old_result_sequences = self._save_sequence_state()

            # 2. Populate proposal_sequences by replicating each result_sequence proposals_per_result times
            self._populate_proposal_sequences()

            # 3. Generate proposals for proposal_sequences in-place by randomly sampling a generator
            generator = random.choice(self.generators)
            generator.sample()

            # 4. Score proposal_sequences
            self.score_energy()

            # 5. Metropolis-Hastings acceptance and update energy score, proposal_sequences, and result_sequences state
            self._select_topk_with_mcmc_acceptance(step, old_result_sequences)

            # Save snapshot and log at tracking interval or final step
            if step % self.tracking_interval == 0 or step == self.num_steps:
                self._save_progress_snapshot(time_step=step)
                self._log_mcmc_progress(step)

            self._last_completed_step = step
            if (
                self._checkpoint_path is not None
                and (step % self._checkpoint_interval_steps == 0 or step == self.num_steps)
            ):
                self._save_checkpoint(step=step)

    def _save_sequence_state(self) -> List[Tuple[Dict[int, Sequence], float]]:
        """Save state of result sequences.

        Returns:
            list[tuple[dict[int, Sequence], float]]: List of tuples, one per result sequence, each containing:
                - segments dict: {segment_id -> deepcopied Sequence object}
                - energy: float (energy_scores[result_idx])
        """
        sequence_state = []
        for result_idx in range(self.num_results):
            segments_dict = {}
            for segment in self.segments:
                seg_id = id(segment)
                segments_dict[seg_id] = copy.deepcopy(segment.result_sequences[result_idx])
            sequence_state.append((segments_dict, self.energy_scores[result_idx]))
        return sequence_state

    def _populate_proposal_sequences(self) -> None:
        """Populate proposal_sequences by replicating each result_sequence proposals_per_result times.

        Updates proposal_sequences in-place.
        Layout: [sequence_0] * proposals_per_result + [sequence_1] * proposals_per_result + ...
        """
        for segment in self.segments:
            for result_idx in range(self.num_results):
                start_idx = result_idx * self._proposals_per_result
                for offset in range(self._proposals_per_result):
                    segment.proposal_sequences[start_idx + offset] = copy.deepcopy(segment.result_sequences[result_idx])

    def _select_topk_with_mcmc_acceptance(
        self,
        step: int,
        old_result_sequences: List[Tuple[Dict[int, Sequence], float]],
    ) -> None:
        """Select the best proposal per trajectory and apply Metropolis-Hastings acceptance.

        For each trajectory (processed independently):

        1. Find the best proposal by energy from the trajectory's proposal pool.
        2. Apply MH acceptance criterion: accept if ``random() < min(1, exp(-dE/T))``.
        3. Update the trajectory state (accept new or keep old).
        4. Classify each proposal's outcome: "accepted", "Metropolis-Hastings
           rejection", "Not best in proposal pool", or unchanged (filter-rejected).
        5. Truncate energy_scores to num_results (discard stale proposal energies).

        Args:
            step (int): Current MCMC step (used for temperature annealing).
            old_result_sequences (list[tuple[dict[int, Sequence], float]]): Saved trajectory state before proposals.
        """
        outcomes = list(self._proposal_outcomes)

        for result_idx in range(self.num_results):
            old_segments_dict, old_result_energy = old_result_sequences[result_idx]
            proposal_pool_start = result_idx * self._proposals_per_result
            proposal_pool_end = (result_idx + 1) * self._proposals_per_result

            # 1. Find the best proposal by energy
            best_energy = float('inf')
            best_proposal_idx = None
            for proposal_idx in range(proposal_pool_start, proposal_pool_end):
                if outcomes[proposal_idx] != "accepted":
                    continue
                if self.energy_scores[proposal_idx] < best_energy:
                    best_energy = self.energy_scores[proposal_idx]
                    best_proposal_idx = proposal_idx

            # 2. Apply MH acceptance criterion
            valid_proposals_exist = best_proposal_idx is not None
            alpha = self._compute_mcmc_alpha(old_result_energy, best_energy, step)
            accepted = valid_proposals_exist and random.random() < alpha

            # 3. Update trajectory state
            if accepted:
                for segment in self.segments:
                    segment.result_sequences[result_idx] = copy.deepcopy(segment.proposal_sequences[best_proposal_idx])
            else:
                best_energy = old_result_energy
                for segment in self.segments:
                    segment.result_sequences[result_idx] = copy.deepcopy(old_segments_dict[id(segment)])
            self.energy_scores[result_idx] = best_energy

            # 4. Classify each proposal's outcome
            for proposal_idx in range(proposal_pool_start, proposal_pool_end):
                if outcomes[proposal_idx] != "accepted":
                    continue
                if proposal_idx == best_proposal_idx and accepted:
                    outcomes[proposal_idx] = "accepted"
                elif proposal_idx == best_proposal_idx:
                    outcomes[proposal_idx] = "Metropolis-Hastings rejection"
                else:
                    outcomes[proposal_idx] = "Not best in proposal pool"

        self._proposal_outcomes = outcomes

        # 5. Truncate to num_results (score_energy() resizes back each step)
        self.energy_scores = self.energy_scores[:self.num_results]

    def _compute_temperature(self, step: int) -> float:
        """Calculate annealed temperature: T(step) = T_max * (T_min/T_max)^((step-1)/(num_steps-1))

        Args:
            step (int): Current MCMC iteration index.

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

        Args:
            current_energy (float): Energy of the current accepted state.
            proposed_energy (float): Energy of the proposed candidate state.
            step (int): Current MCMC iteration index.

        Note:
        - Always accepts improvements (proposed_energy < current_energy)
        - Accepts worse proposals with probability exp(-(ΔE / T)) where ΔE = proposed - current

        Important: When proposals_per_result > 1, this is applied to the BEST proposal
        from the pool, not a randomly selected one. This "best-of-N then MH" strategy is a
        heuristic that accelerates convergence but does not satisfy detailed balance for the
        true Boltzmann distribution. For mathematically rigorous MCMC sampling, use
        proposals_per_result=1.
        """
        # Non-finite energies: guard against inf - inf = NaN.
        if math.isinf(proposed_energy):
            return 0.0  # Reject infinite proposals (covers inf/inf implicitly)
        if math.isinf(current_energy):
            return 1.0  # Any finite proposal beats infinite current

        temperature = self._compute_temperature(step)
        log_acceptance_ratio = -(proposed_energy - current_energy) / temperature
        # Cap to prevent overflow in exp()
        log_acceptance_ratio = min(log_acceptance_ratio, MAX_EXP_ARG)
        return min(1.0, np.exp(log_acceptance_ratio))

    def _log_mcmc_progress(self, step: int) -> None:
        """Log optimization progress."""
        if self.verbose:
            best_energy = min(self.energy_scores)
            mean_energy = np.mean(self.energy_scores)
            worst_energy = max(self.energy_scores)
            std_energy = np.std(self.energy_scores) if len(self.energy_scores) > 1 else 0.0
            current_temp = self._compute_temperature(step)

            # Format output based on num_results
            if self.num_results == 1:
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
