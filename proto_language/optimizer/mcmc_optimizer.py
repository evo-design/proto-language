"""Metropolis-Hastings MCMC optimizer with simulated annealing.

Provides the ``mcmc`` optimization strategy: a propose-score-refine loop that treats
the weighted constraint energy as the objective to minimize and the assigned mutation
generators as proposal distributions. Each step replicates every result sequence into a
proposal pool, mutates the pool in place with one randomly chosen generator, scores the
proposals, and per trajectory selects the lowest-energy proposal before applying the
Metropolis-Hastings accept/reject rule. Acceptance always keeps improvements and accepts
worse proposals with probability ``exp(-dE / T)``, where the temperature anneals from
``max_temperature`` to ``min_temperature`` along ``temperature_schedule`` over ``num_steps``.

Examples:
    >>> from proto_language.constraint import gc_content_constraint
    >>> from proto_language.core import Constraint, Construct, Program, Segment
    >>> from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
    >>> from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
    >>> seg = Segment(length=20, sequence_type="dna")
    >>> gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    >>> gen.assign(seg)
    >>> gc = Constraint(inputs=[seg], function=gc_content_constraint, function_config={"min_gc": 80, "max_gc": 90})
    >>> optimizer = MCMCOptimizer(
    ...     constructs=[Construct([seg])],
    ...     generators=[gen],
    ...     constraints=[gc],
    ...     config=MCMCOptimizerConfig(num_results=1, proposals_per_result=20, num_steps=10),
    ... )
    >>> Program(optimizers=[optimizer], num_results=1).run()
"""

import copy
import logging
import math
from collections.abc import Callable
from typing import Any, final

import numpy as np
from pydantic import model_validator

from proto_language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Sequence,
)
from proto_language.optimizer.optimizer_registry import optimizer
from proto_language.utils.base import BaseOptimizerConfig, ConfigField
from proto_language.utils.scheduling import SCHEDULES, Schedule, Scheduler

logger = logging.getLogger(__name__)

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

        temperature_schedule (Scheduler): Annealing schedule from max to min temperature.

        verbose (bool): Whether to print detailed progress information at each
            step, including energy statistics and temperature. Default: ``False``.

        tracking_interval (int): Number of steps between progress snapshots.

        track_proposals (bool): Whether to record proposal sequences alongside accepted results.

    Note:
        - When ``num_results=1`` (default), behaves like standard single-chain MCMC.
        - When ``num_results > 1``, maintains that many independent trajectories and
          generates ``proposals_per_result`` (default: 1) proposals per result sequence each step.
    """

    # Required parameters
    num_steps: int = ConfigField(
        ge=1,
        title="Number of MCMC Steps",
        description="Number of Metropolis-Hastings steps. Each step proposes, evaluates, and accept/reject samples.",
    )

    # Advanced parameters
    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Independent MCMC trajectories run in parallel; each yields one candidate. Overrides program count.",
    )
    proposals_per_result: int = ConfigField(
        default=1,
        ge=1,
        title="Proposals Per Step",
        description="Proposals per trajectory each step; the best by energy is chosen, then accept/reject is applied.",
    )
    max_temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Max Temperature",
        description="Starting temperature for simulated annealing; higher accepts worse proposals more readily.",
    )
    min_temperature: float = ConfigField(
        default=0.001,
        gt=0.0,
        title="Min Temperature",
        description="Ending dimensionless temperature for simulated annealing; must be greater than 0 and below the max.",
    )
    temperature_schedule: Scheduler = ConfigField(
        default="exponential",
        title="Temperature Schedule",
        description="Annealing schedule from max to min temperature.",
    )

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "MCMCOptimizerConfig":
        """Validate cross-field constraints."""
        if self.min_temperature >= self.max_temperature:
            raise ValueError(
                f"min_temperature ({self.min_temperature}) must be < max_temperature ({self.max_temperature})"
            )
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
        >>> config = MCMCOptimizerConfig(num_results=1, num_steps=100, max_temperature=0.5, min_temperature=0.001)
        >>> mcmc = MCMCOptimizer(
        ...     constructs=constructs, generators=[mutation_gen], constraints=[gc_constraint], config=config
        ... )
        >>> mcmc.run()
        >>> final_sequences = mcmc.constructs[0].joined_sequences

    Note:
        - Typically used with mutation generators (``category="mutation"``)
        - Uses Metropolis-Hastings acceptance: always accepts improvements,
          accepts worse proposals with probability exp(-ΔE/T)
        - Simulated annealing via configurable ``temperature_schedule``
        - Lower energy scores are better (minimization objective)
        - When ``proposals_per_result > 1``, generates multiple proposals per
          trajectory, selects the best one, then applies a single MH accept/reject decision
    """

    # Class attribute required by OptimizerRegistry
    config_class = MCMCOptimizerConfig
    config: MCMCOptimizerConfig

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: MCMCOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the MCMC Optimizer with sub-generators and constraints.

        Args:
            constructs (list[Construct]): List of Construct objects to optimize.
            generators (list[Generator]): List of Generator objects for sequence modification.
            constraints (list[Constraint]): List of Constraint objects for evaluation.
            config (MCMCOptimizerConfig): Configuration object containing algorithm parameters (temperature, num_steps, etc.).
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked steps (governed by ``tracking_interval``).
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
            seed=config.seed,
        )

        self.num_steps: int = config.num_steps
        self.max_temperature: float = config.max_temperature
        self.min_temperature: float = config.min_temperature
        self._temperature_schedule = self._build_temperature_schedule(config)

    def run(self) -> None:
        """Execute Metropolis-Hastings MCMC sampling for sequence optimization.

        Runs the specified number of MCMC steps, where each step:
        1. Maintains `num_results` independent trajectories in `result_sequences`
        2. Creates `proposal_sequences` by replicating each result sequence `proposals_per_result` times
        3. Generates proposals (mutates `proposal_sequences` in-place)
        4. For each trajectory, independently apply MH acceptance and select the best accepted proposal

        Note:
            - Each trajectory (result index) is independent with no cross-trajectory mixing.
            - Temperature anneals via ``temperature_schedule`` (default: exponential)
            - Total proposals per step: num_results x proposals_per_result
            - Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        self._prepare_run()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing

        n_filter = sum(1 for c in self.constraints if c.threshold is not None)
        n_score = len(self.constraints) - n_filter
        t0 = self._temperature_schedule(0, self.num_steps)
        tN = self._temperature_schedule(self.num_steps, self.num_steps)
        logger.info(
            f"MCMCOptimizer: {self.num_steps} steps, "
            f"{self.num_results} trajectories x {self._proposals_per_result} proposals "
            f"= {self.num_proposals}/step, temp {t0:.2f}->{tN:.2f}, "
            f"{len(self.constraints)} constraints ({n_filter} filter, {n_score} scoring)"
        )

        # all(): score only when every design segment is ready. any() let scoring
        # run on empty design segments paired with a fixed target.
        if all(seq.sequence for segment in self.segments for seq in segment.proposal_sequences):
            self.score_energy()
        else:
            self.energy_scores = [float("inf")] * self.num_proposals

        # Truncate to num_results for initial snapshot (score_energy sets to num_proposals)
        self.energy_scores = self.energy_scores[: self.num_results]
        logger.debug(f"MCMCOptimizer initial energy: {self.energy_scores[0]:.4f}")

        # Track initial state
        self._save_progress_snapshot(
            time_step=0,
            optimizer_metadata={
                "type": "mcmc",
                "num_steps": self.num_steps,
                "num_results": self.num_results,
                "proposals_per_result": self._proposals_per_result,
                "temperature": self._temperature_schedule(0, self.num_steps),
                "proposal_count": len(self._proposal_outcomes),
                "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
            },
        )

        # MCMC loop
        for step in range(1, self.num_steps + 1):
            # 1. Save state of result_sequences to revert if rejected by Metropolis-Hastings acceptance criterion
            old_result_sequences = self._save_sequence_state()

            # 2. Populate proposal_sequences by replicating each result_sequence proposals_per_result times
            self._populate_proposal_sequences()

            # 3. Generate proposals for proposal_sequences in-place by randomly sampling a generator
            generator = self._rng.choice(self.generators)
            generator.sample()

            # 4. Score proposal_sequences
            self.score_energy()

            # 5. Metropolis-Hastings acceptance and update energy score, proposal_sequences, and result_sequences state
            self._select_topk_with_mcmc_acceptance(step, old_result_sequences)

            # Save snapshot and log at tracking interval or final step
            if step % self.tracking_interval == 0 or step == self.num_steps:
                self._save_progress_snapshot(
                    time_step=step,
                    optimizer_metadata={
                        "type": "mcmc",
                        "num_steps": self.num_steps,
                        "num_results": self.num_results,
                        "proposals_per_result": self._proposals_per_result,
                        "temperature": self._temperature_schedule(step, self.num_steps),
                        "proposal_count": len(self._proposal_outcomes),
                        "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
                    },
                )
                self._log_mcmc_progress(step)

    def _save_sequence_state(self) -> list[tuple[dict[int, Sequence], float]]:
        """Save state of result sequences.

        Returns:
            list[tuple[dict[int, Sequence], float]]: List of tuples, one per result sequence, each containing:
                - segments dict: {segment_id -> deepcopied Sequence object}
                - energy: float (energy_scores[result_idx])
        """
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
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
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing
        for segment in self.segments:
            for result_idx in range(self.num_results):
                start_idx = result_idx * self._proposals_per_result
                for offset in range(self._proposals_per_result):
                    segment.proposal_sequences[start_idx + offset] = copy.deepcopy(segment.result_sequences[result_idx])

    def _select_topk_with_mcmc_acceptance(
        self,
        step: int,
        old_result_sequences: list[tuple[dict[int, Sequence], float]],
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

        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        for result_idx in range(self.num_results):
            old_segments_dict, old_result_energy = old_result_sequences[result_idx]
            proposal_pool_start = result_idx * self._proposals_per_result
            proposal_pool_end = (result_idx + 1) * self._proposals_per_result

            # 1. Find the best proposal by energy
            best_energy = float("inf")
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
            accepted = valid_proposals_exist and self._rng.random() < alpha

            # 3. Update trajectory state
            if accepted:
                for segment in self.segments:
                    segment.result_sequences[result_idx] = copy.deepcopy(segment.proposal_sequences[best_proposal_idx])  # type: ignore[index]
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
        self.energy_scores = self.energy_scores[: self.num_results]

    @staticmethod
    def _build_temperature_schedule(config: MCMCOptimizerConfig) -> Schedule:
        """Build a temperature schedule mapping MCMC step to temperature.

        Shifts indices so step=1 is exactly ``max_temperature`` and
        step=num_steps is exactly ``min_temperature``.
        """
        base = SCHEDULES[config.temperature_schedule](config.max_temperature, config.min_temperature)
        n = config.num_steps
        if n <= 1:
            return lambda _s, _t: config.max_temperature
        return lambda step, _total: base(max(step - 1, 0), n - 1)

    def _compute_mcmc_alpha(self, current_energy: float, proposed_energy: float, step: int) -> float:
        """Compute Metropolis-Hastings acceptance probability: alpha = min(1, exp(-(E_new - E_old) / T)).

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

        temperature = self._temperature_schedule(step, self.num_steps)
        log_acceptance_ratio = -(proposed_energy - current_energy) / temperature
        # Cap to prevent overflow in exp()
        log_acceptance_ratio = min(log_acceptance_ratio, MAX_EXP_ARG)
        return float(min(1.0, np.exp(log_acceptance_ratio)))

    def _log_mcmc_progress(self, step: int) -> None:
        """Log optimization progress as a multi-line INFO block."""
        logger.info(f"Iteration {step}/{self.num_steps}")
        filter_summary = self._format_filter_summary()
        if filter_summary is not None:
            logger.info(f"  filters: {filter_summary}")
        for line in self._format_scoring_lines():
            logger.info(f"  {line}")
        logger.info(f"  energy:  {self._format_energy_summary()}")
        accepted = self._proposal_outcomes.count("accepted")
        current_temp = self._temperature_schedule(step, self.num_steps)
        logger.info(f"  accepted {accepted}/{self.num_proposals} proposals, T={current_temp:.4f}")

        if self.custom_logging:
            self.custom_logging(step, self.segments)
