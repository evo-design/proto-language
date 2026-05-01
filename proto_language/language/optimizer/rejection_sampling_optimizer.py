"""Rejection Sampling Optimizer that runs multiple independent sampling rounds and returns the best constructs."""

import bisect
import copy
import logging
import math
from collections.abc import Callable
from typing import Any, final

from pydantic import model_validator

from proto_language.base_config import BaseOptimizerConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Sequence,
)
from proto_language.language.optimizer.optimizer_registry import optimizer

logger = logging.getLogger(__name__)


class RejectionSamplingOptimizerConfig(BaseOptimizerConfig):
    """Configuration object for RejectionSamplingOptimizer.

    The Rejection Sampling optimizer generates many proposal sequences and keeps only
    the best ``num_results`` by lowest energy score. It samples in rounds and maintains
    a sorted list to track the best proposals.

    Attributes:
        num_samples (int): Maximum number of samples to generate. Rounded up to the
            nearest multiple of ``samples_per_round``. Must be at least ``num_results``.

        num_results (int | None): Number of top sequences to keep and return (lowest
            energy scores). Overrides program-level ``num_results`` if set.

        samples_per_round (int): Number of proposal sequences to generate
            and evaluate per sampling round. Default: ``1``.

        energy_threshold (float | None): If set, enables early stopping. The
            optimizer stops before reaching ``num_samples`` if all top-``num_results``
            best proposals have energy below this threshold. Default: ``None``
            (no early stopping).

        verbose (bool): Print progress information. Default: ``False``.
        tracking_interval (int): Number of steps between progress snapshots.
        track_proposals (bool): Whether to record proposal sequences alongside accepted results.

    Note:
        If filter constraints reject many proposals (returning inf/nan energies),
        the optimizer may return fewer than ``num_results`` valid results.

    """

    # Required parameters
    num_samples: int = ConfigField(
        ge=1,
        title="Num Samples",
        description="Number of samples to generate.",
    )

    # Advanced parameters
    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Top candidate designs to keep for this optimizer. Overrides program-level count.",
        advanced=True,
    )
    samples_per_round: int = ConfigField(
        default=1,
        ge=1,
        title="Samples Per Round",
        description="Number of proposal sequences to generate and evaluate per sampling round.",
        advanced=True,
    )
    energy_threshold: float | None = ConfigField(
        default=None,
        ge=0.0,
        title="Energy Threshold",
        description="Early stop when all energy scores in results are below threshold.",
        advanced=True,
    )

    @model_validator(mode="after")
    def validate_params(self) -> "RejectionSamplingOptimizerConfig":
        """Validate parameter relationships."""
        # num_results must not exceed num_samples (only validate when num_results is set)
        if self.num_results is not None and self.num_results > self.num_samples:
            raise ValueError(
                f"num_results ({self.num_results}) cannot exceed num_samples ({self.num_samples}). Cannot keep more sequences than generated."
            )
        return self


@optimizer(
    key="rejection-sampling",
    label="Rejection Sampling Optimizer",
    config=RejectionSamplingOptimizerConfig,
    description="Optimizer that runs sampling rounds and keeps the best constructs by energy score",
)
@final
class RejectionSamplingOptimizer(Optimizer):
    """Rejection Sampling optimizer for sequence optimization through extensive sampling.

    Generates many proposal sequences and keeps only the best ``num_results`` by
    lowest energy score. Unlike iterative optimizers (MCMC, beam search), each
    sampling round starts fresh from the original sequences. There is no state
    carried between rounds.

    Each round:
    1. Resets proposals to the original sequence
    2. Applies all generators sequentially
    3. Evaluates proposals with constraints
    4. Updates the sorted results list if any proposals are better than the current worst

    If ``energy_threshold`` is set, the optimizer stops early once all best
    proposals have energy below the threshold.

    Attributes:
        num_samples: Maximum samples to generate.
        num_results: Number of top sequences to keep.
        samples_per_round: Proposals generated and evaluated per round.
        energy_threshold: Early stopping threshold.

    Note:
        If filter constraints reject many proposals, the optimizer may return
        fewer than ``num_results`` valid results.

    Example:
        >>> config = RejectionSamplingOptimizerConfig(num_samples=100, num_results=10, samples_per_round=10)
        >>> optimizer = RejectionSamplingOptimizer(
        ...     constructs=constructs, generators=[mutation_gen], constraints=[gc_constraint], config=config
        ... )
        >>> optimizer.run()
        >>> best_sequences = optimizer.constructs[0].segments[0].result_sequences

        With early stopping:

        >>> config = RejectionSamplingOptimizerConfig(
        ...     num_samples=1000,
        ...     num_results=10,
        ...     samples_per_round=10,
        ...     energy_threshold=0.5,  # Stop when all top-10 have energy < 0.5
        ... )
    """

    # Class attribute required by OptimizerRegistry
    config_class = RejectionSamplingOptimizerConfig

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: RejectionSamplingOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the Rejection Sampling Optimizer.

        Args:
            constructs (list[Construct]): List of Construct objects to optimize.
            generators (list[Generator]): List of Generator objects for sequence modification.
            constraints (list[Constraint]): List of Constraint objects for evaluation.
            config (RejectionSamplingOptimizerConfig): Configuration object containing algorithm parameters.
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked rounds (governed by ``tracking_interval``).
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
            num_proposals=config.samples_per_round,
            num_results=config.num_results,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
            seed=config.seed,
        )
        self.samples_per_round: int = config.samples_per_round
        self.energy_threshold: float | None = config.energy_threshold

        self.num_samples: int = config.num_samples
        if self.num_samples % self.samples_per_round != 0:
            self.num_samples = math.ceil(self.num_samples / self.samples_per_round) * self.samples_per_round
            logger.warning(
                f"num_samples rounded up to {self.num_samples} (nearest multiple of samples_per_round={self.samples_per_round})."
            )

        # Override base class num_steps for progress tracking
        self.num_steps = self.num_samples // self.samples_per_round
        # Sorted list of energies for result_sequences (ascending order,
        # parallel to result_sequences; index i matches segment.result_sequences[i])
        self._result_energies: list[float] = []

    def _insert_into_results(self, pos: int, proposal_idx: int, energy: float) -> None:
        """Insert a proposal into the sorted results at the given position."""
        self._result_energies.insert(pos, energy)
        for segment in self.segments:
            segment.result_sequences.insert(pos, copy.deepcopy(segment.proposal_sequences[proposal_idx]))

    def _remove_worst_result(self) -> None:
        """Remove the worst (last) entry from the sorted results."""
        self._result_energies.pop()
        for segment in self.segments:
            segment.result_sequences.pop()

    def _run_sampling_round(self, round_num: int, save_snapshot: bool = True) -> None:
        """Execute a single sampling round.

        1. Reset proposal sequences to their initial state (fresh each round).
        2. Run all generators sequentially on the proposals.
        3. Score proposals with constraints (sets ``_proposal_outcomes``).
        4. Update the sorted results list and classify outcomes.
        5. Optionally save a progress snapshot from the current sorted state.

        Args:
            round_num (int): The 1-indexed round number (for tracking purposes).
            save_snapshot (bool): Whether to save a progress snapshot after this round.
        """
        assert self._initial_state is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing
        # 1. Reset proposal sequences to their initial state
        for seg_idx, segment in enumerate(self.segments):
            proposals = self._initial_state["segments"][seg_idx]["proposals"]
            segment.proposal_sequences = [Sequence.from_dict(proposals[i]) for i in range(self.num_proposals)]

        # 2. Run all generators sequentially
        for generator in self.generators:
            generator.sample()

        # 3. Score proposals with constraints
        self.score_energy()

        # 4. Update the sorted results list and classify outcomes
        for proposal_idx in range(self.samples_per_round):
            if self._proposal_outcomes[proposal_idx] != "accepted":
                continue
            energy = self.energy_scores[proposal_idx]

            if len(self._result_energies) < self.num_results:
                pos = bisect.bisect_left(self._result_energies, energy)
                self._insert_into_results(pos, proposal_idx, energy)
            elif energy < self._result_energies[-1]:
                self._remove_worst_result()
                pos = bisect.bisect_left(self._result_energies, energy)
                self._insert_into_results(pos, proposal_idx, energy)
            else:
                self._proposal_outcomes[proposal_idx] = "Not in results"

        # 5. Save a progress snapshot and log from the current sorted state
        if save_snapshot:
            self._save_round_snapshot(round_num)

    def _capture_initial_state(self) -> None:
        """Capture state and clear optimizer-specific state for fresh run."""
        super()._capture_initial_state()
        self._result_energies = []
        self.energy_scores = []
        # Builds result_sequences dynamically via sorted insertion
        for segment in self.segments:
            segment.result_sequences = []

    def _restore_initial_state(self) -> None:
        """Restore to captured state and reset optimizer-specific state."""
        super()._restore_initial_state()
        self._result_energies = []
        self.energy_scores = []
        # Builds result_sequences dynamically via sorted insertion
        for segment in self.segments:
            segment.result_sequences = []

    def run(self) -> None:
        """Execute Rejection Sampling optimization through multiple sampling rounds.

        The mode is determined by whether ``energy_threshold`` is set:
        - **Standard mode** (no threshold): Generate ``num_samples`` proposals.
        - **Threshold mode** (threshold set): Stop early when threshold is met,
          or when ``num_samples`` is reached.

        Each round:
        - Resets all proposal_sequences to original_sequence
        - Runs each generator sequentially across segments (generators batch across proposals)
        - Evaluates all proposals with constraints
        - Updates the best results in result_sequences (in-place)
        """
        self._prepare_run()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing
        assert self._initial_state is not None  # noqa: S101 -- mypy type narrowing

        # Deferred validation: num_results vs num_samples (num_results may have been set via Program)
        if self.num_results > self.num_samples:
            raise ValueError(
                f"num_results ({self.num_results}) cannot exceed num_samples ({self.num_samples}). "
                "Cannot keep more sequences than generated."
            )

        # Starts empty (builds results dynamically); no initial snapshot

        proposals_generated = 0
        threshold_met = False
        threshold_mode = self.energy_threshold is not None
        num_sampling_rounds = self.num_samples // self.samples_per_round

        for round_num in range(1, num_sampling_rounds + 1):
            save = round_num % self.tracking_interval == 0 or round_num == num_sampling_rounds
            self._run_sampling_round(round_num, save_snapshot=save)
            proposals_generated += self.samples_per_round

            # Threshold mode: stop early when all results are below threshold
            if (
                threshold_mode
                and len(self._result_energies) == self.num_results
                and self.energy_threshold is not None
                and self._result_energies[-1] < self.energy_threshold
            ):
                threshold_met = True
                if self.verbose:
                    logger.info(
                        f"Threshold met! Worst in top-{self.num_results}: {self._result_energies[-1]:.6f} < {self.energy_threshold:.6f}"
                    )
                # Force a final snapshot if this round wasn't already saved
                if not save:
                    self._save_round_snapshot(round_num)
                break

        if not threshold_met and len(self._result_energies) < self.num_results:
            logger.warning(
                f"Rejection Sampling optimizer completed with only {len(self._result_energies)}/{self.num_results} valid proposals. Filter constraints may be too restrictive or num_samples may not be high enough."
            )

        # Handoff: set energy_scores to the sorted result energies.
        # May be fewer than k if filter constraints rejected too many proposals.
        self.energy_scores = list(self._result_energies)

        # Log statistics
        self._log_optimization_summary(threshold_mode, threshold_met, proposals_generated)

    def _save_round_snapshot(self, round_num: int) -> None:
        """Save a progress snapshot using the sorted result energies."""
        saved_energy_scores = self.energy_scores
        self.energy_scores = list(self._result_energies)
        self._save_progress_snapshot(
            time_step=round_num,
            optimizer_metadata={
                "type": "rejection-sampling",
                "num_rounds": self.num_steps,
                "num_samples": self.num_samples,
                "samples_per_round": self.samples_per_round,
                "num_results": self.num_results,
                "result_count": len(self._result_energies),
                "energy_threshold": self.energy_threshold,
                "proposal_count": len(self._proposal_outcomes),
                "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
            },
        )
        self.energy_scores = saved_energy_scores
        self._log_round_progress(round_num)

    def _log_round_progress(self, round_num: int) -> None:
        """Log round progress."""
        if self.verbose:
            num_results = len(self._result_energies)
            if num_results > 0:
                best_energy = self._result_energies[0]
                worst_energy = self._result_energies[-1]
                total_rounds = self.num_samples // self.samples_per_round
                progress_pct = (round_num / total_rounds) * 100
                logger.info(
                    f"Round {round_num}/{total_rounds} ({progress_pct:.0f}%): {num_results}/{self.num_results} in results, best={best_energy:.4f}, worst={worst_energy:.4f}"
                )

        if self.custom_logging:
            self.custom_logging(round_num, self.segments)

    def _log_optimization_summary(self, threshold_mode: bool, threshold_met: bool, proposals_generated: int) -> None:
        """Log optimization statistics and results."""
        if not self.verbose:
            return
        mode_str = "threshold" if threshold_mode else "standard"
        logger.debug(f"Optimization complete ({mode_str} mode):")
        logger.debug(f"  Total samples generated: {proposals_generated}")
        logger.debug(f"  Proposals per round: {self.samples_per_round}")
        logger.debug(f"  Results kept: {self.num_results}")

        if threshold_mode:
            logger.debug("Threshold mode:")
            logger.debug(f"  Target threshold: {self.energy_threshold:.6f}")
            logger.debug(f"  Num samples (max): {self.num_samples}")
            if threshold_met:
                logger.debug("  Status: Threshold met (early stop)")
            else:
                logger.debug("  Status: Num samples reached without meeting threshold")

        if self.energy_scores:
            best_energy = self.energy_scores[0]
            worst_in_results = self.energy_scores[-1]

            logger.debug(f"Top-{self.num_results} statistics:")
            logger.debug(f"  Best energy:  {best_energy:.6f}")
            if len(self.energy_scores) > 1:
                logger.debug(f"  Worst in results: {worst_in_results:.6f}")

            if self.num_results is not None and self.num_results <= 20:
                logger.debug(f"Top-{self.num_results} constructs:")
                for i, energy in enumerate(self.energy_scores):
                    logger.debug(f"  Rank {i + 1}: Energy={energy:.6f}")
            logger.debug(
                f"Rejection Sampling optimization complete. Returned {len(self.energy_scores)} best constructs."
            )
