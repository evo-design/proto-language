"""Rejection Sampling Optimizer that samples independent proposals and returns the best constructs."""

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
from proto_language.utils.export import build_proposal_results

logger = logging.getLogger(__name__)

# Passed threshold filters but did not enter the retained top-k at evaluation time.
DID_NOT_ENTER_TOP_K = "did_not_enter_top_k"


def _finite_number(value: Any) -> float | None:
    """Return value as finite float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _proposal_snapshot_score(proposal: dict[str, Any]) -> tuple[float | None, str]:
    """Choose the scalar score recorded for a proposal progress snapshot."""
    rejected_by = proposal.get("rejected_by")
    if proposal.get("accepted") or rejected_by == DID_NOT_ENTER_TOP_K:
        score = _finite_number(proposal.get("energy_score"))
        return (score, "energy_score") if score is not None else (None, "none")

    if rejected_by:
        for construct in proposal.get("constructs", []) or []:
            for segment in construct.get("segments", []) or []:
                constraints = segment.get("constraints", {}) or {}
                constraint = constraints.get(rejected_by)
                if isinstance(constraint, dict):
                    score = _finite_number(constraint.get("score"))
                    if score is not None:
                        return score, "failed_filter_score"
    return None, "none"


def _proposal_filter_metadata(proposal: dict[str, Any]) -> tuple[str, str | None]:
    """Return whether the proposal passed threshold filters and the failing filter if any."""
    rejected_by = proposal.get("rejected_by")
    if rejected_by and rejected_by != DID_NOT_ENTER_TOP_K:
        return "failed", str(rejected_by)
    return "passed", None


class RejectionSamplingOptimizerConfig(BaseOptimizerConfig):
    """Configuration object for RejectionSamplingOptimizer.

    The Rejection Sampling optimizer generates many proposal sequences and keeps
    only the best ``num_results`` by lowest energy score. It processes proposals
    in internal batches and reports each proposal as the semantic iteration.

    Attributes:
        num_samples (int): Maximum number of samples to generate. Must be at
            least ``num_results``.

        num_results (int | None): Number of top sequences to keep and return (lowest
            energy scores). Overrides program-level ``num_results`` if set.

        proposal_batch_size (int | None): Number of proposal sequences to
            generate and evaluate per internal batch. If ``None``, inferred
            from the largest positive ``batch_size`` on generators and
            constraints, capped at ``num_samples``.

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
    )
    proposal_batch_size: int | None = ConfigField(
        default=None,
        ge=1,
        title="Proposal Batch Size",
        description="Proposals scored per internal batch. Inferred from component batch sizes if omitted.",
    )
    energy_threshold: float | None = ConfigField(
        default=None,
        ge=0.0,
        title="Energy Threshold",
        description="Early stop when all energy scores in results are below threshold.",
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
    proposal batch starts fresh from the original sequences. There is no state
    carried between rounds.

    Each proposal batch:
    1. Resets proposals to the original sequence
    2. Applies all generators sequentially
    3. Evaluates proposals with constraints
    4. Updates the sorted results list if any proposals are better than the current worst
    5. Reports each proposal as its own history iteration

    If ``energy_threshold`` is set, the optimizer stops early once all best
    proposals have energy below the threshold.

    Attributes:
        num_samples: Maximum samples to generate.
        num_results: Number of top sequences to keep.
        proposal_batch_size: Proposals generated and evaluated per internal batch.
        energy_threshold: Early stopping threshold.

    Note:
        If filter constraints reject many proposals, the optimizer may return
        fewer than ``num_results`` valid results.

    Example:
        >>> config = RejectionSamplingOptimizerConfig(num_samples=100, num_results=10)
        >>> optimizer = RejectionSamplingOptimizer(
        ...     constructs=constructs, generators=[mutation_gen], constraints=[gc_constraint], config=config
        ... )
        >>> optimizer.run()
        >>> best_sequences = optimizer.constructs[0].segments[0].result_sequences

        With early stopping:

        >>> config = RejectionSamplingOptimizerConfig(
        ...     num_samples=1000,
        ...     num_results=10,
        ...     energy_threshold=0.5,  # Stop when all top-10 have energy < 0.5
        ... )
    """

    # Class attribute required by OptimizerRegistry
    config_class = RejectionSamplingOptimizerConfig
    config: RejectionSamplingOptimizerConfig

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
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked proposals (governed by ``tracking_interval``).
            clear_tool_cache (int | bool | list[str]): (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If any validation checks fail or num_results cannot be determined.
        """
        self.config = config
        proposal_batch_size = self._resolve_proposal_batch_size(
            generators=generators,
            constraints=constraints,
            num_samples=config.num_samples,
            configured=config.proposal_batch_size,
        )
        self.config.proposal_batch_size = proposal_batch_size

        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_proposals=proposal_batch_size,
            num_results=config.num_results,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
            seed=config.seed,
        )
        self.proposal_batch_size: int = proposal_batch_size
        self.energy_threshold: float | None = config.energy_threshold

        self.num_samples: int = config.num_samples

        # Override base class num_steps for progress tracking
        self.num_steps = self.num_samples
        self._last_saved_proposal_number: int | None = None
        # Sorted list of energies for result_sequences (ascending order,
        # parallel to result_sequences; index i matches segment.result_sequences[i])
        self._result_energies: list[float] = []

    @staticmethod
    def _read_batch_size(value: Any) -> int | None:
        """Read a positive ``batch_size`` from an object or config dict."""
        value = value.get("batch_size") if isinstance(value, dict) else getattr(value, "batch_size", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
        return int(value)

    @classmethod
    def _resolve_proposal_batch_size(
        cls,
        *,
        generators: list[Generator],
        constraints: list[Constraint],
        num_samples: int,
        configured: int | None,
    ) -> int:
        """Resolve the internal proposal batch size."""
        if configured is not None:
            return min(configured, num_samples)

        discovered: list[int] = []
        for generator in generators:
            value = cls._read_batch_size(generator)
            if value is not None:
                discovered.append(value)

        for constraint in constraints:
            for config in (constraint.function_config, constraint.backward_config):
                value = cls._read_batch_size(config)
                if value is not None:
                    discovered.append(value)

        return min(max(discovered, default=1), num_samples)

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

    def _run_proposal_batch(self, batch_num: int, first_proposal_number: int, batch_size: int) -> int:
        """Execute a single proposal batch and return the last processed proposal number.

        1. Reset proposal sequences to their initial state (fresh each batch).
        2. Run all generators sequentially on the proposals.
        3. Score proposals with constraints (sets ``_proposal_outcomes``).
        4. Update the sorted results list and classify outcomes.
        5. Save one progress snapshot per tracked proposal.

        Args:
            batch_num (int): The 1-indexed internal batch number.
            first_proposal_number (int): The 1-indexed proposal number for batch index 0.
            batch_size (int): Number of proposals to generate in this batch.
        """
        assert self._initial_state is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        # 1. Reset proposal sequences to their initial state
        for seg_idx, segment in enumerate(self.segments):
            source = self._initial_state["segments"][seg_idx]["result"]
            segment.proposal_sequences = [
                Sequence.from_dict(source[(first_proposal_number - 1 + i) % len(source)]) for i in range(batch_size)
            ]

        # 2. Run all generators sequentially
        for generator in self.generators:
            generator.sample()

        # 3. Score proposals with constraints
        self.score_energy()

        # 4. Update the sorted results list and classify outcomes
        last_proposal_number = first_proposal_number - 1
        for proposal_idx in range(batch_size):
            proposal_number = first_proposal_number + proposal_idx
            if self._proposal_outcomes[proposal_idx] != "accepted":
                if self._should_save_proposal(proposal_number):
                    self._save_proposal_snapshot(proposal_number, batch_num, proposal_idx)
                last_proposal_number = proposal_number
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
                self._proposal_outcomes[proposal_idx] = DID_NOT_ENTER_TOP_K

            if self._should_save_proposal(proposal_number):
                self._save_proposal_snapshot(proposal_number, batch_num, proposal_idx)
            last_proposal_number = proposal_number

        return last_proposal_number

    def _capture_initial_state(self) -> None:
        """Capture state and clear optimizer-specific state for fresh run."""
        super()._capture_initial_state()
        self._result_energies = []
        self.energy_scores = []
        self._last_saved_proposal_number = None
        # Builds result_sequences dynamically via sorted insertion
        for segment in self.segments:
            segment.result_sequences = []

    def _restore_initial_state(self) -> None:
        """Restore to captured state and reset optimizer-specific state."""
        super()._restore_initial_state()
        self._result_energies = []
        self.energy_scores = []
        self._last_saved_proposal_number = None
        # Builds result_sequences dynamically via sorted insertion
        for segment in self.segments:
            segment.result_sequences = []

    def run(self) -> None:
        """Execute Rejection Sampling optimization through multiple sampling rounds.

        The mode is determined by whether ``energy_threshold`` is set:
        - **Standard mode** (no threshold): Generate ``num_samples`` proposals.
        - **Threshold mode** (threshold set): Stop early when threshold is met,
          or when ``num_samples`` is reached.

        Each proposal batch:
        - Resets all proposal_sequences to original_sequence
        - Runs each generator sequentially across segments (generators batch across proposals)
        - Evaluates all proposals with constraints
        - Updates the best results in result_sequences (in-place)
        """
        self._prepare_run()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self._initial_state is not None  # noqa: S101 -- mypy type narrowing

        # Deferred validation: num_results vs num_samples (num_results may have been set via Program)
        if self.num_results > self.num_samples:
            raise ValueError(
                f"num_results ({self.num_results}) cannot exceed num_samples ({self.num_samples}). "
                "Cannot keep more sequences than generated."
            )

        n_filter = sum(1 for c in self.constraints if c.threshold is not None)
        n_score = len(self.constraints) - n_filter
        mode_str = f"threshold={self.energy_threshold:.4f}" if self.energy_threshold is not None else "standard mode"
        logger.info(
            f"RejectionSamplingOptimizer: up to {self.num_samples} samples, "
            f"batch={self.proposal_batch_size}, {mode_str}, "
            f"{len(self.constraints)} constraints ({n_filter} filter, {n_score} scoring)"
        )

        # Starts empty (builds results dynamically); no initial snapshot

        proposals_generated = 0
        threshold_met = False
        threshold_mode = self.energy_threshold is not None

        batch_num = 1
        last_batch_num = 0
        while proposals_generated < self.num_samples:
            batch_size = min(self.proposal_batch_size, self.num_samples - proposals_generated)
            last_batch_num = batch_num
            last_proposal_number = self._run_proposal_batch(
                batch_num,
                proposals_generated + 1,
                batch_size,
            )
            proposals_generated = last_proposal_number

            # Threshold mode: stop early when all results are below threshold
            if (
                threshold_mode
                and len(self._result_energies) == self.num_results
                and self.energy_threshold is not None
                and self._result_energies[-1] < self.energy_threshold
            ):
                threshold_met = True
                logger.info(
                    f"Threshold met! Worst in top-{self.num_results}: "
                    f"{self._result_energies[-1]:.6f} < {self.energy_threshold:.6f}"
                )
                if self._last_saved_proposal_number != proposals_generated:
                    self._save_proposal_snapshot(proposals_generated, batch_num, batch_size - 1)
                break

            batch_num += 1

        if not threshold_met and self._last_saved_proposal_number != proposals_generated:
            self._save_proposal_snapshot(proposals_generated, last_batch_num, batch_size - 1)

        if not threshold_met and len(self._result_energies) < self.num_results:
            logger.warning(
                f"Rejection Sampling optimizer completed with only {len(self._result_energies)}/{self.num_results} valid proposals. Filter constraints may be too restrictive or num_samples may not be high enough."
            )

        # Handoff: set energy_scores to the sorted result energies.
        # May be fewer than k if filter constraints rejected too many proposals.
        self.energy_scores = list(self._result_energies)

        # Log statistics
        self._log_optimization_summary(threshold_mode, threshold_met, proposals_generated)

    def _should_save_proposal(self, proposal_number: int) -> bool:
        """Return whether the proposal should be stored as a history timepoint."""
        return proposal_number % self.tracking_interval == 0 or proposal_number == self.num_samples

    def _save_proposal_snapshot(self, proposal_number: int, batch_num: int, batch_proposal_idx: int) -> None:
        """Save a progress snapshot for one proposal sample.

        Proposal rows serialize one sampled candidate, not the sorted result state
        expected by the base progress snapshot helper.
        """
        proposals = build_proposal_results(self.constructs, self._proposal_outcomes, self._proposal_energy_scores)
        proposal = {**proposals[batch_proposal_idx], "proposal_idx": proposal_number - 1}
        score, score_source = _proposal_snapshot_score(proposal)
        filter_status, failed_filter = _proposal_filter_metadata(proposal)
        metadata: dict[str, Any] = {
            "type": "rejection-sampling",
            "iteration_kind": "proposal",
            "proposal_number": proposal_number,
            "proposal_idx": proposal_number - 1,
            "batch_number": batch_num,
            "batch_proposal_idx": batch_proposal_idx,
            "num_samples": self.num_samples,
            "proposal_batch_size": self.proposal_batch_size,
            "num_results": self.num_results,
            "result_count": len(self._result_energies),
            "energy_threshold": self.energy_threshold,
            "proposal_count": 1,
            "filter_status": filter_status,
            "failed_filter": failed_filter,
            "score_source": score_source,
        }
        result = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": score,
                    "constructs": proposal.get("constructs", []),
                }
            ],
            "best_result_idx": 0,
            "time_step": proposal_number,
            "optimizer": metadata,
        }
        if self.track_proposals:
            result["proposal_results"] = [proposal]
        self.history.append(result)
        self._last_saved_proposal_number = proposal_number
        self._log_proposal_progress(proposal_number)

    def _log_proposal_progress(self, proposal_number: int) -> None:
        """Log proposal progress as a multi-line INFO block."""
        progress_pct = (proposal_number / self.num_samples) * 100
        logger.info(f"Proposal {proposal_number}/{self.num_samples} ({progress_pct:.0f}%)")
        filter_summary = self._format_filter_summary()
        if filter_summary is not None:
            logger.info(f"  filters: {filter_summary}")
        for line in self._format_scoring_lines():
            logger.info(f"  {line}")
        if self._result_energies:
            best = self._result_energies[0]
            worst = self._result_energies[-1]
            logger.info(f"  energy:  best={best:.4f} worst={worst:.4f}")
        else:
            logger.info("  energy:  n/a (no accepted proposals)")
        logger.info(f"  results {len(self._result_energies)}/{self.num_results}")

        if self.custom_logging:
            self.custom_logging(proposal_number, self.segments)

    def _log_optimization_summary(self, threshold_mode: bool, threshold_met: bool, proposals_generated: int) -> None:
        """Log optimization statistics and results at DEBUG."""
        mode_str = "threshold" if threshold_mode else "standard"
        logger.debug(f"Optimization complete ({mode_str} mode):")
        logger.debug(f"  Total samples generated: {proposals_generated}")
        logger.debug(f"  Proposal batch size: {self.proposal_batch_size}")
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
