"""
TopK Optimizer that runs multiple independent sampling rounds and returns the top-k best constructs.
"""
from __future__ import annotations

import bisect
import copy
import logging
from typing import Callable, List, Optional, final

from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Sequence,
)
from proto_language.language.optimizer.optimizer_registry import optimizer

logger = logging.getLogger(__name__)


class TopKOptimizerConfig(BaseConfig):
    """Configuration object for TopKOptimizer.

    The TopK optimizer generates many candidate sequences and keeps only the best
    ``num_results`` by lowest energy score. It samples in batches for efficiency and
    maintains a sorted list to track the top candidates.

    Attributes:
        num_samples (int): Maximum number of samples to generate. Rounded up to the
            nearest multiple of ``batch_size``. Must be at least ``num_results``.

        num_results (Optional[int]): Number of top sequences to keep and return (lowest
            energy scores). Overrides program-level ``num_results`` if set.

        batch_size (int): Number of candidate sequences to generate per
            sampling round. Higher values enable more efficient batched inference.
            Default: ``1``.

        energy_threshold (Optional[float]): If set, enables early stopping. The
            optimizer stops before reaching ``num_samples`` if all top-``num_results``
            best candidates have energy below this threshold. Default: ``None``
            (no early stopping).

        verbose (bool): Print progress information. Default: ``False``.

    Note:
        If filter constraints reject many candidates (returning inf/nan energies),
        the optimizer may return fewer than ``num_results`` valid results.

    """
    # Required parameters
    num_samples: int = ConfigField(
        ge=1,
        title="Num Samples",
        description="Number of samples to generate. Rounded up to nearest batch_size multiple.",
    )

    # Advanced parameters
    num_results: Optional[int] = ConfigField(
        default=None,
        ge=1,
        title="Num Results",
        description="Number of top sequences to keep and return (top-K value). Overrides program Num Results.",
        advanced=True,
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of candidate sequences to generate per sampling round.",
        advanced=True,
    )
    energy_threshold: Optional[float] = ConfigField(
        default=None,
        ge=0.0,
        title="Energy Threshold",
        description="Early stop when all energy scores in top-k are below threshold.",
        advanced=True,
    )

    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
        hidden=True,
    )

    @model_validator(mode='after')
    def validate_params(self):
        """Validate parameter relationships."""
        # num_results must not exceed num_samples (only validate when num_results is set)
        if self.num_results is not None and self.num_results > self.num_samples:
            raise ValueError(f"num_results ({self.num_results}) cannot exceed num_samples ({self.num_samples}). Cannot keep more sequences than generated.")
        return self


@optimizer(
    key="topk",
    label="TopK Optimizer",
    config=TopKOptimizerConfig,
    description="Greedy optimizer that runs sampling rounds and maintains the top-k best constructs",
)
@final
class TopKOptimizer(Optimizer):
    """TopK optimizer for sequence optimization through extensive sampling.

    Generates many candidate sequences and keeps only the top ``k`` by lowest
    energy score. Unlike iterative optimizers (MCMC, beam search), each sampling
    round starts fresh from the original sequences—there is no state carried
    between rounds.

    Each round:
    1. Resets candidates to the original sequence
    2. Applies all generators sequentially
    3. Evaluates candidates with constraints
    4. Updates the sorted top-k list if any candidates are better than the current worst

    If ``energy_threshold`` is set, the optimizer stops early once all top-k best
    candidates have energy below the threshold.

    Attributes:
        num_samples (int): Maximum samples to generate (rounded up to batch_size multiple).
        num_results (int): Number of top sequences to keep (k).
        batch_size (int): Samples per round.
        energy_threshold (Optional[float]): Early stopping threshold.

    Note:
        If filter constraints reject many candidates, the optimizer may return
        fewer than ``num_results`` valid results.

    Example:
        >>> config = TopKOptimizerConfig(num_samples=100, num_results=10, batch_size=10)
        >>> optimizer = TopKOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],
        ...     constraints=[gc_constraint],
        ...     config=config
        ... )
        >>> optimizer.run()
        >>> best_sequences = optimizer.constructs[0].segments[0].selected_sequences

        With early stopping:

        >>> config = TopKOptimizerConfig(
        ...     num_samples=1000,
        ...     num_results=10,
        ...     batch_size=10,
        ...     energy_threshold=0.5  # Stop when all top-10 have energy < 0.5
        ... )
    """
    # Class attribute required by OptimizerRegistry
    config_class = TopKOptimizerConfig

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: TopKOptimizerConfig,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
    ) -> None:
        """
        Initialize the TopK Optimizer.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            config: Configuration object containing algorithm parameters.
            custom_logging: Optional custom logging function called after each round.
            clear_tool_cache: (int) Maximum size of cache in bytes, defaults to 100 MB.
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
            num_candidates=config.batch_size,
            num_results=config.num_results,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
        )
        self.batch_size: int = config.batch_size
        self.energy_threshold: Optional[float] = config.energy_threshold

        # Round up num_samples to nearest batch_size multiple
        if config.num_samples % config.batch_size != 0:
            self.num_samples = ((config.num_samples // config.batch_size) + 1) * config.batch_size
            logger.warning(
                f"num_samples ({config.num_samples}) is not divisible by batch_size ({config.batch_size}). "
                f"Rounding up to {self.num_samples}."
            )
        else:
            self.num_samples = config.num_samples
        # Sorted list of energies for selected_sequences (ascending order,
        # parallel to selected_sequences — index i matches segment.selected_sequences[i])
        self._selected_energies: list[float] = []

    def _insert_into_topk(self, pos: int, candidate_idx: int, energy: float) -> None:
        """Insert a candidate into the sorted top-k at the given position."""
        self._selected_energies.insert(pos, energy)
        for segment in self.segments:
            segment.selected_sequences.insert(
                pos, copy.deepcopy(segment.candidate_sequences[candidate_idx])
            )

    def _remove_worst_from_topk(self) -> None:
        """Remove the worst (last) entry from the sorted top-k."""
        self._selected_energies.pop()
        for segment in self.segments:
            segment.selected_sequences.pop()

    def _run_sampling_round(self, round_idx: int) -> None:
        """Execute a single sampling round.

        1. Reset candidate sequences to their initial state (fresh each round).
        2. Run all generators sequentially on the candidates.
        3. Score candidates with constraints (sets ``_candidate_outcomes``).
        4. Update the sorted top-k list and classify outcomes.
        5. Save a progress snapshot from the current sorted state.

        Args:
            round_idx: The index of the current round (for tracking purposes).
        """
        # 1. Reset candidate sequences to their initial state
        for seg_idx, segment in enumerate(self.segments):
            candidates = self._initial_state['segments'][seg_idx]['candidates']
            segment.candidate_sequences = [
                Sequence(
                    sequence=candidates[i]['sequence'],
                    sequence_type=segment.sequence_type,
                    valid_chars=segment.valid_chars
                )
                for i in range(self.num_candidates)
            ]

        # 2. Run all generators sequentially
        for generator in self.generators:
            generator.sample()

        # 3. Score candidates with constraints
        self.score_energy()

        # 4. Update the sorted top-k list and classify outcomes
        for candidate_idx in range(self.batch_size):
            if self._candidate_outcomes[candidate_idx] != "accepted":
                continue
            energy = self.energy_scores[candidate_idx]

            if len(self._selected_energies) < self.num_results:
                pos = bisect.bisect_left(self._selected_energies, energy)
                self._insert_into_topk(pos, candidate_idx, energy)
            elif energy < self._selected_energies[-1]:
                self._remove_worst_from_topk()
                pos = bisect.bisect_left(self._selected_energies, energy)
                self._insert_into_topk(pos, candidate_idx, energy)
            else:
                self._candidate_outcomes[candidate_idx] = "Not in top-k"

        # 5. Save a progress snapshot from the current sorted state
        saved_energy_scores = self.energy_scores
        self.energy_scores = list(self._selected_energies)
        self._save_progress_snapshot(time_step=round_idx + 1)
        self.energy_scores = saved_energy_scores

        self._log_round_progress(round_idx)

    def _capture_initial_state(self) -> None:
        """Capture state and clear TopK-specific state for fresh run."""
        super()._capture_initial_state()
        self._selected_energies = []
        self.energy_scores = []
        # TopK builds selected_sequences dynamically via sorted insertion
        for segment in self.segments:
            segment.selected_sequences = []

    def _restore_initial_state(self) -> None:
        """Restore to captured state and reset TopK-specific state."""
        super()._restore_initial_state()
        self._selected_energies = []
        self.energy_scores = []
        # TopK builds selected_sequences dynamically via sorted insertion
        for segment in self.segments:
            segment.selected_sequences = []

    def run(self) -> None:
        """
        Execute TopK optimization through multiple sampling rounds.

        The mode is determined by whether ``energy_threshold`` is set:
        - **Standard mode** (no threshold): Generate ``num_samples`` candidates.
        - **Threshold mode** (threshold set): Stop early when threshold is met,
          or when ``num_samples`` is reached.

        Each round:
        - Resets all candidate_sequences to original_sequence
        - Runs each generator sequentially across segments (generators batch across candidates)
        - Evaluates all candidates with constraints
        - Updates the top-k in selected_sequences (in-place)
        """
        self._prepare_run()

        # Deferred validation: num_results vs num_samples (num_results may have been set via Program)
        if self.num_results > self.num_samples:
            raise ValueError(
                f"num_results ({self.num_results}) cannot exceed num_samples ({self.num_samples}). "
                "Cannot keep more sequences than generated."
            )

        # t=0 initial snapshot (empty state)
        self._save_progress_snapshot(time_step=0)

        candidates_generated = 0
        threshold_met = False
        num_sampling_rounds = self.num_samples // self.batch_size

        # Determine mode based on energy_threshold
        threshold_mode = self.energy_threshold is not None

        if threshold_mode:
            # Threshold mode: Generate until threshold met or num_samples reached
            for round_idx in range(num_sampling_rounds):
                # Check if threshold is met (only after we have k candidates)
                if len(self._selected_energies) == self.num_results:
                    worst_energy = self._selected_energies[-1]
                    if worst_energy < self.energy_threshold:
                        threshold_met = True
                        if self.verbose:
                            logger.info(f"Threshold met! Worst in top-{self.num_results}: {worst_energy:.6f} < {self.energy_threshold:.6f}")
                        break

                self._run_sampling_round(round_idx)
                candidates_generated += self.batch_size

            # Log warning if list never filled to k (filter constraints too strict)
            if not threshold_met and len(self._selected_energies) < self.num_results:
                logger.warning(f"TopK optimizer completed with only {len(self._selected_energies)}/{self.num_results} valid candidates. Filter constraints may be too restrictive or num_samples may not be high enough.")
        else:
            # Standard mode: Generate exactly num_samples
            for round_idx in range(num_sampling_rounds):
                self._run_sampling_round(round_idx)
                candidates_generated += self.batch_size

            if len(self._selected_energies) < self.num_results:
                logger.warning(f"TopK optimizer completed with only {len(self._selected_energies)}/{self.num_results} valid candidates. Filter constraints may be too restrictive or num_samples may not be high enough.")

        # Handoff: set energy_scores to the sorted selected energies.
        # May be fewer than k if filter constraints rejected too many candidates.
        self.energy_scores = list(self._selected_energies)

        # Log statistics
        if self.verbose:
            self._log_optimization_summary(threshold_mode, threshold_met, candidates_generated)

    def _log_round_progress(self, round_idx: int) -> None:
        """Log round progress."""
        if self.verbose:
            num_results = len(self._selected_energies)
            if num_results > 0:
                best_energy = self._selected_energies[0]
                worst_energy = self._selected_energies[-1]
                total_rounds = self.num_samples // self.batch_size
                progress_pct = ((round_idx + 1) / total_rounds) * 100
                logger.info(f"Round {round_idx+1}/{total_rounds} ({progress_pct:.0f}%): {num_results}/{self.num_results} in top-k, best={best_energy:.4f}, worst={worst_energy:.4f}")

        if self.custom_logging:
            self.custom_logging(round_idx, self.segments)

    def _log_optimization_summary(
        self,
        threshold_mode: bool,
        threshold_met: bool,
        candidates_generated: int
    ) -> None:
        """Log optimization statistics and results."""
        mode_str = "threshold" if threshold_mode else "standard"
        logger.debug(f"Optimization complete ({mode_str} mode):")
        logger.debug(f"  Total samples generated: {candidates_generated}")
        logger.debug(f"  Candidates per round: {self.batch_size}")
        logger.debug(f"  Top-k kept: {self.num_results}")

        if threshold_mode:
            logger.debug(f"Threshold mode:")
            logger.debug(f"  Target threshold: {self.energy_threshold:.6f}")
            logger.debug(f"  Num samples (max): {self.num_samples}")
            if threshold_met:
                logger.debug("  Status: Threshold met (early stop)")
            else:
                logger.debug("  Status: Num samples reached without meeting threshold")

        if self.energy_scores:
            best_energy = self.energy_scores[0]
            worst_in_topk = self.energy_scores[-1]

            logger.debug(f"Top-{self.num_results} statistics:")
            logger.debug(f"  Best energy:  {best_energy:.6f}")
            if len(self.energy_scores) > 1:
                logger.debug(f"  Worst in top-k: {worst_in_topk:.6f}")

            if self.num_results <= 20:
                logger.debug(f"Top-{self.num_results} constructs:")
                for i, energy in enumerate(self.energy_scores):
                    logger.debug(f"  Rank {i+1}: Energy={energy:.6f}")
            logger.debug(f"TopK optimization complete. Returned {len(self.energy_scores)} best constructs.")
