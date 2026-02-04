"""
TopK Optimizer that runs multiple independent sampling rounds and returns the top-k best constructs.
"""
from __future__ import annotations

import copy
import heapq
import logging
import math
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
    ``k`` by lowest energy score. It samples in batches for efficiency and maintains
    a heap to track the top candidates.

    Attributes:
        num_samples (int): Maximum number of samples to generate. Rounded up to the
            nearest multiple of ``batch_size``. Must be at least ``k``.

        k (int): Number of top sequences to keep and return (lowest energy scores).
            Must be at least 1.

        batch_size (int): Number of samples to generate per round. Higher values
            enable more efficient batched inference. Default: ``1``.

        energy_threshold (Optional[float]): If set, enables early stopping. The
            optimizer stops before reaching ``num_samples`` if all ``k`` best
            candidates have energy below this threshold. Default: ``None`` (no
            early stopping).

        verbose (bool): Print progress information. Default: ``False``.

    Note:
        If filter constraints reject many candidates (returning inf/nan energies),
        the optimizer may return fewer than ``k`` valid results. The remaining
        slots are filled with empty placeholder sequences and inf energy scores.

    """
    # Required parameters
    num_samples: int = ConfigField(
        ge=1,
        title="Num Samples",
        description="Number of samples to generate. Rounded up to nearest batch_size multiple.",
    )
    k: int = ConfigField(
        ge=1,
        title="Top-k",
        description="Number of top samples to keep and return.",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of samples to generate per round (enables batching for generators).",
    )

    # Threshold mode parameter (presence determines mode)
    energy_threshold: Optional[float] = ConfigField(
        default=None,
        ge=0.0,
        title="Energy Threshold",
        description="Early stop when all energy scores in top-k are below threshold.",
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
        # k must not exceed num_samples
        if self.k > self.num_samples:
            raise ValueError(f"k ({self.k}) cannot exceed num_samples ({self.num_samples}). Cannot keep more sequences than generated.")
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
    4. Updates the top-k heap if any candidates are better than the current worst

    If ``energy_threshold`` is set, the optimizer stops early once all ``k`` best
    candidates have energy below the threshold.

    Attributes:
        num_samples (int): Maximum samples to generate (rounded up to batch_size multiple).
        k (int): Number of top sequences to keep.
        batch_size (int): Samples per round.
        energy_threshold (Optional[float]): Early stopping threshold.

    Note:
        If filter constraints reject many candidates, the optimizer may have fewer
        than ``k`` valid results. Remaining slots are padded with empty sequences
        and inf energy scores.

    Example:
        >>> config = TopKOptimizerConfig(num_samples=100, k=10, batch_size=10)
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
        ...     k=10,
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
            ValueError: If any validation checks fail.
        """
        # Map TopK variables to base Optimizer:
        # - batch_size → num_candidates (candidate pool size per round)
        # - k → num_selected (top-k to keep in results)
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_candidates=config.batch_size,
            num_selected=config.k,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
        )

        # Store parameters
        self.k: int = config.k
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
        # Max-heap for tracking top-k: stores (-energy, heap_idx) tuples
        # heap_idx corresponds to position in selected_sequences
        self._energy_heap: List[tuple] = []

    def _run_sampling_round(self, round_idx: int) -> None:
        """
        Execute a single sampling round.

        Args:
            round_idx: The index of the current round (for tracking purposes).
        """
        # 1. Create fresh candidate sequences at the start of each round (clean metadata state)
        # Each candidate uses its own initial sequence from the handoff (preserves diversity)
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

        # 2. Sample each generator in sequence (they see all batch_size candidates)
        for generator in self.generators:
            generator.sample()

        # 3. Evaluate all candidates after all generators
        self.score_energy()

        # 4. Process each candidate in the batch - store directly in selected_sequences
        for candidate_idx in range(self.batch_size):
            energy = self.energy_scores[candidate_idx]

            # Skip inf/nan energies
            if math.isinf(energy) or math.isnan(energy):
                continue

            # 5. Maintain top-k in selected_sequences (in-place) with max-heap
            # selected_sequences pool < k (not full yet) - append and push to heap
            if len(self._energy_heap) < self.k:
                heap_idx = len(self._energy_heap)
                heapq.heappush(self._energy_heap, (-energy, heap_idx))
                for segment in self.segments:
                    segment.selected_sequences.append(copy.deepcopy(segment.candidate_sequences[candidate_idx]))

            # Better than worst in top-k (selected_sequences pool) - replace worst
            elif energy < -self._energy_heap[0][0]:
                # Pop worst entry and reuse its slot index for the new better sequence
                # heap_idx is a stable identifier for a position in selected_sequences, not heap position
                _, worst_heap_idx = heapq.heappop(self._energy_heap)
                heapq.heappush(self._energy_heap, (-energy, worst_heap_idx))
                for segment in self.segments:
                    segment.selected_sequences[worst_heap_idx] = copy.deepcopy(segment.candidate_sequences[candidate_idx])

        # Sort for logging (both default and custom)
        self._log_round_progress(round_idx)

    def _capture_initial_state(self) -> None:
        """Capture state and clear TopK-specific state for fresh run."""
        super()._capture_initial_state()
        self._energy_heap = []

    def _restore_initial_state(self) -> None:
        """Restore to captured state and reset TopK-specific state."""
        super()._restore_initial_state()
        self._energy_heap = []

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

        # Clear selected_sequences since TopK populates this dynamically during optimization
        for segment in self.segments:
            segment.selected_sequences = []

        candidates_generated = 0
        threshold_met = False
        num_sampling_rounds = self.num_samples // self.batch_size

        # Determine mode based on energy_threshold
        threshold_mode = self.energy_threshold is not None

        if threshold_mode:
            # Threshold mode: Generate until threshold met or num_samples reached
            for round_idx in range(num_sampling_rounds):
                # Check if threshold is met (only after we have k candidates)
                if len(self._energy_heap) == self.k:
                    worst_energy = -self._energy_heap[0][0]  # Un-negate from heap
                    if worst_energy < self.energy_threshold:
                        threshold_met = True
                        if self.verbose:
                            logger.info(f"Threshold met! Worst in top-{self.k}: {worst_energy:.6f} < {self.energy_threshold:.6f}")
                        break

                self._run_sampling_round(round_idx)
                candidates_generated += self.batch_size

            # Log warning if heap never filled to k (filter constraints too strict)
            if not threshold_met and len(self._energy_heap) < self.k:
                logger.warning(f"TopK optimizer completed with only {len(self._energy_heap)}/{self.k} valid candidates. Filter constraints may be too restrictive. Results will be padded with inf energies.")
        else:
            # Standard mode: Generate exactly num_samples
            for round_idx in range(num_sampling_rounds):
                self._run_sampling_round(round_idx)
                candidates_generated += self.batch_size

        # Sort selected_sequences and energy_scores by energy (best first)
        self._sort_topk_by_energy()

        # Save single final timepoint with top-k results
        self._save_progress_snapshot(time_step=0)

        # Log statistics
        if self.verbose:
            self._log_optimization_summary(threshold_mode, threshold_met, candidates_generated)

    def _sort_topk_by_energy(self) -> None:
        """Sort selected_sequences and energy_scores by energy (best first).

        If fewer than k valid candidates were found, pads with inf energies and
        empty placeholder sequences to ensure len(energy_scores) == k and
        len(selected_sequences) == k.
        """
        if self._energy_heap:
            idx_to_energy = {idx: -neg_energy for neg_energy, idx in self._energy_heap}
            sorted_indices = sorted(idx_to_energy.keys(), key=lambda i: idx_to_energy[i])
            self.energy_scores = [idx_to_energy[i] for i in sorted_indices]
            for segment in self.segments:
                segment.selected_sequences = [segment.selected_sequences[i] for i in sorted_indices]
        else:
            self.energy_scores = []

        # Pad to k if heap was underfilled or empty (candidates had inf/nan energies)
        if len(self.energy_scores) < self.k:
            num_to_pad = self.k - len(self.energy_scores)
            self.energy_scores.extend([float('inf')] * num_to_pad)
            for segment in self.segments:
                for _ in range(num_to_pad):
                    # Use empty placeholder sequence to clearly indicate invalid result
                    segment.selected_sequences.append(
                        Sequence(
                            sequence="",
                            sequence_type=segment.sequence_type,
                            valid_chars=segment.valid_chars
                        )
                    )

    def _log_round_progress(self, round_idx: int) -> None:
        """Log round progress."""
        if self.verbose:
            num_selected = len(self._energy_heap)
            if num_selected > 0:
                energies = [-neg_energy for neg_energy, _ in self._energy_heap]
                best_energy = min(energies)
                worst_energy = -self._energy_heap[0][0]  # Max-heap root is worst
                # Show round progress relative to total
                total_rounds = self.num_samples // self.batch_size
                progress_pct = ((round_idx + 1) / total_rounds) * 100
                logger.info(f"Round {round_idx+1}/{total_rounds} ({progress_pct:.0f}%): {num_selected}/{self.k} in top-k, best={best_energy:.4f}, worst={worst_energy:.4f}")

        if self.custom_logging:
            self._sort_topk_by_energy()
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
        logger.debug(f"  Batch size: {self.batch_size}")
        logger.debug(f"  Top-k kept: {self.k}")

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

            logger.debug(f"Top-{self.k} statistics:")
            logger.debug(f"  Best energy:  {best_energy:.6f}")
            if len(self.energy_scores) > 1:
                logger.debug(f"  Worst in top-k: {worst_in_topk:.6f}")

            if self.k <= 20:
                logger.debug(f"Top-{self.k} constructs:")
                for i, energy in enumerate(self.energy_scores):
                    logger.debug(f"  Rank {i+1}: Energy={energy:.6f}")
            logger.debug(f"TopK optimization complete. Returned {len(self.energy_scores)} best constructs.")
