"""
TopK Optimizer that runs multiple independent sampling rounds and returns the top-k best constructs.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, final
import copy
import heapq
import logging
import math

from pydantic import model_validator

from proto_language.language.core import Optimizer, Construct, Generator, Constraint, Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry

logger = logging.getLogger(__name__)


class TopKOptimizerConfig(BaseConfig):
    """Configuration object for TopKOptimizer.

    This class defines configuration parameters for the TopK optimizer, which
    generates many candidate sequences and retains only the best K by lowest energy
    score.

    The optimizer runs in one of two modes based on whether ``energy_threshold`` is set:

    - **Standard mode** (``energy_threshold=None``): Generate exactly ``num_samples``
      candidates and keep the top ``k``.

    - **Threshold mode** (``energy_threshold`` set): Generate candidates until the
      worst energy in top-k is below ``energy_threshold``, or until ``num_samples``
      is reached (whichever comes first).

    Attributes:
        num_samples (int): Number of samples to generate. In standard mode, the optimizer 
            samples exactly this number of candidates. In threshold mode, the optimizer 
            samples until the energy threshold is met or the number of samples is reached.
            Must be at least ``k``. Will be rounded up to the nearest multiple of ``batch_size`` 
            if not evenly divisible.

        k (int): Number of top sequences to keep and return based on energy scores.
            The optimizer maintains a max-heap of size ``k`` to efficiently track
            the best candidates. Must be at least 1.

        batch_size (int): Number of samples to generate per round. Enables batching
            for efficient parallel generation with generators that support batched
            inference (e.g., language models). Must be at least 1. Default: ``1``.

        energy_threshold (Optional[float]): Target energy threshold for early stopping.
            When set, enables threshold mode where sampling stops when the worst
            (highest) energy in the top-k heap falls below this value. Must be at
            least 0 if set. Default: ``None`` (standard mode).

        verbose (bool): Whether to print detailed progress information including
            round statistics, energy values, and stopping conditions. Default: ``False``.

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


@OptimizerRegistry.register(
    key="topk",
    label="TopK Optimizer",
    config=TopKOptimizerConfig,
    description="Greedy optimizer that runs sampling rounds and maintains the top-k best constructs",
)
@final
class TopKOptimizer(Optimizer):
    """TopK optimizer for sequence optimization through extensive sampling.

    This optimizer generates many candidate sequences through multiple sampling
    rounds and maintains only the top-K sequences by energy score. Unlike iterative
    optimizers (MCMC, beam search), it does not maintain state between rounds—each
    round starts fresh from original sequences.

    In each round, the optimizer generates ``batch_size`` candidates, applies all
    generators sequentially to them, evaluates them with constraints, and updates
    the top-K list if any candidates are better than the current worst in top-K.

    The mode is determined by whether ``energy_threshold`` is set:

    - **Standard mode** (no threshold): Generate ``num_samples`` candidates.
    - **Threshold mode** (threshold set): Stop early when threshold is met.

    Attributes:
        num_samples (int): Number of samples (rounded up to batch_size multiple).
        k (int): Number of top sequences to keep.
        batch_size (int): Samples per round (enables batching).
        energy_threshold (Optional[float]): Target threshold (enables threshold mode).

    Example:
        Standard mode - generate 100 samples:

        >>> config = TopKOptimizerConfig(num_samples=100, k=10, batch_size=10)
        >>> optimizer = TopKOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],
        ...     constraints=[gc_constraint],
        ...     config=config
        ... )
        >>> optimizer.run()
        >>> best_constructs = optimizer.constructs  # Top 10 sequences

        Threshold mode - stop early when threshold met:

        >>> config = TopKOptimizerConfig(
        ...     num_samples=1000,
        ...     energy_threshold=0.5,
        ...     k=10,
        ...     batch_size=10
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
        # Initialize _initial_sequences state before super().__init__() which calls _initialize_sequence_pools
        # Used to reset candidates to initial states in each round.
        self._initial_sequence_states: Dict[int, str] = {}

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

    def _initialize_sequence_pools(self) -> None:
        """
        Initialize sequence pools for TopK optimizer. Overrides the base Optimizer method.

        Captures initial sequence state for resetting candidates each round. Initializes
        selected_sequences to empty - it will be populated during optimization via in-place
        updates to maintain the top-k sequences. Initializes candidate pool with batch_size copies.

        Note: Unlike the base optimizer, selected_sequences grows dynamically during optimization
        as better sequences are discovered, rather than being pre-allocated.
        """
        for seg_idx, segment in enumerate(self.segments):
            # Capture initial state (from previous optimizer or original)
            source = segment.selected_sequences[0] if segment.selected_sequences else segment.original_sequence
            self._initial_sequence_states[seg_idx] = source.sequence

            # Start with empty selected pool (will be populated during optimization)
            segment.selected_sequences = []

            # Initialize candidate pool with batch_size copies for sampling
            segment.candidate_sequences = [
                copy.deepcopy(segment.candidate_sequences[0])
                for _ in range(self.num_candidates)
            ]

    def _run_sampling_round(self, round_idx: int) -> None:
        """
        Execute a single sampling round.

        Args:
            round_idx: The index of the current round (for tracking purposes).
        """
        # 1. Create fresh candidate sequences at the start of each round (clean metadata state)
        for seg_idx, segment in enumerate(self.segments):
            segment.candidate_sequences = [
                Sequence(
                    sequence=self._initial_sequence_states[seg_idx],
                    sequence_type=segment.sequence_type,
                    valid_chars=segment._valid_chars
                )
                for _ in range(self.num_candidates)
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
        # Reset top-k tracking state for fresh run
        self._energy_heap = []
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
                            print(f"\nThreshold met! Worst in top-{self.k}: {worst_energy:.6f} < {self.energy_threshold:.6f}")
                        break

                self._run_sampling_round(round_idx)
                candidates_generated += self.batch_size
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
        """Sort selected_sequences and energy_scores by energy (best first)."""
        if self._energy_heap:
            idx_to_energy = {idx: -neg_energy for neg_energy, idx in self._energy_heap}
            sorted_indices = sorted(idx_to_energy.keys(), key=lambda i: idx_to_energy[i])
            self.energy_scores = [idx_to_energy[i] for i in sorted_indices]
            for segment in self.segments:
                segment.selected_sequences = [segment.selected_sequences[i] for i in sorted_indices]
        else:
            self.energy_scores = []

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
                print(f"  Round {round_idx+1}/{total_rounds} ({progress_pct:.0f}%): {num_selected}/{self.k} in top-k, best={best_energy:.4f}, worst={worst_energy:.4f}")

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
        print(f"\nOptimization complete ({mode_str} mode):")
        print(f"  Total samples generated: {candidates_generated}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Top-k kept: {self.k}")

        if threshold_mode:
            print(f"\nThreshold mode:")
            print(f"  Target threshold: {self.energy_threshold:.6f}")
            print(f"  Num samples (max): {self.num_samples}")
            if threshold_met:
                print("  Status: Threshold met (early stop)")
            else:
                print("  Status: Num samples reached without meeting threshold")

        if self.energy_scores:
            best_energy = self.energy_scores[0]
            worst_in_topk = self.energy_scores[-1]

            print(f"\nTop-{self.k} statistics:")
            print(f"  Best energy:  {best_energy:.6f}")
            if len(self.energy_scores) > 1:
                print(f"  Worst in top-k: {worst_in_topk:.6f}")

            if self.k <= 20:
                print(f"\nTop-{self.k} constructs:")
                for i, energy in enumerate(self.energy_scores):
                    print(f"  Rank {i+1}: Energy={energy:.6f}")
            print(f"\nTopK optimization complete. Returned {len(self.energy_scores)} best constructs.")

