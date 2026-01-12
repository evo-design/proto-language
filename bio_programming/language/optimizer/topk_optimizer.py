"""
TopK Optimizer that runs multiple independent sampling rounds and returns the top-k best constructs.
"""
from __future__ import annotations
from typing import Callable, List, Optional, final
import copy
import heapq
import logging
import math

import numpy as np
from pydantic import model_validator

from proto_language.language.core import Optimizer, Construct, Generator, Constraint
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
            verbose=config.verbose,
        )

        # Store parameters
        self.k: int = config.k
        self.batch_size: int = config.batch_size
        self.custom_logging: Optional[Callable] = custom_logging
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

        # Storage for top-k candidates using a max-heap of size k
        # We negate energies since heapq is a min-heap but we want max-heap behavior
        # This keeps the worst (highest) energy at the root for easy replacement
        self.top_k_heap: List[tuple] = []

    def _initialize_sequence_pools(self) -> None:
        """
        Initialize sequence pools for TopK optimizer.

        TopK starts with an empty selected pool and only populates candidates.
        Unlike MCMC which maintains N selected sequences throughout, TopK builds
        up the top-k list as optimization progresses through sampling rounds.

        The selected pool will be populated by set_topk_constructs() as rounds complete.
        """
        for segment in self.segments:
            # Start with empty selected pool (will be populated by set_topk_constructs)
            segment.selected_sequences = []

            # Initialize candidate pool with batch_size copies for sampling
            segment.candidate_sequences = [
                copy.deepcopy(segment.candidate_sequences[0])
                for _ in range(self.num_candidates)
            ]

    def _run_round(self, round_idx: int) -> None:
        """
        Execute a single sampling round.

        Args:
            round_idx: The index of the current round (for tracking purposes).
        """
        # 1. Reset all candidate sequences to original state at the start of each round
        for segment in self.segments:
            for candidate_seq in segment.candidate_sequences:
                candidate_seq.sequence = copy.deepcopy(segment.original_sequence.sequence)

        # 2. Sample each generator in sequence (they see all batch_size candidates)
        for generator in self.generators:
            generator.sample()

        # 3. Evaluate all candidates after all generators
        self.score_energy()

        # 4. Process each candidate in the batch
        for candidate_idx in range(self.batch_size):
            energy = self.energy_scores[candidate_idx]

            # Skip inf/nan energies
            if math.isinf(energy) or math.isnan(energy):
                continue

            # Save the resulting sequences from this candidate
            candidate_sequences = {
                seg_idx: copy.deepcopy(segment.candidate_sequences[candidate_idx])
                for seg_idx, segment in enumerate(self.segments)
            }

            # 5. Maintain a max-heap of size k to track the k smallest energies
            if len(self.top_k_heap) < self.k:
                # Haven't filled top-k yet, just add (negate energy for max-heap)
                heapq.heappush(self.top_k_heap, (-energy, round_idx, candidate_idx, candidate_sequences))
            elif energy < -self.top_k_heap[0][0]:
                # This energy is smaller (better) than the worst in our top-k
                # Replace the worst with this better one
                heapq.heapreplace(self.top_k_heap, (-energy, round_idx, candidate_idx, candidate_sequences))

        # 6. Sync selected_sequences with current top-k heap for custom logging
        if self.top_k_heap:
            top_k_sorted = sorted(self.top_k_heap, key=lambda x: -x[0])
            self.set_topk_constructs(top_k_sorted)

        # 7. Custom logging after selected_sequences are synced
        if self.custom_logging:
            self.custom_logging(round_idx, self.segments)

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
        - Updates the top-k sampled candidates
        """
        # Clear any previous top-k list
        self.top_k_heap = []
        candidates_generated = 0
        threshold_met = False
        num_sampling_rounds = self.num_samples // self.batch_size

        # Determine mode based on energy_threshold
        threshold_mode = self.energy_threshold is not None

        if threshold_mode:
            # Threshold mode: Generate until threshold met or num_samples reached
            for round_idx in range(num_sampling_rounds):
                # Check if threshold is met (only after we have k candidates)
                if len(self.top_k_heap) == self.k:
                    worst_energy = -self.top_k_heap[0][0]  # Un-negate to get actual energy
                    if worst_energy < self.energy_threshold:
                        threshold_met = True
                        if self.verbose:
                            print(f"\nThreshold met! Worst in top-{self.k}: {worst_energy:.6f} < {self.energy_threshold:.6f}")
                        break

                self._run_round(round_idx)
                candidates_generated += self.batch_size

                if self.verbose and (round_idx + 1) % 10 == 0:
                    worst_energy = -self.top_k_heap[0][0] if len(self.top_k_heap) == self.k else float('inf')
                    print(f"  Round {round_idx}: Generated {candidates_generated} candidates, worst in top-k: {worst_energy:.6f}")
        else:
            # Standard mode: Generate exactly num_samples
            for round_idx in range(num_sampling_rounds):
                self._run_round(round_idx)
                candidates_generated += self.batch_size

        # Convert heap to sorted list (best first: lowest energy to highest)
        top_k_list = sorted(self.top_k_heap, key=lambda x: -x[0])

        # Update constructs with top-k
        self.set_topk_constructs(top_k_list)

        # Save single final timepoint with top-k results
        self._save_progress_snapshot(time_step=0)

        # Log statistics
        if self.verbose:
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

            if top_k_list:
                actual_energies = [-e for e, _, _, _ in top_k_list]
                best_energy = actual_energies[0]
                worst_in_topk = actual_energies[-1]
                mean_energy = np.mean(actual_energies)

                print(f"\nTop-{self.k} statistics:")
                print(f"  Best energy:  {best_energy:.6f}")
                if len(top_k_list) > 1:
                    print(f"  Worst in top-k: {worst_in_topk:.6f}")
                print(f"  Mean energy:  {mean_energy:.6f}")

                if self.k <= 20:
                    print(f"\nTop-{self.k} constructs:")
                    for i, (neg_energy, _, _, _) in enumerate(top_k_list):
                        energy = -neg_energy
                        print(f"  Rank {i+1}: Energy={energy:.6f}")
                print(f"\nTopK optimization complete. Returned {len(top_k_list)} best constructs.")

    def set_topk_constructs(self, top_k_list: List[tuple]) -> None:
        """
        Set the top-k constructs to segments' selected_sequences pool.

        Args:
            top_k_list: List of (neg_energy, round_idx, candidate_idx, candidate_sequences) tuples.
        """
        # Initialize selected pool to empty lists for building
        for segment in self.segments:
            segment.selected_sequences = []

        # Build selected_sequences by appending top-k results
        energies = []
        for neg_energy, round_idx, candidate_idx, candidate_sequences in top_k_list:
            energy = -neg_energy  # Un-negate to get actual energy
            energies.append(energy)

            # Append each segment's sequence to the selected pool
            for seg_idx, segment in enumerate(self.segments):
                segment.selected_sequences.append(copy.deepcopy(candidate_sequences[seg_idx]))

        # Update energy scores
        self.energy_scores = energies
