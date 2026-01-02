"""
TopK Optimizer that runs multiple independent sampling rounds and returns the top-k best constructs.
"""
from __future__ import annotations
from typing import Callable, List, Optional, final
import copy
import heapq
import math

import numpy as np
from pydantic import model_validator

from proto_language.language.core import Optimizer, Construct, Generator, Constraint
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry


class TopKOptimizerConfig(BaseConfig):
    """Configuration object for TopKOptimizer.

    This class defines configuration parameters for the TopK optimizer, which
    generates many candidate sequences and retains only the best K by lowest energy
    score.

    Attributes:
        min_num_samples (int): Minimum number of total samples to generate across
            all rounds. Must be divisible by ``batch_size`` to ensure equal-sized
            batches. If ``energy_threshold`` is set, may generate additional samples
            beyond this minimum until threshold is met. Must be at least 1.

        k (int): Number of top sequences to keep and return based on energy scores.
            Must be at least 1 and cannot exceed ``min_num_samples``. Lower energy
            scores are better (minimization objective). Must be at least 1.

        batch_size (int): Number of samples to generate per round. Enables batching
            for efficient parallel generation. Total rounds equals
            ``min_num_samples / batch_size``. Must be at least 1 and must evenly
            divide ``min_num_samples``.

        energy_threshold (Optional[float]): Optional threshold for early stopping.
            If set, continues sampling beyond ``min_num_samples`` until the worst
            energy in top-K is below this threshold, up to ``max_num_samples`` total.
            If ``None``, generates exactly ``min_num_samples``. Must be at least 0
            if provided. Default: ``None``.

        max_num_samples (Optional[int]): Maximum number of samples to generate when
            using threshold-based stopping. Prevents infinite sampling if threshold
            is never met. Must be divisible by ``batch_size`` and at least
            ``min_num_samples``. If ``None`` and ``energy_threshold`` is set,
            defaults to ``min_num_samples x 10``. Default: ``None``.

        verbose (bool): Whether to print detailed progress information including
            round statistics, energy values, and threshold status. Default: ``False``.

    Note:
        The optimizer generates samples in rounds of size ``batch_size``. For
        threshold-based stopping to work, both ``energy_threshold`` and
        ``max_num_samples`` must be set. All sampling parameters must be divisible
        by ``batch_size`` to ensure consistent batching.
    """
    # Required parameters
    min_num_samples: int = ConfigField(
        ge=1,
        title="Min Samples",
        description="Minimum number of samples to generate.",  # If energy_threshold is set, may generate more candidates until threshold is met.
    )
    k: int = ConfigField(
        ge=1,
        title="Top-k",
        description="Number of top samples to keep and return. Must be greater than Num samples.",
    )
    # TODO: Determine how to handle this for the client.
    batch_size: int = ConfigField(
        ge=1,
        title="Batch Size",
        description="Number of samples to generate per round (enables batching for generators).",  # "min_num_samples must be divisible by batch_size.",
    )

    # Advanced parameters
    energy_threshold: Optional[float] = ConfigField(
        default=None,
        ge=0.0,
        title="Energy Threshold",
        description="Continue sampling until worst energy in top-k is below this threshold.",  # If set, optimizer will generate at least min_num_samples, then continue until threshold is met or max_num_samples is reached.
        advanced=True,
    )
    max_num_samples: Optional[int] = ConfigField(
        default=None,
        ge=1,
        title="Max Samples",
        description="Maximum number of samples to generate until hard stop. Defaults to min_num_samples * 10",
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
        # k must not exceed total candidates
        if self.k > self.min_num_samples:
            raise ValueError(
                f"k ({self.k}) cannot exceed min_num_samples ({self.min_num_samples}). "
                "Cannot keep more sequences than generated."
            )

        # min_num_samples must be divisible by batch_size
        if self.min_num_samples % self.batch_size != 0:
            raise ValueError(
                f"min_num_samples ({self.min_num_samples}) must be divisible by "
                f"batch_size ({self.batch_size}). This ensures equal-sized batches."
            )

        # max_num_samples must be divisible by batch_size if set
        if self.max_num_samples is not None:
            if self.max_num_samples % self.batch_size != 0:
                raise ValueError(
                    f"max_num_samples ({self.max_num_samples}) must be divisible by "
                    f"batch_size ({self.batch_size}). This ensures equal-sized batches."
                )
            if self.max_num_samples < self.min_num_samples:
                raise ValueError(
                    f"max_num_samples ({self.max_num_samples}) must be >= min_num_samples ({self.min_num_samples})"
                )

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
    This continues for ``min_num_samples / batch_size`` rounds, and optionally
    continues until an energy threshold is met or ``max_num_samples`` is reached.

    Attributes:
        min_num_samples (int): Minimum total samples to generate.
        batch_size (int): Samples per round (enables batching).
        k (int): Number of top sequences to keep.
        rounds (int): Number of rounds (``min_num_samples / batch_size``).
        energy_threshold (Optional[float]): Optional threshold for early stopping.
        max_num_samples (Optional[int]): Maximum samples with threshold stopping.
        top_k_heap (List[tuple]): Max-heap tracking top-K candidates.

    Example:
        >>> config = TopKOptimizerConfig(
        ...     min_num_samples=100,
        ...     k=10,
        ...     batch_size=10
        ... )
        >>> optimizer = TopKOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen],
        ...     constraints=[gc_constraint],
        ...     config=config
        ... )
        >>> optimizer.run()
        >>> best_constructs = optimizer.constructs  # Top 10 sequences

    Note:
        - Lower energy scores are better (minimization objective)
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

        # Store TopK-specific parameters
        self.min_num_samples: int = config.min_num_samples
        self.batch_size: int = config.batch_size
        self.k: int = config.k
        self.rounds: int = config.min_num_samples // config.batch_size  # Derived from total and batch
        self.custom_logging: Optional[Callable] = custom_logging

        # Threshold-based stopping parameters
        self.energy_threshold: Optional[float] = config.energy_threshold
        self.max_num_samples: Optional[int] = config.max_num_samples or (config.min_num_samples * 10 if config.energy_threshold else None)

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

        # 3. Evaluate all batch_size candidates after all generators
        self.score_energy()  # Returns list of length batch_size

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

        This method:
        1. Phase 1: Runs minimum 'rounds' number of independent sampling iterations
        2. Phase 2 (optional): If energy_threshold is set, continues sampling until:
           - Worst energy in top-k is below threshold, OR
           - max_num_samples limit is reached
        3. In each round:
           - Resets all candidate_sequences to original_sequence
           - Applies each generator sequentially (generators batch across candidates)
           - Evaluates all batch_size candidates with constraints
           - Updates the top-k list if any candidates are good enough
        4. After all rounds, updates constructs with the top-k best

        With batch_size > 1, generators can batch their operations for efficiency.
        """
        # Clear any previous top-k list
        self.top_k_heap = []
        candidates_generated = 0

        # Phase 1: Generate min_num_samples
        for round_idx in range(self.rounds):
            self._run_round(round_idx)
            candidates_generated += self.batch_size

        # Phase 2: Continue if threshold not met (only if energy_threshold is set)
        threshold_met = False
        if self.energy_threshold is not None and self.max_num_samples is not None:
            round_idx = self.rounds

            while candidates_generated < self.max_num_samples:
                # Check if worst in top-k meets threshold
                if len(self.top_k_heap) == self.k:
                    worst_energy = -self.top_k_heap[0][0]  # Un-negate to get actual energy
                    if worst_energy < self.energy_threshold:
                        threshold_met = True
                        if self.verbose:
                            print(f"\nThreshold met! Worst in top-{self.k}: {worst_energy:.6f} < {self.energy_threshold:.6f}")
                        break

                # Generate another batch
                self._run_round(round_idx)
                candidates_generated += self.batch_size

                if self.verbose and (round_idx + 1) % 10 == 0:
                    worst_energy = -self.top_k_heap[0][0] if len(self.top_k_heap) == self.k else float('inf')
                    print(f"  Round {round_idx}: Generated {candidates_generated} candidates, worst in top-k: {worst_energy:.6f}")

                round_idx += 1

        # Convert heap to sorted list (best first: lowest energy to highest)
        # Sort by actual energy (un-negate the first element)
        top_k_list = sorted(self.top_k_heap, key=lambda x: -x[0])

        # Update constructs with top-k
        self.set_topk_constructs(top_k_list)

        # Save single final timepoint with top-k results
        self._save_progress_snapshot(time_step=0)

        # Log statistics
        if self.verbose:
            print("\nOptimization complete:")
            print(f"  Total samples generated: {candidates_generated}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Rounds executed: {candidates_generated // self.batch_size}")
            print(f"  Top-k kept: {self.k}")

            # Show threshold mode info if applicable
            if self.energy_threshold is not None:
                print("\nThreshold mode:")
                print(f"  Target threshold: {self.energy_threshold:.6f}")
                print(f"  Max samples: {self.max_num_samples}")
                if threshold_met:
                    print("  Status: ✓ Threshold met")
                else:
                    print("  Status: ✗ Max samples reached without meeting threshold")

            if top_k_list:
                # Un-negate energies for display (they're stored as negative in heap)
                actual_energies = [-e for e, _, _, _ in top_k_list]
                best_energy = actual_energies[0]
                worst_in_topk = actual_energies[-1]
                mean_energy = np.mean(actual_energies)

                print(f"\nTop-{self.k} statistics:")
                print(f"  Best energy:  {best_energy:.6f}")
                if len(top_k_list) > 1:
                    print(f"  Worst in top-k: {worst_in_topk:.6f}")
                print(f"  Mean energy:  {mean_energy:.6f}")

                # Show individual rankings
                if self.k <= 20:  # Only show individual rankings for small k
                    print(f"\nTop-{self.k} constructs:")
                    for i, (neg_energy, _, _, _) in enumerate(top_k_list):
                        energy = -neg_energy  # Un-negate to get actual energy
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
