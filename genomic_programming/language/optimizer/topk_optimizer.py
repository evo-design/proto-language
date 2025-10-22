"""
TopK Optimizer that runs multiple independent sampling rounds and returns the top-k best constructs.
"""

from typing import List, Optional, final
import copy
import heapq

import numpy as np
from pydantic import Field, model_validator

from ..core import Optimizer, Construct, Generator, Constraint
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry


class TopKOptimizerConfig(BaseConfig):
    """Configuration for TopKOptimizer"""
    min_candidates: int = Field(
        default=100,
        ge=1,
        description="Minimum number of candidates to generate. If energy_threshold is set, "
                    "may generate more candidates until threshold is met."
    )
    k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of top candidates to keep and return (must be <= min_candidates)"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of candidates to generate per round (enables batching for generators). "
                    "min_candidates must be divisible by batch_size."
    )
    energy_threshold: Optional[float] = Field(
        default=None,
        description="Optional: Continue sampling until worst energy in top-k is below this threshold. "
                    "If set, optimizer will generate at least min_candidates, then continue until "
                    "threshold is met or max_candidates is reached."
    )
    max_candidates: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional: Maximum total candidates to generate (safety valve for energy_threshold). "
                    "If not set and energy_threshold is set, defaults to min_candidates * 10."
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print progress information"
    )

    @model_validator(mode='after')
    def validate_params(self):
        """Validate parameter relationships."""
        # k must not exceed total candidates
        if self.k > self.min_candidates:
            raise ValueError(
                f"k ({self.k}) cannot exceed min_candidates ({self.min_candidates}). "
                f"Cannot keep more sequences than generated."
            )

        # min_candidates must be divisible by batch_size
        if self.min_candidates % self.batch_size != 0:
            raise ValueError(
                f"min_candidates ({self.min_candidates}) must be divisible by "
                f"batch_size ({self.batch_size}). This ensures equal-sized batches."
            )

        # max_candidates must be divisible by batch_size if set
        if self.max_candidates is not None:
            if self.max_candidates % self.batch_size != 0:
                raise ValueError(
                    f"max_candidates ({self.max_candidates}) must be divisible by "
                    f"batch_size ({self.batch_size}). This ensures equal-sized batches."
                )
            if self.max_candidates < self.min_candidates:
                raise ValueError(
                    f"max_candidates ({self.max_candidates}) must be >= min_candidates ({self.min_candidates})"
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
    """
    TopK Optimizer for sequence optimization with efficient batching and threshold-based stopping.

    This optimizer implements a sampling approach that:
    1. Generates min_candidates total sequences across multiple rounds
    2. In each round: generates batch_size candidates, applies all generators, then evaluates
    3. Maintains a running list of the top-k best constructs
    4. Optionally continues sampling until energy_threshold is met
    5. Returns the final top-k constructs after all rounds

    Examples:
        Basic usage (fixed number of candidates):
        >>> config = TopKOptimizerConfig(
        ...     min_candidates=100,  # Generate 100 total candidates
        ...     k=10,                  # Keep top 10 constructs
        ...     batch_size=10,         # Generate 10 candidates at a time
        ... )
        >>> optimizer = TopKOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen, extension_gen, refine_gen],  # Applied in sequence
        ...     constraints=[gc_constraint, structure_constraint],
        ...     config=config,
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> optimizer.run()  # 10 rounds × 10 candidate = 100 total
        >>> best_constructs = optimizer.constructs  # Contains top 10

        Threshold-based stopping (sample until goal met):
        >>> config = TopKOptimizerConfig(
        ...     min_candidates=100,     # Minimum to generate
        ...     k=10,                     # Keep top 10
        ...     batch_size=10,            # Generate 10 at a time
        ...     energy_threshold=0.5,     # Continue until worst < 0.5
        ...     max_candidates=1000,      # Safety: stop after 1000 total
        ... )
        >>> optimizer = TopKOptimizer(
        ...     constructs=constructs,
        ...     generators=[evo2_gen],
        ...     constraints=[gc_constraint],
        ...     config=config,
        ... )
        >>> optimizer.run()  # Samples until worst in top-10 < 0.5 or 1000 total
        >>> best_constructs = optimizer.constructs
    """
    # Class attribute required by OptimizerRegistry
    config_class = TopKOptimizerConfig

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: TopKOptimizerConfig,
        constraint_weights: Optional[List[float]] = None,
        clear_tool_cache: bool | List[str] = True,
    ) -> None:
        """
        Initialize the TopK Optimizer.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            config: Configuration object containing algorithm parameters.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            clear_tool_cache: (bool) Whether to clear the tool cache on each iteration.
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
            constraint_weights=constraint_weights,
            clear_tool_cache=clear_tool_cache,
        )

        # Store TopK-specific parameters
        self.min_candidates = config.min_candidates
        self.batch_size = config.batch_size
        self.k = config.k
        self.rounds = config.min_candidates // config.batch_size  # Derived from total and batch
        self.verbose = config.verbose

        # Threshold-based stopping parameters
        self.energy_threshold = config.energy_threshold
        self.max_candidates = config.max_candidates or (config.min_candidates * 10 if config.energy_threshold else None)

        # Storage for top-k candidates using a max-heap of size k
        # We negate energies since heapq is a min-heap but we want max-heap behavior
        # This keeps the worst (highest) energy at the root for easy replacement
        self.top_k_heap: List[tuple] = []

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

            # Save the resulting sequences from this candidate
            candidate_sequences = {
                id(segment): copy.deepcopy(segment.candidate_sequences[candidate_idx])
                for segment in self.segments
            }

            # 5. Maintain a max-heap of size k to track the k smallest energies
            if len(self.top_k_heap) < self.k:
                # Haven't filled top-k yet, just add (negate energy for max-heap)
                heapq.heappush(self.top_k_heap, (-energy, round_idx, candidate_idx, candidate_sequences))
            elif energy < -self.top_k_heap[0][0]:
                # This energy is smaller (better) than the worst in our top-k
                # Replace the worst with this better one
                heapq.heapreplace(self.top_k_heap, (-energy, round_idx, candidate_idx, candidate_sequences))

    def run(self) -> None:
        """
        Execute TopK optimization through multiple sampling rounds.

        This method:
        1. Phase 1: Runs minimum 'rounds' number of independent sampling iterations
        2. Phase 2 (optional): If energy_threshold is set, continues sampling until:
           - Worst energy in top-k is below threshold, OR
           - max_candidates limit is reached
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

        # Phase 1: Generate minimum min_candidates
        for round_idx in range(self.rounds):
            self._run_round(round_idx)
            candidates_generated += self.batch_size

        # Phase 2: Continue if threshold not met (only if energy_threshold is set)
        threshold_met = False
        if self.energy_threshold is not None and self.max_candidates is not None:
            round_idx = self.rounds

            while candidates_generated < self.max_candidates:
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
                round_idx += 1

                if self.verbose and round_idx % 10 == 0:
                    worst_energy = -self.top_k_heap[0][0] if len(self.top_k_heap) == self.k else float('inf')
                    print(f"  Round {round_idx}: Generated {candidates_generated} candidates, worst in top-k: {worst_energy:.6f}")

        # Convert heap to sorted list (best first: lowest energy to highest)
        # Sort by actual energy (un-negate the first element)
        top_k_list = sorted(self.top_k_heap, key=lambda x: -x[0])

        # Update constructs with top-k
        self.set_topk_constructs(top_k_list)

        # Log statistics
        if self.verbose:
            print(f"\nOptimization complete:")
            print(f"  Total candidates generated: {candidates_generated}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Rounds executed: {candidates_generated // self.batch_size}")
            print(f"  Top-k kept: {self.k}")

            # Show threshold mode info if applicable
            if self.energy_threshold is not None:
                print(f"\nThreshold mode:")
                print(f"  Target threshold: {self.energy_threshold:.6f}")
                print(f"  Max candidates: {self.max_candidates}")
                if threshold_met:
                    print(f"  Status: ✓ Threshold met")
                else:
                    print(f"  Status: ✗ Max candidates reached without meeting threshold")

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

    # TODO: Remove this method and use self.selected_sequences directly
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
            for segment in self.segments:
                seg_id = id(segment)
                segment.selected_sequences.append(copy.deepcopy(candidate_sequences[seg_id]))

        # Update energy scores
        self.energy_scores = energies
