"""
TopK Optimizer that runs multiple independent sampling rounds and returns the top-k best constructs.
"""

from typing import List, Optional, final
import copy
import heapq

import numpy as np
from pydantic import Field

from ..core import Optimizer, Construct, Generator, Constraint
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry


class TopKOptimizerConfig(BaseConfig):
    """Configuration for TopKOptimizer"""
    rounds: int = Field(
        default=100,
        ge=1,
        description="Number of sampling rounds to perform"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of top candidates to keep and return (must be <= rounds)"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print progress information"
    )


@OptimizerRegistry.register(
    key="topk",
    label="TopK Optimizer",
    config=TopKOptimizerConfig,
    description="Greedy optimizer that runs sampling rounds and maintains the top-k best constructs",
)
@final
class TopKOptimizer(Optimizer):
    """
    TopK Optimizer for sequence optimization.

    This optimizer implements a sampling approach that:
    1. Runs 'rounds' number of sampling iterations
    2. In each round: samples all generators sequentially, then evaluates
    3. Maintains a running list of the top-k best constructs
    4. Returns the final top-k constructs after all rounds

    Key Features:
    - Each round applies ALL generators in sequence (compositional generation)
    - Maintains only the k best constructs in memory (efficient for large number of rounds)
    - Each round is independent, starting from the original constructs

    Examples:
        >>> config = TopKOptimizerConfig(
        ...     rounds=100,      # Run 100 sampling rounds
        ...     k=10,            # Keep top 10 constructs
        ... )
        >>> optimizer = TopKOptimizer(
        ...     constructs=constructs,
        ...     generators=[mutation_gen, extension_gen, refine_gen],  # Applied in sequence
        ...     constraints=[gc_constraint, structure_constraint],
        ...     config=config,
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> optimizer.run()  # Runs 100 rounds, keeps best 10
        >>> best_constructs = optimizer.constructs  # Contains top 10
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
    ) -> None:
        """
        Initialize the TopK Optimizer.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            config: Configuration object containing algorithm parameters.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.

        Raises:
            ValueError: If any validation checks fail.
        """
        # TODO: Support num_candidates > 1 for increased efficiency
        # Initialize with num_candidates=1 since we process one construct at a time
        # Each round num_selected=1, so after k rounds we will return k constructs
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_candidates=1,
            num_selected=config.k,
            constraint_weights=constraint_weights,
        )
        self.rounds = config.rounds

        # TODO: Remove this and use self.num_selected directly
        self.k = config.k
        self.verbose = config.verbose

        # Storage for top-k candidates using a max-heap of size k
        # We negate energies since heapq is a min-heap but we want max-heap behavior
        # This keeps the worst (highest) energy at the root for easy replacement
        self.top_k_heap: List[tuple] = []

        if self.k > self.rounds:
            raise ValueError(f"k ({self.k}) cannot be greater than rounds ({self.rounds}). There needs to be at least {self.k} rounds.")

    def run(self) -> None:
        """
        Execute TopK optimization through multiple sampling rounds.

        This method:
        1. Runs 'rounds' number of independent sampling iterations
        2. In each round:
           - Resets to original_sequence (leveraging segment's built-in state)
           - Applies each generator sequentially to modify constructs
           - Evaluates the final construct with constraints
           - Updates the top-k list if the construct is good enough
        3. After all rounds, updates constructs with the top-k best
        """
        # Clear any previous top-k list
        self.top_k_heap = []

        # Run sampling rounds
        for round_idx in range(self.rounds):
            # 1. Reset candidate pool to original state at the start of each round
            for segment in self.segments:
                segment.candidate_sequences[0] = copy.deepcopy(segment.original_sequence)

            # 2. Sample each generator in sequence
            for generator in self.generators:
                generator.sample()

            # 3. Evaluate the constructs after all generators
            self.score_energy()
            energy = self.energy_scores[0]  # Only one (num_candidates=1)

            # 4. Save the resulting sequences from this round 
            round_sequences = {
                id(segment): copy.deepcopy(segment.candidate_sequences[0])
                for segment in self.segments
            }

            # 5. Maintain a max-heap of size k to track the k smallest energies
            if len(self.top_k_heap) < self.k:
                # Haven't filled top-k yet, just add (negate energy for max-heap)
                heapq.heappush(self.top_k_heap, (-energy, round_idx, round_sequences))
            elif energy < -self.top_k_heap[0][0]:
                # This energy is smaller (better) than the worst in our top-k
                # Replace the worst with this better one
                heapq.heapreplace(self.top_k_heap, (-energy, round_idx, round_sequences))

        # 6. Convert heap to sorted list (best first: lowest energy to highest)
        # Sort by actual energy (un-negate the first element)
        top_k_list = sorted(self.top_k_heap, key=lambda x: -x[0])

        # 7. Update constructs with top-k
        self.set_topk_constructs(top_k_list)

        # Log statistics
        if self.verbose:
            print(f"\nOptimization complete after {self.rounds} rounds")

            if top_k_list:
                # Un-negate energies for display (they're stored as negative in heap)
                actual_energies = [-e for e, _, _ in top_k_list]
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
                    for i, (neg_energy, round_idx, _) in enumerate(top_k_list):
                        energy = -neg_energy  # Un-negate to get actual energy
                        print(f"  Rank {i+1}: Energy={energy:.6f} (from round {round_idx + 1})")
                print(f"\nTopK optimization complete. Returned {len(top_k_list)} best constructs.")

    # TODO: Remove this method and use self.selected_sequences directly
    def set_topk_constructs(self, top_k_list: List[tuple]) -> None:
        """
        Set the top-k constructs to segments' selected_sequences pool.

        Args:
            top_k_list: List of (neg_energy, round_idx, round_sequences) tuples.
        """
        # Initialize selected pool to empty lists for building
        for segment in self.segments:
            segment.selected_sequences = []

        # Build selected_sequences by appending top-k results
        energies = []
        for neg_energy, _, round_sequences in top_k_list:
            energy = -neg_energy  # Un-negate to get actual energy
            energies.append(energy)

            # Append each segment's sequence to the selected pool
            for segment in self.segments:
                seg_id = id(segment)
                segment.selected_sequences.append(copy.deepcopy(round_sequences[seg_id]))

        # Update energy scores
        self.energy_scores = energies
