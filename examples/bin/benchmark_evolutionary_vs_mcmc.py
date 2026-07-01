"""Benchmark comparing NSGA-II multi-objective optimization vs MCMC scalarizations.

This benchmark validates the NSGA-II selection mode added to EvolutionaryOptimizer
by comparing it against the practitioner's baseline: running MCMC with multiple
weight vectors and pooling the non-dominated points.

## Task: Concave Pareto front (NSGA-II's best case)

Two genuinely competing constraints on one DNA sequence with scores transformed
to produce a concave Pareto front:
- objective_1: concave transform of GC% distance from [10, 30] target
- objective_2: concave transform of GC% distance from [70, 90] target

The constraints are identical to the linear baseline, but scores are transformed
as f(s) = 1 - (1-s)² to create a ZDT2-style concave front that bows strongly
away from the line connecting the two extremes, with maximum deviation ~0.18.

## Prediction (pre-registered before running)

On a concave front, weighted-sum scalarization (MCMC's optimization strategy)
provably cannot find the middle trade-offs - it can only reach the two extremal
solutions. NSGA-II's Pareto-dominance ranking should fill the concave middle,
yielding:
  - Higher hypervolume for NSGA-II vs multi-weight MCMC
  - Visual scatter showing MCMC clustering at extremes, NSGA-II covering middle

Even though multi-weight MCMC pools its full trajectory (not just final points),
it should gap the concave middle because every scalarization λ·f₁ + (1-λ)·f₂
optimizes to one of the two extremes on a concave front.

## Methods (budget-matched, front-size-comparable)

1. **EA-nsga2**: EvolutionaryOptimizer with selection="nsga2"
   - Extracts non-dominated set from full population trajectory
   - Evaluations: measured from optimizer.history

2. **Multi-weight MCMC**: K independent MCMC chains with different weight vectors
   - Each chain contributes non-dominated points from its trajectory
   - Pooled across chains to form final front
   - Evaluations: measured from optimizer.history

3. **Single-weight MCMC**: One chain with equal weights (0.5, 0.5)
   - Non-dominated set from trajectory
   - Evaluations: measured from optimizer.history

All methods contribute comparable numbers of candidate points (pooled from
trajectories), ensuring hypervolume comparison is fair. A hard budget assertion
enforces that measured evaluations agree within 5% across all three methods.

## Metrics

- **Hypervolume**: Volume dominated by front relative to reference point (1.0, 1.0)
  - Higher is better, measures both convergence and spread
  - Primary comparison metric

- **Front size**: Number of non-dominated solutions
  - Reported for transparency

- **2D scatter**: Visual comparison of fronts in objective space
  - NSGA-II drawn last (large blue circles) to avoid occlusion

## Budget matching

All methods use identical total constraint evaluations, verified by reading
actual eval counts from optimizer.history (not nominal parameters). Budget
mismatch >5% raises an assertion failure.

## Outputs

Writes to current directory:
- benchmark_nsga2_summary.json: Aggregate statistics and conclusions
- benchmark_nsga2_detailed.json: Per-trial results
- benchmark_nsga2_fronts.png: 2D scatter plot of fronts

Usage:
    python examples/bin/benchmark_evolutionary_vs_mcmc.py
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from proto_tools.transforms.masking import MaskingStrategy
from scipy import stats  # type: ignore[import-untyped]

from proto_language.constraint import gc_content_constraint
from proto_language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import (
    EvolutionaryOptimizer,
    EvolutionaryOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Task parameters
# ============================================================================

SEQUENCE_LENGTH = 30
TARGET_BUDGET = 1000  # Target evaluations per trial
NUM_TRIALS = 20

# GC content targets (conflicting)
LOW_GC_MIN, LOW_GC_MAX = 10, 30
HIGH_GC_MIN, HIGH_GC_MAX = 70, 90

# Reference point for hypervolume (worst possible scores)
REFERENCE_POINT = (1.0, 1.0)

# Budget tolerance for assertion (5% mismatch allowed)
BUDGET_TOLERANCE = 0.05


# ============================================================================
# Concave transform for Pareto front geometry
# ============================================================================


def concave_transform(score: float) -> float:
    """Transform linear score to produce strongly concave Pareto front.

    Applies f(s) = 1 - (1-s)² transformation to create ZDT2-style concave
    front geometry. The squared term produces pronounced curvature, making
    the front bow strongly away from the line connecting extremes. This
    creates wide objective spread and ensures middle trade-offs are
    unreachable by weighted-sum scalarization.

    Args:
        score: Linear constraint score in [0, 1]

    Returns:
        Transformed score in [0, 1] with strong concave Pareto geometry
    """
    # Clamp to [0, 1] to handle numerical noise
    s = max(0.0, min(1.0, score))
    return 1.0 - (1.0 - s) * (1.0 - s)


# ============================================================================
# Pareto front extraction and hypervolume
# ============================================================================


def extract_objective_pair(seq: Any, constraints: list[Constraint]) -> tuple[float, float]:
    """Extract (low_gc_score, high_gc_score) for one sequence."""
    low_score = seq._constraints_metadata[constraints[0].label]["score"]
    high_score = seq._constraints_metadata[constraints[1].label]["score"]
    return (low_score, high_score)


def is_dominated(point: tuple[float, float], other: tuple[float, float]) -> bool:
    """Check if point is dominated by other (minimization)."""
    return all(o <= p for o, p in zip(other, point, strict=True)) and any(
        o < p for o, p in zip(other, point, strict=True)
    )


def extract_pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Extract non-dominated points from a set."""
    return [point for point in points if not any(is_dominated(point, other) for other in points)]


def hypervolume_2d(front: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Compute 2D hypervolume (dominated area) relative to reference point.

    Uses the standard 2D hypervolume algorithm: sort by first objective,
    compute rectangles. Reference point must dominate all front points.
    """
    if not front:
        return 0.0

    # Filter out points dominated by reference
    valid_front = [(x, y) for x, y in front if x < reference[0] and y < reference[1]]
    if not valid_front:
        return 0.0

    # Sort by first objective
    sorted_front = sorted(valid_front)

    volume = 0.0
    prev_y = reference[1]

    for x, y in sorted_front:
        if y < prev_y:
            volume += (reference[0] - x) * (prev_y - y)
            prev_y = y

    return volume


def count_actual_evaluations(optimizer: Any) -> int:
    """Count actual constraint evaluations from optimizer history.

    Reads the measured evaluation count from history, not nominal parameters.
    This is the only trustworthy eval count.
    """
    total = 0
    for snapshot in optimizer.history:
        results = snapshot.get("results", [])
        # Each result represents one evaluated proposal
        total += len(results)
    return total


# ============================================================================
# Concave-transformed GC constraints
# ============================================================================


def concave_low_gc_constraint(sequences: list[Any], config: Any) -> list[Any]:
    """Low-GC constraint with concave transform for ZDT2-style front."""
    from proto_language.core.constraint import ConstraintOutput

    # Evaluate underlying GC constraint
    results = gc_content_constraint(sequences, config)

    # Transform scores to create concave geometry
    transformed_results = []
    for result in results:
        original_score = result.score
        transformed_score = concave_transform(original_score)

        # Create new metadata with transform flag
        new_metadata = dict(result.metadata)
        new_metadata["original_score"] = original_score
        new_metadata["transformed"] = True

        # Return new ConstraintOutput with transformed score
        transformed_results.append(
            ConstraintOutput(
                score=transformed_score,
                metadata=new_metadata,
                structures=result.structures,
                logits=result.logits,
                metadata_recipient=result.metadata_recipient,
            )
        )

    return transformed_results


def concave_high_gc_constraint(sequences: list[Any], config: Any) -> list[Any]:
    """High-GC constraint with concave transform for ZDT2-style front."""
    from proto_language.core.constraint import ConstraintOutput

    # Evaluate underlying GC constraint
    results = gc_content_constraint(sequences, config)

    # Transform scores to create concave geometry
    transformed_results = []
    for result in results:
        original_score = result.score
        transformed_score = concave_transform(original_score)

        # Create new metadata with transform flag
        new_metadata = dict(result.metadata)
        new_metadata["original_score"] = original_score
        new_metadata["transformed"] = True

        # Return new ConstraintOutput with transformed score
        transformed_results.append(
            ConstraintOutput(
                score=transformed_score,
                metadata=new_metadata,
                structures=result.structures,
                logits=result.logits,
                metadata_recipient=result.metadata_recipient,
            )
        )

    return transformed_results


# ============================================================================
# NSGA-II run
# ============================================================================


def run_nsga2(seed: int, target_budget: int) -> dict[str, Any]:
    """Run EA with NSGA-II selection, extract non-dominated set from trajectory."""
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")

    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    low_gc = Constraint(
        inputs=[segment],
        function=concave_low_gc_constraint,
        function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
        weight=1.0,
        label="low_gc",
    )

    high_gc = Constraint(
        inputs=[segment],
        function=concave_high_gc_constraint,
        function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
        weight=1.0,
        label="high_gc",
    )

    # Configure EA to match MCMC budget
    # MCMC does: proposals_per_result=1, num_steps iterations → target_budget evals
    # EA counts evaluations as: population_size + num_generations * population_size
    # (each generation evaluates full population, not just offspring)
    # Solve: target = pop + gens * pop → gens = (target - pop) / pop
    population_size = 20
    elitism_count = 2
    # Calculate generations to match target budget
    # target_budget = pop + gens * pop → gens = (target - pop) / pop
    num_generations = (target_budget - population_size) // population_size

    config = EvolutionaryOptimizerConfig(
        population_size=population_size,
        num_generations=num_generations,
        elitism_count=elitism_count,
        selection="nsga2",
        seed=seed,
        verbose=False,
    )

    optimizer = EvolutionaryOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[low_gc, high_gc],
        config=config,
    )

    program = Program(optimizers=[optimizer], num_results=population_size, seed=seed)
    program.run()

    # Extract all evaluated points from history (not just final population)
    all_points: list[tuple[float, float]] = []
    for snapshot in optimizer.history:
        results = snapshot.get("results", [])
        for result in results:
            for construct in result.get("constructs", []):
                for segment in construct.get("segments", []):
                    constraints = segment.get("constraints", {})
                    if "low_gc" in constraints and "high_gc" in constraints:
                        low_score = constraints["low_gc"]["score"]
                        high_score = constraints["high_gc"]["score"]
                        all_points.append((low_score, high_score))

    # Extract Pareto front from all evaluated points
    front = extract_pareto_front(all_points)
    hv = hypervolume_2d(front, REFERENCE_POINT)
    actual_evals = count_actual_evaluations(optimizer)

    return {
        "front": front,
        "front_size": len(front),
        "hypervolume": hv,
        "actual_evaluations": actual_evals,
        "all_points_evaluated": len(all_points),
    }


# ============================================================================
# Multi-weight MCMC run
# ============================================================================


def run_multiweight_mcmc(seed: int, target_budget: int, num_weights: int = 5) -> dict[str, Any]:
    """Run multiple MCMC chains with different weight vectors, pool non-dominated from trajectories."""
    # Distribute budget across chains
    steps_per_chain = target_budget // num_weights

    # Generate weight vectors spanning the space
    weight_pairs = [(i / (num_weights - 1), 1 - i / (num_weights - 1)) for i in range(num_weights)]

    all_points: list[tuple[float, float]] = []
    total_evals = 0

    for chain_idx, (w_low, w_high) in enumerate(weight_pairs):
        segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        low_gc = Constraint(
            inputs=[segment],
            function=concave_low_gc_constraint,
            function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
            weight=w_low,
            label="low_gc",
        )

        high_gc = Constraint(
            inputs=[segment],
            function=concave_high_gc_constraint,
            function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
            weight=w_high,
            label="high_gc",
        )

        config = MCMCOptimizerConfig(
            num_steps=steps_per_chain,
            proposals_per_result=1,  # Explicit: 1 eval per step
            seed=seed + chain_idx,
            verbose=False,
        )

        optimizer = MCMCOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[low_gc, high_gc],
            config=config,
        )

        program = Program(optimizers=[optimizer], num_results=1, seed=seed + chain_idx)
        program.run()

        # Collect all evaluated points from this chain's trajectory
        for snapshot in optimizer.history:
            results = snapshot.get("results", [])
            for result in results:
                for construct in result.get("constructs", []):
                    for segment in construct.get("segments", []):
                        constraints = segment.get("constraints", {})
                        if "low_gc" in constraints and "high_gc" in constraints:
                            low_score = constraints["low_gc"]["score"]
                            high_score = constraints["high_gc"]["score"]
                            all_points.append((low_score, high_score))

        total_evals += count_actual_evaluations(optimizer)

    # Extract Pareto front from pooled trajectory points
    front = extract_pareto_front(all_points)
    hv = hypervolume_2d(front, REFERENCE_POINT)

    return {
        "front": front,
        "front_size": len(front),
        "hypervolume": hv,
        "actual_evaluations": total_evals,
        "num_chains": num_weights,
        "all_points_evaluated": len(all_points),
    }


# ============================================================================
# Single-weight MCMC run
# ============================================================================


def run_singleweight_mcmc(seed: int, target_budget: int) -> dict[str, Any]:
    """Run single MCMC chain with equal weights, extract non-dominated from trajectory."""
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    low_gc = Constraint(
        inputs=[segment],
        function=concave_low_gc_constraint,
        function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
        weight=0.5,
        label="low_gc",
    )

    high_gc = Constraint(
        inputs=[segment],
        function=concave_high_gc_constraint,
        function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
        weight=0.5,
        label="high_gc",
    )

    config = MCMCOptimizerConfig(
        num_steps=target_budget,
        proposals_per_result=1,  # Explicit: 1 eval per step
        seed=seed,
        verbose=False,
    )

    optimizer = MCMCOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[low_gc, high_gc],
        config=config,
    )

    program = Program(optimizers=[optimizer], num_results=1, seed=seed)
    program.run()

    # Collect all evaluated points from trajectory
    all_points: list[tuple[float, float]] = []
    for snapshot in optimizer.history:
        results = snapshot.get("results", [])
        for result in results:
            for construct in result.get("constructs", []):
                for segment in construct.get("segments", []):
                    constraints = segment.get("constraints", {})
                    if "low_gc" in constraints and "high_gc" in constraints:
                        low_score = constraints["low_gc"]["score"]
                        high_score = constraints["high_gc"]["score"]
                        all_points.append((low_score, high_score))

    front = extract_pareto_front(all_points)
    hv = hypervolume_2d(front, REFERENCE_POINT)
    actual_evals = count_actual_evaluations(optimizer)

    return {
        "front": front,
        "front_size": len(front),
        "hypervolume": hv,
        "actual_evaluations": actual_evals,
        "all_points_evaluated": len(all_points),
    }


# ============================================================================
# Statistical analysis and conclusions
# ============================================================================


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),  # Sample std
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def welch_t_test(group1: list[float], group2: list[float]) -> tuple[float, float]:
    """Compute Welch's t-test (unequal variances) between two groups.

    Returns:
        (t_statistic, p_value): Two-sided test
    """
    result = stats.ttest_ind(group1, group2, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def compute_conclusion(
    nsga2_hvs: list[float],
    multiweight_hvs: list[float],
    singleweight_hvs: list[float],
) -> dict[str, Any]:
    """Compute data-driven conclusion with statistical significance testing."""
    nsga2_mean = np.mean(nsga2_hvs)
    multiweight_mean = np.mean(multiweight_hvs)
    singleweight_mean = np.mean(singleweight_hvs)

    # Effect sizes (fractional difference)
    nsga2_vs_multi_effect = (nsga2_mean - multiweight_mean) / max(multiweight_mean, 1e-9)
    nsga2_vs_single_effect = (nsga2_mean - singleweight_mean) / max(singleweight_mean, 1e-9)

    # Statistical significance (Welch's t-test, alpha=0.05)
    _, p_nsga2_vs_multi = welch_t_test(nsga2_hvs, multiweight_hvs)
    _, p_nsga2_vs_single = welch_t_test(nsga2_hvs, singleweight_hvs)

    # Significance gate: p < 0.05 AND effect > 5%
    nsga2_beats_multi = p_nsga2_vs_multi < 0.05 and nsga2_vs_multi_effect > 0.05
    nsga2_beats_single = p_nsga2_vs_single < 0.05 and nsga2_vs_single_effect > 0.05

    # Framing
    if nsga2_beats_multi:
        recommendation = (
            f"Select NSGA-II for multi-objective problems when you want a diverse Pareto front. "
            f"NSGA-II achieves {nsga2_vs_multi_effect:.1%} higher hypervolume than multi-weight MCMC "
            f"(p={p_nsga2_vs_multi:.3f}, {NUM_TRIALS} trials)."
        )
    elif nsga2_beats_single:
        recommendation = (
            f"NSGA-II finds Pareto-optimal trade-offs better than single-weight MCMC "
            f"({nsga2_vs_single_effect:.1%} hypervolume improvement, p={p_nsga2_vs_single:.3f}), "
            f"but is comparable to multi-weight MCMC (p={p_nsga2_vs_multi:.3f}). "
            f"Use NSGA-II when you want the front in one run."
        )
    else:
        recommendation = (
            f"NSGA-II and multi-weight MCMC perform comparably "
            f"(hypervolume difference {nsga2_vs_multi_effect:.1%}, p={p_nsga2_vs_multi:.3f}). "
            f"Both are valid approaches for multi-objective optimization."
        )

    return {
        "recommendation": recommendation,
        "nsga2_vs_multiweight_effect": nsga2_vs_multi_effect,
        "nsga2_vs_singleweight_effect": nsga2_vs_single_effect,
        "p_nsga2_vs_multiweight": p_nsga2_vs_multi,
        "p_nsga2_vs_singleweight": p_nsga2_vs_single,
        "nsga2_mean_hv": nsga2_mean,
        "multiweight_mean_hv": multiweight_mean,
        "singleweight_mean_hv": singleweight_mean,
    }


# ============================================================================
# Plotting
# ============================================================================


def plot_fronts(
    nsga2_fronts: list[list[tuple[float, float]]],
    multiweight_fronts: list[list[tuple[float, float]]],
    singleweight_fronts: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    """Create 2D scatter plot of fronts from all methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    _fig, ax = plt.subplots(figsize=(10, 8))

    # Plot MCMC fronts first (light colors)
    for front in multiweight_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="red", alpha=0.1, s=20)

    for front in singleweight_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="gray", alpha=0.1, s=20)

    # Plot representative MCMC fronts (darker)
    if multiweight_fronts[0]:
        x, y = zip(*multiweight_fronts[0], strict=True)
        ax.scatter(x, y, c="red", s=80, label="Multi-weight MCMC", marker="^",
                  edgecolors="black", linewidth=1, alpha=0.6)

    if singleweight_fronts[0]:
        x, y = zip(*singleweight_fronts[0], strict=True)
        ax.scatter(x, y, c="gray", s=80, label="Single-weight MCMC", marker="s",
                  edgecolors="black", linewidth=1, alpha=0.6)

    # Plot NSGA-II LAST so it's on top (light colors)
    for front in nsga2_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="blue", alpha=0.15, s=25)

    # Plot one representative NSGA-II front (darker, large, on top)
    if nsga2_fronts[0]:
        x, y = zip(*nsga2_fronts[0], strict=True)
        ax.scatter(x, y, c="blue", s=120, label="NSGA-II", edgecolors="black",
                  linewidth=2, zorder=10)

    ax.set_xlabel("Low GC score (lower is better)", fontsize=12)
    ax.set_ylabel("High GC score (lower is better)", fontsize=12)
    ax.set_title(f"Pareto Fronts: NSGA-II vs MCMC Baselines ({NUM_TRIALS} trials)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved plot to {output_path}")


# ============================================================================
# Main benchmark
# ============================================================================


def main() -> None:
    """Run NSGA-II benchmark and save results."""
    logger.info("=" * 80)
    logger.info("NSGA-II Multi-Objective Optimization Benchmark")
    logger.info("=" * 80)
    logger.info(
        f"\nTask: Concave Pareto front (GC targets: low={LOW_GC_MIN}-{LOW_GC_MAX}%, "
        f"high={HIGH_GC_MIN}-{HIGH_GC_MAX}%)"
    )
    logger.info("Objectives transformed via f(s) = 1-(1-s)² for strong concave geometry")
    logger.info(f"Target budget: ~{TARGET_BUDGET} evals/trial, {NUM_TRIALS} trials")
    logger.info("Budget assertion: methods must agree within 5% (hard failure if mismatched)")
    logger.info("Methods extract non-dominated sets from full trajectories (comparable front sizes)")
    logger.info("\nPrediction: NSGA-II fills concave middle, MCMC clusters at extremes\n")

    # Run trials
    nsga2_results = []
    multiweight_results = []
    singleweight_results = []

    for trial in range(NUM_TRIALS):
        seed = 1000 + trial
        logger.info(f"Trial {trial + 1}/{NUM_TRIALS}")

        # NSGA-II
        result = run_nsga2(seed, TARGET_BUDGET)
        nsga2_results.append(result)
        logger.info(
            f"  NSGA-II: HV={result['hypervolume']:.4f}, front_size={result['front_size']}, "
            f"evals={result['actual_evaluations']}"
        )

        # Multi-weight MCMC
        result = run_multiweight_mcmc(seed, TARGET_BUDGET, num_weights=5)
        multiweight_results.append(result)
        logger.info(
            f"  Multi-weight MCMC: HV={result['hypervolume']:.4f}, front_size={result['front_size']}, "
            f"evals={result['actual_evaluations']}"
        )

        # Single-weight MCMC
        result = run_singleweight_mcmc(seed, TARGET_BUDGET)
        singleweight_results.append(result)
        logger.info(
            f"  Single-weight MCMC: HV={result['hypervolume']:.4f}, front_size={result['front_size']}, "
            f"evals={result['actual_evaluations']}"
        )

        # Budget assertion: all three methods must agree within tolerance
        nsga2_evals_trial = nsga2_results[-1]["actual_evaluations"]
        multi_evals_trial = multiweight_results[-1]["actual_evaluations"]
        single_evals_trial = singleweight_results[-1]["actual_evaluations"]

        max_evals = max(nsga2_evals_trial, multi_evals_trial, single_evals_trial)
        min_evals = min(nsga2_evals_trial, multi_evals_trial, single_evals_trial)
        budget_mismatch = (max_evals - min_evals) / max_evals

        if budget_mismatch > BUDGET_TOLERANCE:
            raise AssertionError(
                f"Budget mismatch exceeds {BUDGET_TOLERANCE*100:.0f}% tolerance in trial {trial+1}: "
                f"NSGA-II={nsga2_evals_trial}, Multi-weight={multi_evals_trial}, "
                f"Single-weight={single_evals_trial} (mismatch={budget_mismatch*100:.1f}%)"
            )

    # Aggregate statistics
    nsga2_hvs = [r["hypervolume"] for r in nsga2_results]
    multiweight_hvs = [r["hypervolume"] for r in multiweight_results]
    singleweight_hvs = [r["hypervolume"] for r in singleweight_results]

    nsga2_sizes = [r["front_size"] for r in nsga2_results]
    multiweight_sizes = [r["front_size"] for r in multiweight_results]
    singleweight_sizes = [r["front_size"] for r in singleweight_results]

    nsga2_evals = [r["actual_evaluations"] for r in nsga2_results]
    multiweight_evals = [r["actual_evaluations"] for r in multiweight_results]
    singleweight_evals = [r["actual_evaluations"] for r in singleweight_results]

    summary = {
        "task": {
            "description": "Conflicting GC-content objectives",
            "sequence_length": SEQUENCE_LENGTH,
            "low_gc_target": f"{LOW_GC_MIN}-{LOW_GC_MAX}%",
            "high_gc_target": f"{HIGH_GC_MIN}-{HIGH_GC_MAX}%",
        },
        "target_budget": TARGET_BUDGET,
        "num_trials": NUM_TRIALS,
        "nsga2": {
            "hypervolume": compute_statistics(nsga2_hvs),
            "front_size": compute_statistics(nsga2_sizes),
            "actual_evaluations": compute_statistics(nsga2_evals),
        },
        "multiweight_mcmc": {
            "hypervolume": compute_statistics(multiweight_hvs),
            "front_size": compute_statistics(multiweight_sizes),
            "actual_evaluations": compute_statistics(multiweight_evals),
            "num_chains": 5,
        },
        "singleweight_mcmc": {
            "hypervolume": compute_statistics(singleweight_hvs),
            "front_size": compute_statistics(singleweight_sizes),
            "actual_evaluations": compute_statistics(singleweight_evals),
        },
        "conclusion": compute_conclusion(nsga2_hvs, multiweight_hvs, singleweight_hvs),
    }

    # Save results
    summary_path = Path("benchmark_nsga2_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_path}")

    detailed_path = Path("benchmark_nsga2_detailed.json")
    with detailed_path.open("w") as f:
        json.dump(
            {
                "nsga2": nsga2_results,
                "multiweight_mcmc": multiweight_results,
                "singleweight_mcmc": singleweight_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved detailed results to {detailed_path}")

    # Plot
    plot_path = Path("benchmark_nsga2_fronts.png")
    plot_fronts(
        [r["front"] for r in nsga2_results],
        [r["front"] for r in multiweight_results],
        [r["front"] for r in singleweight_results],
        plot_path,
    )

    # Print conclusion
    logger.info("\n" + "=" * 80)
    logger.info("CONCLUSION")
    logger.info("=" * 80)
    logger.info(summary["conclusion"]["recommendation"])  # type: ignore[index]
    logger.info("\nMean Hypervolume:")
    logger.info(f"  NSGA-II:            {summary['nsga2']['hypervolume']['mean']:.4f}")  # type: ignore[index]
    logger.info(f"  Multi-weight MCMC:  {summary['multiweight_mcmc']['hypervolume']['mean']:.4f}")  # type: ignore[index]
    logger.info(f"  Single-weight MCMC: {summary['singleweight_mcmc']['hypervolume']['mean']:.4f}")  # type: ignore[index]
    logger.info("\nMean Actual Evaluations:")
    logger.info(f"  NSGA-II:            {summary['nsga2']['actual_evaluations']['mean']:.0f}")  # type: ignore[index]
    logger.info(f"  Multi-weight MCMC:  {summary['multiweight_mcmc']['actual_evaluations']['mean']:.0f}")  # type: ignore[index]
    logger.info(f"  Single-weight MCMC: {summary['singleweight_mcmc']['actual_evaluations']['mean']:.0f}")  # type: ignore[index]
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
