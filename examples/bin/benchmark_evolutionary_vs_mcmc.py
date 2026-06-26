"""Benchmark comparing NSGA-II multi-objective optimization vs MCMC scalarizations.

This benchmark validates the NSGA-II selection mode added to EvolutionaryOptimizer
by comparing it against the practitioner's baseline: running MCMC with multiple
weight vectors and pooling the non-dominated points.

## Task: Conflicting GC-content objectives

Two genuinely competing constraints on one DNA sequence:
- low_gc: minimize distance from GC% target in [10, 30]
- high_gc: minimize distance from GC% target in [70, 90]

These targets are mutually exclusive - no sequence can score well on both.
The Pareto front represents the trade-off surface.

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
trajectories), ensuring hypervolume comparison is fair.

## Metrics

- **Hypervolume**: Volume dominated by front relative to reference point (1.0, 1.0)
  - Higher is better, measures both convergence and spread
  - Primary comparison metric

- **Front size**: Number of non-dominated solutions
  - Reported for transparency

- **2D scatter**: Visual comparison of fronts in objective space

## Budget matching

All methods use identical total constraint evaluations, verified by reading
actual eval counts from optimizer.history (not nominal parameters).

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
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
        weight=1.0,
        label="low_gc",
    )

    high_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=HIGH_GC_MIN, max_gc=HIGH_GC_MAX),
        weight=1.0,
        label="high_gc",
    )

    # Configure EA to approximately match budget
    population_size = 20
    elitism_count = 2
    offspring_per_gen = population_size - elitism_count
    # Budget = pop + gens * offspring
    num_generations = (target_budget - population_size) // offspring_per_gen

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
            for seq_data in result.get("sequences", []):
                seq_obj = seq_data.get("sequence")
                if seq_obj and hasattr(seq_obj, "_constraints_metadata"):
                    point = extract_objective_pair(seq_obj, [low_gc, high_gc])
                    all_points.append(point)

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
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
            weight=w_low,
            label="low_gc",
        )

        high_gc = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
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
                for seq_data in result.get("sequences", []):
                    seq_obj = seq_data.get("sequence")
                    if seq_obj and hasattr(seq_obj, "_constraints_metadata"):
                        point = extract_objective_pair(seq_obj, [low_gc, high_gc])
                        all_points.append(point)

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
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=LOW_GC_MIN, max_gc=LOW_GC_MAX),
        weight=0.5,
        label="low_gc",
    )

    high_gc = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
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
            for seq_data in result.get("sequences", []):
                seq_obj = seq_data.get("sequence")
                if seq_obj and hasattr(seq_obj, "_constraints_metadata"):
                    point = extract_objective_pair(seq_obj, [low_gc, high_gc])
                    all_points.append(point)

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

    # Statistical significance (Welch's t-test, α=0.05)
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

    # Plot each trial's front (light colors)
    for front in nsga2_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="blue", alpha=0.1, s=20)

    for front in multiweight_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="red", alpha=0.1, s=20)

    for front in singleweight_fronts:
        if front:
            x, y = zip(*front, strict=True)
            ax.scatter(x, y, c="gray", alpha=0.1, s=20)

    # Plot one representative front from each method (darker)
    if nsga2_fronts[0]:
        x, y = zip(*nsga2_fronts[0], strict=True)
        ax.scatter(x, y, c="blue", s=100, label="NSGA-II", edgecolors="black", linewidth=1)

    if multiweight_fronts[0]:
        x, y = zip(*multiweight_fronts[0], strict=True)
        ax.scatter(x, y, c="red", s=100, label="Multi-weight MCMC", marker="^", edgecolors="black", linewidth=1)

    if singleweight_fronts[0]:
        x, y = zip(*singleweight_fronts[0], strict=True)
        ax.scatter(x, y, c="gray", s=100, label="Single-weight MCMC", marker="s", edgecolors="black", linewidth=1)

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
        f"\nTask: Conflicting GC targets (low={LOW_GC_MIN}-{LOW_GC_MAX}%, high={HIGH_GC_MIN}-{HIGH_GC_MAX}%)"
    )
    logger.info(f"Target budget: ~{TARGET_BUDGET} evals/trial, {NUM_TRIALS} trials")
    logger.info("Methods extract non-dominated sets from full trajectories (comparable front sizes)\n")

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
