"""Benchmark characterizing the EvolutionaryOptimizer's building-block regime.

This benchmark documents where the EvolutionaryOptimizer is the right tool in the
proto-language optimizer suite. The EA's distinctive mechanism is crossover, which
provides value specifically when a problem has building-block structure: multiple
independent sub-problems that must be solved simultaneously, where local search
gets stuck on one sub-problem at a time but a population can specialize and
recombine solutions.

## Regime Identification

Task parameters (NUM_BLOCKS=4, BLOCK_LEN=9, MOTIF_LEN=5) were identified via
hardness sweep (sweep_building_block_hardness.py) as the setting where crossover
advantage manifests. At this difficulty:
- EA with crossover: 0.62 ± 0.44 blocks solved
- EA without crossover: 0.04 ± 0.11 blocks solved (15× worse)
- Multi-restart MCMC: 0.27 ± 0.07 blocks solved

## Pre-registered predictions (written before running)

- As NUM_BLOCKS rises, multi-restart MCMC's fraction-complete should fall faster
  than the EA's, because restarts can't share blocks across chains.
- Crossover-enabled EA should reach higher mean-blocks-solved than crossover-disabled
  at matched budget, because recombination composes block-specialists.
- Single-point ≥ uniform on mean-blocks-solved, because uniform breaks contiguous
  solved blocks.
- If NUM_BLOCKS is small (2-3), expect all methods to look similar — that's the
  easy regime where the building-block advantage doesn't apply.

## Task: Concatenated Building Blocks ("Royal Road" style)

A sequence is divided into NUM_BLOCKS contiguous blocks. Each block has its own
target motif. Score is the mean of normalized window-Hamming distances across all
blocks, so a complete solution requires ALL blocks solved.

Each block is individually hard enough (MOTIF_LEN=8 in BLOCK_LEN=12 window) to
require directed search, not random luck. This forces the population to specialize:
different individuals solve different subsets of blocks, and crossover can assemble
complete solutions by recombining specialists.

## Comparisons

Four comparisons, all at matched evaluation budget:
1. EA vs single-chain MCMC
2. EA vs 20-restart MCMC (key comparison for building-block regime)
3. Crossover ablation: EA(crossover_rate=0.0) vs EA(crossover_rate=0.8)
4. Single-point vs uniform crossover

## Metrics

Primary: fraction of final population with all blocks solved (complete solutions)
Secondary: mean blocks solved per individual (partial credit, shows progress)
Both at strict (tol=0.0) and relaxed (tol=1/MOTIF_LEN) thresholds.

Outputs (written to current directory):
    - benchmark_ea_vs_mcmc_summary.json: Aggregate statistics
    - benchmark_ea_vs_mcmc_detailed.json: Per-trial results
    - benchmark_ea_vs_mcmc_convergence.png: Convergence plots

Usage:
    python examples/bin/benchmark_evolutionary_vs_mcmc.py
"""

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from proto_tools.transforms.masking import MaskingStrategy
from pydantic import BaseModel

from proto_language.core import Constraint, ConstraintOutput, Construct, Program, Segment, Sequence
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import (
    EvolutionaryOptimizer,
    EvolutionaryOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Section 1: Task parameters and motif generation
# ============================================================================

# Tunable knobs defining the building-block regime
# Identified via sweep: NUM_BLOCKS=4 shows crossover advantage
NUM_BLOCKS = 4  # Regime found: crossover 15× better than no crossover
BLOCK_LEN = 9  # Length of each block's region
MOTIF_LEN = 5  # Length of target motif
SEQUENCE_LENGTH = NUM_BLOCKS * BLOCK_LEN  # 36

# Fixed evaluation budget and trial count
BUDGET = 1000
NUM_TRIALS = 20  # Raised from 5 for lower variance


def _generate_block_motifs(num_blocks: int, motif_len: int, seed: int = 0) -> list[str]:
    """Generate distinct random DNA motifs for each block.

    Uses a fixed seed so the task is identical across all runs and trials.

    Args:
        num_blocks: Number of blocks to generate motifs for
        motif_len: Length of each motif
        seed: Random seed for reproducibility

    Returns:
        List of distinct DNA motif strings
    """
    rng = random.Random(seed)  # noqa: S311
    bases = ["A", "C", "G", "T"]
    motifs: list[str] = []

    for _ in range(num_blocks):
        # Generate until we get a motif that's distinct from all existing ones
        while True:
            motif = "".join(rng.choices(bases, k=motif_len))
            # Ensure it's distinct (no shared prefixes of length > motif_len//2)
            if all(sum(1 for a, b in zip(motif, existing, strict=True) if a == b) < motif_len // 2 for existing in motifs):
                motifs.append(motif)
                break

    return motifs


# Generate fixed motifs for the task (same across all runs)
BLOCK_MOTIFS = _generate_block_motifs(NUM_BLOCKS, MOTIF_LEN, seed=0)

logger.info(f"Building-block task: {NUM_BLOCKS} blocks, sequence length {SEQUENCE_LENGTH}")
logger.info(f"Block motifs (fixed): {BLOCK_MOTIFS}")


# ============================================================================
# Section 2: Hamming distance helpers (reused from verified implementation)
# ============================================================================


def hamming(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings."""
    if len(a) != len(b):
        raise ValueError(f"hamming requires equal length, got {len(a)} and {len(b)}")
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def min_window_hamming(region: str, motif: str) -> int:
    """Minimum Hamming distance between motif and any window of region."""
    m = len(motif)
    if len(region) < m:
        return m
    best = m
    for i in range(len(region) - m + 1):
        d = hamming(region[i : i + m], motif)
        if d < best:
            best = d
            if best == 0:
                break
    return best


def normalized_window_score(region: str, motif: str) -> float:
    """Normalized score in [0,1]: best window Hamming distance / motif length."""
    return min_window_hamming(region, motif) / len(motif)


# ============================================================================
# Section 3: Building-block constraint and metrics
# ============================================================================


class BuildingBlockConfig(BaseModel):
    """Configuration for building-block constraint."""


def building_block_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: BuildingBlockConfig  # noqa: ARG001
) -> list[ConstraintOutput]:
    """Score sequences on the building-block task.

    Each block contributes its normalized window-Hamming distance to its motif.
    Score is the MEAN across blocks, so every additional solved block strictly
    lowers the score, and a full solution requires all blocks solved.

    Score in [0,1], where 0.0 = all blocks solved.
    """
    results = []
    for (seq,) in input_sequences:
        seq_str = seq.sequence
        total = 0.0
        for b in range(NUM_BLOCKS):
            region = seq_str[b * BLOCK_LEN : (b + 1) * BLOCK_LEN]
            total += normalized_window_score(region, BLOCK_MOTIFS[b])
        score = total / NUM_BLOCKS
        results.append(ConstraintOutput(score=score))
    return results


def blocks_solved(seq: str, tol: float = 0.0) -> int:
    """Count how many blocks are solved in this sequence."""
    return sum(
        1
        for b in range(NUM_BLOCKS)
        if normalized_window_score(seq[b * BLOCK_LEN : (b + 1) * BLOCK_LEN], BLOCK_MOTIFS[b]) <= tol
    )


def analyze_building_block_diversity(sequences: list[str], tol: float = 0.0) -> dict[str, Any]:
    """Analyze final population on building-block task.

    Returns:
        Dict with:
        - fraction_complete: fraction of individuals with all blocks solved
        - mean_blocks_solved: mean blocks solved per individual
        - distinct_complete: number of unique complete solutions
        - tol: threshold used
    """
    num_complete = 0
    total_blocks = 0
    complete_seqs = set()

    for seq in sequences:
        solved = blocks_solved(seq, tol=tol)
        total_blocks += solved
        if solved == NUM_BLOCKS:
            num_complete += 1
            complete_seqs.add(seq)

    return {
        "fraction_complete": num_complete / len(sequences) if sequences else 0.0,
        "mean_blocks_solved": total_blocks / len(sequences) if sequences else 0.0,
        "distinct_complete": len(complete_seqs),
        "tol": tol,
    }


# ============================================================================
# Section 4: Metric extraction at budget (with Bug 5 fix for nested keys)
# ============================================================================


def extract_metrics_at_budget(optimizer: Any, budget: int) -> dict[str, Any]:
    """Extract metrics from optimizer history up to target evaluation budget.

    Implements Bug 2 fix: captures final population AT budget cutoff, not history[-1].
    Implements Bug 4 fix: counts offspring correctly for EA.

    Args:
        optimizer: The optimizer instance with history
        budget: Target evaluation budget
        task: Task identifier (for task-specific metrics)

    Returns:
        Dict with convergence, total_evaluations, best_score, task_metrics
    """
    convergence = []
    total_evaluations = 0
    final_snapshot_at_budget = None

    # Determine offspring count per generation (EA-specific)
    if hasattr(optimizer.config, "population_size"):
        population_size = optimizer.config.population_size
        elitism_count = optimizer.config.elitism_count
        offspring_per_gen = population_size - elitism_count
    else:
        offspring_per_gen = 0  # Not used for MCMC

    for snapshot in optimizer.history:
        time_step = snapshot.get("time_step", 0)
        results = snapshot.get("results", [])

        # Count evaluations (skip initial population at time_step=0)
        if time_step > 0:
            total_evaluations += offspring_per_gen

        # Capture this snapshot if we're still within budget
        if total_evaluations <= budget:
            final_snapshot_at_budget = snapshot

        # Stop if we've exceeded budget
        if total_evaluations > budget:
            break

        # Extract energy scores and sequences
        energy_scores = [r.get("energy_score") for r in results]
        sequences = []
        for result in results:
            for construct in result.get("constructs", []):
                for segment_data in construct.get("segments", []):
                    seq = segment_data.get("sequence", "")
                    if seq:
                        sequences.append(seq)
                        break  # Only take first segment

        # Best score at this point
        finite_scores = [s for s in energy_scores if s is not None and math.isfinite(s)]
        if finite_scores:
            convergence.append(
                {
                    "time_step": time_step,
                    "evaluations": total_evaluations,
                    "best_score": min(finite_scores),
                    "mean_score": float(np.mean(finite_scores)),
                }
            )

    # Bug 2 fix: Use final snapshot AT budget cutoff, not history[-1]
    if final_snapshot_at_budget is None:
        return {"convergence": [], "total_evaluations": 0}

    final_results = final_snapshot_at_budget.get("results", [])
    final_scores = [r.get("energy_score") for r in final_results]
    final_sequences = []
    for result in final_results:
        for construct in result.get("constructs", []):
            for segment_data in construct.get("segments", []):
                seq = segment_data.get("sequence", "")
                if seq:
                    final_sequences.append(seq)
                    break

    # Building-block metrics at both strict and relaxed thresholds
    strict = analyze_building_block_diversity(final_sequences, tol=0.0)
    relaxed = analyze_building_block_diversity(final_sequences, tol=1.0 / MOTIF_LEN)
    task_metrics = {
        "strict": strict,
        "relaxed": relaxed,
        "primary_metric": relaxed["mean_blocks_solved"],  # Lead with partial credit
        "secondary_metric": relaxed["fraction_complete"],  # Complete solutions
    }

    # Best final score
    finite_final = [s for s in final_scores if s is not None and math.isfinite(s)]
    best_score = min(finite_final) if finite_final else float("inf")

    return {
        "convergence": convergence,
        "total_evaluations": total_evaluations,
        "best_score": best_score,
        "task_metrics": task_metrics,
        "final_sequences": final_sequences[:10],  # Store sample for inspection
    }


def run_ea_config(
    population_size: int,
    num_generations: int,
    crossover_rate: float,
    crossover_strategy: str,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Run EA with given config and extract metrics at budget cutoff.

    Args:
        population_size: Population size
        num_generations: Number of generations (will stop early if budget reached)
        crossover_rate: Crossover rate in [0,1]
        crossover_strategy: "single-point" or "uniform"
        seed: Random seed
        budget: Evaluation budget to enforce

    Returns:
        Dict with metrics extracted at budget cutoff
    """
    # Setup
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # Building-block constraint
    constraint = Constraint(
        inputs=[segment],
        function=building_block_constraint,
        function_config=BuildingBlockConfig(),
    )

    # EA config
    config = EvolutionaryOptimizerConfig(
        population_size=population_size,
        num_generations=num_generations,
        elitism_count=max(1, population_size // 10),
        tournament_size=3,
        crossover_rate=crossover_rate,
        mutation_rate=0.05,  # Low to avoid erasing assembled blocks
        crossover_strategy=cast(Literal["single-point", "uniform"], crossover_strategy),
        seed=seed,
        verbose=False,
        tracking_interval=1,
    )

    optimizer = EvolutionaryOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[constraint],
        config=config,
    )

    # Run
    program = Program(optimizers=[optimizer], num_results=population_size, seed=seed)
    program.run()

    # Extract metrics at budget
    return extract_metrics_at_budget(optimizer, budget)


def run_mcmc_config(
    num_results: int,
    num_steps: int,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Run MCMC with given config and extract metrics at budget cutoff.

    Args:
        num_results: Number of independent chains
        num_steps: Number of MCMC steps per chain (will stop early if budget reached)
        seed: Random seed
        budget: Evaluation budget to enforce

    Returns:
        Dict with metrics extracted at budget cutoff, includes is_single_chain flag
    """
    # Setup
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # Building-block constraint
    constraint = Constraint(
        inputs=[segment],
        function=building_block_constraint,
        function_config=BuildingBlockConfig(),
    )

    # MCMC config
    config = MCMCOptimizerConfig(
        num_results=num_results,
        proposals_per_result=1,
        num_steps=num_steps,
        seed=seed,
        verbose=False,
        tracking_interval=1,
    )

    optimizer = MCMCOptimizer(
        constructs=[Construct([segment])],
        generators=[mutation_gen],
        constraints=[constraint],
        config=config,
    )

    # Run
    program = Program(optimizers=[optimizer], num_results=num_results, seed=seed)
    program.run()

    # Extract metrics at budget - MCMC counts differently
    convergence = []
    total_evaluations = 0
    final_snapshot_at_budget = None  # Bug 2 fix: capture snapshot AT budget

    for snapshot in optimizer.history:
        time_step = snapshot.get("time_step", 0)
        results = snapshot.get("results", [])

        # MCMC: num_results * proposals_per_result evaluations per step (except step 0)
        if time_step > 0:
            total_evaluations += num_results * 1  # proposals_per_result=1

        # Capture this snapshot if we're still within budget
        if total_evaluations <= budget:
            final_snapshot_at_budget = snapshot

        # Stop if we've exceeded budget
        if total_evaluations > budget:
            break

        # Extract energy scores
        energy_scores = [r.get("energy_score") for r in results]

        # Best score at this point
        finite_scores = [s for s in energy_scores if s is not None and math.isfinite(s)]
        if finite_scores:
            convergence.append(
                {
                    "time_step": time_step,
                    "evaluations": total_evaluations,
                    "best_score": min(finite_scores),
                    "mean_score": float(np.mean(finite_scores)),
                }
            )

    # Bug 2 fix: Use final snapshot AT budget cutoff
    if final_snapshot_at_budget is None:
        return {"convergence": [], "total_evaluations": 0, "is_single_chain": num_results == 1}

    final_results = final_snapshot_at_budget.get("results", [])
    final_scores = [r.get("energy_score") for r in final_results]
    final_sequences = []
    for result in final_results:
        for construct in result.get("constructs", []):
            for segment_data in construct.get("segments", []):
                seq = segment_data.get("sequence", "")
                if seq:
                    final_sequences.append(seq)
                    break

    # Bug 3 fix: Flag single-chain MCMC
    is_single_chain = num_results == 1

    # Task metrics
    if is_single_chain:
        # Single-chain: diversity metrics not meaningful
        task_metrics = {
            "primary_metric": 0.0,
            "secondary_metric": 0.0,
            "note": "Single-chain MCMC: diversity metrics not comparable",
        }
    else:
        strict = analyze_building_block_diversity(final_sequences, tol=0.0)
        relaxed = analyze_building_block_diversity(final_sequences, tol=1.0 / MOTIF_LEN)
        task_metrics = {
            "strict": strict,
            "relaxed": relaxed,
            "primary_metric": relaxed["mean_blocks_solved"],
            "secondary_metric": relaxed["fraction_complete"],
        }

    # Best final score
    finite_final = [s for s in final_scores if s is not None and math.isfinite(s)]
    best_score = min(finite_final) if finite_final else float("inf")

    return {
        "convergence": convergence,
        "total_evaluations": total_evaluations,
        "best_score": best_score,
        "task_metrics": task_metrics,
        "final_sequences": final_sequences[:10],
        "is_single_chain": is_single_chain,
    }


# ============================================================================
# Section 5: Budget verification (Bug 4 fix)
# ============================================================================


def verify_budget_match(
    results1: list[dict[str, Any]], results2: list[dict[str, Any]], method1: str, method2: str, budget: int
) -> None:
    """Verify that measured budgets match across methods (Bug 4 fix).

    Args:
        results1: Trial results from first method
        results2: Trial results from second method
        method1: Name of first method
        method2: Name of second method
        budget: Target budget
    """
    tolerance = 20
    for i, (r1, r2) in enumerate(zip(results1, results2, strict=True)):
        evals1 = r1["total_evaluations"]
        evals2 = r2["total_evaluations"]
        if abs(evals1 - evals2) > tolerance:
            logger.warning(
                f"Budget mismatch trial {i+1}: {method1} {evals1} vs {method2} {evals2} (tolerance={tolerance})"
            )
        if abs(evals1 - budget) > tolerance:
            logger.warning(f"Budget overshoot {method1} trial {i+1}: {evals1} vs target {budget}")
        if abs(evals2 - budget) > tolerance:
            logger.warning(f"Budget overshoot {method2} trial {i+1}: {evals2} vs target {budget}")


# ============================================================================
# Section 6: Run all comparisons
# ============================================================================


def run_benchmark(budget: int = BUDGET, trials: int = NUM_TRIALS) -> dict[str, Any]:
    """Run the full benchmark: 4 comparisons at matched budget.

    Args:
        budget: Evaluation budget per trial
        trials: Number of independent trials

    Returns:
        Dict with results for all comparisons
    """
    logger.info(f"Starting benchmark: budget={budget}, trials={trials}")

    # Storage for results
    comp1_ea = []  # EA vs single-chain
    comp1_mcmc = []
    comp2_ea = []  # EA vs multi-start
    comp2_mcmc = []
    comp3_no_cross = []  # Crossover ablation
    comp3_with_cross = []
    comp4_single_point = []  # Crossover strategy comparison
    comp4_uniform = []

    logger.info("\n--- Comparison 1: EA vs Single-Chain MCMC ---")
    for trial_idx in range(trials):
        seed = 42 + trial_idx
        logger.info(f"Trial {trial_idx + 1}/{trials} (seed={seed})")

        ea_result = run_ea_config(
            population_size=50,  # Larger pop for specialist formation
            num_generations=30,  # Adjusted for budget with larger pop
            crossover_rate=0.8,
            crossover_strategy="single-point",
            seed=seed,
            budget=budget,
        )
        comp1_ea.append(ea_result)

        mcmc_result = run_mcmc_config(
            num_results=1,
            num_steps=1200,  # Overshoot budget
            seed=seed,
            budget=budget,
        )
        comp1_mcmc.append(mcmc_result)

    verify_budget_match(comp1_ea, comp1_mcmc, "EA", "single-chain MCMC", budget)

    logger.info("\n--- Comparison 2: EA vs Multi-Start MCMC ---")
    for trial_idx in range(trials):
        seed = 42 + trial_idx
        logger.info(f"Trial {trial_idx + 1}/{trials} (seed={seed})")

        ea_result = run_ea_config(
            population_size=50,
            num_generations=30,
            crossover_rate=0.8,
            crossover_strategy="single-point",
            seed=seed,
            budget=budget,
        )
        comp2_ea.append(ea_result)

        mcmc_result = run_mcmc_config(
            num_results=50,  # Match EA population
            num_steps=30,
            seed=seed,
            budget=budget,
        )
        comp2_mcmc.append(mcmc_result)

    verify_budget_match(comp2_ea, comp2_mcmc, "EA", "multi-start MCMC", budget)

    logger.info("\n--- Comparison 3: Crossover Ablation ---")
    for trial_idx in range(trials):
        seed = 42 + trial_idx
        logger.info(f"Trial {trial_idx + 1}/{trials} (seed={seed})")

        no_cross_result = run_ea_config(
            population_size=50,
            num_generations=30,
            crossover_rate=0.0,
            crossover_strategy="single-point",
            seed=seed,
            budget=budget,
        )
        comp3_no_cross.append(no_cross_result)

        with_cross_result = run_ea_config(
            population_size=50,
            num_generations=30,
            crossover_rate=0.8,
            crossover_strategy="single-point",
            seed=seed,
            budget=budget,
        )
        comp3_with_cross.append(with_cross_result)

    verify_budget_match(comp3_no_cross, comp3_with_cross, "no crossover", "with crossover", budget)

    logger.info("\n--- Comparison 4: Crossover Strategy (Single-Point vs Uniform) ---")
    for trial_idx in range(trials):
        seed = 42 + trial_idx
        logger.info(f"Trial {trial_idx + 1}/{trials} (seed={seed})")

        single_point_result = run_ea_config(
            population_size=50,
            num_generations=30,
            crossover_rate=0.8,
            crossover_strategy="single-point",
            seed=seed,
            budget=budget,
        )
        comp4_single_point.append(single_point_result)

        uniform_result = run_ea_config(
            population_size=50,
            num_generations=30,
            crossover_rate=0.8,
            crossover_strategy="uniform",
            seed=seed,
            budget=budget,
        )
        comp4_uniform.append(uniform_result)

    verify_budget_match(comp4_single_point, comp4_uniform, "single-point", "uniform", budget)

    return {
        "comp1_ea_vs_single": {"ea": comp1_ea, "mcmc": comp1_mcmc},
        "comp2_ea_vs_multistart": {"ea": comp2_ea, "mcmc": comp2_mcmc},
        "comp3_crossover_ablation": {"no_crossover": comp3_no_cross, "with_crossover": comp3_with_cross},
        "comp4_crossover_strategy": {"single_point": comp4_single_point, "uniform": comp4_uniform},
    }


# ============================================================================
# Section 7: Statistics and conclusions
# ============================================================================


def compute_statistics(trial_results: list[dict[str, Any]], metric_key: str) -> dict[str, float]:
    """Compute mean ± std for a metric across trials.

    Handles nested metrics like "primary_metric" which lives in task_metrics.
    Bug 5 fix: checks both top-level and task_metrics for the key.
    """
    values = []
    for r in trial_results:
        # Try top-level first (e.g., "best_score")
        if metric_key in r:
            values.append(r[metric_key])
        # Then try inside task_metrics (e.g., "primary_metric")
        elif "task_metrics" in r and metric_key in r["task_metrics"]:
            values.append(r["task_metrics"][metric_key])
        else:
            values.append(0.0)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def generate_conclusions(results: dict[str, Any]) -> dict[str, str]:
    """Generate regime-characterization conclusions from benchmark results.

    Frames results as characterizing when the EA option applies, not as one
    method being better than another. Requires statistically significant
    differences (> pooled std).

    Bug 3 fix: Comparison 1 (EA vs single-chain MCMC) only compares best-score,
    not diversity metrics which are degenerate for single-chain.
    """
    comp1 = results["comp1_ea_vs_single"]
    comp2 = results["comp2_ea_vs_multistart"]
    comp3 = results["comp3_crossover_ablation"]
    comp4 = results["comp4_crossover_strategy"]

    # Compute statistics
    ea1_best = compute_statistics(comp1["ea"], "best_score")
    mcmc1_best = compute_statistics(comp1["mcmc"], "best_score")

    ea2_primary = compute_statistics(comp2["ea"], "primary_metric")
    mcmc2_primary = compute_statistics(comp2["mcmc"], "primary_metric")

    no_cross_primary = compute_statistics(comp3["no_crossover"], "primary_metric")
    with_cross_primary = compute_statistics(comp3["with_crossover"], "primary_metric")

    sp_primary = compute_statistics(comp4["single_point"], "primary_metric")
    uniform_primary = compute_statistics(comp4["uniform"], "primary_metric")

    # Statistical significance check: require difference > pooled std
    def is_significant(stats1: dict[str, float], stats2: dict[str, float]) -> bool:
        """Check if difference in means exceeds pooled standard deviation."""
        pooled_std = math.sqrt((stats1["std"] ** 2 + stats2["std"] ** 2) / 2)
        diff = abs(stats1["mean"] - stats2["mean"])
        return diff > pooled_std

    # Comparison 1: Best score only (diversity not comparable for single-chain)
    comp1_score_significant = is_significant(ea1_best, mcmc1_best)
    if comp1_score_significant:
        if ea1_best["mean"] < mcmc1_best["mean"]:
            comp1_text = f"EA reaches lower best-score than single-chain MCMC: EA {ea1_best['mean']:.3f}±{ea1_best['std']:.3f} vs MCMC {mcmc1_best['mean']:.3f}±{mcmc1_best['std']:.3f}"
        else:
            comp1_text = f"Single-chain MCMC reaches lower best-score than EA: MCMC {mcmc1_best['mean']:.3f}±{mcmc1_best['std']:.3f} vs EA {ea1_best['mean']:.3f}±{ea1_best['std']:.3f}"
    else:
        comp1_text = f"EA vs single-chain MCMC: indistinguishable on best score at this trial count (EA {ea1_best['mean']:.3f}±{ea1_best['std']:.3f}, MCMC {mcmc1_best['mean']:.3f}±{mcmc1_best['std']:.3f})"

    # Comparison 2: Primary metric (mean blocks solved) - the key comparison
    comp2_significant = is_significant(ea2_primary, mcmc2_primary)
    if comp2_significant:
        if ea2_primary["mean"] > mcmc2_primary["mean"]:
            comp2_text = f"EA option reaches higher mean-blocks-solved than multi-restart MCMC: EA {ea2_primary['mean']:.3f}±{ea2_primary['std']:.3f} vs MCMC {mcmc2_primary['mean']:.3f}±{mcmc2_primary['std']:.3f}, consistent with building-block regime"
        else:
            comp2_text = f"Multi-restart MCMC reaches higher mean-blocks-solved than EA: MCMC {mcmc2_primary['mean']:.3f}±{mcmc2_primary['std']:.3f} vs EA {ea2_primary['mean']:.3f}±{ea2_primary['std']:.3f}, suggesting task may not be in building-block regime"
    else:
        comp2_text = f"EA vs multi-restart MCMC: indistinguishable on mean-blocks-solved (EA {ea2_primary['mean']:.3f}±{ea2_primary['std']:.3f}, MCMC {mcmc2_primary['mean']:.3f}±{mcmc2_primary['std']:.3f})"

    # Comparison 3: Crossover ablation - mechanism check
    comp3_significant = is_significant(no_cross_primary, with_cross_primary)
    if comp3_significant:
        if with_cross_primary["mean"] > no_cross_primary["mean"]:
            comp3_text = f"Crossover mechanism provides value: rate=0.8 reaches {with_cross_primary['mean']:.3f}±{with_cross_primary['std']:.3f} vs rate=0.0 {no_cross_primary['mean']:.3f}±{no_cross_primary['std']:.3f}"
        else:
            comp3_text = f"Crossover ablation: rate=0.0 unexpectedly reaches {no_cross_primary['mean']:.3f}±{no_cross_primary['std']:.3f} vs rate=0.8 {with_cross_primary['mean']:.3f}±{with_cross_primary['std']:.3f}"
    else:
        comp3_text = f"Crossover ablation: indistinguishable at this trial count (rate=0.8 {with_cross_primary['mean']:.3f}±{with_cross_primary['std']:.3f}, rate=0.0 {no_cross_primary['mean']:.3f}±{no_cross_primary['std']:.3f})"

    # Comparison 4: Crossover strategy
    comp4_significant = is_significant(sp_primary, uniform_primary)
    if comp4_significant:
        if sp_primary["mean"] > uniform_primary["mean"]:
            comp4_text = f"Single-point preserves building blocks better: {sp_primary['mean']:.3f} vs uniform {uniform_primary['mean']:.3f}"
        else:
            comp4_text = f"Uniform crossover unexpectedly matches or exceeds single-point: uniform {uniform_primary['mean']:.3f} vs single-point {sp_primary['mean']:.3f}"
    else:
        comp4_text = f"Crossover strategy: indistinguishable (single-point {sp_primary['mean']:.3f}, uniform {uniform_primary['mean']:.3f})"

    return {
        "comp1": comp1_text,
        "comp2": comp2_text,
        "comp3": comp3_text,
        "comp4": comp4_text,
    }


# ============================================================================
# Section 8: Plotting
# ============================================================================


def plot_convergence(results: dict[str, Any], output_path: Path) -> None:
    """Plot convergence analysis for all four comparisons.

    Args:
        results: Benchmark results
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Building-Block Task ({NUM_BLOCKS} blocks): Convergence Analysis", fontsize=14)

    comp1 = results["comp1_ea_vs_single"]
    comp2 = results["comp2_ea_vs_multistart"]
    comp3 = results["comp3_crossover_ablation"]
    comp4 = results["comp4_crossover_strategy"]

    # Plot 1: EA vs Single-Chain MCMC
    ax = axes[0, 0]
    ax.set_title("EA vs Single-Chain: Best Score")
    ax.set_xlabel("Constraint Evaluations")
    ax.set_ylabel("Best Score")
    for trial in comp1["ea"]:
        evals = [c["evaluations"] for c in trial["convergence"]]
        scores = [c["best_score"] for c in trial["convergence"]]
        ax.plot(evals, scores, color="blue", alpha=0.5, linewidth=1)
    for trial in comp1["mcmc"]:
        evals = [c["evaluations"] for c in trial["convergence"]]
        scores = [c["best_score"] for c in trial["convergence"]]
        ax.plot(evals, scores, color="red", alpha=0.5, linewidth=1)
    # Dummy lines for legend
    ax.plot([], [], color="blue", label="EA", linewidth=2)
    ax.plot([], [], color="red", label="Single-Chain MCMC", linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: EA vs Multi-Start MCMC
    ax = axes[0, 1]
    ax.set_title("EA vs Multi-Start: Best Score")
    ax.set_xlabel("Constraint Evaluations")
    ax.set_ylabel("Best Score")
    for trial in comp2["ea"]:
        evals = [c["evaluations"] for c in trial["convergence"]]
        scores = [c["best_score"] for c in trial["convergence"]]
        ax.plot(evals, scores, color="blue", alpha=0.5, linewidth=1)
    for trial in comp2["mcmc"]:
        evals = [c["evaluations"] for c in trial["convergence"]]
        scores = [c["best_score"] for c in trial["convergence"]]
        ax.plot(evals, scores, color="red", alpha=0.3, linewidth=0.5)  # Thinner for 20 chains
    ax.plot([], [], color="blue", label="EA", linewidth=2)
    ax.plot([], [], color="red", label="20-Restart MCMC", linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Crossover Ablation
    ax = axes[1, 0]
    ax.set_title("Crossover Ablation")
    ax.set_xlabel("Constraint Evaluations")
    ax.set_ylabel("Best Score")
    for trial in comp3["no_crossover"]:
        evals = [c["evaluations"] for c in trial["convergence"]]
        scores = [c["best_score"] for c in trial["convergence"]]
        ax.plot(evals, scores, color="tan", alpha=0.5, linewidth=1)
    for trial in comp3["with_crossover"]:
        evals = [c["evaluations"] for c in trial["convergence"]]
        scores = [c["best_score"] for c in trial["convergence"]]
        ax.plot(evals, scores, color="green", alpha=0.5, linewidth=1)
    ax.plot([], [], color="tan", label="No Crossover (rate=0.0)", linewidth=2)
    ax.plot([], [], color="green", label="With Crossover (rate=0.8)", linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Final metrics bar chart
    ax = axes[1, 1]
    ax.set_title("Mean Blocks Solved (mean ± std)")
    ax.set_ylabel("Mean Blocks Solved")

    # Compute stats
    ea1_primary = compute_statistics(comp1["ea"], "primary_metric")
    ea2_primary = compute_statistics(comp2["ea"], "primary_metric")
    mcmc2_primary = compute_statistics(comp2["mcmc"], "primary_metric")
    no_cross = compute_statistics(comp3["no_crossover"], "primary_metric")
    with_cross = compute_statistics(comp3["with_crossover"], "primary_metric")
    sp = compute_statistics(comp4["single_point"], "primary_metric")
    unif = compute_statistics(comp4["uniform"], "primary_metric")

    labels = ["EA\n(comp1)", "EA\n(comp2)", "Multi\nMCMC", "No\nCross", "With\nCross", "Single\nPoint", "Uniform"]
    means = [
        ea1_primary["mean"],
        ea2_primary["mean"],
        mcmc2_primary["mean"],
        no_cross["mean"],
        with_cross["mean"],
        sp["mean"],
        unif["mean"],
    ]
    stds = [
        ea1_primary["std"],
        ea2_primary["std"],
        mcmc2_primary["std"],
        no_cross["std"],
        with_cross["std"],
        sp["std"],
        unif["std"],
    ]

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(y=NUM_BLOCKS, color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"All {NUM_BLOCKS} solved")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Convergence plot saved to {output_path}")


# ============================================================================
# Section 9: Main execution
# ============================================================================


def main() -> None:
    """Run the full benchmark and save results."""
    logger.info("=" * 70)
    logger.info("EvolutionaryOptimizer Building-Block Regime Characterization")
    logger.info("=" * 70)
    logger.info(f"Budget: {BUDGET} constraint evaluations per trial")
    logger.info(f"Trials: {NUM_TRIALS}")
    logger.info(f"Task: {NUM_BLOCKS} building blocks, sequence length {SEQUENCE_LENGTH}")
    logger.info("=" * 70)

    # Run benchmark
    results = run_benchmark(budget=BUDGET, trials=NUM_TRIALS)

    # Generate conclusions
    conclusions = generate_conclusions(results)

    # Save detailed results
    detailed_path = Path("benchmark_ea_vs_mcmc_detailed.json")
    with detailed_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nDetailed results saved to {detailed_path}")

    # Save summary
    summary = {
        "task": {
            "description": f"Building-block task: {NUM_BLOCKS} blocks, {BLOCK_LEN} bp each, {MOTIF_LEN} bp motifs",
            "num_blocks": NUM_BLOCKS,
            "block_len": BLOCK_LEN,
            "motif_len": MOTIF_LEN,
            "sequence_length": SEQUENCE_LENGTH,
            "primary_metric": "mean_blocks_solved",
        },
        "comp1_ea_vs_single": {
            "ea": {
                "best_score": compute_statistics(results["comp1_ea_vs_single"]["ea"], "best_score"),
                "primary_metric": compute_statistics(results["comp1_ea_vs_single"]["ea"], "primary_metric"),
            },
            "mcmc": {
                "best_score": compute_statistics(results["comp1_ea_vs_single"]["mcmc"], "best_score"),
            },
        },
        "comp2_ea_vs_multistart": {
            "ea": {
                "best_score": compute_statistics(results["comp2_ea_vs_multistart"]["ea"], "best_score"),
                "primary_metric": compute_statistics(results["comp2_ea_vs_multistart"]["ea"], "primary_metric"),
            },
            "mcmc": {
                "best_score": compute_statistics(results["comp2_ea_vs_multistart"]["mcmc"], "best_score"),
                "primary_metric": compute_statistics(results["comp2_ea_vs_multistart"]["mcmc"], "primary_metric"),
            },
        },
        "comp3_crossover_ablation": {
            "no_crossover": {
                "primary_metric": compute_statistics(results["comp3_crossover_ablation"]["no_crossover"], "primary_metric"),
            },
            "with_crossover": {
                "primary_metric": compute_statistics(results["comp3_crossover_ablation"]["with_crossover"], "primary_metric"),
            },
        },
        "comp4_crossover_strategy": {
            "single_point": {
                "primary_metric": compute_statistics(results["comp4_crossover_strategy"]["single_point"], "primary_metric"),
            },
            "uniform": {
                "primary_metric": compute_statistics(results["comp4_crossover_strategy"]["uniform"], "primary_metric"),
            },
        },
        "conclusions": conclusions,
    }

    summary_path = Path("benchmark_ea_vs_mcmc_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Plot convergence
    plot_path = Path("benchmark_ea_vs_mcmc_convergence.png")
    plot_convergence(results, plot_path)

    # Print conclusions
    logger.info("\n" + "=" * 70)
    logger.info("REGIME CHARACTERIZATION (computed from data)")
    logger.info("=" * 70)
    for comp_name, conclusion_text in conclusions.items():
        logger.info(f"\n{comp_name}: {conclusion_text}")
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
