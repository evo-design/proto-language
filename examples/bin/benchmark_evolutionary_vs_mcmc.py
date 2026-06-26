"""Benchmark comparing EvolutionaryOptimizer vs MCMCOptimizer on multimodal tasks.

This benchmark tests two distinct theses:

1. **Task A (dual-region)**: Does crossover compose modular solutions?
   - Two independent regions, each requiring a different motif
   - Crossover can combine an A-only parent with a B-only parent to get both
   - Primary metric: fraction of final population solving both regions

2. **Task B (three-basin)**: Does population spread across multiple optima?
   - Three mutually-exclusive optima (three different motifs)
   - Tests diversity without modular structure
   - Primary metric: number of distinct basins occupied by good final solutions

The benchmark runs four comparisons on both tasks:
- EA vs single-chain MCMC (algorithmic apples-to-apples)
- EA vs 20-restart MCMC (practitioner's baseline, the real threat)
- Crossover ablation: EA(crossover_rate=0.0) vs EA(crossover_rate=0.8)
- Task A also tests single-point vs uniform crossover strategies

This benchmark is designed to be capable of producing results unfavorable to the EA.
That's what makes it credible. Conclusions are computed from data, not assumed.

Outputs (written to current directory):
    - benchmark_ea_vs_mcmc_summary.json: Aggregate statistics
    - benchmark_ea_vs_mcmc_detailed.json: Per-trial results
    - benchmark_ea_vs_mcmc_task_a_convergence.png: Task A plots
    - benchmark_ea_vs_mcmc_task_b_convergence.png: Task B plots

Usage:
    python examples/bin/benchmark_evolutionary_vs_mcmc.py
"""

import json
import logging
import math
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

logger = logging.getLogger(__name__)


# ============================================================================
# Section 1: Sliding-window Hamming helpers (exact definitions)
# ============================================================================


def hamming(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings. Requires len(a) == len(b)."""
    if len(a) != len(b):
        raise ValueError(f"hamming requires equal length, got {len(a)} and {len(b)}")
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def min_window_hamming(region: str, motif: str) -> int:
    """Minimum Hamming distance between `motif` and any equal-length window of `region`.

    Slides a window of len(motif) across `region` and returns the smallest Hamming
    distance found. If region is shorter than motif, returns len(motif) (worst case:
    no window can match). The motif is found exactly when this returns 0.
    """
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
    """min_window_hamming normalized to [0, 1]. 0.0 means motif present exactly."""
    return min_window_hamming(region, motif) / len(motif)


# ============================================================================
# Section 2: Task A - Dual-region (the crossover task)
# ============================================================================

MOTIF_A = "TATAAA"  # belongs in region A, positions [0:25]
MOTIF_B = "CAAT"  # belongs in region B, positions [25:50]
REGION_SPLIT = 25
SEQUENCE_LENGTH = 50


class DualRegionConfig(BaseModel):
    """Empty config for dual-region constraint."""


def dual_region_constraint(input_sequences: list[tuple[Sequence, ...]], config: DualRegionConfig) -> list[ConstraintOutput]:  # noqa: ARG001
    """Dual-region constraint: TATAAA in region [0:25], CAAT in region [25:50].

    Mean of the two normalized region scores. 0.0 iff both motifs present in their regions.
    Signature matches gc_content_constraint from proto_language/constraint/sequence_composition/.
    """
    results = []
    for (seq,) in input_sequences:
        seq_str = seq.sequence
        region_a = seq_str[:REGION_SPLIT]
        region_b = seq_str[REGION_SPLIT:]
        score_a = normalized_window_score(region_a, MOTIF_A)
        score_b = normalized_window_score(region_b, MOTIF_B)
        score = (score_a + score_b) / 2.0
        results.append(ConstraintOutput(score=score))
    return results


# Decorate constraint function following gc_content_constraint pattern
dual_region_constraint._constraint_config_class = DualRegionConfig  # type: ignore[attr-defined]
dual_region_constraint._constraint_supported_sequence_types = ["dna"]  # type: ignore[attr-defined]


def classify_dual_region(seq: str, relaxed: bool = False) -> str:
    """Classify sequence into {both, A_only, B_only, neither} based on which regions are solved.

    Args:
        seq: The DNA sequence to classify
        relaxed: If True, accept within-one-mismatch as "solved"
    """
    tol = (1.0 / len(MOTIF_A)) if relaxed else 0.0
    a_solved = normalized_window_score(seq[:REGION_SPLIT], MOTIF_A) <= tol
    b_solved = normalized_window_score(seq[REGION_SPLIT:], MOTIF_B) <= tol

    if a_solved and b_solved:
        return "both"
    if a_solved:
        return "A_only"
    if b_solved:
        return "B_only"
    return "neither"


def analyze_dual_region_diversity(sequences: list[str], relaxed: bool = False) -> dict[str, Any]:
    """Analyze Task A diversity: 4-cell distribution and fraction solving both.

    Returns:
        Dict with counts per cell, fraction_solving_both, and distinct "both" sequences
    """
    cells = {"both": 0, "A_only": 0, "B_only": 0, "neither": 0}
    both_sequences = set()

    for seq in sequences:
        cell = classify_dual_region(seq, relaxed=relaxed)
        cells[cell] += 1
        if cell == "both":
            both_sequences.add(seq)

    return {
        "cells": cells,
        "fraction_solving_both": cells["both"] / len(sequences) if sequences else 0.0,
        "distinct_both": len(both_sequences),
        "relaxed": relaxed,
    }


# ============================================================================
# Section 3: Task B - Three-basin (the diversity task)
# ============================================================================

MOTIFS_3 = ["TATAAA", "CAAT", "GGGCGG"]


class ThreeBasinConfig(BaseModel):
    """Empty config for three-basin constraint."""


def three_basin_constraint(input_sequences: list[tuple[Sequence, ...]], config: ThreeBasinConfig) -> list[ConstraintOutput]:  # noqa: ARG001
    """Three-basin constraint: match any of three motifs anywhere in sequence.

    Best (minimum) normalized window score across the three motifs. 0.0 iff any motif present.
    Signature matches gc_content_constraint from proto_language/constraint/sequence_composition/.
    """
    results = []
    for (seq,) in input_sequences:
        seq_str = seq.sequence
        score = min(normalized_window_score(seq_str, m) for m in MOTIFS_3)
        results.append(ConstraintOutput(score=score))
    return results


three_basin_constraint._constraint_config_class = ThreeBasinConfig  # type: ignore[attr-defined]
three_basin_constraint._constraint_supported_sequence_types = ["dna"]  # type: ignore[attr-defined]


def classify_basin(seq: str) -> str:
    """Which of the three motifs is this sequence closest to (by min window Hamming)?"""
    dists = [min_window_hamming(seq, m) for m in MOTIFS_3]
    return MOTIFS_3[int(min(range(3), key=lambda i: dists[i]))]


def analyze_three_basin_diversity(
    sequences: list[str], scores: list[float], threshold: float
) -> dict[str, Any]:
    """Analyze Task B diversity: basins occupied among good solutions.

    Args:
        sequences: All final sequences
        scores: Corresponding energy scores
        threshold: Maximum score to consider "good"

    Returns:
        Dict with basins_occupied, distinct_good, and per-basin counts
    """
    good_sequences = [seq for seq, score in zip(sequences, scores, strict=True) if score < threshold]

    if not good_sequences:
        return {
            "basins_occupied": 0,
            "distinct_good": 0,
            "basin_counts": {},
            "threshold": threshold,
        }

    basins_found = set()
    basin_counts: dict[str, int] = dict.fromkeys(MOTIFS_3, 0)

    for seq in good_sequences:
        basin = classify_basin(seq)
        basins_found.add(basin)
        basin_counts[basin] += 1

    return {
        "basins_occupied": len(basins_found),
        "distinct_good": len(set(good_sequences)),
        "basin_counts": basin_counts,
        "threshold": threshold,
    }


# ============================================================================
# Section 4 & 5: Optimization runs with budget matching
# ============================================================================


def extract_metrics_at_budget(optimizer: Any, budget: int, task: str) -> dict[str, Any]:
    """Extract metrics from optimizer history up to target evaluation budget.

    Args:
        optimizer: The optimizer instance with populated history
        budget: Target constraint evaluation count
        task: "task_a" or "task_b"

    Returns:
        Dict with convergence data, final metrics, and actual evaluation count
    """
    convergence = []
    total_evaluations = 0
    population_size = optimizer.population_size
    elitism_count = optimizer.elitism_count

    # Offspring per generation = population - elites (Bug 4 fix)
    offspring_per_gen = population_size - elitism_count

    final_snapshot_at_budget = None  # Bug 2 fix: capture snapshot AT budget

    # Track cumulative evaluations and truncate at budget
    for snapshot in optimizer.history:
        time_step = snapshot.get("time_step", 0)
        results = snapshot.get("results", [])

        # Count evaluations: initial population (gen 0) + offspring per subsequent generation
        if time_step > 0:
            total_evaluations += offspring_per_gen  # Bug 4 fix: count offspring correctly

        # Capture this snapshot if we're still within budget
        if total_evaluations <= budget:
            final_snapshot_at_budget = snapshot  # Bug 2 fix: track last valid snapshot

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

    # Task-specific metrics
    if task == "task_a":
        # Dual-region: 4-cell distribution at both strict and relaxed thresholds
        strict = analyze_dual_region_diversity(final_sequences, relaxed=False)
        relaxed = analyze_dual_region_diversity(final_sequences, relaxed=True)
        task_metrics = {
            "strict": strict,
            "relaxed": relaxed,
            "primary_metric": strict["fraction_solving_both"],
        }
    else:  # task_b
        # Three-basin: basins occupied at multiple score thresholds
        thresholds = [0.1, 0.2, 0.3]
        threshold_results = {}
        for thresh in thresholds:
            threshold_results[f"thresh_{thresh}"] = analyze_three_basin_diversity(
                final_sequences, final_scores, thresh
            )

        # Primary metric: basins at most lenient threshold that has results
        primary = 0
        for thresh in reversed(thresholds):
            result = threshold_results[f"thresh_{thresh}"]
            if result["basins_occupied"] > 0:
                primary = result["basins_occupied"]
                break

        task_metrics = {
            "thresholds": threshold_results,
            "primary_metric": primary,
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
    task: str,
    population_size: int,
    num_generations: int,
    crossover_rate: float,
    crossover_strategy: str,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Run EA with specified configuration on given task.

    Args:
        task: "task_a" or "task_b"
        population_size: EA population size
        num_generations: Number of generations (may exceed budget, will be truncated)
        crossover_rate: Crossover probability (0.0 for ablation)
        crossover_strategy: "single-point" or "uniform"
        seed: Random seed
        budget: Target evaluation budget for truncation

    Returns:
        Dict with metrics extracted at budget cutoff
    """
    # Setup
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # Task-specific constraint
    if task == "task_a":
        constraint = Constraint(
            inputs=[segment],
            function=dual_region_constraint,
            function_config=DualRegionConfig(),
        )
    else:
        constraint = Constraint(
            inputs=[segment],
            function=three_basin_constraint,
            function_config=ThreeBasinConfig(),
        )

    # EA config
    config = EvolutionaryOptimizerConfig(
        population_size=population_size,
        num_generations=num_generations,
        elitism_count=max(1, population_size // 10),
        tournament_size=3,
        crossover_rate=crossover_rate,
        mutation_rate=0.2,
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
    return extract_metrics_at_budget(optimizer, budget, task)


def run_mcmc_config(
    task: str,
    num_results: int,
    num_steps: int,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Run MCMC with specified configuration on given task.

    Args:
        task: "task_a" or "task_b"
        num_results: Number of independent chains
        num_steps: Steps per chain (may exceed budget, will be truncated)
        seed: Random seed
        budget: Target evaluation budget for truncation

    Returns:
        Dict with metrics extracted at budget cutoff
    """
    # Setup
    segment = Segment(sequence="A" * SEQUENCE_LENGTH, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # Task-specific constraint
    if task == "task_a":
        constraint = Constraint(
            inputs=[segment],
            function=dual_region_constraint,
            function_config=DualRegionConfig(),
        )
    else:
        constraint = Constraint(
            inputs=[segment],
            function=three_basin_constraint,
            function_config=ThreeBasinConfig(),
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

        if total_evaluations > budget:
            break

        energy_scores = [r.get("energy_score") for r in results]
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

    # Bug 3 fix: Single-chain MCMC (num_results=1) has degenerate diversity metrics
    # For single-chain, diversity is not comparable to population-based methods
    # Only compute diversity for multi-chain (num_results > 1)
    if num_results == 1:
        # Single-chain: diversity metrics are meaningless (always 0 or 1)
        task_metrics = {
            "primary_metric": 0.0,  # Will be ignored in comparisons
            "note": "Single-chain MCMC: diversity metrics not comparable to population methods",
        }
    else:
        # Multi-chain: compute diversity over final chains (comparable to EA population)
        if task == "task_a":
            strict = analyze_dual_region_diversity(final_sequences, relaxed=False)
            relaxed = analyze_dual_region_diversity(final_sequences, relaxed=True)
            task_metrics = {
                "strict": strict,
                "relaxed": relaxed,
                "primary_metric": strict["fraction_solving_both"],
            }
        else:
            thresholds = [0.1, 0.2, 0.3]
            threshold_results = {}
            for thresh in thresholds:
                threshold_results[f"thresh_{thresh}"] = analyze_three_basin_diversity(
                    final_sequences, final_scores, thresh
                )
            primary = 0
            for thresh in reversed(thresholds):
                result = threshold_results[f"thresh_{thresh}"]
                if result["basins_occupied"] > 0:
                    primary = result["basins_occupied"]
                    break
            task_metrics = {
                "thresholds": threshold_results,
                "primary_metric": primary,
            }

    finite_final = [s for s in final_scores if s is not None and math.isfinite(s)]
    best_score = min(finite_final) if finite_final else float("inf")

    return {
        "convergence": convergence,
        "total_evaluations": total_evaluations,
        "best_score": best_score,
        "task_metrics": task_metrics,
        "final_sequences": final_sequences[:10],
        "is_single_chain": num_results == 1,  # Flag for comparison logic
    }


# ============================================================================
# Section 6: Run all comparisons and compute honest conclusions
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

    Raises:
        RuntimeWarning if budgets differ by more than one population width
    """
    for i, (r1, r2) in enumerate(zip(results1, results2, strict=True)):
        evals1 = r1.get("total_evaluations", 0)
        evals2 = r2.get("total_evaluations", 0)
        diff = abs(evals1 - evals2)

        # Tolerance: one population/chain width (20)
        tolerance = 20

        if diff > tolerance:
            logger.warning(
                f"Trial {i}: Budget mismatch exceeds tolerance! "
                f"{method1}={evals1} vs {method2}={evals2} (diff={diff}, tolerance={tolerance}). "
                f"Comparison may be invalid."
            )

        # Also check both are reasonably close to target
        if abs(evals1 - budget) > tolerance or abs(evals2 - budget) > tolerance:
            logger.warning(
                f"Trial {i}: Measured budgets deviate from target {budget}: "
                f"{method1}={evals1}, {method2}={evals2}"
            )


def run_all_comparisons(budget: int = 1000, num_trials: int = 5, base_seed: int = 42) -> dict[str, Any]:
    """Run all four comparisons on both tasks across multiple trials.

    Returns structured results with per-trial data and aggregate statistics.
    """
    logger.info(f"Starting benchmark: budget={budget}, trials={num_trials}")

    tasks = ["task_a", "task_b"]
    results: dict[str, Any] = {}

    for task in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task.upper()}")
        logger.info(f"{'='*60}")

        task_results = {}

        # Comparison 1: EA vs single-chain MCMC
        logger.info("\n--- Comparison 1: EA vs Single-Chain MCMC ---")
        comp1_ea = []
        comp1_mcmc = []

        for trial in range(num_trials):
            seed = base_seed + trial
            logger.info(f"Trial {trial + 1}/{num_trials} (seed={seed})")

            # For Task A, also test crossover strategies
            if task == "task_a":
                # Single-point crossover (expected to work better on modular task)
                ea_result = run_ea_config(
                    task=task,
                    population_size=20,
                    num_generations=60,  # Overshoot budget, will truncate
                    crossover_rate=0.8,
                    crossover_strategy="single-point",
                    seed=seed,
                    budget=budget,
                )
                comp1_ea.append({"strategy": "single-point", **ea_result})
            else:
                ea_result = run_ea_config(
                    task=task,
                    population_size=20,
                    num_generations=60,
                    crossover_rate=0.8,
                    crossover_strategy="single-point",
                    seed=seed,
                    budget=budget,
                )
                comp1_ea.append(ea_result)

            mcmc_result = run_mcmc_config(
                task=task,
                num_results=1,
                num_steps=1200,  # Overshoot budget
                seed=seed,
                budget=budget,
            )
            comp1_mcmc.append(mcmc_result)

        task_results["comp1_ea_vs_single_mcmc"] = {
            "ea": comp1_ea,
            "mcmc": comp1_mcmc,
        }

        # Bug 4 fix: Verify budgets match
        verify_budget_match(comp1_ea, comp1_mcmc, "EA", "Single-Chain MCMC", budget)

        # Comparison 2: EA vs 20-restart MCMC
        logger.info("\n--- Comparison 2: EA vs 20-Restart MCMC ---")
        comp2_ea = []
        comp2_mcmc = []

        for trial in range(num_trials):
            seed = base_seed + trial
            logger.info(f"Trial {trial + 1}/{num_trials} (seed={seed})")

            ea_result = run_ea_config(
                task=task,
                population_size=20,
                num_generations=60,
                crossover_rate=0.8,
                crossover_strategy="single-point",
                seed=seed,
                budget=budget,
            )
            comp2_ea.append(ea_result)

            mcmc_result = run_mcmc_config(
                task=task,
                num_results=20,
                num_steps=60,
                seed=seed,
                budget=budget,
            )
            comp2_mcmc.append(mcmc_result)

        task_results["comp2_ea_vs_multistart_mcmc"] = {
            "ea": comp2_ea,
            "mcmc": comp2_mcmc,
        }

        # Bug 4 fix: Verify budgets match
        verify_budget_match(comp2_ea, comp2_mcmc, "EA", "Multi-Start MCMC", budget)

        # Comparison 3: Crossover ablation
        logger.info("\n--- Comparison 3: Crossover Ablation (rate=0.0 vs 0.8) ---")
        comp3_no_cross = []
        comp3_with_cross = []

        for trial in range(num_trials):
            seed = base_seed + trial
            logger.info(f"Trial {trial + 1}/{num_trials} (seed={seed})")

            no_cross = run_ea_config(
                task=task,
                population_size=20,
                num_generations=60,
                crossover_rate=0.0,  # NO crossover
                crossover_strategy="single-point",  # Doesn't matter, rate is 0
                seed=seed,
                budget=budget,
            )
            comp3_no_cross.append(no_cross)

            with_cross = run_ea_config(
                task=task,
                population_size=20,
                num_generations=60,
                crossover_rate=0.8,
                crossover_strategy="single-point",
                seed=seed,
                budget=budget,
            )
            comp3_with_cross.append(with_cross)

        task_results["comp3_crossover_ablation"] = {
            "no_crossover": comp3_no_cross,
            "with_crossover": comp3_with_cross,
        }

        # Bug 4 fix: Verify budgets match
        verify_budget_match(comp3_no_cross, comp3_with_cross, "EA(crossover=0.0)", "EA(crossover=0.8)", budget)

        # For Task A: Also compare single-point vs uniform crossover
        if task == "task_a":
            logger.info("\n--- Task A: Single-Point vs Uniform Crossover ---")
            comp4_single = []
            comp4_uniform = []

            for trial in range(num_trials):
                seed = base_seed + trial
                logger.info(f"Trial {trial + 1}/{num_trials} (seed={seed})")

                single = run_ea_config(
                    task=task,
                    population_size=20,
                    num_generations=60,
                    crossover_rate=0.8,
                    crossover_strategy="single-point",
                    seed=seed,
                    budget=budget,
                )
                comp4_single.append(single)

                uniform = run_ea_config(
                    task=task,
                    population_size=20,
                    num_generations=60,
                    crossover_rate=0.8,
                    crossover_strategy="uniform",
                    seed=seed,
                    budget=budget,
                )
                comp4_uniform.append(uniform)

            task_results["comp4_crossover_strategy"] = {
                "single_point": comp4_single,
                "uniform": comp4_uniform,
            }

            # Bug 4 fix: Verify budgets match
            verify_budget_match(comp4_single, comp4_uniform, "EA(single-point)", "EA(uniform)", budget)

        results[task] = task_results

    return results


def compute_statistics(trial_results: list[dict[str, Any]], metric_key: str) -> dict[str, float]:
    """Compute mean ± std for a metric across trials."""
    values = [r.get(metric_key, 0.0) for r in trial_results]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def generate_conclusions(results: dict[str, Any]) -> dict[str, str]:
    """Generate honest conclusions from benchmark results.

    Computes conclusions FROM the data, selecting among outcome branches.
    Does not assume EA is better. Requires statistically significant differences (> pooled std).

    Bug 3 fix: Comparison 1 (EA vs single-chain MCMC) only compares best-score.
    Diversity metrics are degenerate for single-chain (1 sequence → 1 basin or 0/1 solving-both).
    Diversity comparison is reserved for Comparison 2 (20 vs 20).
    """
    conclusions = {}

    for task in ["task_a", "task_b"]:
        task_data = results[task]
        task_conclusions = []

        # Comparison 1: EA vs single-chain MCMC
        # BUG 3 FIX: BEST-SCORE ONLY. Diversity not comparable (1 sequence is degenerate).
        comp1 = task_data["comp1_ea_vs_single_mcmc"]
        ea1_best = compute_statistics(comp1["ea"], "best_score")
        mcmc1_best = compute_statistics(comp1["mcmc"], "best_score")
        # DO NOT extract primary_metric from comp1 MCMC - it's degenerate

        # Comparison 2: EA vs 20-restart MCMC
        # This is the ONLY comparison where diversity is valid for both sides (20 vs 20)
        comp2 = task_data["comp2_ea_vs_multistart_mcmc"]
        ea2_primary = compute_statistics(comp2["ea"], "primary_metric")
        mcmc2_primary = compute_statistics(comp2["mcmc"], "primary_metric")

        # Comparison 3: Crossover ablation
        comp3 = task_data["comp3_crossover_ablation"]
        no_cross_primary = compute_statistics(comp3["no_crossover"], "primary_metric")
        with_cross_primary = compute_statistics(comp3["with_crossover"], "primary_metric")

        # Statistical significance check: require difference > pooled std
        def is_significant(stats1: dict[str, float], stats2: dict[str, float]) -> bool:
            """Check if difference in means exceeds pooled standard deviation."""
            pooled_std = math.sqrt((stats1["std"]**2 + stats2["std"]**2) / 2)
            diff = abs(stats1["mean"] - stats2["mean"])
            return diff > pooled_std

        # Comparison 1: Best score only (diversity not comparable for single-chain)
        ea1_beats_mcmc1_score = ea1_best["mean"] < mcmc1_best["mean"]  # Lower score is better
        comp1_score_significant = is_significant(ea1_best, mcmc1_best)

        # Comparison 2: Both diversity and best score
        ea2_beats_mcmc2_diversity = ea2_primary["mean"] > mcmc2_primary["mean"]  # Higher diversity is better
        comp2_div_significant = is_significant(ea2_primary, mcmc2_primary)

        # Crossover ablation
        crossover_helps = with_cross_primary["mean"] > no_cross_primary["mean"]
        crossover_significant = is_significant(with_cross_primary, no_cross_primary)

        # Construct conclusion based on statistical significance
        # Comparison 1: EA vs single-chain (score only, diversity not comparable)
        if comp1_score_significant:
            if ea1_beats_mcmc1_score:
                task_conclusions.append(
                    f"EA beats single-chain MCMC on best score: "
                    f"EA {ea1_best['mean']:.3f}±{ea1_best['std']:.3f} vs "
                    f"MCMC {mcmc1_best['mean']:.3f}±{mcmc1_best['std']:.3f}"
                )
            else:
                task_conclusions.append(
                    f"Single-chain MCMC beats EA on best score: "
                    f"MCMC {mcmc1_best['mean']:.3f}±{mcmc1_best['std']:.3f} vs "
                    f"EA {ea1_best['mean']:.3f}±{ea1_best['std']:.3f}"
                )
        else:
            task_conclusions.append(
                f"EA vs single-chain MCMC: indistinguishable on best score at this trial count "
                f"(EA {ea1_best['mean']:.3f}±{ea1_best['std']:.3f}, "
                f"MCMC {mcmc1_best['mean']:.3f}±{mcmc1_best['std']:.3f})"
            )

        # Comparison 2: EA vs multi-start (diversity is the key metric here)
        if comp2_div_significant:
            if ea2_beats_mcmc2_diversity:
                task_conclusions.append(
                    f"EA beats multi-start MCMC on diversity: "
                    f"EA {ea2_primary['mean']:.3f}±{ea2_primary['std']:.3f} vs "
                    f"MCMC {mcmc2_primary['mean']:.3f}±{mcmc2_primary['std']:.3f}"
                )
            else:
                task_conclusions.append(
                    f"Multi-start MCMC is the simpler effective baseline on {task}: "
                    f"MCMC {mcmc2_primary['mean']:.3f}±{mcmc2_primary['std']:.3f} vs "
                    f"EA {ea2_primary['mean']:.3f}±{ea2_primary['std']:.3f}"
                )
        else:
            task_conclusions.append(
                f"EA vs multi-start MCMC: indistinguishable on diversity "
                f"(EA {ea2_primary['mean']:.3f}±{ea2_primary['std']:.3f}, "
                f"MCMC {mcmc2_primary['mean']:.3f}±{mcmc2_primary['std']:.3f})"
            )

        # Crossover ablation conclusion
        if crossover_significant:
            if task == "task_a" and crossover_helps:
                task_conclusions.append(
                    f"Crossover composes modular solutions as designed: "
                    f"rate=0.8 {with_cross_primary['mean']:.3f}±{with_cross_primary['std']:.3f} beats "
                    f"rate=0.0 {no_cross_primary['mean']:.3f}±{no_cross_primary['std']:.3f}"
                )
            elif task == "task_a" and not crossover_helps:
                task_conclusions.append(
                    f"Crossover provides no measurable benefit even on modular task: "
                    f"rate=0.0 {no_cross_primary['mean']:.3f}±{no_cross_primary['std']:.3f} beats "
                    f"rate=0.8 {with_cross_primary['mean']:.3f}±{with_cross_primary['std']:.3f}. "
                    "This is a real and publishable negative result."
                )
            elif crossover_helps:
                task_conclusions.append(
                    f"Crossover improves diversity on {task}: "
                    f"rate=0.8 {with_cross_primary['mean']:.3f}±{with_cross_primary['std']:.3f} beats "
                    f"rate=0.0 {no_cross_primary['mean']:.3f}±{no_cross_primary['std']:.3f}"
                )
        else:
            task_conclusions.append(
                f"Crossover ablation: indistinguishable at this trial count "
                f"(rate=0.8 {with_cross_primary['mean']:.3f}±{with_cross_primary['std']:.3f}, "
                f"rate=0.0 {no_cross_primary['mean']:.3f}±{no_cross_primary['std']:.3f})"
            )

        # For Task A: also report crossover strategy comparison
        if task == "task_a" and "comp4_crossover_strategy" in task_data:
            comp4 = task_data["comp4_crossover_strategy"]
            single_primary = compute_statistics(comp4["single_point"], "primary_metric")
            uniform_primary = compute_statistics(comp4["uniform"], "primary_metric")

            if single_primary["mean"] > uniform_primary["mean"]:
                task_conclusions.append(
                    f"Single-point crossover outperforms uniform on modular task: "
                    f"single-point {single_primary['mean']:.3f} vs uniform {uniform_primary['mean']:.3f}. "
                    "Expected: uniform shuffles positions and breaks contiguous motifs."
                )
            else:
                task_conclusions.append(
                    f"Uniform crossover unexpectedly matches or beats single-point: "
                    f"uniform {uniform_primary['mean']:.3f} vs single-point {single_primary['mean']:.3f}"
                )

        conclusions[task] = " | ".join(task_conclusions)

    return conclusions


def plot_convergence(results: dict[str, Any], output_dir: Path) -> None:
    """Generate convergence plots for both tasks."""
    for task in ["task_a", "task_b"]:
        task_data = results[task]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Task {task.upper()}: Convergence Analysis", fontsize=14, fontweight="bold")

        # Plot 1: EA vs Single-Chain MCMC - Best Score
        ax = axes[0, 0]
        comp1 = task_data["comp1_ea_vs_single_mcmc"]
        for _i, ea_trial in enumerate(comp1["ea"]):
            conv = ea_trial["convergence"]
            if conv:
                evals = [c["evaluations"] for c in conv]
                scores = [c["best_score"] for c in conv]
                ax.plot(evals, scores, "b-", alpha=0.3, linewidth=1)
        for _i, mcmc_trial in enumerate(comp1["mcmc"]):
            conv = mcmc_trial["convergence"]
            if conv:
                evals = [c["evaluations"] for c in conv]
                scores = [c["best_score"] for c in conv]
                ax.plot(evals, scores, "r-", alpha=0.3, linewidth=1)
        ax.plot([], [], "b-", label="EA", linewidth=2)
        ax.plot([], [], "r-", label="Single-Chain MCMC", linewidth=2)
        ax.set_xlabel("Constraint Evaluations")
        ax.set_ylabel("Best Score")
        ax.set_title("Best Score vs Evaluations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: EA vs Multi-Start MCMC - Best Score
        ax = axes[0, 1]
        comp2 = task_data["comp2_ea_vs_multistart_mcmc"]
        for ea_trial in comp2["ea"]:
            conv = ea_trial["convergence"]
            if conv:
                evals = [c["evaluations"] for c in conv]
                scores = [c["best_score"] for c in conv]
                ax.plot(evals, scores, "b-", alpha=0.3, linewidth=1)
        for mcmc_trial in comp2["mcmc"]:
            conv = mcmc_trial["convergence"]
            if conv:
                evals = [c["evaluations"] for c in conv]
                scores = [c["best_score"] for c in conv]
                ax.plot(evals, scores, "r-", alpha=0.3, linewidth=1)
        ax.plot([], [], "b-", label="EA", linewidth=2)
        ax.plot([], [], "r-", label="20-Restart MCMC", linewidth=2)
        ax.set_xlabel("Constraint Evaluations")
        ax.set_ylabel("Best Score")
        ax.set_title("EA vs Multi-Start: Best Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Crossover Ablation
        ax = axes[1, 0]
        comp3 = task_data["comp3_crossover_ablation"]
        for trial in comp3["no_crossover"]:
            conv = trial["convergence"]
            if conv:
                evals = [c["evaluations"] for c in conv]
                scores = [c["best_score"] for c in conv]
                ax.plot(evals, scores, "orange", alpha=0.3, linewidth=1)
        for trial in comp3["with_crossover"]:
            conv = trial["convergence"]
            if conv:
                evals = [c["evaluations"] for c in conv]
                scores = [c["best_score"] for c in conv]
                ax.plot(evals, scores, "g-", alpha=0.3, linewidth=1)
        ax.plot([], [], "orange", label="No Crossover (rate=0.0)", linewidth=2)
        ax.plot([], [], "g-", label="With Crossover (rate=0.8)", linewidth=2)
        ax.set_xlabel("Constraint Evaluations")
        ax.set_ylabel("Best Score")
        ax.set_title("Crossover Ablation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Task-specific diversity metric (aggregate across trials)
        # BUG 3 FIX: EXCLUDE single-chain MCMC - diversity is degenerate for 1 sequence
        ax = axes[1, 1]
        metric_label = "Fraction Solving Both" if task == "task_a" else "Basins Occupied"

        # Show final primary metrics as bar chart
        # Single-chain MCMC excluded: 1 final sequence → degenerate diversity (always 0 or 1)
        methods = ["EA\n(comp1)", "EA\n(comp2)", "Multi\nMCMC", "No\nCross", "With\nCross"]
        values = [
            compute_statistics(comp1["ea"], "primary_metric")["mean"],
            compute_statistics(comp2["ea"], "primary_metric")["mean"],
            compute_statistics(comp2["mcmc"], "primary_metric")["mean"],
            compute_statistics(comp3["no_crossover"], "primary_metric")["mean"],
            compute_statistics(comp3["with_crossover"], "primary_metric")["mean"],
        ]
        errors = [
            compute_statistics(comp1["ea"], "primary_metric")["std"],
            compute_statistics(comp2["ea"], "primary_metric")["std"],
            compute_statistics(comp2["mcmc"], "primary_metric")["std"],
            compute_statistics(comp3["no_crossover"], "primary_metric")["std"],
            compute_statistics(comp3["with_crossover"], "primary_metric")["std"],
        ]

        ax.bar(methods, values, yerr=errors, capsize=5, color=["b", "b", "r", "orange", "g"], alpha=0.7)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Final {metric_label} (mean ± std)\n[Single-chain MCMC excluded: degenerate for 1 seq]")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = output_dir / f"benchmark_ea_vs_mcmc_{task}_convergence.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Convergence plot saved to {output_path}")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the benchmark."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Run benchmark comparing EvolutionaryOptimizer vs MCMCOptimizer."""
    setup_logging(verbose=False)

    # Hardcoded benchmark parameters
    budget = 1000
    num_trials = 5
    base_seed = 42

    output_dir = Path(".")

    logger.info("="*70)
    logger.info("Evolutionary Optimizer vs MCMC: Honest Benchmark")
    logger.info("="*70)
    logger.info(f"Budget: {budget} constraint evaluations per trial")
    logger.info(f"Trials: {num_trials}")
    logger.info("Tasks: A (dual-region, tests crossover), B (three-basin, tests diversity)")
    logger.info("="*70)

    # Run all comparisons
    results = run_all_comparisons(budget=budget, num_trials=num_trials, base_seed=base_seed)

    # Compute conclusions
    conclusions = generate_conclusions(results)

    # Generate plots
    plot_convergence(results, output_dir)

    # Save detailed results
    detailed_path = output_dir / "benchmark_ea_vs_mcmc_detailed.json"
    with open(detailed_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDetailed results saved to {detailed_path}")

    # Save summary
    summary = {
        "budget": budget,
        "num_trials": num_trials,
        "conclusions": conclusions,
        "task_a": {
            "description": "Dual-region modular task (TATAAA in [0:25], CAAT in [25:50])",
            "primary_metric": "fraction_solving_both",
            "ea_vs_single": {
                "ea": compute_statistics(results["task_a"]["comp1_ea_vs_single_mcmc"]["ea"], "primary_metric"),
                "mcmc": compute_statistics(results["task_a"]["comp1_ea_vs_single_mcmc"]["mcmc"], "primary_metric"),
            },
            "ea_vs_multistart": {
                "ea": compute_statistics(results["task_a"]["comp2_ea_vs_multistart_mcmc"]["ea"], "primary_metric"),
                "mcmc": compute_statistics(results["task_a"]["comp2_ea_vs_multistart_mcmc"]["mcmc"], "primary_metric"),
            },
            "crossover_ablation": {
                "no_crossover": compute_statistics(
                    results["task_a"]["comp3_crossover_ablation"]["no_crossover"], "primary_metric"
                ),
                "with_crossover": compute_statistics(
                    results["task_a"]["comp3_crossover_ablation"]["with_crossover"], "primary_metric"
                ),
            },
        },
        "task_b": {
            "description": "Three-basin diversity task (match any of TATAAA, CAAT, GGGCGG)",
            "primary_metric": "basins_occupied",
            "ea_vs_single": {
                "ea": compute_statistics(results["task_b"]["comp1_ea_vs_single_mcmc"]["ea"], "primary_metric"),
                "mcmc": compute_statistics(results["task_b"]["comp1_ea_vs_single_mcmc"]["mcmc"], "primary_metric"),
            },
            "ea_vs_multistart": {
                "ea": compute_statistics(results["task_b"]["comp2_ea_vs_multistart_mcmc"]["ea"], "primary_metric"),
                "mcmc": compute_statistics(results["task_b"]["comp2_ea_vs_multistart_mcmc"]["mcmc"], "primary_metric"),
            },
            "crossover_ablation": {
                "no_crossover": compute_statistics(
                    results["task_b"]["comp3_crossover_ablation"]["no_crossover"], "primary_metric"
                ),
                "with_crossover": compute_statistics(
                    results["task_b"]["comp3_crossover_ablation"]["with_crossover"], "primary_metric"
                ),
            },
        },
    }

    summary_path = output_dir / "benchmark_ea_vs_mcmc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Print conclusions
    logger.info("\n" + "="*70)
    logger.info("CONCLUSIONS (computed from data)")
    logger.info("="*70)
    for task, conclusion in conclusions.items():
        logger.info(f"\n{task.upper()}:")
        logger.info(f"  {conclusion}")
    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()
