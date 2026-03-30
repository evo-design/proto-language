"""
Timing comparison for Evo2 single-segment beam search with and without KV caching.

This script compares the performance of single-segment beam search optimization
with use_kv_caching enabled vs disabled to demonstrate the speedup from KV caching.

The BeamSearchOptimizer generates a single long segment by splitting it into beams
of `beam_length` tokens and performing beam search at each beam boundary.

Usage:
    python examples/scripts/beam_search_kv_caching.py
"""

import time

import numpy as np

from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
    GCContentConfig,
)
from proto_language.language.core import Constraint, Construct, Segment
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.language.optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
)

# ==============================
# GLOBAL CONFIGURATION VARIABLES
# ==============================

# Beam search parameters
TOTAL_TOKEN_COUNT: int = 1_000
BEAM_LENGTH: int = 100
NUM_RESULTS: int = 2
N_PROPOSALS_PER_RESULT: int = 2

# Score aggregation method: "mean" or "last"
SCORE_BY: str = "mean"

# Initial prompt for beam search
INITIAL_PROMPT: str = "ATCGATCGATCG"

# Target GC content for constraint (percentage)
TARGET_GC_MIN: float = 40.0
TARGET_GC_MAX: float = 60.0

# Number of timing runs for averaging
NUM_TIMING_RUNS: int = 1

# ============================================================================


def run_beam_search(
    use_kv_caching: bool,
    num_results: int,
    proposals_per_result: int,
    beam_length: int,
    total_token_count: int,
    prompt: str,
    target_gc_min: float,
    target_gc_max: float,
    score_by: str = "mean",
    verbose: bool = False
) -> tuple[float, list[str]]:
    """
    Run single-segment beam search optimization and return elapsed time and sequences.

    Args:
        use_kv_caching: Whether to enable KV caching
        num_results: Number of result sequences to return
        proposals_per_result: Proposals to generate per result
        beam_length: Number of tokens to generate per beam
        total_token_count: Total tokens to generate
        prompt: Initial prompt for beam search
        target_gc_min: Minimum target GC content (percentage)
        target_gc_max: Maximum target GC content (percentage)
        score_by: Score aggregation method ("mean" or "last")
        verbose: Whether to print progress

    Returns:
        Tuple of (elapsed_time, generated_sequences)
    """
    # Create a single segment for the full sequence
    segment_length = len(prompt) + total_token_count
    segment = Segment(length=segment_length, sequence_type="dna", label="full_sequence")
    construct = Construct(segments=[segment])

    # Create generator
    gen_config = Evo2GeneratorConfig(
        prompts=[prompt],
        prepend_prompt=True,
        stop_at_eos=False,
    )
    generator = Evo2Generator(config=gen_config)
    generator.assign(segment)

    # Create constraint
    gc_config = GCContentConfig(min_gc=target_gc_min, max_gc=target_gc_max)
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=gc_config,
    )

    # Create optimizer
    optimizer_config = BeamSearchOptimizerConfig(
        prompt=prompt,
        beam_length=beam_length,
        num_results=num_results,
        proposals_per_result=proposals_per_result,
        score_by=score_by,
        use_kv_caching=use_kv_caching,
        prepend_prompt=True,
        verbose=verbose,
    )

    optimizer = BeamSearchOptimizer(
        target_segment=segment,
        constructs=[construct],
        generators=[generator],
        constraints=[constraint],
        config=optimizer_config,
    )

    # Time the optimization
    start_time = time.time()
    try:
        optimizer.run()
        elapsed_time = time.time() - start_time

        # Get generated sequences
        sequences = [seq.sequence for seq in segment.result_sequences]
        return elapsed_time, sequences

    except Exception as e:
        print("\nERROR during beam search optimization:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("\nThis may be due to:")
        print("  - GPU memory issues (try reducing TOTAL_TOKEN_COUNT or BEAM_LENGTH)")
        print("  - CUDA errors (check GPU status with 'nvidia-smi')")
        print("  - Model loading issues (check model cache)")
        raise


def main():
    """Run timing comparison between cached and non-cached beam search."""
    num_beams = (TOTAL_TOKEN_COUNT + BEAM_LENGTH - 1) // BEAM_LENGTH

    print("=" * 80)
    print("EVO2 SINGLE-SEGMENT BEAM SEARCH KV CACHING TIMING COMPARISON")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Total sequence length (including prompt): {TOTAL_TOKEN_COUNT + len(INITIAL_PROMPT):,}")
    print(f"  Total tokens to generate: {TOTAL_TOKEN_COUNT:,}")
    print(f"  Tokens per beam: {BEAM_LENGTH:,}")
    print(f"  Number of beams: {num_beams}")
    print(f"  Num results: {NUM_RESULTS}")
    print(f"  Proposals per result: {N_PROPOSALS_PER_RESULT}")
    print(f"  Score aggregation: {SCORE_BY}")
    print(f"  Target GC content: {TARGET_GC_MIN:.1f}% - {TARGET_GC_MAX:.1f}%")
    print(f"  Number of timing runs: {NUM_TIMING_RUNS}")
    print()

    # Run WITHOUT KV caching
    print("-" * 80)
    print("Running beam search WITHOUT KV caching...")
    print("-" * 80)
    uncached_times = []

    for run in range(NUM_TIMING_RUNS):
        print(f"\nRun {run + 1}/{NUM_TIMING_RUNS}...")
        elapsed, sequences = run_beam_search(
            use_kv_caching=False,
            num_results=NUM_RESULTS,
            proposals_per_result=N_PROPOSALS_PER_RESULT,
            beam_length=BEAM_LENGTH,
            total_token_count=TOTAL_TOKEN_COUNT,
            prompt=INITIAL_PROMPT,
            target_gc_min=TARGET_GC_MIN,
            target_gc_max=TARGET_GC_MAX,
            score_by=SCORE_BY,
            verbose=True,
        )
        uncached_times.append(elapsed)
        print(f"  Completed in {elapsed:.2f} seconds")

    # Run WITH KV caching
    print()
    print("-" * 80)
    print("Running beam search WITH KV caching...")
    print("-" * 80)
    cached_times = []
    cached_sequences = None

    for run in range(NUM_TIMING_RUNS):
        print(f"\nRun {run + 1}/{NUM_TIMING_RUNS}...")
        elapsed, sequences = run_beam_search(
            use_kv_caching=True,
            num_results=NUM_RESULTS,
            proposals_per_result=N_PROPOSALS_PER_RESULT,
            beam_length=BEAM_LENGTH,
            total_token_count=TOTAL_TOKEN_COUNT,
            prompt=INITIAL_PROMPT,
            target_gc_min=TARGET_GC_MIN,
            target_gc_max=TARGET_GC_MAX,
            score_by=SCORE_BY,
            verbose=True,
        )
        cached_times.append(elapsed)
        cached_sequences = sequences
        print(f"  Completed in {elapsed:.2f} seconds")

    # Calculate statistics
    cached_mean = np.mean(cached_times)
    cached_std = np.std(cached_times)
    uncached_mean = np.mean(uncached_times)
    uncached_std = np.std(uncached_times)
    speedup = uncached_mean / cached_mean if cached_mean > 0 else float('inf')

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print("WITH KV Caching:")
    print(f"  Mean time: {cached_mean:.2f} +/- {cached_std:.2f} seconds")
    print(f"  All runs: {[f'{t:.2f}s' for t in cached_times]}")
    print()
    print("WITHOUT KV Caching:")
    print(f"  Mean time: {uncached_mean:.2f} +/- {uncached_std:.2f} seconds")
    print(f"  All runs: {[f'{t:.2f}s' for t in uncached_times]}")
    print()
    print(f"SPEEDUP with KV caching: {speedup:.2f}x")
    print(f"Time saved: {uncached_mean - cached_mean:.2f} seconds ({(1 - cached_mean/uncached_mean)*100:.1f}% reduction)")
    print()

    # Print sequence info
    print("-" * 80)
    print("GENERATED SEQUENCES (with KV caching)")
    print("-" * 80)
    if cached_sequences:
        for i, seq in enumerate(cached_sequences):
            gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100 if seq else 0
            print(f"  Beam {i + 1}: length={len(seq)}, GC={gc_content:.1f}%")
            print(f"    First 50bp: {seq[:50]}...")
            print(f"    Last 50bp: ...{seq[-50:]}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
