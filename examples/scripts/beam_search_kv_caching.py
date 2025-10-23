"""
Timing comparison for Evo2 beam search with and without KV caching.

This script compares the performance of beam search optimization with cached_generation
enabled vs disabled to demonstrate the speedup from KV caching.
"""

import time
import numpy as np

from proto_language.language.core import Construct, Segment, Constraint
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.language.optimizer import BeamSearchOptimizer, BeamSearchOptimizerConfig
from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES - Edit these for easy experimentation
# ============================================================================

# Beam search parameters
TOTAL_TOKEN_COUNT: int = 1_000 
BEAM_LENGTH: int = 200
BEAM_WIDTH: int = 2
N_CANDIDATES_PER_BEAM: int = 2

# Initial prompt for beam search
INITIAL_PROMPT: str = "ATCGATCGATCG"

# Target GC content for constraint
TARGET_GC_CONTENT: float = 0.5

# Number of timing runs for averaging
NUM_TIMING_RUNS: int = 2

# ============================================================================


def run_beam_search(
    use_kv_caching: bool,
    beam_width: int,
    candidates_per_beam: int,
    beam_length: int,
    total_token_count: int,
    prompt: str,
    target_gc: float,
    verbose: bool = False
) -> float:
    """
    Run beam search optimization and return elapsed time.

    Args:
        use_kv_caching: Whether to enable KV caching
        beam_width: Number of beams to maintain (K)
        candidates_per_beam: Candidates to generate per beam (N)
        chunk_size: Size of each segment
        desired_token_count: Total tokens to generate
        prompt: Initial prompt for beam search
        target_gc: Target GC content
        verbose: Whether to print progress

    Returns:
        Elapsed time in seconds
    """
    # Create segments - one for each chunk
    num_segments = total_token_count // beam_length
    segments = [Segment(sequence="", label=f"chunk_{i+1}") for i in range(num_segments)]
    construct = Construct(segments=segments)

    # Create generator
    gen_config = Evo2GeneratorConfig(
        prompts=[prompt],
        num_tokens=beam_length,  # Tokens per segment
    )
    generator = Evo2Generator(config=gen_config)

    # Assign generator to first segment (required for validation)
    generator._assigned_segment = segments[0]

    # Create constraint
    gc_config = GCContentConfig(
        min_gc=target_gc * 100 - 10,  # +/- 10% range around target
        max_gc=target_gc * 100 + 10
    )
    constraint = Constraint(
        inputs=[segments[0]],  # Will be updated dynamically by beam search
        scoring_function=gc_content_constraint,
        scoring_function_config=gc_config
    )

    # Create optimizer
    optimizer_config = BeamSearchOptimizerConfig(
        beam_width=beam_width,
        candidates_per_beam=candidates_per_beam,
        use_kv_caching=use_kv_caching,
        verbose=verbose
    )

    optimizer = BeamSearchOptimizer(
        construct=construct,
        generator=generator,
        prompt=prompt,
        constraints=[constraint],
        config=optimizer_config
    )

    # Time the optimization
    start_time = time.time()
    try:
        optimizer.run()
        elapsed_time = time.time() - start_time
        return elapsed_time
    except Exception as e:
        print(f"\nERROR during beam search optimization:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"\nThis may be due to:")
        print(f"  - GPU memory issues (try reducing DESIRED_TOKEN_COUNT or N_CHUNKS_PER_SAMPLE)")
        print(f"  - CUDA errors (check GPU status with 'nvidia-smi')")
        print(f"  - Model loading issues (check model cache)")
        raise


def main():
    """Run timing comparison between cached and non-cached beam search."""

    print("=" * 80)
    print("EVO2 BEAM SEARCH KV CACHING TIMING COMPARISON")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Desired token count: {TOTAL_TOKEN_COUNT:,}")
    print(f"  Chunk size: {BEAM_LENGTH:,}")
    print(f"  Beam width (n_samples_retained): {BEAM_WIDTH}")
    print(f"  Candidates per beam (n_chunks_per_sample): {N_CANDIDATES_PER_BEAM}")
    print(f"  Number of segments: {TOTAL_TOKEN_COUNT // BEAM_LENGTH}")
    print(f"  Number of timing runs: {NUM_TIMING_RUNS}")
    print()

    # Run WITHOUT KV caching
    print()
    print("-" * 80)
    print("Running beam search WITHOUT KV caching...")
    print("-" * 80)
    uncached_times = []

    for run in range(NUM_TIMING_RUNS):
        print(f"\nRun {run + 1}/{NUM_TIMING_RUNS}...")
        elapsed = run_beam_search(
            use_kv_caching=False,
            beam_width=BEAM_WIDTH,
            candidates_per_beam=N_CANDIDATES_PER_BEAM,
            beam_length=BEAM_LENGTH,
            total_token_count=TOTAL_TOKEN_COUNT,
            prompt=INITIAL_PROMPT,
            target_gc=TARGET_GC_CONTENT,
            verbose=False
        )
        uncached_times.append(elapsed)
        print(f"  Completed in {elapsed:.2f} seconds")

    # Run WITH KV caching
    print("-" * 80)
    print("Running beam search WITH KV caching...")
    print("-" * 80)
    cached_times = []

    for run in range(NUM_TIMING_RUNS):
        print(f"\nRun {run + 1}/{NUM_TIMING_RUNS}...")
        elapsed = run_beam_search(
            use_kv_caching=True,
            beam_width=BEAM_WIDTH,
            candidates_per_beam=N_CANDIDATES_PER_BEAM,
            beam_length=BEAM_LENGTH,
            total_token_count=TOTAL_TOKEN_COUNT,
            prompt=INITIAL_PROMPT,
            target_gc=TARGET_GC_CONTENT,
            verbose=False
        )
        cached_times.append(elapsed)
        print(f"  Completed in {elapsed:.2f} seconds")

    # Calculate statistics
    cached_mean = np.mean(cached_times)
    cached_std = np.std(cached_times)
    uncached_mean = np.mean(uncached_times)
    uncached_std = np.std(uncached_times)
    speedup = uncached_mean / cached_mean

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"WITH KV Caching:")
    print(f"  Mean time: {cached_mean:.2f} +/- {cached_std:.2f} seconds")
    print(f"  All runs: {[f'{t:.2f}s' for t in cached_times]}")
    print()
    print(f"WITHOUT KV Caching:")
    print(f"  Mean time: {uncached_mean:.2f} +/- {uncached_std:.2f} seconds")
    print(f"  All runs: {[f'{t:.2f}s' for t in uncached_times]}")
    print()
    print(f"SPEEDUP with KV caching: {speedup:.2f}x")
    print(f"Time saved: {uncached_mean - cached_mean:.2f} seconds ({(1 - cached_mean/uncached_mean)*100:.1f}% reduction)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
