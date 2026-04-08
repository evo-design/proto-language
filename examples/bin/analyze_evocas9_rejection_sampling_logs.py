"""Analyze evocas9 rejection sampling SLURM logs for per-filter pass rates.

Parses the 8-stage filter pipeline from evocas9_rejection_sampling.py SLURM logs and
reports per-filter pass rates, both per-job and aggregated across all jobs.

These logs are produced by evocas9_rejection_sampling.py (examples/scripts/evocas9_rejection_sampling.py),
which generates Cas9 proposals via rejection sampling optimization on SLURM. Each job writes
a log file named slurm_evocas9_rejection_sampling_*_{SLURM_JOB_ID}.log.

The 8-stage filter pipeline:
  1. orf          : ORF >= 3000 nt
  2. cas9_phmm   : Cas9 profile HMM match
  3. crispr_array : Contains CRISPR repeat
  4. identity     : Sequence identity to SpCas9 within threshold
  5. gap_gini     : Gap distribution (Gini coefficient)
  6. domain       : Required domains (RuvC + HNH)
  7. tracr        : Has tracrRNA
  8. structure    : AF3 pLDDT / radius of gyration / helix length

Usage:
    # Auto-discover logs in current directory:
    python examples/bin/analyze_evocas9_rejection_sampling_logs.py

    # Specify log files explicitly:
    python examples/bin/analyze_evocas9_rejection_sampling_logs.py slurm_evocas9_rejection_sampling_150_1689545.log

    # Scan a directory:
    python examples/bin/analyze_evocas9_rejection_sampling_logs.py --log-dir /path/to/logs/
"""

import argparse
import glob
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Glob pattern for evocas9 rejection sampling SLURM log files.
LOG_GLOB = "slurm_evocas9_rejection_sampling_*.log"

# Ordered filter names matching the evocas9 rejection sampling pipeline.
FILTER_ORDER = [
    "orf",
    "cas9_phmm",
    "crispr_array",
    "identity",
    "gap_gini",
    "domain",
    "tracr",
    "structure",
]

FILTER_DESCRIPTIONS = {
    "orf": "ORF >= 3000 nt",
    "cas9_phmm": "Cas9 profile HMM",
    "crispr_array": "CRISPR repeat",
    "identity": "Sequence identity",
    "gap_gini": "Gap distribution",
    "domain": "RuvC + HNH domains",
    "tracr": "tracrRNA detected",
    "structure": "AF3 pLDDT/RG/helix",
}

# Regex to match filter lines like "orf_filter: 38/150 have ORFs"
FILTER_PATTERN = re.compile(r"INFO:\s+(?:Filter \d+: )?(\w+?)(?:_filter)?:\s+(\d+)/(\d+)\s+")


def parse_log(log_path: Path) -> dict[str, dict[str, int]]:
    """Parse a single SLURM log and return per-filter stats.

    Returns dict mapping filter name -> {"passed": N, "total": M, "batches": B}.
    """
    text = log_path.read_text()
    stats = defaultdict(lambda: {"passed": 0, "total": 0, "batches": 0})

    for m in FILTER_PATTERN.finditer(text):
        name = m.group(1)
        passed = int(m.group(2))
        total = int(m.group(3))
        stats[name]["passed"] += passed
        stats[name]["total"] += total
        stats[name]["batches"] += 1

    return dict(stats)


def extract_job_id(log_path: Path) -> str:
    """Extract job ID from log filename (last numeric segment before .log)."""
    parts = log_path.stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return log_path.stem


def print_job_table(job_id: str, stats: dict[str, dict[str, int]]) -> None:
    """Print a formatted table for a single job."""
    print(f"\n{'=' * 70}")
    print(f"Job {job_id}")
    print(f"{'=' * 70}")
    print(f"{'Filter':<16} {'Description':<22} {'Passed':>8} {'Total':>8} {'Rate':>8}")
    print("-" * 70)
    for name in FILTER_ORDER:
        if name not in stats:
            continue
        s = stats[name]
        rate = s["passed"] / s["total"] * 100 if s["total"] > 0 else 0
        desc = FILTER_DESCRIPTIONS.get(name, "")
        print(f"{name:<16} {desc:<22} {s['passed']:>8} {s['total']:>8} {rate:>7.1f}%")


def print_aggregate_table(all_stats: dict[str, dict[str, int]], n_jobs: int, total_samples: int) -> None:
    """Print a formatted aggregate table across all jobs."""
    print(f"\n{'=' * 70}")
    print(f"Aggregate across {n_jobs} jobs ({total_samples:,} total samples)")
    print(f"{'=' * 70}")
    print(f"{'Filter':<16} {'Description':<22} {'Passed':>8} {'Total':>8} {'Rate':>8}")
    print("-" * 70)
    for name in FILTER_ORDER:
        if name not in all_stats:
            continue
        s = all_stats[name]
        rate = s["passed"] / s["total"] * 100 if s["total"] > 0 else 0
        desc = FILTER_DESCRIPTIONS.get(name, "")
        print(f"{name:<16} {desc:<22} {s['passed']:>8} {s['total']:>8} {rate:>7.1f}%")

    # End-to-end yield.
    first_filter = FILTER_ORDER[0]
    last_filter = FILTER_ORDER[-1]
    if first_filter in all_stats and last_filter in all_stats:
        total_in = all_stats[first_filter]["total"]
        total_out = all_stats[last_filter]["passed"]
        if total_in > 0:
            yield_pct = total_out / total_in * 100
            print(f"\n{total_in:,} samples in -> {total_out} proposals out ({yield_pct:.3f}% end-to-end yield)")


def discover_logs(log_dir: str) -> list[str]:
    """Find all evocas9 rejection sampling log files in the given directory."""
    pattern = str(Path(log_dir) / LOG_GLOB)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No logs matching '{LOG_GLOB}' found in {log_dir}")
    return paths


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze evocas9 rejection sampling SLURM logs for per-filter pass rates.",
    )
    parser.add_argument(
        "log_files",
        nargs="*",
        help="Log files to analyze (default: auto-discover in current directory)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to scan for slurm_evocas9_rejection_sampling_*.log files",
    )
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve log files.
    if parsed.log_files:
        log_paths = [Path(p) for p in parsed.log_files]
    else:
        log_dir = parsed.log_dir or "."
        log_paths = [Path(p) for p in discover_logs(log_dir)]

    for p in log_paths:
        if not p.exists():
            raise FileNotFoundError(f"Log file not found: {p}")

    logger.info(f"Analyzing {len(log_paths)} log file(s)")

    # Parse each log and print per-job tables.
    aggregate = defaultdict(lambda: {"passed": 0, "total": 0, "batches": 0})
    total_samples = 0

    for log_path in log_paths:
        job_id = extract_job_id(log_path)
        stats = parse_log(log_path)

        if not stats:
            logger.warning(f"No filter stats found in {log_path}")
            continue

        print_job_table(job_id, stats)

        # Accumulate into aggregate.
        for name, s in stats.items():
            aggregate[name]["passed"] += s["passed"]
            aggregate[name]["total"] += s["total"]
            aggregate[name]["batches"] += s["batches"]

        # Track total samples from first filter.
        first = FILTER_ORDER[0]
        if first in stats:
            total_samples += stats[first]["total"]

    # Print aggregate if multiple jobs.
    if len(log_paths) > 1:
        print_aggregate_table(dict(aggregate), len(log_paths), total_samples)


if __name__ == "__main__":
    main()
