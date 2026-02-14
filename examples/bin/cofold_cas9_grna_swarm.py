"""Distributed Cas9-sgRNA cofolding via evoswarm (SLURM).

Upstream: examples/scripts/evocas9_topk.py generates Cas9 candidates via TopK
optimization. Each SLURM job writes a *_candidates.tsv (named with its SLURM job ID,
e.g. cas9_topk_2000_1689545_candidates.tsv). Multiple jobs run in parallel, producing
one TSV each with ~8-13 candidates per job.

This script collates all candidate TSVs, constructs chimeric sgRNAs, cofolds each
Cas9-sgRNA pair with AF3, aligns to the SpCas9-sgRNA reference (PDB 4OO8) using
USalign, and ranks by structural similarity (TM-score). Each cofold runs on its own
SLURM worker (1 GPU) via evoswarm for parallel execution.

Usage:
    # Auto-discover all *_candidates.tsv in a directory:
    python examples/bin/cofold_cas9_grna_swarm.py --input-dir .

    # Or specify TSVs explicitly:
    python examples/bin/cofold_cas9_grna_swarm.py --input-tsvs a.tsv b.tsv c.tsv

    # With options:
    python examples/bin/cofold_cas9_grna_swarm.py --input-dir . --output-dir my_output/ --num-workers 20
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from evoswarm import Swarm

logger = logging.getLogger(__name__)

# Glob pattern for TSVs produced by evocas9_topk SLURM jobs.
# Each job writes cas9_topk_{n_samples}_{SLURM_JOB_ID}_candidates.tsv.
CANDIDATE_TSV_GLOB = "*_candidates.tsv"

DEFAULT_REFERENCE_PDB = "cofold_cas9_grna_output/4OO8.pdb"


def discover_candidate_tsvs(input_dir: str) -> List[str]:
    """Find all *_candidates.tsv files in the given directory."""
    pattern = str(Path(input_dir) / CANDIDATE_TSV_GLOB)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No candidate TSVs matching '{CANDIDATE_TSV_GLOB}' found in {input_dir}"
        )
    logger.info(f"Discovered {len(paths)} candidate TSVs in {input_dir}")
    for p in paths:
        logger.info(f"  {p}")
    return paths

SUMMARY_COLUMNS = [
    "global_id",
    "job_id",
    "candidate_idx",
    "temperature",
    "top_k",
    "identity",
    "protein_length",
    "sgrna_length",
    "avg_plddt",
    "ptm",
    "iptm",
    "ranking_score",
    "tm_score_candidate",
    "tm_score_reference",
    "rmsd",
    "cofold_pdb_path",
    "superposed_path",
]


def collate_candidates(
    tsv_paths: List[str],
    repo_root: str,
    output_dir: str,
    reference_pdb: str,
    seeds: List[int],
    use_msa: bool,
) -> List[Dict]:
    """Read all candidate TSVs and build a flat list of work items."""
    candidates = []
    global_id = 0
    for tsv_path in tsv_paths:
        # Extract job_id from filename: cas9_topk_2000_1689545_candidates.tsv -> 1689545
        tsv_name = Path(tsv_path).stem  # cas9_topk_2000_1689545_candidates
        parts = tsv_name.split("_")
        # job_id is the numeric part before "candidates"
        job_id = parts[-2] if len(parts) >= 2 else tsv_name

        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if not row.get("crispr_repeat") or not row.get("tracr_rna_sequence"):
                    logger.warning(
                        f"Skipping candidate {row.get('candidate_idx')} from "
                        f"{tsv_path}: missing crRNA or tracrRNA"
                    )
                    continue

                candidates.append({
                    "global_id": global_id,
                    "job_id": job_id,
                    "candidate_idx": row.get("candidate_idx", str(global_id)),
                    "temperature": row.get("temperature", ""),
                    "top_k": row.get("top_k", ""),
                    "identity": row.get("identity", ""),
                    "protein_sequence": row["protein_sequence"],
                    "crispr_repeat": row["crispr_repeat"],
                    "tracr_rna_sequence": row["tracr_rna_sequence"],
                    # Worker config injected into each dict.
                    "_repo_root": repo_root,
                    "_output_dir": output_dir,
                    "_reference_pdb": reference_pdb,
                    "_seeds": seeds,
                    "_use_msa": use_msa,
                })
                global_id += 1

    logger.info(f"Collated {len(candidates)} candidates from {len(tsv_paths)} TSVs")
    return candidates


def cofold_worker(data: dict) -> dict:
    """Worker function for evoswarm: cofold one Cas9-sgRNA candidate.

    Must be a module-level function (picklable). Handles its own imports
    since it runs on remote SLURM nodes.

    Returns a result dict with AF3 metrics + TM-scores, or an error dict.
    """
    import sys as _sys

    # Ensure repo root is on sys.path for imports on remote nodes.
    repo_root = data["_repo_root"]
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)

    try:
        from examples.bin.cofold_cas9_grna import (
            cofold_candidate,
            construct_sgrna,
            run_usalign,
        )
        from pathlib import Path as _Path

        global_id = data["global_id"]
        output_dir = _Path(data["_output_dir"])
        reference_pdb = _Path(data["_reference_pdb"])
        seeds = data["_seeds"]
        use_msa = data["_use_msa"]

        # Construct sgRNA.
        sgrna = construct_sgrna(
            data["crispr_repeat"],
            data["tracr_rna_sequence"],
            max_tracr_length=80,
        )

        # Run AF3 cofolding.
        af3_result = cofold_candidate(
            candidate_idx=global_id,
            protein_sequence=data["protein_sequence"],
            sgrna_sequence=sgrna,
            output_dir=output_dir,
            seeds=seeds,
            use_msa=use_msa,
            verbose=False,
        )

        if af3_result is None:
            return {
                "global_id": global_id,
                "job_id": data["job_id"],
                "candidate_idx": data["candidate_idx"],
                "error": "AF3 returned no structures",
            }

        # USalign against reference.
        candidate_pdb = _Path(af3_result["pdb_path"])
        candidate_dir = candidate_pdb.parent
        superposed_prefix = candidate_dir / "superposed_to_4OO8"

        try:
            usalign_metrics = run_usalign(
                candidate_pdb, reference_pdb, superposed_prefix
            )
        except Exception as e:
            usalign_metrics = {
                "tm_score_1": None,
                "tm_score_2": None,
                "rmsd": None,
            }
            return {
                "global_id": global_id,
                "job_id": data["job_id"],
                "candidate_idx": data["candidate_idx"],
                "error": f"USalign failed: {e}",
                **{k: af3_result.get(k) for k in ["avg_plddt", "ptm", "iptm", "ranking_score"]},
                "cofold_pdb_path": af3_result.get("pdb_path"),
            }

        # Find superposed PDB.
        superposed_pdb = None
        for suffix in ["_all_atm.pdb", ".pdb", "_atm.pdb"]:
            candidate_path = candidate_dir / f"superposed_to_4OO8{suffix}"
            if candidate_path.exists():
                superposed_pdb = str(candidate_path)
                break

        return {
            "global_id": global_id,
            "job_id": data["job_id"],
            "candidate_idx": data["candidate_idx"],
            "temperature": data["temperature"],
            "top_k": data["top_k"],
            "identity": data["identity"],
            "protein_length": len(data["protein_sequence"]),
            "sgrna_length": len(sgrna),
            "avg_plddt": af3_result.get("avg_plddt"),
            "ptm": af3_result.get("ptm"),
            "iptm": af3_result.get("iptm"),
            "ranking_score": af3_result.get("ranking_score"),
            "tm_score_candidate": usalign_metrics.get("tm_score_1"),
            "tm_score_reference": usalign_metrics.get("tm_score_2"),
            "rmsd": usalign_metrics.get("rmsd"),
            "cofold_pdb_path": af3_result.get("pdb_path"),
            "superposed_path": superposed_pdb,
        }

    except Exception as e:
        return {
            "global_id": data.get("global_id"),
            "job_id": data.get("job_id"),
            "candidate_idx": data.get("candidate_idx"),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def write_summary(results: List[Dict], output_dir: Path) -> Path:
    """Write ranked summary TSV and print table to stdout."""
    # Separate successes and failures.
    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    if failures:
        logger.warning(f"{len(failures)} candidates failed:")
        for f in failures:
            logger.warning(
                f"  global_id={f.get('global_id')}, "
                f"job_id={f.get('job_id')}, "
                f"candidate_idx={f.get('candidate_idx')}: "
                f"{f.get('error', 'unknown error')}"
            )

    # Sort successes by tm_score_reference descending.
    successes.sort(
        key=lambda r: r.get("tm_score_reference") or 0.0,
        reverse=True,
    )

    summary_path = output_dir / "swarm_summary.tsv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SUMMARY_COLUMNS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(successes)

    logger.info(f"Summary written to {summary_path} ({len(successes)} rows)")

    # Print ranked table.
    print(f"\n{'=' * 100}")
    print(f"Cas9-sgRNA Swarm Cofolding Summary ({len(successes)} successes, {len(failures)} failures)")
    print(f"{'=' * 100}")
    header = (
        f"{'GID':>4s}  {'Job':>8s}  {'Idx':>4s}  {'Prot':>5s}  {'sgRNA':>5s}  "
        f"{'pLDDT':>6s}  {'pTM':>5s}  {'ipTM':>5s}  "
        f"{'Rank':>6s}  {'TM(c)':>6s}  {'TM(r)':>6s}  {'RMSD':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in successes:
        def fmt(v, w=5, d=3):
            return f"{v:{w}.{d}f}" if v is not None else f"{'N/A':>{w}s}"

        print(
            f"{r['global_id']:>4}  "
            f"{r.get('job_id', ''):>8s}  "
            f"{str(r.get('candidate_idx', '')):>4s}  "
            f"{r.get('protein_length', 0):>5d}  "
            f"{r.get('sgrna_length', 0):>5d}  "
            f"{fmt(r.get('avg_plddt'), 6, 1)}  "
            f"{fmt(r.get('ptm'))}  "
            f"{fmt(r.get('iptm'))}  "
            f"{fmt(r.get('ranking_score'), 6, 3)}  "
            f"{fmt(r.get('tm_score_candidate'), 6, 3)}  "
            f"{fmt(r.get('tm_score_reference'), 6, 3)}  "
            f"{fmt(r.get('rmsd'), 6, 2)}"
        )
    print()
    print("TM(c) = TM-score normalized by candidate length")
    print("TM(r) = TM-score normalized by reference (4OO8) length")
    print(f"{'=' * 100}\n")

    return summary_path


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cofold Cas9-sgRNA candidates via evoswarm (distributed SLURM).",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        help="Directory to scan for *_candidates.tsv files (from evocas9_topk runs)",
    )
    input_group.add_argument(
        "--input-tsvs",
        nargs="+",
        help="Explicit list of candidate TSV files",
    )
    parser.add_argument(
        "--output-dir",
        default="cofold_cas9_grna_swarm_output",
        help="Output directory (default: cofold_cas9_grna_swarm_output/)",
    )
    parser.add_argument(
        "--reference-pdb",
        default=DEFAULT_REFERENCE_PDB,
        help=f"Path to 4OO8.pdb (default: {DEFAULT_REFERENCE_PDB})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=40,
        help="Number of SLURM workers (default: 40)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="AF3 random seeds (default: 0)",
    )
    parser.add_argument(
        "--no-msa",
        action="store_true",
        help="Disable MSA generation (faster, less accurate)",
    )
    parser.add_argument(
        "--partition",
        default="preemptible",
        help="SLURM partition (default: preemptible)",
    )
    parser.add_argument(
        "--time",
        default="14:00:00",
        help="SLURM time limit (default: 14:00:00)",
    )
    parser.add_argument(
        "--mem-per-cpu",
        default="4gb",
        help="SLURM memory per CPU (default: 4gb)",
    )
    parser.add_argument(
        "--exclude",
        default="GPUCACE",
        help="SLURM nodes to exclude (default: GPUCACE)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths to absolute for remote workers.
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    reference_pdb = str(Path(parsed.reference_pdb).resolve())
    output_dir_abs = str(output_dir.resolve())

    if not Path(reference_pdb).exists():
        raise FileNotFoundError(
            f"Reference PDB not found: {reference_pdb}. "
            f"Run cofold_cas9_grna.py first to download 4OO8.pdb, or provide --reference-pdb."
        )

    # Resolve input TSV paths.
    if parsed.input_dir:
        tsv_paths = [str(Path(p).resolve()) for p in discover_candidate_tsvs(parsed.input_dir)]
    else:
        tsv_paths = []
        for tsv in parsed.input_tsvs:
            p = Path(tsv)
            if not p.exists():
                raise FileNotFoundError(f"Input TSV not found: {tsv}")
            tsv_paths.append(str(p.resolve()))

    # Section 1: Collate candidates.
    candidates = collate_candidates(
        tsv_paths=tsv_paths,
        repo_root=repo_root,
        output_dir=output_dir_abs,
        reference_pdb=reference_pdb,
        seeds=parsed.seeds,
        use_msa=not parsed.no_msa,
    )

    if not candidates:
        logger.error("No valid candidates found in input TSVs")
        sys.exit(1)

    logger.info(f"Dispatching {len(candidates)} candidates to {parsed.num_workers} workers")

    # Section 3: Swarm dispatch.
    swarm = Swarm(
        input_data=candidates,
        slurm_partition=parsed.partition,
        output_log_dir=str(output_dir / "evoswarm_log"),
        slurm_gpus_per_node=1,
        slurm_mem_per_cpu=parsed.mem_per_cpu,
        slurm_cpus_per_task=32,
        slurm_time=parsed.time,
    )

    results = swarm.map(
        cofold_worker,
        num_workers=parsed.num_workers,
        max_unresponsive_iterations=100_000,
        slurm_additional_parameters={"exclude": parsed.exclude},
    )

    if results is None:
        logger.error("Swarm returned no results")
        sys.exit(1)

    logger.info(f"Swarm completed: {len(results)} results")

    # Section 4: Collect and summarize results.
    write_summary(results, output_dir)


if __name__ == "__main__":
    main()
