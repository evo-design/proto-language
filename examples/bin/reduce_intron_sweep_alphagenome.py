#!/usr/bin/env python3
"""Reduce and visualize AlphaGenome intron sweep outputs for a given sweep ID.

This script parses per-task sweep logs produced by:
  examples/bin/intron_sweep_slurm_alphagenome.sh

Given a sweep ID, it writes:
  1) per-config summary table (energy + AG SSU metrics)
  2) per-iteration long table (energy + AG mean per iteration)
  3) ranked tables for best configs by energy and by AG SSU score
  4) parameter-value aggregate table
  5) comparison plots across sweep configs/params
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ITERATION_PATTERN = re.compile(
    r"Iteration\s+(\d+)\s+\|\s+energy:\s*([-+0-9.eE]+),\s*T:\s*([-+0-9.eE]+)"
)
AG_SCORE_PATTERN = re.compile(
    r"alphagenome_(?:splice_site_usage|interval_track)_score:\s*([-+0-9.eE]+)"
)
LEFT_FLANK_PATTERN = re.compile(r"sequence\s+\(left_flank\):")
TASK_DIR_PATTERN = re.compile(r"task_(\d+)$")
SPECIFICITY_TOKENS = ("max_brain", "min_brain", "max_blood", "min_blood")


def _natural_task_sort_key(path: Path) -> Tuple[int, str]:
    match = TASK_DIR_PATTERN.search(path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def _safe_int(raw: str) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _safe_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _mean_or_nan(values: Iterable[float]) -> float:
    finite = [v for v in values if _is_finite(v)]
    if not finite:
        return math.nan
    return float(np.mean(np.asarray(finite, dtype=float)))


def _min_or_nan(values: Iterable[float]) -> float:
    finite = [v for v in values if _is_finite(v)]
    if not finite:
        return math.nan
    return float(np.min(np.asarray(finite, dtype=float)))


def _parse_config_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_stdout_iterations(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    current: Optional[Dict[str, Any]] = None

    def finalize_current() -> None:
        nonlocal current
        if current is None:
            return
        scores = current.pop("ag_scores")
        if scores:
            current["ag_mean"] = float(np.mean(np.asarray(scores, dtype=float)))
            current["ag_count"] = len(scores)
        else:
            current["ag_mean"] = math.nan
            current["ag_count"] = 0
        rows.append(current)
        current = None

    for line in path.read_text().splitlines():
        m_iter = ITERATION_PATTERN.search(line)
        if m_iter is not None:
            finalize_current()
            current = {
                "iteration": int(m_iter.group(1)),
                "energy": float(m_iter.group(2)),
                "temperature_logged": float(m_iter.group(3)),
                "ag_scores": [],
            }
            continue

        if current is None:
            continue

        m_ag = AG_SCORE_PATTERN.search(line)
        if m_ag is not None:
            current["ag_scores"].append(float(m_ag.group(1)))

    finalize_current()
    return rows


def _parse_stderr_iterations(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    for line in path.read_text().splitlines():
        m_iter = ITERATION_PATTERN.search(line)
        if m_iter is None:
            continue
        rows.append(
            {
                "iteration": int(m_iter.group(1)),
                "energy": float(m_iter.group(2)),
                "temperature_logged": float(m_iter.group(3)),
                "ag_mean": math.nan,
                "ag_count": 0,
            }
        )
    return rows


def _parse_stdout_ag_groups(path: Path) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if not path.exists():
        return groups

    current_scores: Optional[List[float]] = None

    def finalize_group() -> None:
        nonlocal current_scores
        if current_scores is None:
            return
        if current_scores:
            groups.append(
                {
                    "ag_mean": float(np.mean(np.asarray(current_scores, dtype=float))),
                    "ag_count": len(current_scores),
                }
            )
        else:
            groups.append({"ag_mean": math.nan, "ag_count": 0})
        current_scores = None

    for line in path.read_text().splitlines():
        if LEFT_FLANK_PATTERN.search(line):
            finalize_group()
            current_scores = []
            continue
        if current_scores is None:
            continue
        m_ag = AG_SCORE_PATTERN.search(line)
        if m_ag is not None:
            current_scores.append(float(m_ag.group(1)))

    finalize_group()
    return groups


def _parse_task_iterations(stdout_log: Path, stderr_log: Path) -> List[Dict[str, Any]]:
    """Parse per-iteration energy/temperature + AG means from task logs.

    Current sweep logs write:
      - iteration/energy/temperature to stderr.log
      - per-context AlphaGenome scores to stdout.log

    We align AG groups to iterations by order. If stderr parsing yields no
    iterations, fall back to legacy stdout-only parsing.
    """
    iter_rows = _parse_stderr_iterations(stderr_log)
    if not iter_rows:
        return _parse_stdout_iterations(stdout_log)

    ag_groups = _parse_stdout_ag_groups(stdout_log)
    for idx, row in enumerate(iter_rows):
        if idx < len(ag_groups):
            row["ag_mean"] = float(ag_groups[idx]["ag_mean"])
            row["ag_count"] = int(ag_groups[idx]["ag_count"])
        else:
            row["ag_mean"] = math.nan
            row["ag_count"] = 0
    return iter_rows


def _count_specificity_terms(specificity_type: str) -> int:
    spec = str(specificity_type).lower()
    return sum(1 for token in SPECIFICITY_TOKENS if token in spec)


def _infer_constraint_counts(
    stdout_log: Path,
    config_env: Dict[str, str],
    iter_rows: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Infer active constraint counts for a task.

    Prefers parsing unique constraint labels from stdout log. Falls back to
    config-driven inference when labels are unavailable.
    """
    boundary_labels: set[str] = set()
    specificity_labels: set[str] = set()
    alphagenome_labels: set[str] = set()

    if stdout_log.exists():
        for line in stdout_log.read_text().splitlines():
            stripped = line.strip()
            if ":" not in stripped:
                continue
            label = stripped.split(":", 1)[0].strip()
            if label.startswith("splice_boundary__"):
                boundary_labels.add(label)
            elif label.startswith("splice_specificity_"):
                specificity_labels.add(label)
            elif label.startswith("alphagenome_"):
                alphagenome_labels.add(label)

    n_boundary = len(boundary_labels)
    n_specificity = len(specificity_labels)
    n_alphagenome = len(alphagenome_labels)

    multicontext = _safe_bool(config_env.get("MULTICONTEXT", ""))
    context_count = 3 if multicontext else 1

    enable_splice_transformer = _safe_bool(config_env.get("ENABLE_SPLICE_TRANSFORMER", ""))
    enable_splice_specificity = _safe_bool(config_env.get("ENABLE_SPLICE_SPECIFICITY", ""))
    enable_alphagenome = _safe_bool(config_env.get("ENABLE_ALPHAGENOME", ""))

    if n_boundary == 0 and enable_splice_transformer:
        n_boundary = context_count

    if n_specificity == 0 and enable_splice_transformer and enable_splice_specificity:
        n_specificity = context_count * _count_specificity_terms(
            config_env.get("SPECIFICITY_TYPE", "")
        )

    if n_alphagenome == 0 and enable_alphagenome:
        ag_counts = [
            int(r.get("ag_count", 0))
            for r in iter_rows
            if int(r.get("ag_count", 0)) > 0
        ]
        if ag_counts:
            n_alphagenome = int(round(float(np.median(np.asarray(ag_counts, dtype=float)))))
        else:
            # Conservative fallback for legacy/missing logs.
            genomic_context_count = 4 if multicontext else 1
            n_alphagenome = context_count * genomic_context_count * max(
                1,
                _count_specificity_terms(config_env.get("SPECIFICITY_TYPE", "")),
            )

    n_total = n_boundary + n_specificity + n_alphagenome
    return {
        "constraint_count_total": int(n_total),
        "constraint_count_boundary": int(n_boundary),
        "constraint_count_specificity": int(n_specificity),
        "constraint_count_alphagenome": int(n_alphagenome),
    }


def _normalize_energy_metrics(
    metrics: Dict[str, Any],
    constraint_count_total: int,
) -> Dict[str, float]:
    def norm(value: Any) -> float:
        as_float = _safe_float(value)
        if constraint_count_total <= 0 or not _is_finite(as_float):
            return math.nan
        return float(as_float / float(constraint_count_total))

    return {
        "start_energy_per_constraint": norm(metrics.get("start_energy", math.nan)),
        "final_energy_per_constraint": norm(metrics.get("final_energy", math.nan)),
        "min_energy_per_constraint": norm(metrics.get("min_energy", math.nan)),
        "energy_delta_final_per_constraint": norm(
            metrics.get("energy_delta_final", math.nan)
        ),
        "energy_delta_best_per_constraint": norm(
            metrics.get("energy_delta_best", math.nan)
        ),
        "energy_at_best_ag_per_constraint": norm(
            metrics.get("energy_at_best_ag", math.nan)
        ),
    }


def _compute_summary_metrics(iter_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "iterations_logged": len(iter_rows),
        "first_iter": None,
        "last_iter": None,
        "start_energy": math.nan,
        "final_energy": math.nan,
        "min_energy": math.nan,
        "best_energy_iter": None,
        "energy_delta_final": math.nan,
        "energy_delta_best": math.nan,
        "final_temperature_logged": math.nan,
        "iterations_with_ag": 0,
        "final_ag_mean": math.nan,
        "min_ag_mean": math.nan,
        "best_ag_iter": None,
        "ag_at_best_energy": math.nan,
        "energy_at_best_ag": math.nan,
        "final_ag_track_count": 0,
    }
    if not iter_rows:
        return metrics

    iterations = np.asarray([int(r["iteration"]) for r in iter_rows], dtype=int)
    energies = np.asarray([float(r["energy"]) for r in iter_rows], dtype=float)
    temperatures = np.asarray(
        [float(r["temperature_logged"]) for r in iter_rows], dtype=float
    )
    ag_means = np.asarray([float(r["ag_mean"]) for r in iter_rows], dtype=float)
    ag_counts = np.asarray([int(r["ag_count"]) for r in iter_rows], dtype=int)

    metrics["first_iter"] = int(iterations[0])
    metrics["last_iter"] = int(iterations[-1])
    metrics["start_energy"] = float(energies[0])
    metrics["final_energy"] = float(energies[-1])
    metrics["final_temperature_logged"] = float(temperatures[-1])

    best_energy_idx = int(np.argmin(energies))
    metrics["min_energy"] = float(energies[best_energy_idx])
    metrics["best_energy_iter"] = int(iterations[best_energy_idx])
    metrics["energy_delta_final"] = float(energies[-1] - energies[0])
    metrics["energy_delta_best"] = float(np.min(energies) - energies[0])

    finite_ag_mask = np.isfinite(ag_means)
    metrics["iterations_with_ag"] = int(np.sum(finite_ag_mask))
    metrics["final_ag_track_count"] = int(ag_counts[-1])

    if np.isfinite(ag_means[-1]):
        metrics["final_ag_mean"] = float(ag_means[-1])
    if np.isfinite(ag_means[best_energy_idx]):
        metrics["ag_at_best_energy"] = float(ag_means[best_energy_idx])

    if np.any(finite_ag_mask):
        finite_idxs = np.where(finite_ag_mask)[0]
        best_ag_rel_idx = int(np.argmin(ag_means[finite_idxs]))
        best_ag_idx = int(finite_idxs[best_ag_rel_idx])
        metrics["min_ag_mean"] = float(ag_means[best_ag_idx])
        metrics["best_ag_iter"] = int(iterations[best_ag_idx])
        metrics["energy_at_best_ag"] = float(energies[best_ag_idx])

    return metrics


def _write_tsv(path: Path, rows: List[Dict[str, Any]], field_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in field_names})


def _plot_energy_trajectories(
    per_config_iters: Dict[str, List[Dict[str, Any]]],
    summary_rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    energy_metric_key: str = "energy",
    summary_metric_key: str = "min_energy",
    title: str = "Sweep Energy Trajectories (all configs)",
    y_label: str = "Total energy",
    best_label: str = "best energy",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    best_cfg: Optional[str] = None
    best_energy = math.inf
    for row in summary_rows:
        value = _safe_float(str(row.get(summary_metric_key, "")))
        if _is_finite(value) and value < best_energy:
            best_energy = value
            best_cfg = str(row["config_id"])

    for config_id, rows in per_config_iters.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["iteration"]))
        x = np.asarray([int(r["iteration"]) for r in rows_sorted], dtype=float)
        y = np.asarray(
            [_safe_float(str(r.get(energy_metric_key, ""))) for r in rows_sorted],
            dtype=float,
        )
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        x = x[mask]
        y = y[mask]
        if config_id == best_cfg:
            ax.plot(
                x,
                y,
                linewidth=2.0,
                color="#d62728",
                alpha=0.95,
                label=f"{best_label} config {config_id}",
            )
        else:
            ax.plot(x, y, linewidth=0.9, color="#1f77b4", alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    if best_cfg is not None:
        ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _normalize_iteration_energies(
    per_config_iters: Dict[str, List[Dict[str, Any]]],
    summary_rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    constraints_by_cfg: Dict[str, int] = {}
    for row in summary_rows:
        cfg = str(row.get("config_id", ""))
        constraints_by_cfg[cfg] = int(row.get("constraint_count_total", 0))

    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for config_id, rows in per_config_iters.items():
        denom = constraints_by_cfg.get(str(config_id), 0)
        for row in rows:
            row_out = dict(row)
            energy = _safe_float(str(row.get("energy", "")))
            if denom > 0 and _is_finite(energy):
                row_out["energy_per_constraint"] = float(energy / float(denom))
            else:
                row_out["energy_per_constraint"] = math.nan
            out[config_id].append(row_out)
    return out


def _plot_ag_trajectories(
    per_config_iters: Dict[str, List[Dict[str, Any]]],
    summary_rows: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    best_cfg: Optional[str] = None
    best_ag = math.inf
    for row in summary_rows:
        value = row.get("min_ag_mean", math.nan)
        if _is_finite(value) and value < best_ag:
            best_ag = value
            best_cfg = str(row["config_id"])

    for config_id, rows in per_config_iters.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["iteration"]))
        x = np.asarray([int(r["iteration"]) for r in rows_sorted], dtype=float)
        y = np.asarray([float(r["ag_mean"]) for r in rows_sorted], dtype=float)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        x = x[mask]
        y = y[mask]
        if config_id == best_cfg:
            ax.plot(
                x,
                y,
                linewidth=2.0,
                color="#2ca02c",
                alpha=0.95,
                label=f"best AG config {config_id}",
            )
        else:
            ax.plot(x, y, linewidth=0.9, color="#2ca02c", alpha=0.18)

    ax.set_title("Sweep AlphaGenome SSU Trajectories")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("AG SSU score (mean over contexts)")
    if best_cfg is not None:
        ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _plot_pareto_min_energy_vs_ag(
    summary_rows: List[Dict[str, Any]],
    out_path: Path,
    max_annotations: int,
    *,
    energy_metric_key: str = "min_energy",
    title: str = "Config Pareto View: min AG score vs min energy",
    y_label: str = "min total energy (lower is better)",
) -> None:
    plot_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        energy_value = _safe_float(str(row.get(energy_metric_key, "")))
        ag_value = _safe_float(str(row.get("min_ag_mean", "")))
        if _is_finite(energy_value) and _is_finite(ag_value):
            plot_rows.append(row)
    if not plot_rows:
        return

    x = np.asarray([float(r["min_ag_mean"]) for r in plot_rows], dtype=float)
    y = np.asarray(
        [_safe_float(str(r.get(energy_metric_key, ""))) for r in plot_rows],
        dtype=float,
    )
    temps = np.asarray([_safe_float(r.get("temperature", "")) for r in plot_rows], dtype=float)
    st_flags = np.asarray(
        [str(r.get("enable_splice_specificity", "")).lower() == "true" for r in plot_rows],
        dtype=bool,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    finite_temp_mask = np.isfinite(temps)
    scatter = None
    if np.any(finite_temp_mask):
        scatter = ax.scatter(
            x[finite_temp_mask],
            y[finite_temp_mask],
            c=temps[finite_temp_mask],
            cmap="viridis",
            s=52,
            edgecolors=np.where(st_flags[finite_temp_mask], "black", "white"),
            linewidths=0.8,
            alpha=0.9,
        )
    if np.any(~finite_temp_mask):
        ax.scatter(
            x[~finite_temp_mask],
            y[~finite_temp_mask],
            color="#7f7f7f",
            s=52,
            marker="x",
            alpha=0.9,
        )

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Sweep temperature")

    ax.set_title(title)
    ax.set_xlabel("min AG SSU score (lower is better)")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)

    energy_order = np.argsort(y)
    ag_order = np.argsort(x)
    energy_rank = np.empty_like(energy_order)
    energy_rank[energy_order] = np.arange(len(plot_rows))
    ag_rank = np.empty_like(ag_order)
    ag_rank[ag_order] = np.arange(len(plot_rows))
    combined_rank = energy_rank + ag_rank
    annotate_indices = np.argsort(combined_rank)[: max(0, max_annotations)]
    for idx in annotate_indices:
        cfg = str(plot_rows[int(idx)].get("config_id", "?"))
        ax.annotate(cfg, (x[idx], y[idx]), textcoords="offset points", xytext=(4, 3), fontsize=7)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb", markeredgecolor="black", label="ST specificity: True", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb", markeredgecolor="white", label="ST specificity: False", markersize=7),
    ]
    ax.legend(handles=legend_items, loc="best", fontsize=8)

    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _plot_metrics_vs_temperature(
    summary_rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    energy_metric_key: str = "min_energy",
    energy_label: str = "min total energy",
    title: str = "Best metrics vs sweep temperature",
) -> None:
    rows = []
    for row in summary_rows:
        temp = _safe_float(row.get("temperature", ""))
        if not _is_finite(temp):
            continue
        rows.append({**row, "temperature_value": temp})
    if not rows:
        return

    temps = np.asarray([float(r["temperature_value"]) for r in rows], dtype=float)
    st_flags = np.asarray(
        [str(r.get("enable_splice_specificity", "")).lower() == "true" for r in rows],
        dtype=bool,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True, constrained_layout=True)
    metrics = [
        (energy_metric_key, energy_label),
        ("min_ag_mean", "min AG SSU score"),
    ]
    colors = {True: "#1f77b4", False: "#ff7f0e"}

    for ax, (metric_key, label) in zip(axes, metrics):
        metric_vals = np.asarray([_safe_float(r.get(metric_key, "")) for r in rows], dtype=float)
        for st_value in [True, False]:
            mask = (st_flags == st_value) & np.isfinite(metric_vals)
            if not np.any(mask):
                continue
            ax.scatter(
                temps[mask],
                metric_vals[mask],
                s=38,
                color=colors[st_value],
                alpha=0.8,
                label=f"ST specificity {st_value}",
            )

        uniq_temps = sorted({float(t) for t in temps})
        mean_vals: List[float] = []
        mean_temps: List[float] = []
        for temp in uniq_temps:
            mask = (temps == temp) & np.isfinite(metric_vals)
            if not np.any(mask):
                continue
            mean_temps.append(temp)
            mean_vals.append(float(np.mean(metric_vals[mask])))
        if mean_temps:
            ax.plot(mean_temps, mean_vals, color="black", linewidth=1.2, alpha=0.9, label="mean")

        ax.set_ylabel(label)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("temperature")
    axes[-1].set_xscale("log")
    fig.suptitle(title)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _plot_param_value_bars(
    param_rows: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    out_path: Path,
) -> None:
    filtered = [r for r in param_rows if _is_finite(_safe_float(str(r.get(metric_key, ""))))]
    if not filtered:
        return

    sorted_rows = sorted(filtered, key=lambda r: _safe_float(str(r[metric_key])))
    labels = [f"{r['param_name']}={r['param_value']}" for r in sorted_rows]
    values = np.asarray([_safe_float(str(r[metric_key])) for r in sorted_rows], dtype=float)

    fig_height = max(4.5, 0.33 * len(labels))
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4c78a8", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric_key)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.6)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _build_param_value_summary(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    params = [
        "initialization",
        "temperature",
        "enable_splice_specificity",
        "specificity_type",
        "alphagenome_track_strand",
        "alphagenome_brain_weight",
        "alphagenome_blood_weight",
    ]
    out_rows: List[Dict[str, Any]] = []
    for param in params:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in summary_rows:
            value = str(row.get(param, ""))
            grouped[value].append(row)
        for value, rows in grouped.items():
            final_energy = [_safe_float(str(r.get("final_energy", ""))) for r in rows]
            min_energy = [_safe_float(str(r.get("min_energy", ""))) for r in rows]
            final_energy_norm = [
                _safe_float(str(r.get("final_energy_per_constraint", ""))) for r in rows
            ]
            min_energy_norm = [
                _safe_float(str(r.get("min_energy_per_constraint", ""))) for r in rows
            ]
            final_ag = [_safe_float(str(r.get("final_ag_mean", ""))) for r in rows]
            min_ag = [_safe_float(str(r.get("min_ag_mean", ""))) for r in rows]
            out_rows.append(
                {
                    "param_name": param,
                    "param_value": value,
                    "n_configs": len(rows),
                    "mean_final_energy": _mean_or_nan(final_energy),
                    "mean_min_energy": _mean_or_nan(min_energy),
                    "best_min_energy": _min_or_nan(min_energy),
                    "mean_final_energy_per_constraint": _mean_or_nan(final_energy_norm),
                    "mean_min_energy_per_constraint": _mean_or_nan(min_energy_norm),
                    "best_min_energy_per_constraint": _min_or_nan(min_energy_norm),
                    "mean_final_ag_mean": _mean_or_nan(final_ag),
                    "mean_min_ag_mean": _mean_or_nan(min_ag),
                    "best_min_ag_mean": _min_or_nan(min_ag),
                }
            )
    out_rows.sort(key=lambda r: (str(r["param_name"]), str(r["param_value"])))
    return out_rows


def _row_status(row: Dict[str, Any]) -> str:
    if not row.get("stdout_log_path"):
        return "missing_stdout"
    if int(row.get("iterations_logged", 0)) <= 0:
        return "no_iterations"
    n_steps = _safe_int(str(row.get("n_steps", "")))
    last_iter = row.get("last_iter")
    if n_steps is not None and isinstance(last_iter, int) and last_iter < n_steps:
        return "incomplete"
    return "complete"


def _to_summary_field_order() -> List[str]:
    return [
        "sweep_id",
        "task_id",
        "config_id",
        "status",
        "initialization",
        "temperature",
        "temperature_value",
        "n_steps",
        "multicontext",
        "intron_generator",
        "specificity_type",
        "enable_splice_transformer",
        "enable_splice_specificity",
        "enable_alphagenome",
        "target_cell",
        "target_ontology_terms",
        "offtarget_cell",
        "offtarget_ontology_terms",
        "alphagenome_track_strand",
        "alphagenome_brain_weight",
        "alphagenome_blood_weight",
        "iterations_logged",
        "first_iter",
        "last_iter",
        "start_energy",
        "start_energy_per_constraint",
        "final_energy",
        "final_energy_per_constraint",
        "min_energy",
        "min_energy_per_constraint",
        "best_energy_iter",
        "energy_delta_final",
        "energy_delta_final_per_constraint",
        "energy_delta_best",
        "energy_delta_best_per_constraint",
        "final_temperature_logged",
        "iterations_with_ag",
        "final_ag_mean",
        "min_ag_mean",
        "best_ag_iter",
        "ag_at_best_energy",
        "energy_at_best_ag",
        "energy_at_best_ag_per_constraint",
        "final_ag_track_count",
        "constraint_count_total",
        "constraint_count_boundary",
        "constraint_count_specificity",
        "constraint_count_alphagenome",
        "stdout_log_path",
        "stderr_log_path",
        "config_env_path",
    ]


def _to_iteration_field_order() -> List[str]:
    return [
        "sweep_id",
        "task_id",
        "config_id",
        "iteration",
        "energy",
        "temperature_logged",
        "ag_mean",
        "ag_count",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce/visualize intron AlphaGenome sweep results for one sweep ID.",
    )
    parser.add_argument("--sweep_id", required=True, help="Sweep ID (e.g., sweep_20260220_010203).")
    parser.add_argument("--log_root", default="log/intron_alphagenome", help="Root directory containing sweep task logs.")
    parser.add_argument("--run_root", default="runs/intron_alphagenome", help="Root directory containing per-config run dirs.")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional output directory. Defaults to runs/intron_alphagenome/<sweep_id>/reduction.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-k configs to keep in ranking tables.")
    parser.add_argument(
        "--max_pareto_annotations",
        type=int,
        default=15,
        help="Max config labels to annotate in pareto plot.",
    )
    args = parser.parse_args()

    sweep_id = args.sweep_id
    sweep_log_dir = Path(args.log_root) / sweep_id
    if not sweep_log_dir.exists():
        raise FileNotFoundError(f"Sweep log directory not found: {sweep_log_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (Path(args.run_root) / sweep_id / "reduction")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(
        [p for p in sweep_log_dir.glob("task_*") if p.is_dir()],
        key=_natural_task_sort_key,
    )
    if not task_dirs:
        raise ValueError(f"No task directories found under: {sweep_log_dir}")

    summary_rows: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []
    per_config_iters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for task_dir in task_dirs:
        match = TASK_DIR_PATTERN.search(task_dir.name)
        task_id = int(match.group(1)) if match else -1

        config_env_path = task_dir / "config.env"
        stdout_log = task_dir / "stdout.log"
        stderr_log = task_dir / "stderr.log"
        config_env = _parse_config_env(config_env_path)

        config_id = config_env.get("CONFIG_ID", str(task_id))
        temperature_raw = config_env.get("TEMPERATURE", "")
        ag_track_strand = config_env.get("ALPHAGENOME_TRACK_STRAND", "")
        ag_brain_weight = config_env.get("ALPHAGENOME_BRAIN_WEIGHT", "")
        ag_blood_weight = config_env.get("ALPHAGENOME_BLOOD_WEIGHT", "")

        parsed_iters = _parse_task_iterations(stdout_log, stderr_log)
        metrics = _compute_summary_metrics(parsed_iters)
        constraint_counts = _infer_constraint_counts(stdout_log, config_env, parsed_iters)
        normalized_metrics = _normalize_energy_metrics(
            metrics,
            constraint_counts["constraint_count_total"],
        )

        summary_row: Dict[str, Any] = {
            "sweep_id": sweep_id,
            "task_id": task_id,
            "config_id": config_id,
            "initialization": config_env.get("INITIALIZATION", ""),
            "temperature": temperature_raw,
            "temperature_value": _safe_float(temperature_raw),
            "n_steps": config_env.get("N_STEPS", ""),
            "multicontext": config_env.get("MULTICONTEXT", ""),
            "intron_generator": config_env.get("INTRON_GENERATOR", ""),
            "specificity_type": config_env.get("SPECIFICITY_TYPE", ""),
            "enable_splice_transformer": config_env.get("ENABLE_SPLICE_TRANSFORMER", ""),
            "enable_splice_specificity": config_env.get("ENABLE_SPLICE_SPECIFICITY", ""),
            "enable_alphagenome": config_env.get("ENABLE_ALPHAGENOME", ""),
            "target_cell": config_env.get("TARGET_CELL", ""),
            "target_ontology_terms": config_env.get("TARGET_ONTOLOGY_TERMS", ""),
            "offtarget_cell": config_env.get("OFFTARGET_CELL", ""),
            "offtarget_ontology_terms": config_env.get("OFFTARGET_ONTOLOGY_TERMS", ""),
            "alphagenome_track_strand": ag_track_strand,
            "alphagenome_brain_weight": ag_brain_weight,
            "alphagenome_blood_weight": ag_blood_weight,
            "stdout_log_path": str(stdout_log) if stdout_log.exists() else "",
            "stderr_log_path": str(stderr_log) if stderr_log.exists() else "",
            "config_env_path": str(config_env_path) if config_env_path.exists() else "",
            **metrics,
            **normalized_metrics,
            **constraint_counts,
        }
        summary_row["status"] = _row_status(summary_row)
        summary_rows.append(summary_row)

        for iter_row in parsed_iters:
            long_row = {
                "sweep_id": sweep_id,
                "task_id": task_id,
                "config_id": config_id,
                "iteration": int(iter_row["iteration"]),
                "energy": float(iter_row["energy"]),
                "temperature_logged": float(iter_row["temperature_logged"]),
                "ag_mean": float(iter_row["ag_mean"]),
                "ag_count": int(iter_row["ag_count"]),
            }
            iteration_rows.append(long_row)
            per_config_iters[str(config_id)].append(long_row)

    summary_fields = _to_summary_field_order()
    iteration_fields = _to_iteration_field_order()
    _write_tsv(output_dir / "config_summary.tsv", summary_rows, summary_fields)
    _write_tsv(output_dir / "iteration_metrics.tsv", iteration_rows, iteration_fields)

    complete_rows = [r for r in summary_rows if r.get("status") == "complete"]

    energy_rank_rows = [
        r for r in complete_rows if _is_finite(_safe_float(str(r.get("min_energy", ""))))
    ]
    energy_rank_rows.sort(key=lambda r: float(r["min_energy"]))
    _write_tsv(output_dir / "top_by_min_energy.tsv", energy_rank_rows[: args.top_k], summary_fields)

    energy_norm_rank_rows = [
        r
        for r in complete_rows
        if _is_finite(_safe_float(str(r.get("min_energy_per_constraint", ""))))
    ]
    energy_norm_rank_rows.sort(key=lambda r: float(r["min_energy_per_constraint"]))
    _write_tsv(
        output_dir / "top_by_min_energy_per_constraint.tsv",
        energy_norm_rank_rows[: args.top_k],
        summary_fields,
    )

    ag_rank_rows = [
        r for r in complete_rows if _is_finite(_safe_float(str(r.get("min_ag_mean", ""))))
    ]
    ag_rank_rows.sort(key=lambda r: float(r["min_ag_mean"]))
    _write_tsv(output_dir / "top_by_min_ag.tsv", ag_rank_rows[: args.top_k], summary_fields)

    param_summary_rows = _build_param_value_summary(complete_rows)
    param_fields = [
        "param_name",
        "param_value",
        "n_configs",
        "mean_final_energy",
        "mean_min_energy",
        "best_min_energy",
        "mean_final_energy_per_constraint",
        "mean_min_energy_per_constraint",
        "best_min_energy_per_constraint",
        "mean_final_ag_mean",
        "mean_min_ag_mean",
        "best_min_ag_mean",
    ]
    _write_tsv(output_dir / "param_value_summary.tsv", param_summary_rows, param_fields)

    per_config_iters_norm = _normalize_iteration_energies(per_config_iters, complete_rows)
    _plot_energy_trajectories(
        per_config_iters_norm,
        complete_rows,
        plots_dir / "energy_trajectories.svg",
        energy_metric_key="energy_per_constraint",
        summary_metric_key="min_energy_per_constraint",
        title="Sweep Energy/Constraint Trajectories (all configs)",
        y_label="Total energy per constraint",
        best_label="best normalized-energy",
    )
    _plot_energy_trajectories(
        per_config_iters,
        complete_rows,
        plots_dir / "energy_trajectories_raw.svg",
        energy_metric_key="energy",
        summary_metric_key="min_energy",
        title="Sweep Energy Trajectories (all configs, raw)",
        y_label="Total energy",
        best_label="best raw-energy",
    )
    _plot_ag_trajectories(per_config_iters, complete_rows, plots_dir / "ag_mean_trajectories.svg")
    _plot_pareto_min_energy_vs_ag(
        complete_rows,
        plots_dir / "pareto_min_energy_vs_min_ag.svg",
        max_annotations=args.max_pareto_annotations,
        energy_metric_key="min_energy_per_constraint",
        title="Config Pareto View: min AG score vs min energy/constraint",
        y_label="min energy per constraint (lower is better)",
    )
    _plot_pareto_min_energy_vs_ag(
        complete_rows,
        plots_dir / "pareto_min_energy_raw_vs_min_ag.svg",
        max_annotations=args.max_pareto_annotations,
        energy_metric_key="min_energy",
        title="Config Pareto View: min AG score vs min energy (raw)",
        y_label="min total energy (lower is better)",
    )
    _plot_metrics_vs_temperature(
        complete_rows,
        plots_dir / "metrics_vs_temperature.svg",
        energy_metric_key="min_energy_per_constraint",
        energy_label="min energy per constraint",
        title="Best normalized metrics vs sweep temperature",
    )
    _plot_metrics_vs_temperature(
        complete_rows,
        plots_dir / "metrics_vs_temperature_raw_energy.svg",
        energy_metric_key="min_energy",
        energy_label="min total energy",
        title="Best metrics vs sweep temperature (raw energy)",
    )
    _plot_param_value_bars(
        param_summary_rows,
        metric_key="best_min_energy_per_constraint",
        title="Best min energy per constraint by parameter value",
        out_path=plots_dir / "best_min_energy_by_param_value.svg",
    )
    _plot_param_value_bars(
        param_summary_rows,
        metric_key="best_min_energy_per_constraint",
        title="Best min energy per constraint by parameter value",
        out_path=plots_dir / "best_min_energy_per_constraint_by_param_value.svg",
    )
    _plot_param_value_bars(
        param_summary_rows,
        metric_key="best_min_energy",
        title="Best min energy by parameter value (raw)",
        out_path=plots_dir / "best_min_energy_raw_by_param_value.svg",
    )
    _plot_param_value_bars(
        param_summary_rows,
        metric_key="best_min_ag_mean",
        title="Best min AG score by parameter value",
        out_path=plots_dir / "best_min_ag_by_param_value.svg",
    )

    print(f"[OK] Sweep ID: {sweep_id}")
    print(f"[OK] Wrote: {output_dir / 'config_summary.tsv'}")
    print(f"[OK] Wrote: {output_dir / 'iteration_metrics.tsv'}")
    print(f"[OK] Wrote: {output_dir / 'top_by_min_energy.tsv'}")
    print(f"[OK] Wrote: {output_dir / 'top_by_min_energy_per_constraint.tsv'}")
    print(f"[OK] Wrote: {output_dir / 'top_by_min_ag.tsv'}")
    print(f"[OK] Wrote: {output_dir / 'param_value_summary.tsv'}")
    print(f"[OK] Wrote plots under: {plots_dir}")


if __name__ == "__main__":
    main()
