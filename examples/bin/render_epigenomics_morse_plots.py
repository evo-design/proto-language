#!/usr/bin/env python3
"""Render standard epigenomics MORSE run plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_TOOLS_ROOT = REPO_ROOT / "proto-tools"
for path in (str(PROTO_TOOLS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass
class ProposalPoint:
    raw_index: int
    parent: int
    energy: float


@dataclass
class SelectedPoint:
    rank: int
    parent: int
    raw_index: int
    energy: float


@dataclass
class StepRecord:
    step: int
    total_steps: int
    proposals: list[ProposalPoint]
    selected: list[SelectedPoint]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Run directory produced by epigenomics_morse_proto.py."
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <run-dir>/plots.")
    parser.add_argument(
        "--optimization-history",
        type=Path,
        default=None,
        help="Optional explicit optimization_history.csv path.",
    )
    parser.add_argument(
        "--candidates-per-beam",
        type=int,
        default=None,
        help="Sampled proposals per parent beam if not inferable from optimizer.proposals_per_beam.",
    )
    parser.add_argument("--first-steps", type=int, default=20, help="Also render a truncated trajectory view.")
    parser.add_argument("--plot-end-bp", type=int, default=20_000, help="Track overlay x-axis limit in bp.")
    parser.add_argument(
        "--device", default=None, help="Device for Borzoi and Enformer inference. Defaults to run scoring_device."
    )
    parser.add_argument("--skip-trajectory", action="store_true", help="Skip beam trajectory rendering.")
    parser.add_argument("--skip-overlay", action="store_true", help="Skip Borzoi and Enformer overlay rendering.")
    parser.add_argument("--hide-legend", action="store_true", help="Hide legends.")
    return parser.parse_args()


def _read_fasta_sequence(path: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    sequence = "".join(lines).upper()
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return sequence


def _parse_float(raw: str | None) -> float:
    if raw is None or raw == "":
        raise ValueError("Encountered missing proposal score in optimization history")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Encountered non-finite proposal score: {raw}")
    return value


def _infer_candidates_per_beam(history_path: Path, candidates_override: int | None) -> int:
    if candidates_override is not None:
        return candidates_override

    with history_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("optimizer.proposals_per_beam")
            if value:
                return int(value)

    raise ValueError("Could not infer candidates per beam; pass --candidates-per-beam explicitly.")


def _parse_optimization_history(history_path: Path, *, candidates_per_beam: int) -> list[StepRecord]:
    proposals_by_step: dict[int, list[ProposalPoint]] = {}
    selected_by_step: dict[int, list[SelectedPoint]] = {}
    total_steps_by_step: dict[int, int] = {}

    with history_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("pool") != "proposal":
                continue

            step = int(row["timepoint"])
            total_steps_by_step[step] = int(row.get("optimizer.num_beams") or step)
            raw_index = int(row["proposal_idx"])
            energy = _parse_float(row.get("energy_score"))
            proposal = ProposalPoint(
                raw_index=raw_index,
                parent=raw_index // candidates_per_beam,
                energy=energy,
            )
            proposals_by_step.setdefault(step, []).append(proposal)

            if row.get("accepted", "").lower() == "true":
                selected_by_step.setdefault(step, []).append(
                    SelectedPoint(
                        rank=0,
                        parent=proposal.parent,
                        raw_index=proposal.raw_index,
                        energy=proposal.energy,
                    )
                )

    steps = []
    for step in sorted(proposals_by_step):
        selected = sorted(selected_by_step.get(step, []), key=lambda point: (point.energy, point.raw_index))
        for rank, point in enumerate(selected, start=1):
            point.rank = rank
        steps.append(
            StepRecord(
                step=step,
                total_steps=total_steps_by_step.get(step, step),
                proposals=sorted(proposals_by_step[step], key=lambda point: point.raw_index),
                selected=selected,
            )
        )

    if not steps:
        raise ValueError(f"No proposal rows were parsed from {history_path}; export with include_proposals=True.")
    return steps


def _plot_beam_trajectory(
    steps: list[StepRecord],
    *,
    title: str,
    output_path: Path,
    hide_legend: bool,
) -> None:
    total_steps = steps[-1].total_steps
    visible_max_step = steps[-1].step
    figure_width = max(16.0, total_steps * 0.12)

    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(figure_width, 8.0))

    proposal_xs: list[float] = []
    proposal_ys: list[float] = []
    selected_xs: list[float] = []
    selected_ys: list[float] = []
    faint_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    selected_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    base_color = "#1e293b"
    proposal_color = "#94a3b8"

    for step_idx, step in enumerate(steps):
        previous_selected = steps[step_idx - 1].selected if step_idx > 0 else []
        previous_by_parent = {selected.rank - 1: selected for selected in previous_selected}
        previous_x = float(steps[step_idx - 1].step) if step_idx > 0 else 0.0
        step_x = float(step.step)

        for proposal in step.proposals:
            proposal_xs.append(step_x)
            proposal_ys.append(proposal.energy)
            parent_selected = previous_by_parent.get(proposal.parent)
            if parent_selected is not None:
                faint_segments.append(((previous_x, parent_selected.energy), (step_x, proposal.energy)))

        for selected in step.selected:
            selected_xs.append(step_x)
            selected_ys.append(selected.energy)
            parent_selected = previous_by_parent.get(selected.parent)
            if parent_selected is not None:
                selected_segments.append(((previous_x, parent_selected.energy), (step_x, selected.energy)))

    if faint_segments:
        ax.add_collection(LineCollection(faint_segments, colors=[(0.58, 0.64, 0.72, 0.32)], linewidths=0.45, zorder=1))
    if proposal_xs:
        ax.scatter(proposal_xs, proposal_ys, s=30, color=proposal_color, alpha=0.35, linewidths=0.0, zorder=2)
    if selected_segments:
        ax.add_collection(
            LineCollection(selected_segments, colors=[(0.12, 0.16, 0.23, 0.95)], linewidths=1.8, zorder=4)
        )
    if selected_xs:
        ax.scatter(selected_xs, selected_ys, s=66, color=base_color, edgecolors=base_color, linewidths=0.0, zorder=5)

    ax.set_xlim(0.5, visible_max_step + 0.5)
    ax.set_xlabel("Beam Search Step")
    ax.set_ylabel("Energy Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if not hide_legend:
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color=proposal_color,
                    markersize=7,
                    alpha=0.35,
                    label="Sampled child",
                ),
                Line2D([0], [0], marker="o", linestyle="None", color=base_color, markersize=9, label="Selected child"),
                Line2D([0], [0], color=proposal_color, alpha=0.32, linewidth=1.0, label="Parent-child edge"),
                Line2D([0], [0], color=base_color, linewidth=2.0, label="Selected edge"),
            ],
            loc="upper right",
            frameon=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in run config: {config_path}")
    return payload


def _load_construct_parts(run_dir: Path, run_config: dict[str, Any]) -> tuple[str, str, str]:
    target_sequence = _read_fasta_sequence(run_dir / "best_target.fa")
    left_path = Path(str(run_config["left_flank_path"]))
    right_path = Path(str(run_config["right_flank_path"]))
    if left_path.exists() and right_path.exists():
        return _read_fasta_sequence(left_path), target_sequence, _read_fasta_sequence(right_path)

    full_sequence = _read_fasta_sequence(run_dir / "best_sequence.fa")
    target_start = full_sequence.find(target_sequence)
    if target_start < 0:
        raise ValueError("Could not locate best target sequence inside best full sequence.")
    return full_sequence[:target_start], target_sequence, full_sequence[target_start + len(target_sequence) :]


def _slice_track_signals(
    signals: np.ndarray,
    *,
    output_start: int,
    output_resolution: float,
    target_start: int,
    target_end: int,
    plot_end_bp: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_abs = np.arange(signals.shape[1], dtype=np.float32) * float(output_resolution) + float(output_start)
    slice_end = min(target_end, target_start + plot_end_bp)
    start_idx = max(0, int(np.floor((target_start - output_start) / output_resolution)))
    end_idx = min(signals.shape[1], int(np.ceil((slice_end - output_start) / output_resolution)))
    if end_idx <= start_idx:
        raise ValueError("Model output did not overlap the requested plot window.")
    return x_abs[start_idx:end_idx] - float(target_start), signals[:, start_idx:end_idx]


def _plot_track_overlay(
    *,
    run_dir: Path,
    output_dir: Path,
    run_name: str,
    plot_end_bp: int,
    device_override: str | None,
    hide_legend: bool,
) -> Path:
    from proto_tools.tools.sequence_scoring.borzoi import (
        BORZOI_CONTEXT,
        BORZOI_OUTPUT_FLANK,
        BorzoiEnsembleConfig,
        BorzoiInput,
        SequenceTargetRange,
        SequenceWindow,
        run_borzoi_ensemble,
    )
    from proto_tools.tools.sequence_scoring.enformer import (
        ENFORMER_CONTEXT,
        ENFORMER_OUTPUT_FLANK,
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )
    from proto_tools.utils.tool_instance import ToolInstance

    from proto_language.constraint import (
        BorzoiChromatinAccessibilityMorseConfig,
        EnformerChromatinAccessibilityMorseConfig,
    )
    from proto_language.constraint.sequence_annotation.chromatin_accessibility_morse_utils import (
        compute_morse_windows,
        prepare_context_padded_candidate,
        reduce_2d_by_method,
    )
    from proto_language.core import Sequence

    run_config = _load_run_config(run_dir)
    borzoi_config = BorzoiChromatinAccessibilityMorseConfig(
        organism=str(run_config["organism"]),
        borzoi_output_tracks=run_config.get("borzoi_tracks"),
    )
    enformer_config = EnformerChromatinAccessibilityMorseConfig(
        organism=str(run_config["organism"]),
        enformer_output_tracks=run_config.get("enformer_tracks"),
        enformer_track_reduce_method=run_config.get("enformer_track_reduce_method", "mean"),
    )
    left_sequence, target_sequence, right_sequence = _load_construct_parts(run_dir, run_config)
    borzoi_sequence, borzoi_target_start, borzoi_target_end = prepare_context_padded_candidate(
        (Sequence(left_sequence), Sequence(target_sequence), Sequence(right_sequence)),
        trim_prefix_bp=0,
        output_flank=BORZOI_OUTPUT_FLANK,
        context_length=BORZOI_CONTEXT,
        model_name="Borzoi",
    )
    device = device_override or str(run_config.get("scoring_device") or run_config.get("evo_device") or "cuda:0")

    with ToolInstance.persist():
        borzoi_result = run_borzoi_ensemble(
            BorzoiInput(
                sequences=[
                    SequenceWindow(
                        sequence=borzoi_sequence,
                        target_range=SequenceTargetRange(start=borzoi_target_start, end=borzoi_target_end),
                    )
                ],
            ),
            BorzoiEnsembleConfig(
                output_tracks=borzoi_config.borzoi_output_tracks,
                species=borzoi_config.organism,
                avg_output_tracks=True,
                batch_size=1,
                device=device,
            ),
        )
        enformer_sequence, enformer_target_start, enformer_target_end = prepare_context_padded_candidate(
            (Sequence(left_sequence), Sequence(target_sequence), Sequence(right_sequence)),
            trim_prefix_bp=0,
            output_flank=ENFORMER_OUTPUT_FLANK,
            context_length=ENFORMER_CONTEXT,
            model_name="Enformer",
        )
        enformer_result = run_enformer(
            EnformerInput(
                sequences=[
                    SequenceWindow(
                        sequence=enformer_sequence,
                        target_range=SequenceTargetRange(start=enformer_target_start, end=enformer_target_end),
                    )
                ],
            ),
            EnformerConfig(
                output_tracks=enformer_config.enformer_output_tracks,
                species=enformer_config.organism,
                batch_size=1,
                device=device,
            ),
        )

    borzoi_prediction = borzoi_result.results[0]
    borzoi_preds = np.asarray(borzoi_prediction.predictions, dtype=np.float32)
    if borzoi_preds.ndim != 3:
        raise ValueError(f"Unexpected Borzoi ensemble prediction shape: {borzoi_preds.shape}")
    borzoi_signals = borzoi_preds.mean(axis=1)
    borzoi_x, borzoi_y = _slice_track_signals(
        borzoi_signals,
        output_start=borzoi_prediction.output_start,
        output_resolution=float(borzoi_prediction.output_resolution),
        target_start=borzoi_target_start,
        target_end=borzoi_target_end,
        plot_end_bp=plot_end_bp,
    )

    enformer_prediction = enformer_result.results[0]
    enformer_pred = np.asarray(enformer_prediction.prediction, dtype=np.float32)
    if enformer_pred.ndim != 2:
        raise ValueError(f"Unexpected Enformer prediction shape: {enformer_pred.shape}")
    enformer_signal = reduce_2d_by_method(
        enformer_pred,
        axis=1,
        method=enformer_config.enformer_track_reduce_method,
    )
    enformer_x, enformer_y = _slice_track_signals(
        enformer_signal[None, :],
        output_start=enformer_prediction.output_start,
        output_resolution=float(enformer_prediction.output_resolution),
        target_start=enformer_target_start,
        target_end=enformer_target_end,
        plot_end_bp=plot_end_bp,
    )
    enformer_y = enformer_y[0]

    borzoi_max = float(np.max(borzoi_y)) if borzoi_y.size else 0.0
    enformer_max = float(np.max(enformer_y)) if enformer_y.size else 0.0
    enformer_scale = borzoi_max / enformer_max if enformer_max > 0.0 else 1.0
    enformer_scaled = enformer_y * enformer_scale
    highs, lows = compute_morse_windows(
        pattern=str(run_config["pattern"]),
        pattern_start_bp=int(run_config["pattern_start_bp"]),
        dot_bp=int(run_config["dot_bp"]),
        dash_bp=int(run_config["dash_bp"]),
        intra_symbol_gap_bp=int(run_config["intra_symbol_gap_bp"]),
        inter_letter_gap_bp=int(run_config["inter_letter_gap_bp"]),
    )

    fig, ax = plt.subplots(figsize=(16, 4.8))
    colors = ["#4f6bed", "#2f7f7b", "#8b5cf6", "#b56a2b"]
    for idx, signal in enumerate(borzoi_y):
        ax.plot(
            borzoi_x,
            signal,
            linewidth=1.4,
            alpha=0.92,
            color=colors[idx % len(colors)],
            label=f"Borzoi replicate {idx + 1}",
        )
    ax.plot(
        enformer_x,
        enformer_scaled,
        color="#111827",
        linewidth=2.2,
        alpha=0.95,
        label=f"Enformer scaled x{enformer_scale:.2f}",
    )

    for idx, (start, end) in enumerate(highs):
        if end > 0 and start < plot_end_bp:
            ax.axvspan(
                max(0, start),
                min(plot_end_bp, end),
                color="#4CAF50",
                alpha=0.14,
                label="dot/dash" if idx == 0 else None,
            )
    for idx, (start, end) in enumerate(lows):
        if end > 0 and start < plot_end_bp:
            ax.axvspan(
                max(0, start), min(plot_end_bp, end), color="#F44336", alpha=0.07, label="gap" if idx == 0 else None
            )

    plot_end_label = f"{plot_end_bp / 1000:g} kb"
    ax.axvline(plot_end_bp, color="#9CA3AF", linestyle=":", linewidth=1.0, label=f"{plot_end_label} end")
    ax.set_xlim(0, plot_end_bp)
    ax.set_xlabel("Position in designed region (bp)")
    ax.set_ylabel("Accessibility signal")
    ax.set_title(f"{run_name} Borzoi + Enformer overlay over {plot_end_label}")
    if not hide_legend:
        ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()

    plot_end_slug = f"{plot_end_bp // 1000}kb" if plot_end_bp % 1000 == 0 else f"{plot_end_bp}bp"
    output_path = output_dir / f"borzoi_enformer_overlay_{plot_end_slug}.svg"
    fig.savefig(output_path, dpi=200, format="svg")
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path: Path | None = None
    first_steps_path: Path | None = None
    if not args.skip_trajectory:
        history_path = (
            args.optimization_history.resolve()
            if args.optimization_history is not None
            else run_dir / "optimization_history.csv"
        )
        candidates_per_beam = _infer_candidates_per_beam(
            history_path,
            candidates_override=args.candidates_per_beam,
        )
        steps = _parse_optimization_history(history_path, candidates_per_beam=candidates_per_beam)
        trajectory_path = output_dir / "beam_sampling_trajectory.svg"
        _plot_beam_trajectory(
            steps,
            title=f"{run_dir.name}: sampled beam trajectory",
            output_path=trajectory_path,
            hide_legend=args.hide_legend,
        )
        if args.first_steps > 0:
            truncated_steps = [step for step in steps if step.step <= args.first_steps]
            if truncated_steps:
                first_steps_path = output_dir / f"beam_sampling_trajectory_first{args.first_steps}.svg"
                _plot_beam_trajectory(
                    truncated_steps,
                    title=f"{run_dir.name}: sampled beam trajectory (first {args.first_steps} steps)",
                    output_path=first_steps_path,
                    hide_legend=args.hide_legend,
                )

    track_overlay_path: Path | None = None
    if not args.skip_overlay:
        track_overlay_path = _plot_track_overlay(
            run_dir=run_dir,
            output_dir=output_dir,
            run_name=run_dir.name,
            plot_end_bp=args.plot_end_bp,
            device_override=args.device,
            hide_legend=args.hide_legend,
        )

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "beam_sampling_trajectory_svg": str(trajectory_path) if trajectory_path is not None else None,
        "beam_sampling_trajectory_first_steps_svg": str(first_steps_path) if first_steps_path is not None else None,
        "borzoi_enformer_overlay_svg": str(track_overlay_path) if track_overlay_path is not None else None,
    }
    (output_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
