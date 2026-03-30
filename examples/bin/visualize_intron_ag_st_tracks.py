#!/usr/bin/env python3
"""
Visualize aligned AlphaGenome and SpliceTransformer intron tracks.

This script rebuilds the 1 kb target sequence used by the intron design
program, then plots four aligned track groups on the same target coordinate
system:
  1) AlphaGenome RNA-seq target/offtarget tracks
  2) AlphaGenome splice-site-usage target/offtarget tracks
  3) SpliceTransformer donor/acceptor channels
  4) SpliceTransformer target/offtarget tissue channels

AlphaGenome predictions are still run on the integrated genomic-context
sequence used during design. For plotting, the requested target window is
cropped back out and mapped onto the 1 kb target coordinates so the AlphaGenome
and SpliceTransformer rows can be compared directly.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tap import Tap

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proto_tools.tools.rna_splicing.splice_transformer import (
    SPLICE_TISSUE_CHANNEL_INDEX,
    SpliceTransformerConfig,
    SpliceTransformerInput,
    SpliceTransformerTissue,
    SpliceTransformerType,
    run_splice_transformer,
)
from proto_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)
from proto_tools.utils.tool_instance import ToolInstance

from examples.scripts.program_intron_alphagenome import (
    DEFAULT_CELL_ONTOLOGY_TERMS,
    DEFAULT_GENOMIC_CONTEXT_PATHS,
)
from examples.scripts.program_intron_design import process_splice_transformer_input

DEFAULT_PLASMID_CONTEXT_PATHS = ",".join(
    [
        "examples/data/plasmid_context_cmv_20260308.txt",
        "examples/data/plasmid_context_Ef1a.txt",
        "examples/data/plasmid_context_sffv.txt",
    ]
)

DNA_PATTERN = re.compile(r"[ACGTNacgtn]+")
LOG_INTRON_PATTERN = re.compile(r"sequence \(intron\):\s*([ACGTNacgtn]+)")

TARGET_COLOR = "#1f77b4"
OFFTARGET_COLOR = "#d62728"
DONOR_COLOR = "#9467bd"
ACCEPTOR_COLOR = "#2ca02c"
DONOR_EVAL_COLOR = "#6a3d9a"
ACCEPTOR_EVAL_COLOR = "#1b9e77"


class VisualizeIntronAGSTTracksArgs(Tap):
    stdout_log: str = ""
    design_sequences_path: str = ""
    intron_sequences_csv: str = ""
    log_selection: Literal["last", "all"] = "last"
    max_designs: int = 1

    plasmid_context_paths: str = DEFAULT_PLASMID_CONTEXT_PATHS
    genomic_context_paths: str = DEFAULT_GENOMIC_CONTEXT_PATHS
    gene_sequence_path: str = "examples/data/mscarlet_ires_zsgreen.txt"
    gene_insertion_pos: int = 159 * 3

    target_cell: str = "shsy5y"
    offtarget_cell: str = "k562"
    target_ontology_terms: str = ""
    offtarget_ontology_terms: str = ""

    target_tissue: str = "BRAIN"
    offtarget_tissue: str = "BLOOD"

    alphagenome_model_version: str = "all_folds"
    alphagenome_organism: Literal["human", "mouse"] = "human"
    alphagenome_device: str = "cuda"
    alphagenome_track_strand: Literal["positive", "negative", "all"] = "positive"

    splice_transformer_device: str = "cuda"

    smoothing_window: int = 1
    view_mode: Literal["full", "gene", "intron"] = "gene"
    gene_flank_bp: int = 160
    intron_flank_bp: int = 80
    aggregate_only: bool = False

    output_dir: str = "examples/outputs/intron_ag_st_tracks"
    filename_prefix: str = ""


def _split_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _normalize_intron_sequence(raw_sequence: str) -> str:
    sequence = raw_sequence.strip().upper()
    if not sequence:
        raise ValueError("Encountered empty intron sequence.")
    invalid = set(sequence) - set("ACGTN")
    if invalid:
        raise ValueError(
            f"Intron sequence has invalid DNA characters: {sorted(invalid)}."
        )
    if len(sequence) < 4:
        warnings.warn(
            f"Very short intron sequence ({len(sequence)} bp) encountered; attempting canonicalization.",
            stacklevel=2,
        )

    if sequence.startswith("GT") and sequence.endswith("AG"):
        return sequence

    canonicalized = sequence
    missing_gt = not canonicalized.startswith("GT")
    missing_ag = not canonicalized.endswith("AG")
    if missing_gt:
        canonicalized = f"GT{canonicalized}"
    if missing_ag:
        canonicalized = f"{canonicalized}AG"

    warnings.warn(
        "Non-canonical intron detected; auto-canonicalizing to GT...AG "
        f"(missing_prefix_GT={missing_gt}, missing_suffix_AG={missing_ag}, "
        f"old_len={len(sequence)}, new_len={len(canonicalized)}).",
        stacklevel=2,
    )
    return canonicalized


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _load_introns_from_stdout_log(path: str) -> List[str]:
    introns: List[str] = []
    for line in Path(path).read_text().splitlines():
        match = LOG_INTRON_PATTERN.search(line)
        if not match:
            continue
        introns.append(_normalize_intron_sequence(match.group(1)))
    return introns


def _load_introns_from_sequence_file(path: str) -> List[str]:
    lines = Path(path).read_text().splitlines()
    if any(line.startswith(">") for line in lines):
        sequences: List[str] = []
        current: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if current:
                    sequences.append(_normalize_intron_sequence("".join(current)))
                current = []
                continue
            current.append(stripped)
        if current:
            sequences.append(_normalize_intron_sequence("".join(current)))
        return sequences

    sequences = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        matches = DNA_PATTERN.findall(stripped)
        if not matches:
            continue
        candidate = max(matches, key=len)
        if len(candidate) < 8:
            continue
        try:
            sequences.append(_normalize_intron_sequence(candidate))
        except ValueError:
            continue
    return sequences


def _load_design_introns(args: VisualizeIntronAGSTTracksArgs) -> List[str]:
    introns: List[str] = []

    if args.intron_sequences_csv:
        introns.extend(
            _normalize_intron_sequence(seq) for seq in _split_csv(args.intron_sequences_csv)
        )

    if args.design_sequences_path:
        introns.extend(_load_introns_from_sequence_file(args.design_sequences_path))

    if args.stdout_log:
        log_introns = _load_introns_from_stdout_log(args.stdout_log)
        if args.log_selection == "last":
            if log_introns:
                introns.append(log_introns[-1])
        else:
            introns.extend(log_introns)

    introns = _dedupe_keep_order(introns)
    if args.max_designs > 0:
        introns = introns[-args.max_designs :]
    return introns


def _resolve_terms(cell_name: str, override_terms_csv: str) -> List[str]:
    override_terms = _split_csv(override_terms_csv)
    if override_terms:
        return override_terms
    key = cell_name.strip().lower()
    if key in DEFAULT_CELL_ONTOLOGY_TERMS:
        return DEFAULT_CELL_ONTOLOGY_TERMS[key]
    if ":" in cell_name:
        return [cell_name.strip()]
    if key:
        raise ValueError(
            f"Unsupported cell alias '{cell_name}'. Provide a known alias or explicit ontology term."
        )
    raise ValueError("Cell name / ontology term cannot be empty.")


def _resolve_tissue_enum(raw_value: str) -> SpliceTransformerTissue:
    token = raw_value.strip().upper().replace("-", "_").replace(" ", "_")
    if not token:
        raise ValueError("Tissue name cannot be empty.")
    if token.startswith("SPLICETRANSFORMERTISSUE."):
        token = token.split(".", 1)[1]
    valid_tissues = set(get_args(SpliceTransformerTissue))
    if token not in valid_tissues:
        valid = ", ".join(sorted(valid_tissues))
        raise ValueError(f"Unsupported tissue '{raw_value}'. Valid values: {valid}")
    return token


def _tissue_channel_index(tissue: SpliceTransformerTissue) -> Optional[int]:
    return SPLICE_TISSUE_CHANNEL_INDEX[tissue]


def _extract_tissue_signal(
    prediction: np.ndarray,
    tissue: SpliceTransformerTissue,
) -> np.ndarray:
    channel = _tissue_channel_index(tissue)
    if channel is None:
        tissue_channels = sorted(
            index for index in SPLICE_TISSUE_CHANNEL_INDEX.values() if index is not None
        )
        return np.mean(prediction[:, tissue_channels], axis=1)
    return prediction[:, channel]


def _read_context_sequence(path: str) -> str:
    sequence = Path(path).read_text().strip().upper()
    if not sequence:
        raise ValueError(f"Context file is empty: {path}")
    invalid_chars = set(sequence) - set("ACGTN")
    if invalid_chars:
        raise ValueError(
            f"Context file contains invalid DNA characters {sorted(invalid_chars)}: {path}"
        )
    return sequence


def _integrate_cassette_into_context(
    genomic_context: str,
    cassette_sequence: str,
) -> Tuple[str, int]:
    if len(cassette_sequence) > len(genomic_context):
        raise ValueError(
            f"Cassette length {len(cassette_sequence)} exceeds genomic context length {len(genomic_context)}."
        )
    insert_start = (len(genomic_context) - len(cassette_sequence)) // 2
    insert_end = insert_start + len(cassette_sequence)
    integrated = (
        genomic_context[:insert_start]
        + cassette_sequence
        + genomic_context[insert_end:]
    )
    if len(integrated) != len(genomic_context):
        raise RuntimeError("Integrated sequence length mismatch.")
    return integrated, insert_start


def _safe_numeric_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0 or arr.size == 0:
        return None
    if not np.isfinite(arr).any():
        return None
    return arr


def _collect_value_arrays(node: Any, arrays: List[np.ndarray]) -> None:
    if isinstance(node, dict):
        if "values" in node:
            arr = _safe_numeric_array(node["values"])
            if arr is not None:
                arrays.append(arr)
        for child in node.values():
            _collect_value_arrays(child, arrays)
        return

    if isinstance(node, list):
        for child in node:
            _collect_value_arrays(child, arrays)


def _extract_rna_matrix(result_payload: Dict[str, Any]) -> np.ndarray:
    predictions = result_payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("AlphaGenome result payload missing 'predictions' dictionary.")

    rna_payload = predictions.get("rna_seq")
    if rna_payload is None:
        for key, value in predictions.items():
            if str(key).lower() == "rna_seq":
                rna_payload = value
                break
    if rna_payload is None:
        raise ValueError("AlphaGenome prediction payload missing RNA_SEQ output.")

    arrays: List[np.ndarray] = []
    _collect_value_arrays(rna_payload, arrays)
    if not arrays:
        raise ValueError("Unable to extract RNA values from AlphaGenome payload.")

    matrix = max(arrays, key=lambda arr: (arr.shape[0], arr.size))
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)

    if matrix.shape[1] == 0:
        warnings.warn(
            "AlphaGenome RNA payload has zero columns; using zero-valued placeholder track.",
            stacklevel=2,
        )
        matrix = np.zeros((matrix.shape[0], 1), dtype=float)

    if not np.isfinite(matrix).any():
        warnings.warn(
            "AlphaGenome RNA track contains no finite values; replacing with zeros.",
            stacklevel=2,
        )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_output_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _extract_splice_site_usage_track_payload(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    predictions = result_payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("AlphaGenome result payload missing 'predictions' dictionary.")

    requested_key = _normalize_output_key("SPLICE_SITE_USAGE")
    for key, value in predictions.items():
        if _normalize_output_key(str(key)) != requested_key:
            continue
        if not isinstance(value, dict):
            raise ValueError("AlphaGenome SPLICE_SITE_USAGE payload is not a dictionary.")
        return value

    raise ValueError("AlphaGenome prediction payload missing SPLICE_SITE_USAGE output.")


def _extract_track_metadata_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = payload.get("metadata")
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return [row for row in metadata if isinstance(row, dict)]
    if isinstance(metadata, dict):
        records = metadata.get("records")
        if isinstance(records, list):
            return [row for row in records if isinstance(row, dict)]
    return []


def _extract_track_matrix(payload: Dict[str, Any]) -> np.ndarray:
    arr = _safe_numeric_array(payload.get("values"))
    if arr is None:
        raise ValueError("Unable to extract SPLICE_SITE_USAGE values from payload.")

    matrix = arr
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return matrix


def _strand_to_symbol(strand: str) -> str:
    mapping = {
        "positive": "+",
        "negative": "-",
        "all": ".",
    }
    return mapping[strand]


def _select_track_columns(
    matrix: np.ndarray,
    metadata_records: List[Dict[str, Any]],
    strand: str,
) -> np.ndarray:
    if strand == "all":
        return matrix

    strand_symbol = _strand_to_symbol(strand)
    if not metadata_records:
        raise ValueError(
            "SPLICE_SITE_USAGE metadata is missing; cannot apply strand-specific track selection."
        )

    selected_indices: List[int] = []
    for idx, row in enumerate(metadata_records):
        if idx >= matrix.shape[1]:
            break
        if str(row.get("strand", "")).strip() == strand_symbol:
            selected_indices.append(idx)

    if not selected_indices:
        raise ValueError(
            f"No SPLICE_SITE_USAGE tracks matched strand='{strand_symbol}' in metadata."
        )
    return matrix[:, selected_indices]


def _extract_ssu_signal(
    result_payload: Dict[str, Any],
    strand: str,
    expected_length: int,
) -> np.ndarray:
    try:
        payload = _extract_splice_site_usage_track_payload(result_payload)
        matrix = _extract_track_matrix(payload)
    except ValueError as exc:
        warnings.warn(f"{exc} Using zero-valued SSU track.", stacklevel=2)
        return np.zeros(expected_length, dtype=float)

    if matrix.shape[0] != expected_length:
        warnings.warn(
            "SPLICE_SITE_USAGE row count did not match sequence length; "
            f"got {matrix.shape[0]}, expected {expected_length}. Using zeros.",
            stacklevel=2,
        )
        return np.zeros(expected_length, dtype=float)

    metadata_records = _extract_track_metadata_records(payload)
    try:
        selected_matrix = _select_track_columns(matrix, metadata_records, strand)
    except ValueError as exc:
        warnings.warn(f"{exc} Using zero-valued SSU track.", stacklevel=2)
        return np.zeros(expected_length, dtype=float)

    if selected_matrix.shape[1] == 0:
        warnings.warn("Selected SSU track matrix had zero columns; using zeros.", stacklevel=2)
        return np.zeros(expected_length, dtype=float)

    signal = np.mean(selected_matrix, axis=1)
    signal = np.clip(signal, 0.0, 1.0)
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_alphagenome_payloads(
    sequences: List[str],
    ontology_terms: List[str],
    args: VisualizeIntronAGSTTracksArgs,
) -> List[Dict[str, Any]]:
    if not sequences:
        return []

    prediction_input = AlphaGenomePredictSequencesInput(sequences=sequences)
    prediction_config = AlphaGenomePredictSequencesConfig(
        model_version=args.alphagenome_model_version,
        requested_outputs=["RNA_SEQ", "SPLICE_SITE_USAGE"],
        ontology_terms=ontology_terms,
        organism=args.alphagenome_organism,
        device=args.alphagenome_device,
    )
    output = run_alphagenome_predict_sequences(prediction_input, prediction_config)
    return [result.result for result in output.results]


def _window_to_rows(
    start: int,
    end: int,
    sequence_length: int,
    num_rows: int,
) -> Tuple[int, int]:
    row_start = int(math.floor(start * num_rows / sequence_length))
    row_end = int(math.ceil(end * num_rows / sequence_length))
    row_start = max(0, min(row_start, num_rows - 1))
    row_end = max(row_start + 1, min(row_end, num_rows))
    return row_start, row_end


def _window_mean_matrix(
    matrix: np.ndarray,
    window: Tuple[int, int],
    sequence_length: int,
) -> float:
    row_start, row_end = _window_to_rows(
        start=window[0],
        end=window[1],
        sequence_length=sequence_length,
        num_rows=matrix.shape[0],
    )
    return float(np.mean(matrix[row_start:row_end, :]))


def _window_mean_signal(
    signal: np.ndarray,
    window: Tuple[int, int],
) -> float:
    start, end = window
    start = max(0, min(start, signal.shape[0]))
    end = max(start + 1, min(end, signal.shape[0]))
    return float(np.mean(signal[start:end]))


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _resample_signal_to_target(
    source_x: np.ndarray,
    source_signal: np.ndarray,
    target_length: int,
) -> np.ndarray:
    target_x = np.arange(target_length, dtype=float)
    if source_signal.shape[0] == 0:
        return np.zeros(target_length, dtype=float)
    if source_signal.shape[0] == 1:
        return np.full(target_length, float(source_signal[0]), dtype=float)
    return np.interp(target_x, source_x, source_signal)


def _extract_rna_signal_on_target(
    matrix: np.ndarray,
    sequence_length: int,
    target_window: Tuple[int, int],
    target_length: int,
    smoothing_window: int,
) -> np.ndarray:
    num_rows = matrix.shape[0]
    row_width_bp = sequence_length / float(num_rows)
    row_start, row_end = _window_to_rows(
        start=target_window[0],
        end=target_window[1],
        sequence_length=sequence_length,
        num_rows=num_rows,
    )
    row_centers_bp = (np.arange(num_rows, dtype=float) + 0.5) * row_width_bp
    segment_x = row_centers_bp[row_start:row_end] - float(target_window[0])
    segment_signal = np.mean(matrix[row_start:row_end, :], axis=1)
    signal = _resample_signal_to_target(segment_x, segment_signal, target_length)
    return _smooth_signal(signal, smoothing_window)


def _aggregate_regions(region_dicts: List[Dict[str, Tuple[int, int]]]) -> Dict[str, Tuple[int, int]]:
    if not region_dicts:
        raise ValueError("Cannot aggregate empty region list.")
    names = sorted({name for regions in region_dicts for name in regions})
    aggregated: Dict[str, Tuple[int, int]] = {}
    for name in names:
        starts = [regions[name][0] for regions in region_dicts if name in regions]
        ends = [regions[name][1] for regions in region_dicts if name in regions]
        if not starts or not ends:
            continue
        start = int(round(float(np.mean(starts))))
        end = int(round(float(np.mean(ends))))
        if end <= start:
            end = start + 1
        aggregated[name] = (start, end)
    return aggregated


def _draw_spans(ax: Any, regions: Dict[str, Tuple[int, int]]) -> None:
    style = {
        "gene": ("#a1d99b", 0.14),
        "left_exon": ("#31a354", 0.24),
        "right_exon": ("#31a354", 0.24),
        "intron": ("#bdbdbd", 0.26),
    }
    for name, (start, end) in regions.items():
        if end <= start or name not in style:
            continue
        color, alpha = style[name]
        ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0)


def _apply_std_band(
    ax: Any,
    x_bp: np.ndarray,
    mean_signal: np.ndarray,
    std_signal: Optional[np.ndarray],
    color: str,
) -> None:
    if std_signal is None:
        return
    ax.fill_between(
        x_bp,
        mean_signal - std_signal,
        mean_signal + std_signal,
        color=color,
        alpha=0.15,
        linewidth=0,
    )


def _set_xlim(
    axes: List[Any],
    regions: Dict[str, Tuple[int, int]],
    sequence_length: int,
    view_mode: str,
    gene_flank_bp: int,
    intron_flank_bp: int,
) -> None:
    if view_mode == "full":
        start, end = 0, sequence_length
    elif view_mode == "intron":
        intron_start, intron_end = regions["intron"]
        start = max(0, intron_start - intron_flank_bp)
        end = min(sequence_length, intron_end + intron_flank_bp)
    else:
        gene_start, gene_end = regions["gene"]
        start = max(0, gene_start - gene_flank_bp)
        end = min(sequence_length, gene_end + gene_flank_bp)

    for ax in axes:
        ax.set_xlim(start, end)


def _plot_tracks_svg(
    x_bp: np.ndarray,
    ag_rna_target: np.ndarray,
    ag_rna_offtarget: np.ndarray,
    ag_ssu_target: np.ndarray,
    ag_ssu_offtarget: np.ndarray,
    st_donor: np.ndarray,
    st_acceptor: np.ndarray,
    st_tissue_target: np.ndarray,
    st_tissue_offtarget: np.ndarray,
    sequence_length: int,
    regions: Dict[str, Tuple[int, int]],
    donor_eval_pos: int,
    acceptor_eval_pos: int,
    metrics: Dict[str, float],
    target_cell_label: str,
    offtarget_cell_label: str,
    target_tissue_label: str,
    offtarget_tissue_label: str,
    title: str,
    view_mode: str,
    gene_flank_bp: int,
    intron_flank_bp: int,
    out_path: Path,
    ag_rna_target_std: Optional[np.ndarray] = None,
    ag_rna_offtarget_std: Optional[np.ndarray] = None,
    ag_ssu_target_std: Optional[np.ndarray] = None,
    ag_ssu_offtarget_std: Optional[np.ndarray] = None,
    st_donor_std: Optional[np.ndarray] = None,
    st_acceptor_std: Optional[np.ndarray] = None,
    st_tissue_target_std: Optional[np.ndarray] = None,
    st_tissue_offtarget_std: Optional[np.ndarray] = None,
) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.8, 1.5, 1.8]},
        constrained_layout=True,
    )
    ax_ag_rna, ax_ag_ssu, ax_st_splice, ax_st_tissue = axes

    for ax in axes:
        _draw_spans(ax, regions)

    ax_ag_rna.plot(x_bp, ag_rna_target, color=TARGET_COLOR, linewidth=1.5, label=target_cell_label)
    ax_ag_rna.plot(
        x_bp,
        ag_rna_offtarget,
        color=OFFTARGET_COLOR,
        linewidth=1.5,
        label=offtarget_cell_label,
    )
    _apply_std_band(ax_ag_rna, x_bp, ag_rna_target, ag_rna_target_std, TARGET_COLOR)
    _apply_std_band(ax_ag_rna, x_bp, ag_rna_offtarget, ag_rna_offtarget_std, OFFTARGET_COLOR)
    ax_ag_rna.set_ylabel("AG RNA")
    ax_ag_rna.set_title(title, fontsize=11)
    ax_ag_rna.legend(
        handles=[
            Line2D([0], [0], color=TARGET_COLOR, linewidth=1.5, label=target_cell_label),
            Line2D([0], [0], color=OFFTARGET_COLOR, linewidth=1.5, label=offtarget_cell_label),
            Patch(facecolor="#31a354", alpha=0.24, label="Exon"),
            Patch(facecolor="#bdbdbd", alpha=0.26, label="Intron"),
        ],
        loc="upper right",
        fontsize=8,
        ncol=2,
    )

    ax_ag_ssu.plot(x_bp, ag_ssu_target, color=TARGET_COLOR, linewidth=1.4, label=target_cell_label)
    ax_ag_ssu.plot(
        x_bp,
        ag_ssu_offtarget,
        color=OFFTARGET_COLOR,
        linewidth=1.4,
        label=offtarget_cell_label,
    )
    _apply_std_band(ax_ag_ssu, x_bp, ag_ssu_target, ag_ssu_target_std, TARGET_COLOR)
    _apply_std_band(ax_ag_ssu, x_bp, ag_ssu_offtarget, ag_ssu_offtarget_std, OFFTARGET_COLOR)
    ax_ag_ssu.axvline(donor_eval_pos, color=DONOR_EVAL_COLOR, linestyle="--", linewidth=0.9)
    ax_ag_ssu.axvline(acceptor_eval_pos, color=ACCEPTOR_EVAL_COLOR, linestyle="--", linewidth=0.9)
    ax_ag_ssu.set_ylabel("AG SSU")
    ax_ag_ssu.set_ylim(-0.02, 1.02)
    ax_ag_ssu.legend(
        handles=[
            Line2D([0], [0], color=TARGET_COLOR, linewidth=1.4, label=target_cell_label),
            Line2D([0], [0], color=OFFTARGET_COLOR, linewidth=1.4, label=offtarget_cell_label),
            Line2D([0], [0], color=DONOR_EVAL_COLOR, linestyle="--", linewidth=0.9, label="Donor eval"),
            Line2D(
                [0],
                [0],
                color=ACCEPTOR_EVAL_COLOR,
                linestyle="--",
                linewidth=0.9,
                label="Acceptor eval",
            ),
        ],
        loc="upper right",
        fontsize=8,
        ncol=2,
    )

    ax_st_splice.plot(x_bp, st_donor, color=DONOR_COLOR, linewidth=1.4, label="ST donor")
    ax_st_splice.plot(x_bp, st_acceptor, color=ACCEPTOR_COLOR, linewidth=1.4, label="ST acceptor")
    _apply_std_band(ax_st_splice, x_bp, st_donor, st_donor_std, DONOR_COLOR)
    _apply_std_band(ax_st_splice, x_bp, st_acceptor, st_acceptor_std, ACCEPTOR_COLOR)
    ax_st_splice.axvline(donor_eval_pos, color=DONOR_EVAL_COLOR, linestyle="--", linewidth=0.9)
    ax_st_splice.axvline(acceptor_eval_pos, color=ACCEPTOR_EVAL_COLOR, linestyle="--", linewidth=0.9)
    ax_st_splice.set_ylabel("ST D/A")
    ax_st_splice.set_ylim(-0.02, 1.02)
    ax_st_splice.legend(
        handles=[
            Line2D([0], [0], color=DONOR_COLOR, linewidth=1.4, label="ST donor"),
            Line2D([0], [0], color=ACCEPTOR_COLOR, linewidth=1.4, label="ST acceptor"),
            Line2D([0], [0], color=DONOR_EVAL_COLOR, linestyle="--", linewidth=0.9, label="Donor eval"),
            Line2D(
                [0],
                [0],
                color=ACCEPTOR_EVAL_COLOR,
                linestyle="--",
                linewidth=0.9,
                label="Acceptor eval",
            ),
        ],
        loc="upper right",
        fontsize=8,
        ncol=2,
    )

    ax_st_tissue.plot(
        x_bp,
        st_tissue_target,
        color=TARGET_COLOR,
        linewidth=1.5,
        label=target_tissue_label,
    )
    ax_st_tissue.plot(
        x_bp,
        st_tissue_offtarget,
        color=OFFTARGET_COLOR,
        linewidth=1.5,
        label=offtarget_tissue_label,
    )
    _apply_std_band(ax_st_tissue, x_bp, st_tissue_target, st_tissue_target_std, TARGET_COLOR)
    _apply_std_band(ax_st_tissue, x_bp, st_tissue_offtarget, st_tissue_offtarget_std, OFFTARGET_COLOR)
    ax_st_tissue.axvline(donor_eval_pos, color=DONOR_EVAL_COLOR, linestyle="--", linewidth=0.9)
    ax_st_tissue.axvline(acceptor_eval_pos, color=ACCEPTOR_EVAL_COLOR, linestyle="--", linewidth=0.9)
    ax_st_tissue.set_ylabel("ST Tissue")
    ax_st_tissue.set_xlabel("Position in 1 kb target (bp)")
    ax_st_tissue.set_ylim(-0.02, 1.02)
    ax_st_tissue.legend(
        handles=[
            Line2D([0], [0], color=TARGET_COLOR, linewidth=1.5, label=target_tissue_label),
            Line2D([0], [0], color=OFFTARGET_COLOR, linewidth=1.5, label=offtarget_tissue_label),
            Line2D([0], [0], color=DONOR_EVAL_COLOR, linestyle="--", linewidth=0.9, label="Donor eval"),
            Line2D(
                [0],
                [0],
                color=ACCEPTOR_EVAL_COLOR,
                linestyle="--",
                linewidth=0.9,
                label="Acceptor eval",
            ),
        ],
        loc="upper right",
        fontsize=8,
        ncol=2,
    )

    _set_xlim(
        list(axes),
        regions=regions,
        sequence_length=sequence_length,
        view_mode=view_mode,
        gene_flank_bp=gene_flank_bp,
        intron_flank_bp=intron_flank_bp,
    )

    metric_lines = [
        f"AG RNA intron: {metrics['ag_target_intron_rna']:.4f} vs {metrics['ag_offtarget_intron_rna']:.4f}",
        f"AG SSU splice mean: {metrics['ag_target_splice_mean']:.4f} vs {metrics['ag_offtarget_splice_mean']:.4f}",
        f"ST donor/acceptor@site: {metrics['st_donor_site_prob']:.4f} / {metrics['st_acceptor_site_prob']:.4f}",
        f"ST tissue splice mean: {metrics['st_target_splice_mean']:.4f} vs {metrics['st_offtarget_splice_mean']:.4f}",
    ]
    if "ag_target_intron_rna_std" in metrics:
        metric_lines.append(
            f"AG SSU SD: {metrics['ag_target_splice_mean_std']:.4f} / {metrics['ag_offtarget_splice_mean_std']:.4f}"
        )
    ax_st_tissue.text(
        0.01,
        0.98,
        "\n".join(metric_lines),
        transform=ax_st_tissue.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _aggregate_design_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        raise ValueError("Cannot aggregate empty context entries.")

    reference_x = entries[0]["x_bp"]
    metric_keys = [
        "ag_target_intron_rna",
        "ag_offtarget_intron_rna",
        "ag_target_exon_rna",
        "ag_offtarget_exon_rna",
        "ag_rna_ratio_delta",
        "ag_target_splice_mean",
        "ag_offtarget_splice_mean",
        "st_target_intron_signal",
        "st_offtarget_intron_signal",
        "st_target_exon_signal",
        "st_offtarget_exon_signal",
        "st_ratio_delta",
        "st_donor_site_prob",
        "st_acceptor_site_prob",
        "st_boundary_score",
        "st_target_specificity_max_score",
        "st_offtarget_specificity_min_score",
        "st_target_splice_mean",
        "st_offtarget_splice_mean",
    ]
    metric_values: Dict[str, List[float]] = {key: [] for key in metric_keys}
    region_dicts: List[Dict[str, Tuple[int, int]]] = []
    donor_eval_positions: List[int] = []
    acceptor_eval_positions: List[int] = []

    signal_keys = [
        "ag_rna_target",
        "ag_rna_offtarget",
        "ag_ssu_target",
        "ag_ssu_offtarget",
        "st_donor",
        "st_acceptor",
        "st_tissue_target",
        "st_tissue_offtarget",
    ]
    stacked_signals: Dict[str, List[np.ndarray]] = {key: [] for key in signal_keys}

    for entry in entries:
        for key in signal_keys:
            signal = entry[key]
            if signal.shape[0] != reference_x.shape[0]:
                signal = np.interp(reference_x, entry["x_bp"], signal)
            stacked_signals[key].append(signal)

        for key in metric_keys:
            metric_values[key].append(float(entry["metrics"][key]))

        region_dicts.append(entry["regions"])
        donor_eval_positions.append(int(entry["donor_eval_pos"]))
        acceptor_eval_positions.append(int(entry["acceptor_eval_pos"]))

    result: Dict[str, Any] = {
        "x_bp": reference_x,
        "sequence_length": int(entries[0]["sequence_length"]),
        "regions": _aggregate_regions(region_dicts),
        "context_count": len(entries),
        "plasmid_context_count": len({entry["plasmid_label"] for entry in entries}),
        "genomic_context_count": len({entry["genomic_label"] for entry in entries}),
        "donor_eval_pos": int(round(float(np.mean(donor_eval_positions)))),
        "acceptor_eval_pos": int(round(float(np.mean(acceptor_eval_positions)))),
    }

    for key in signal_keys:
        signal_stack = np.vstack(stacked_signals[key])
        result[f"{key}_mean"] = np.mean(signal_stack, axis=0)
        result[f"{key}_std"] = np.std(signal_stack, axis=0)

    aggregated_metrics: Dict[str, float] = {}
    for key, values in metric_values.items():
        arr = np.asarray(values, dtype=float)
        aggregated_metrics[key] = float(np.mean(arr))
        aggregated_metrics[f"{key}_std"] = float(np.std(arr))
    result["metrics"] = aggregated_metrics
    return result


def _to_row(values: Dict[str, Any], field_names: List[str]) -> Dict[str, Any]:
    return {field: values.get(field, "") for field in field_names}


def main() -> None:
    args = VisualizeIntronAGSTTracksArgs(explicit_bool=True).parse_args()
    if args.smoothing_window < 1:
        raise ValueError("--smoothing_window must be >= 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intron_sequences = _load_design_introns(args)
    if not intron_sequences:
        raise ValueError(
            "No intron designs found. Provide one of: "
            "--stdout_log, --design_sequences_path, --intron_sequences_csv."
        )

    plasmid_context_paths = _split_csv(args.plasmid_context_paths)
    genomic_context_paths = _split_csv(args.genomic_context_paths)
    if not plasmid_context_paths:
        raise ValueError("No plasmid context paths provided.")
    if not genomic_context_paths:
        raise ValueError("No genomic context paths provided.")

    genomic_contexts = [
        (Path(path).stem, path, _read_context_sequence(path))
        for path in genomic_context_paths
    ]
    target_terms = _resolve_terms(args.target_cell, args.target_ontology_terms)
    offtarget_terms = _resolve_terms(args.offtarget_cell, args.offtarget_ontology_terms)
    target_tissue = _resolve_tissue_enum(args.target_tissue)
    offtarget_tissue = _resolve_tissue_enum(args.offtarget_tissue)

    selected_fasta = output_dir / "selected_introns.fasta"
    with selected_fasta.open("w") as handle:
        for idx, intron in enumerate(intron_sequences):
            handle.write(f">design_{idx:03d}\n{intron}\n")

    summary_rows: List[Dict[str, Any]] = []
    field_names = [
        "design_index",
        "design_hash",
        "intron_length",
        "plasmid_context",
        "genomic_context",
        "target_terms",
        "offtarget_terms",
        "target_tissue",
        "offtarget_tissue",
        "ag_target_intron_rna",
        "ag_offtarget_intron_rna",
        "ag_target_exon_rna",
        "ag_offtarget_exon_rna",
        "ag_rna_ratio_delta",
        "ag_target_splice_mean",
        "ag_offtarget_splice_mean",
        "st_target_intron_signal",
        "st_offtarget_intron_signal",
        "st_target_exon_signal",
        "st_offtarget_exon_signal",
        "st_ratio_delta",
        "st_donor_site_prob",
        "st_acceptor_site_prob",
        "st_boundary_score",
        "st_target_specificity_max_score",
        "st_offtarget_specificity_min_score",
        "svg_path",
    ]
    aggregated_summary_rows: List[Dict[str, Any]] = []
    aggregated_field_names = [
        "design_index",
        "design_hash",
        "intron_length",
        "context_count",
        "plasmid_context_count",
        "genomic_context_count",
        "target_terms",
        "offtarget_terms",
        "target_tissue",
        "offtarget_tissue",
        "ag_target_intron_rna_mean",
        "ag_target_intron_rna_std",
        "ag_offtarget_intron_rna_mean",
        "ag_offtarget_intron_rna_std",
        "ag_target_exon_rna_mean",
        "ag_target_exon_rna_std",
        "ag_offtarget_exon_rna_mean",
        "ag_offtarget_exon_rna_std",
        "ag_rna_ratio_delta_mean",
        "ag_rna_ratio_delta_std",
        "ag_target_splice_mean_mean",
        "ag_target_splice_mean_std",
        "ag_offtarget_splice_mean_mean",
        "ag_offtarget_splice_mean_std",
        "st_target_intron_signal_mean",
        "st_target_intron_signal_std",
        "st_offtarget_intron_signal_mean",
        "st_offtarget_intron_signal_std",
        "st_target_exon_signal_mean",
        "st_target_exon_signal_std",
        "st_offtarget_exon_signal_mean",
        "st_offtarget_exon_signal_std",
        "st_ratio_delta_mean",
        "st_ratio_delta_std",
        "st_donor_site_prob_mean",
        "st_donor_site_prob_std",
        "st_acceptor_site_prob_mean",
        "st_acceptor_site_prob_std",
        "st_boundary_score_mean",
        "st_boundary_score_std",
        "st_target_specificity_max_score_mean",
        "st_target_specificity_max_score_std",
        "st_offtarget_specificity_min_score_mean",
        "st_offtarget_specificity_min_score_std",
        "svg_path",
    ]

    with ToolInstance.persist():
        for design_index, intron_sequence in enumerate(intron_sequences):
            design_hash = hashlib.sha1(intron_sequence.encode()).hexdigest()[:10]

            plasmid_records: List[Dict[str, Any]] = []
            context_records: List[Dict[str, Any]] = []

            for plasmid_context_path in plasmid_context_paths:
                splice_args = SimpleNamespace(
                    plasmid_context_path=plasmid_context_path,
                    gene_sequence_path=args.gene_sequence_path,
                    gene_insertion_pos=args.gene_insertion_pos,
                )
                (
                    left_context,
                    right_context,
                    target_seq,
                    gene_start_pos,
                    gene_end_pos,
                    donor_start_pos,
                    acceptor_end_pos,
                ) = process_splice_transformer_input(intron_sequence, splice_args)

                target_length = len(target_seq)
                donor_eval_pos = max(0, min(target_length - 1, donor_start_pos - 1))
                acceptor_eval_pos = max(0, min(target_length - 1, acceptor_end_pos + 1))
                intron_window = (donor_start_pos, acceptor_end_pos + 1)
                gene_window = (gene_start_pos, gene_end_pos + 1)
                left_exon = (gene_start_pos, donor_start_pos)
                right_exon = (acceptor_end_pos + 1, gene_end_pos + 1)
                exon_windows: List[Tuple[int, int]] = []
                if left_exon[1] > left_exon[0]:
                    exon_windows.append(left_exon)
                if right_exon[1] > right_exon[0]:
                    exon_windows.append(right_exon)
                if not exon_windows:
                    raise RuntimeError("No valid exon windows for visualization.")

                plasmid_label = Path(plasmid_context_path).stem
                plasmid_record = {
                    "plasmid_label": plasmid_label,
                    "target_seq": target_seq,
                    "left_context": left_context,
                    "right_context": right_context,
                    "target_length": target_length,
                    "regions": {
                        "gene": gene_window,
                        "left_exon": left_exon,
                        "intron": intron_window,
                        "right_exon": right_exon,
                    },
                    "gene_window": gene_window,
                    "intron_window": intron_window,
                    "exon_windows": exon_windows,
                    "donor_eval_pos": donor_eval_pos,
                    "acceptor_eval_pos": acceptor_eval_pos,
                }
                plasmid_records.append(plasmid_record)

                cassette = left_context + target_seq + right_context
                for genomic_label, _, genomic_context in genomic_contexts:
                    integrated_sequence, insert_start = _integrate_cassette_into_context(
                        genomic_context=genomic_context,
                        cassette_sequence=cassette,
                    )
                    target_start = insert_start + len(left_context)
                    target_end = target_start + target_length
                    context_records.append(
                        {
                            "plasmid_label": plasmid_label,
                            "genomic_label": genomic_label,
                            "integrated_sequence": integrated_sequence,
                            "integrated_length": len(integrated_sequence),
                            "target_window_integrated": (target_start, target_end),
                            "gene_window_integrated": (
                                target_start + gene_window[0],
                                target_start + gene_window[1],
                            ),
                            "left_exon_integrated": (
                                target_start + left_exon[0],
                                target_start + left_exon[1],
                            ),
                            "right_exon_integrated": (
                                target_start + right_exon[0],
                                target_start + right_exon[1],
                            ),
                            "intron_window_integrated": (
                                target_start + intron_window[0],
                                target_start + intron_window[1],
                            ),
                            "splice_positions_integrated": [
                                target_start + donor_eval_pos,
                                target_start + acceptor_eval_pos,
                            ],
                            "plasmid_record": plasmid_record,
                        }
                    )

            st_input = SpliceTransformerInput(
                target_seqs=[record["target_seq"] for record in plasmid_records],
                left_contexts=[record["left_context"] for record in plasmid_records],
                right_contexts=[record["right_context"] for record in plasmid_records],
            )
            st_config = SpliceTransformerConfig(
                context_length=len(plasmid_records[0]["left_context"]),
                device=args.splice_transformer_device,
            )
            st_output = run_splice_transformer(st_input, st_config)
            st_prediction = np.asarray(st_output.prediction)
            if st_prediction.ndim != 3:
                raise ValueError(
                    "Unexpected SpliceTransformer prediction rank; "
                    f"expected 3D, got shape {st_prediction.shape}."
                )
            if st_prediction.shape[0] != len(plasmid_records):
                raise ValueError(
                    "SpliceTransformer batch size mismatch: "
                    f"{st_prediction.shape[0]} != {len(plasmid_records)}."
                )

            for plasmid_record, prediction in zip(plasmid_records, st_prediction, strict=True):
                target_signal = _smooth_signal(
                    _extract_tissue_signal(prediction, target_tissue),
                    args.smoothing_window,
                )
                offtarget_signal = _smooth_signal(
                    _extract_tissue_signal(prediction, offtarget_tissue),
                    args.smoothing_window,
                )
                donor_signal = _smooth_signal(
                    prediction[:, SpliceTransformerType.DONOR.value],
                    args.smoothing_window,
                )
                acceptor_signal = _smooth_signal(
                    prediction[:, SpliceTransformerType.ACCEPTOR.value],
                    args.smoothing_window,
                )

                intron_window = plasmid_record["intron_window"]
                exon_windows = plasmid_record["exon_windows"]
                donor_eval_pos = plasmid_record["donor_eval_pos"]
                acceptor_eval_pos = plasmid_record["acceptor_eval_pos"]
                splice_positions = [donor_eval_pos, acceptor_eval_pos]
                target_intron_signal = _window_mean_signal(target_signal, intron_window)
                offtarget_intron_signal = _window_mean_signal(offtarget_signal, intron_window)
                target_exon_signal = float(
                    np.mean([_window_mean_signal(target_signal, w) for w in exon_windows])
                )
                offtarget_exon_signal = float(
                    np.mean([_window_mean_signal(offtarget_signal, w) for w in exon_windows])
                )
                eps = 1e-9
                target_ratio = target_intron_signal / max(target_exon_signal, eps)
                offtarget_ratio = offtarget_intron_signal / max(offtarget_exon_signal, eps)
                ratio_delta = target_ratio - offtarget_ratio
                donor_site_prob = float(np.mean(donor_signal[splice_positions[:1]]))
                acceptor_site_prob = float(np.mean(acceptor_signal[splice_positions[1:2]]))
                target_splice_mean = float(np.mean(target_signal[splice_positions]))
                offtarget_splice_mean = float(np.mean(offtarget_signal[splice_positions]))

                plasmid_record["st_data"] = {
                    "x_bp": np.arange(plasmid_record["target_length"], dtype=float),
                    "target_signal": target_signal,
                    "offtarget_signal": offtarget_signal,
                    "donor_signal": donor_signal,
                    "acceptor_signal": acceptor_signal,
                    "metrics": {
                        "st_target_intron_signal": target_intron_signal,
                        "st_offtarget_intron_signal": offtarget_intron_signal,
                        "st_target_exon_signal": target_exon_signal,
                        "st_offtarget_exon_signal": offtarget_exon_signal,
                        "st_ratio_delta": ratio_delta,
                        "st_donor_site_prob": donor_site_prob,
                        "st_acceptor_site_prob": acceptor_site_prob,
                        "st_boundary_score": 1.0 - ((donor_site_prob + acceptor_site_prob) / 2.0),
                        "st_target_specificity_max_score": 1.0 - target_splice_mean,
                        "st_offtarget_specificity_min_score": offtarget_splice_mean,
                        "st_target_splice_mean": target_splice_mean,
                        "st_offtarget_splice_mean": offtarget_splice_mean,
                    },
                }

            target_payloads = _predict_alphagenome_payloads(
                [record["integrated_sequence"] for record in context_records],
                target_terms,
                args,
            )
            offtarget_payloads = _predict_alphagenome_payloads(
                [record["integrated_sequence"] for record in context_records],
                offtarget_terms,
                args,
            )
            if len(target_payloads) != len(context_records) or len(offtarget_payloads) != len(context_records):
                raise RuntimeError("AlphaGenome output count did not match context count.")

            design_context_entries: List[Dict[str, Any]] = []

            for context_record, target_payload, offtarget_payload in zip(
                context_records,
                target_payloads,
                offtarget_payloads,
                strict=True,
            ):
                plasmid_record = context_record["plasmid_record"]
                st_data = plasmid_record["st_data"]
                target_length = plasmid_record["target_length"]
                x_bp = st_data["x_bp"]

                integrated_length = context_record["integrated_length"]
                target_window_integrated = context_record["target_window_integrated"]
                intron_window_integrated = context_record["intron_window_integrated"]
                exon_windows_integrated = [
                    context_record["left_exon_integrated"],
                    context_record["right_exon_integrated"],
                ]
                exon_windows_integrated = [
                    window for window in exon_windows_integrated if window[1] > window[0]
                ]
                splice_positions_integrated = context_record["splice_positions_integrated"]

                target_rna_matrix = _extract_rna_matrix(target_payload)
                offtarget_rna_matrix = _extract_rna_matrix(offtarget_payload)
                ag_rna_target = _extract_rna_signal_on_target(
                    matrix=target_rna_matrix,
                    sequence_length=integrated_length,
                    target_window=target_window_integrated,
                    target_length=target_length,
                    smoothing_window=args.smoothing_window,
                )
                ag_rna_offtarget = _extract_rna_signal_on_target(
                    matrix=offtarget_rna_matrix,
                    sequence_length=integrated_length,
                    target_window=target_window_integrated,
                    target_length=target_length,
                    smoothing_window=args.smoothing_window,
                )

                target_ssu_full = _extract_ssu_signal(
                    target_payload,
                    strand=args.alphagenome_track_strand,
                    expected_length=integrated_length,
                )
                offtarget_ssu_full = _extract_ssu_signal(
                    offtarget_payload,
                    strand=args.alphagenome_track_strand,
                    expected_length=integrated_length,
                )
                ag_ssu_target = _smooth_signal(
                    target_ssu_full[target_window_integrated[0] : target_window_integrated[1]],
                    args.smoothing_window,
                )
                ag_ssu_offtarget = _smooth_signal(
                    offtarget_ssu_full[target_window_integrated[0] : target_window_integrated[1]],
                    args.smoothing_window,
                )

                ag_target_intron_rna = _window_mean_matrix(
                    target_rna_matrix,
                    intron_window_integrated,
                    integrated_length,
                )
                ag_offtarget_intron_rna = _window_mean_matrix(
                    offtarget_rna_matrix,
                    intron_window_integrated,
                    integrated_length,
                )
                ag_target_exon_rna = float(
                    np.mean(
                        [
                            _window_mean_matrix(target_rna_matrix, window, integrated_length)
                            for window in exon_windows_integrated
                        ]
                    )
                )
                ag_offtarget_exon_rna = float(
                    np.mean(
                        [
                            _window_mean_matrix(offtarget_rna_matrix, window, integrated_length)
                            for window in exon_windows_integrated
                        ]
                    )
                )
                eps = 1e-9
                ag_target_ratio = ag_target_intron_rna / max(ag_target_exon_rna, eps)
                ag_offtarget_ratio = ag_offtarget_intron_rna / max(ag_offtarget_exon_rna, eps)
                ag_rna_ratio_delta = ag_target_ratio - ag_offtarget_ratio
                ag_target_splice_mean = float(
                    np.mean([target_ssu_full[pos] for pos in splice_positions_integrated])
                )
                ag_offtarget_splice_mean = float(
                    np.mean([offtarget_ssu_full[pos] for pos in splice_positions_integrated])
                )

                metrics = {
                    "ag_target_intron_rna": ag_target_intron_rna,
                    "ag_offtarget_intron_rna": ag_offtarget_intron_rna,
                    "ag_target_exon_rna": ag_target_exon_rna,
                    "ag_offtarget_exon_rna": ag_offtarget_exon_rna,
                    "ag_rna_ratio_delta": ag_rna_ratio_delta,
                    "ag_target_splice_mean": ag_target_splice_mean,
                    "ag_offtarget_splice_mean": ag_offtarget_splice_mean,
                    **st_data["metrics"],
                }

                entry = {
                    "x_bp": x_bp,
                    "ag_rna_target": ag_rna_target,
                    "ag_rna_offtarget": ag_rna_offtarget,
                    "ag_ssu_target": ag_ssu_target,
                    "ag_ssu_offtarget": ag_ssu_offtarget,
                    "st_donor": st_data["donor_signal"],
                    "st_acceptor": st_data["acceptor_signal"],
                    "st_tissue_target": st_data["target_signal"],
                    "st_tissue_offtarget": st_data["offtarget_signal"],
                    "sequence_length": target_length,
                    "regions": plasmid_record["regions"],
                    "donor_eval_pos": plasmid_record["donor_eval_pos"],
                    "acceptor_eval_pos": plasmid_record["acceptor_eval_pos"],
                    "metrics": metrics,
                    "plasmid_label": context_record["plasmid_label"],
                    "genomic_label": context_record["genomic_label"],
                }

                if not args.aggregate_only:
                    svg_name = (
                        f"{args.filename_prefix}design_{design_index:03d}_{design_hash}_"
                        f"{context_record['plasmid_label']}_{context_record['genomic_label']}.svg"
                    )
                    svg_path = output_dir / svg_name
                    title = (
                        f"Design {design_index:03d} ({design_hash}) | "
                        f"plasmid={context_record['plasmid_label']} | "
                        f"genomic={context_record['genomic_label']}"
                    )
                    _plot_tracks_svg(
                        x_bp=x_bp,
                        ag_rna_target=ag_rna_target,
                        ag_rna_offtarget=ag_rna_offtarget,
                        ag_ssu_target=ag_ssu_target,
                        ag_ssu_offtarget=ag_ssu_offtarget,
                        st_donor=st_data["donor_signal"],
                        st_acceptor=st_data["acceptor_signal"],
                        st_tissue_target=st_data["target_signal"],
                        st_tissue_offtarget=st_data["offtarget_signal"],
                        sequence_length=target_length,
                        regions=plasmid_record["regions"],
                        donor_eval_pos=plasmid_record["donor_eval_pos"],
                        acceptor_eval_pos=plasmid_record["acceptor_eval_pos"],
                        metrics=metrics,
                        target_cell_label=f"{args.target_cell} ({','.join(target_terms)})",
                        offtarget_cell_label=f"{args.offtarget_cell} ({','.join(offtarget_terms)})",
                        target_tissue_label=target_tissue,
                        offtarget_tissue_label=offtarget_tissue,
                        title=title,
                        view_mode=args.view_mode,
                        gene_flank_bp=args.gene_flank_bp,
                        intron_flank_bp=args.intron_flank_bp,
                        out_path=svg_path,
                    )
                    print(f"[OK] Wrote {svg_path}")
                    summary_rows.append(
                        {
                            "design_index": design_index,
                            "design_hash": design_hash,
                            "intron_length": len(intron_sequence),
                            "plasmid_context": context_record["plasmid_label"],
                            "genomic_context": context_record["genomic_label"],
                            "target_terms": ",".join(target_terms),
                            "offtarget_terms": ",".join(offtarget_terms),
                            "target_tissue": target_tissue,
                            "offtarget_tissue": offtarget_tissue,
                            **metrics,
                            "svg_path": str(svg_path),
                        }
                    )

                design_context_entries.append(entry)

            if design_context_entries:
                aggregated = _aggregate_design_entries(design_context_entries)
                aggregate_svg_name = (
                    f"{args.filename_prefix}design_{design_index:03d}_{design_hash}_"
                    "all_plasmid_all_genomic_mean_std.svg"
                )
                aggregate_svg_path = output_dir / aggregate_svg_name
                aggregate_title = (
                    f"Design {design_index:03d} ({design_hash}) | "
                    f"mean +/- SD across {aggregated['context_count']} contexts "
                    f"({aggregated['plasmid_context_count']} plasmid x "
                    f"{aggregated['genomic_context_count']} genomic)"
                )
                _plot_tracks_svg(
                    x_bp=aggregated["x_bp"],
                    ag_rna_target=aggregated["ag_rna_target_mean"],
                    ag_rna_offtarget=aggregated["ag_rna_offtarget_mean"],
                    ag_ssu_target=aggregated["ag_ssu_target_mean"],
                    ag_ssu_offtarget=aggregated["ag_ssu_offtarget_mean"],
                    st_donor=aggregated["st_donor_mean"],
                    st_acceptor=aggregated["st_acceptor_mean"],
                    st_tissue_target=aggregated["st_tissue_target_mean"],
                    st_tissue_offtarget=aggregated["st_tissue_offtarget_mean"],
                    sequence_length=aggregated["sequence_length"],
                    regions=aggregated["regions"],
                    donor_eval_pos=aggregated["donor_eval_pos"],
                    acceptor_eval_pos=aggregated["acceptor_eval_pos"],
                    metrics=aggregated["metrics"],
                    target_cell_label=f"{args.target_cell} ({','.join(target_terms)})",
                    offtarget_cell_label=f"{args.offtarget_cell} ({','.join(offtarget_terms)})",
                    target_tissue_label=target_tissue,
                    offtarget_tissue_label=offtarget_tissue,
                    title=aggregate_title,
                    view_mode=args.view_mode,
                    gene_flank_bp=args.gene_flank_bp,
                    intron_flank_bp=args.intron_flank_bp,
                    out_path=aggregate_svg_path,
                    ag_rna_target_std=aggregated["ag_rna_target_std"],
                    ag_rna_offtarget_std=aggregated["ag_rna_offtarget_std"],
                    ag_ssu_target_std=aggregated["ag_ssu_target_std"],
                    ag_ssu_offtarget_std=aggregated["ag_ssu_offtarget_std"],
                    st_donor_std=aggregated["st_donor_std"],
                    st_acceptor_std=aggregated["st_acceptor_std"],
                    st_tissue_target_std=aggregated["st_tissue_target_std"],
                    st_tissue_offtarget_std=aggregated["st_tissue_offtarget_std"],
                )
                print(f"[OK] Wrote {aggregate_svg_path}")
                summary_rows.append(
                    {
                        "design_index": design_index,
                        "design_hash": design_hash,
                        "intron_length": len(intron_sequence),
                        "plasmid_context": "__all_plasmid_contexts_mean__",
                        "genomic_context": "__all_genomic_contexts_mean__",
                        "target_terms": ",".join(target_terms),
                        "offtarget_terms": ",".join(offtarget_terms),
                        "target_tissue": target_tissue,
                        "offtarget_tissue": offtarget_tissue,
                        **{
                            key: aggregated["metrics"][key]
                            for key in [
                                "ag_target_intron_rna",
                                "ag_offtarget_intron_rna",
                                "ag_target_exon_rna",
                                "ag_offtarget_exon_rna",
                                "ag_rna_ratio_delta",
                                "ag_target_splice_mean",
                                "ag_offtarget_splice_mean",
                                "st_target_intron_signal",
                                "st_offtarget_intron_signal",
                                "st_target_exon_signal",
                                "st_offtarget_exon_signal",
                                "st_ratio_delta",
                                "st_donor_site_prob",
                                "st_acceptor_site_prob",
                                "st_boundary_score",
                                "st_target_specificity_max_score",
                                "st_offtarget_specificity_min_score",
                            ]
                        },
                        "svg_path": str(aggregate_svg_path),
                    }
                )
                aggregated_summary_rows.append(
                    {
                        "design_index": design_index,
                        "design_hash": design_hash,
                        "intron_length": len(intron_sequence),
                        "context_count": aggregated["context_count"],
                        "plasmid_context_count": aggregated["plasmid_context_count"],
                        "genomic_context_count": aggregated["genomic_context_count"],
                        "target_terms": ",".join(target_terms),
                        "offtarget_terms": ",".join(offtarget_terms),
                        "target_tissue": target_tissue,
                        "offtarget_tissue": offtarget_tissue,
                        **{
                            f"{key}_mean": aggregated["metrics"][key]
                            for key in [
                                "ag_target_intron_rna",
                                "ag_offtarget_intron_rna",
                                "ag_target_exon_rna",
                                "ag_offtarget_exon_rna",
                                "ag_rna_ratio_delta",
                                "ag_target_splice_mean",
                                "ag_offtarget_splice_mean",
                                "st_target_intron_signal",
                                "st_offtarget_intron_signal",
                                "st_target_exon_signal",
                                "st_offtarget_exon_signal",
                                "st_ratio_delta",
                                "st_donor_site_prob",
                                "st_acceptor_site_prob",
                                "st_boundary_score",
                                "st_target_specificity_max_score",
                                "st_offtarget_specificity_min_score",
                            ]
                        },
                        **{
                            f"{key}_std": aggregated["metrics"][f"{key}_std"]
                            for key in [
                                "ag_target_intron_rna",
                                "ag_offtarget_intron_rna",
                                "ag_target_exon_rna",
                                "ag_offtarget_exon_rna",
                                "ag_rna_ratio_delta",
                                "ag_target_splice_mean",
                                "ag_offtarget_splice_mean",
                                "st_target_intron_signal",
                                "st_offtarget_intron_signal",
                                "st_target_exon_signal",
                                "st_offtarget_exon_signal",
                                "st_ratio_delta",
                                "st_donor_site_prob",
                                "st_acceptor_site_prob",
                                "st_boundary_score",
                                "st_target_specificity_max_score",
                                "st_offtarget_specificity_min_score",
                            ]
                        },
                        "svg_path": str(aggregate_svg_path),
                    }
                )

    summary_tsv = output_dir / "track_summary.tsv"
    with summary_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, delimiter="\t")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(_to_row(row, field_names))

    aggregated_summary_tsv = output_dir / "track_summary_aggregated.tsv"
    with aggregated_summary_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregated_field_names, delimiter="\t")
        writer.writeheader()
        for row in aggregated_summary_rows:
            writer.writerow(_to_row(row, aggregated_field_names))

    print(f"[OK] Wrote summary TSV: {summary_tsv}")
    print(f"[OK] Wrote aggregate summary TSV: {aggregated_summary_tsv}")


if __name__ == "__main__":
    main()
