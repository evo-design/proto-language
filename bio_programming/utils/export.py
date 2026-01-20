"""
Export utilities for metadata at different granularities.

Supports exporting to CSV, TSV, JSON, and Excel formats with both wide and long styles.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, IO, List, Literal, Optional, Union

# Type aliases
Style = Literal["wide", "long"]
Format = Literal["csv", "tsv", "json", "xlsx"]
BatchResults = Dict[str, Any]  # Output from Program.extract_batch_results()


def _get_segment_data(batch_results: BatchResults, construct: str, segment: str, batch_idx: int) -> Optional[Dict]:
    """Extract segment data from batch_results structure."""
    for batch in batch_results.get("batch_results", []):
        if batch["batch_idx"] != batch_idx:
            continue
        for c in batch["constructs"]:
            if c["label"] != construct:
                continue
            for s in c["segments"]:
                if s["label"] == segment:
                    return {
                        "sequence": s["sequence"],
                        "constraints": s["constraints"],
                        "energy_score": batch["energy_score"],
                    }
    return None


def _flatten_constraints_wide(constraints: Dict[str, Dict], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten constraints dict to wide format: {constraint.metric: value}.
    
    Constraint structure:
        {
            "constraint_name": {
                "score": 0.1,
                "weight": 1.0,
                "weighted_score": 0.1,
                "multi_input": False,
                "data": {"gc_content": 52.3, ...}
            }
        }
    """
    flat = {}
    for constraint_label, constraint_data in constraints.items():
        base = f"{prefix}{constraint_label}" if prefix else constraint_label
        
        # Add top-level fields (score, weight, etc.)
        for key in ["score", "weight", "weighted_score", "multi_input"]:
            if key in constraint_data:
                flat[f"{base}.{key}"] = constraint_data[key]
        
        # Add nested "data" fields
        data = constraint_data.get("data", {})
        for metric_name, value in data.items():
            flat[f"{base}.{metric_name}"] = value
    
    return flat


def _flatten_constraints_long(constraints: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    Flatten constraints dict to long format: list of {constraint_label, metric, value} rows.
    
    Each constraint becomes one row with its score fields and custom data merged.
    """
    rows = []
    for constraint_label, constraint_data in constraints.items():
        row = {"constraint_label": constraint_label}
        
        # Add top-level fields
        for key in ["score", "weight", "weighted_score", "multi_input"]:
            if key in constraint_data:
                row[key] = constraint_data[key]
        
        # Add nested "data" fields
        data = constraint_data.get("data", {})
        row.update(data)
        
        rows.append(row)
    return rows


def _collect_all_columns(rows: List[Dict]) -> List[str]:
    """Collect all unique column names from rows, preserving insertion order."""
    columns = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return columns


def flatten_segment_metadata(
    batch_results: BatchResults,
    construct: str,
    segment: str,
    batch_idx: int = 0,
    style: Style = "wide",
) -> List[Dict[str, Any]]:
    """
    Flatten metadata for a single segment.

    Args:
        batch_results: Output from Program.extract_batch_results()
        construct: Construct label
        segment: Segment label
        batch_idx: Batch index (default 0)
        style: "wide" (single row) or "long" (one row per constraint)

    Returns:
        List of dicts suitable for CSV/JSON export
    """
    data = _get_segment_data(batch_results, construct, segment, batch_idx)
    if not data:
        return []

    constraints = data["constraints"]

    if style == "wide":
        row = {"sequence": data["sequence"]}
        row.update(_flatten_constraints_wide(constraints))
        return [row]
    else:  # long
        rows = _flatten_constraints_long(constraints)
        for row in rows:
            row["sequence"] = data["sequence"]
        return rows


def flatten_construct_metadata(
    batch_results: BatchResults,
    construct: str,
    batch_idx: int = 0,
    style: Style = "wide",
) -> List[Dict[str, Any]]:
    """
    Flatten metadata for all segments in a construct.

    Args:
        batch_results: Output from Program.extract_batch_results()
        construct: Construct label
        batch_idx: Batch index (default 0)
        style: "wide" (one row per segment) or "long" (one row per segment × constraint)

    Returns:
        List of dicts suitable for CSV/JSON export
    """
    rows = []

    for batch in batch_results.get("batch_results", []):
        if batch["batch_idx"] != batch_idx:
            continue
        for c in batch["constructs"]:
            if c["label"] != construct:
                continue
            for seg in c["segments"]:
                if style == "wide":
                    row = {
                        "segment_label": seg["label"],
                        "sequence": seg["sequence"],
                    }
                    row.update(_flatten_constraints_wide(seg["constraints"]))
                    rows.append(row)
                else:  # long
                    for constraint_row in _flatten_constraints_long(seg["constraints"]):
                        constraint_row["segment_label"] = seg["label"]
                        constraint_row["sequence"] = seg["sequence"]
                        rows.append(constraint_row)

    return rows


def flatten_program_metadata(
    batch_results: BatchResults,
    style: Style = "wide",
) -> List[Dict[str, Any]]:
    """
    Flatten metadata for all segments across all batches.

    Args:
        batch_results: Output from Program.extract_batch_results()
        style: "wide" (one row per batch) or "long" (one row per batch × construct × segment)

    Returns:
        List of dicts suitable for CSV/JSON export
    """
    rows = []

    for batch in batch_results.get("batch_results", []):
        batch_idx = batch["batch_idx"]
        energy_score = batch["energy_score"]

        if style == "wide":
            # One row per batch, with construct.segment.constraint.metric columns
            row = {
                "batch_idx": batch_idx,
                "energy_score": energy_score,
            }
            for c in batch["constructs"]:
                for seg in c["segments"]:
                    prefix = f"{c['label']}.{seg['label']}."
                    row[f"{prefix}sequence"] = seg["sequence"]
                    row.update(_flatten_constraints_wide(seg["constraints"], prefix=prefix))
            rows.append(row)
        else:  # long
            # One row per batch × construct × segment
            for c in batch["constructs"]:
                for seg in c["segments"]:
                    row = {
                        "batch_idx": batch_idx,
                        "energy_score": energy_score,
                        "construct_label": c["label"],
                        "segment_label": seg["label"],
                        "sequence": seg["sequence"],
                    }
                    row.update(_flatten_constraints_wide(seg["constraints"]))
                    rows.append(row)

    return rows


def flatten_batch_over_time(
    history: List[Dict[str, Any]],
    batch_idx: int = 0,
    style: Style = "wide",
) -> List[Dict[str, Any]]:
    """
    Flatten metadata for a single batch across optimization history.

    Args:
        history: Optimizer history list (from optimizer.history)
        batch_idx: Batch index to track (default 0)
        style: "wide" (one row per timepoint) or "long" (one row per timepoint × segment)

    Returns:
        List of dicts suitable for CSV/JSON export
    """
    rows = []

    for entry in history:
        time_step = entry["time_step"]
        energy_scores = entry.get("energy_scores", [])
        energy_score = energy_scores[batch_idx] if batch_idx < len(energy_scores) else None
        constructs = entry.get("constructs", [])

        if style == "wide":
            row = {
                "timepoint": time_step,
                "energy_score": energy_score,
            }
            for c_data in constructs:
                c_label = c_data.get("label", "construct")
                for seg_data in c_data.get("segments", []):
                    seg_label = seg_data.get("label", "segment")
                    # Get the sequence at this batch_idx
                    selected = seg_data.get("selected_sequences", [])
                    if batch_idx < len(selected):
                        seq_data = selected[batch_idx]
                        prefix = f"{c_label}.{seg_label}."
                        row[f"{prefix}sequence"] = seq_data.get("sequence", "")
                        constraints = seq_data.get("metadata", {}).get("constraints", {})
                        row.update(_flatten_constraints_wide(constraints, prefix=prefix))
            rows.append(row)
        else:  # long
            for c_data in constructs:
                c_label = c_data.get("label", "construct")
                for seg_data in c_data.get("segments", []):
                    seg_label = seg_data.get("label", "segment")
                    selected = seg_data.get("selected_sequences", [])
                    if batch_idx < len(selected):
                        seq_data = selected[batch_idx]
                        row = {
                            "timepoint": time_step,
                            "energy_score": energy_score,
                            "construct_label": c_label,
                            "segment_label": seg_label,
                            "sequence": seq_data.get("sequence", ""),
                        }
                        constraints = seq_data.get("metadata", {}).get("constraints", {})
                        row.update(_flatten_constraints_wide(constraints))
                        rows.append(row)

    return rows


# =============================================================================
# Format Writers
# =============================================================================

def to_csv(rows: List[Dict], output: Union[Path, IO, None] = None) -> str:
    """
    Write rows to CSV format.

    Args:
        rows: List of dicts with consistent keys
        output: Path or file-like object. If None, returns string.

    Returns:
        CSV string if output is None
    """
    if not rows:
        return ""

    columns = _collect_all_columns(rows)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        # Fill missing values with empty string
        writer.writerow({k: row.get(k, "") for k in columns})

    csv_str = buffer.getvalue()

    if output is None:
        return csv_str
    elif isinstance(output, Path):
        output.write_text(csv_str)
        return csv_str
    else:
        output.write(csv_str)
        return csv_str


def to_tsv(rows: List[Dict], output: Union[Path, IO, None] = None) -> str:
    """
    Write rows to TSV format.

    Args:
        rows: List of dicts with consistent keys
        output: Path or file-like object. If None, returns string.

    Returns:
        TSV string if output is None
    """
    if not rows:
        return ""

    columns = _collect_all_columns(rows)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter='\t', extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in columns})

    tsv_str = buffer.getvalue()

    if output is None:
        return tsv_str
    elif isinstance(output, Path):
        output.write_text(tsv_str)
        return tsv_str
    else:
        output.write(tsv_str)
        return tsv_str


def to_json(rows: List[Dict], output: Union[Path, IO, None] = None, indent: int = 2) -> str:
    """
    Write rows to JSON format.

    Args:
        rows: List of dicts
        output: Path or file-like object. If None, returns string.
        indent: JSON indentation (default 2)

    Returns:
        JSON string if output is None
    """
    json_str = json.dumps(rows, indent=indent, default=str)

    if output is None:
        return json_str
    elif isinstance(output, Path):
        output.write_text(json_str)
        return json_str
    else:
        output.write(json_str)
        return json_str


def to_xlsx(rows: List[Dict], output: Union[Path, IO]) -> None:
    """
    Write rows to Excel format.

    Args:
        rows: List of dicts with consistent keys
        output: Path or file-like object (required for xlsx)

    Raises:
        ImportError: If openpyxl is not installed
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        raise ImportError(
            "openpyxl is required for Excel export. "
            "Install with: pip install openpyxl"
        )

    if not rows:
        return

    columns = _collect_all_columns(rows)
    wb = Workbook()
    ws = wb.active

    # Write header
    for col_idx, col_name in enumerate(columns, start=1):
        ws.cell(row=1, column=col_idx, value=col_name)

    # Write data
    for row_idx, row in enumerate(rows, start=2):
        for col_idx, col_name in enumerate(columns, start=1):
            ws.cell(row=row_idx, column=col_idx, value=row.get(col_name, ""))

    if isinstance(output, Path):
        wb.save(str(output))
    else:
        wb.save(output)


# =============================================================================
# High-level export function
# =============================================================================

def write_export(
    rows: List[Dict],
    format: Format,
    path: Optional[Path] = None,
) -> Union[str, None]:
    """
    Write rows to the specified format.

    Args:
        rows: List of dicts to export
        format: Output format ("csv", "tsv", "json", "xlsx")
        path: Output path. If None, returns string (not supported for xlsx).

    Returns:
        String content for csv/tsv/json when path is None, else None
    """
    if format == "csv":
        return to_csv(rows, path)
    elif format == "tsv":
        return to_tsv(rows, path)
    elif format == "json":
        return to_json(rows, path)
    elif format == "xlsx":
        if path is None:
            raise ValueError("xlsx format requires a file path")
        to_xlsx(rows, path)
        return None
    else:
        raise ValueError(f"Unsupported format: {format}")
