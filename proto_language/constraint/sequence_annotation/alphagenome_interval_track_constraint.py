"""AlphaGenome interval track-value constraint."""

import math
from typing import Any, Literal

import numpy as np
from proto_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)
from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    DEFAULT_ALPHAGENOME_MODEL_VERSION,
    SUPPORTED_CONTEXT_LENGTHS,
    OutputTypeName,
)
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import BaseConfig, ConfigField


class AlphaGenomeIntervalTrackConfig(BaseConfig):
    """Configuration for interval track-value scoring with AlphaGenome.

    Attributes:
        intervals (list[tuple[int, int]]): 0-based half-open intervals scored relative to the target segment.
        ontology_terms (list[str]): AlphaGenome ontology term(s) to score.
        left_context (str): DNA context placed upstream of the target segment.
        right_context (str): DNA context placed downstream of the target segment.
        context_length (int): Full sequence length submitted to AlphaGenome.
        requested_output (OutputTypeName): AlphaGenome output type to score (e.g. RNA_SEQ).
        track_name_keywords (list[str] | None): Keywords selecting individual tracks from a bundled output; None = all.
        direction (Literal['maximize', 'minimize']): Whether to maximize or minimize interval mean signal.
        maximize_inflection_value (float): Signal value for maximize-mode sigmoid inflection.
        maximize_sigmoid_scale (float): Scale for maximize-mode sigmoid transform.
        minimize_threshold_value (float): Upper bound for minimize-mode linear score.
        contrastive_ontology_terms (list[str] | None): Optional off-target term(s) scoring the target-vs-off margin.
        margin_inflection_value (float): Signal-difference where the contrastive sigmoid has its inflection.
        margin_sigmoid_scale (float): Scale for the contrastive-mode margin sigmoid transform.
        model_version (str): AlphaGenome model version.
        organism (Literal['human', 'mouse']): Organism for AlphaGenome prediction.
        prediction_timeout (int): Timeout (seconds) for each prediction call.
        device (str): Device for AlphaGenome (JAX) inference (e.g. 'cpu', 'cuda').
    """

    intervals: list[tuple[int, int]] = ConfigField(
        title="Intervals",
        description="0-based half-open intervals scored relative to the target segment.",
    )
    left_context: str = ConfigField(
        default="",
        title="Left Context",
        description="DNA context placed upstream of the target segment before AlphaGenome prediction.",
    )
    right_context: str = ConfigField(
        default="",
        title="Right Context",
        description="DNA context placed downstream of the target segment before AlphaGenome prediction.",
    )
    context_length: int = ConfigField(
        default=16384,
        title="Context Length",
        description="Full sequence length submitted to AlphaGenome after context composition.",
    )
    ontology_terms: list[str] = ConfigField(
        title="Ontology Terms",
        description="AlphaGenome ontology term(s) to score.",
    )
    requested_output: OutputTypeName = ConfigField(
        title="Requested Output",
        default="RNA_SEQ",
        description="AlphaGenome output type to score.",
    )
    track_name_keywords: list[str] | None = ConfigField(
        title="Track Name Keywords",
        default=None,
        description="Keywords selecting individual tracks (e.g. ['H3K4me1']) from a bundled output; None = all tracks.",
    )
    direction: Literal["maximize", "minimize"] = ConfigField(
        title="Direction",
        default="maximize",
        description="Whether to maximize or minimize the interval mean signal.",
    )
    maximize_inflection_value: float = ConfigField(
        title="Maximize Sigmoid Inflection",
        default=5.0,
        ge=0.0,
        description="Signal value where maximize-mode sigmoid has its inflection.",
    )
    maximize_sigmoid_scale: float = ConfigField(
        title="Maximize Sigmoid Scale",
        default=1.0,
        gt=0.0,
        description="Scale for maximize-mode sigmoid transform.",
    )
    minimize_threshold_value: float = ConfigField(
        title="Minimize Threshold",
        default=1.0,
        gt=0.0,
        description="Upper bound for minimize-mode linear score; values above are clipped.",
    )
    contrastive_ontology_terms: list[str] | None = ConfigField(
        title="Contrastive Ontology Terms",
        default=None,
        description="Optional off-target cell-type term(s); when set, scores the target-vs-off-target signal margin.",
    )
    margin_inflection_value: float = ConfigField(
        title="Margin Sigmoid Inflection",
        default=0.0,
        description="Signal-difference (target - off-target) where the contrastive sigmoid has its inflection.",
    )
    margin_sigmoid_scale: float = ConfigField(
        title="Margin Sigmoid Scale",
        default=1.0,
        gt=0.0,
        description="Scale for the contrastive-mode margin sigmoid transform.",
    )
    model_version: str = ConfigField(
        title="Model Version",
        default=DEFAULT_ALPHAGENOME_MODEL_VERSION,
        description="AlphaGenome model version.",
    )
    organism: Literal["human", "mouse"] = ConfigField(
        title="Organism",
        default="human",
        description="Organism for AlphaGenome prediction.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device for AlphaGenome prediction.",
    )
    prediction_timeout: int = ConfigField(
        title="Prediction Timeout",
        default=3600,
        ge=1,
        description="Timeout (seconds) for each AlphaGenome prediction call.",
    )

    @field_validator("ontology_terms", mode="before")
    @classmethod
    def _normalize_terms(cls, terms: list[str] | str) -> list[str]:
        if isinstance(terms, str):
            terms = [terms]
        normalized = [t.strip() for t in terms if t and t.strip()]
        if not normalized:
            raise ValueError("ontology_terms cannot be empty.")
        return normalized

    @field_validator("track_name_keywords")
    @classmethod
    def _normalize_track_keywords(cls, keywords: list[str] | None) -> list[str] | None:
        if keywords is None:
            return None
        normalized = [keyword.strip() for keyword in keywords if keyword and keyword.strip()]
        return normalized or None

    @field_validator("intervals")
    @classmethod
    def _validate_intervals(cls, intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            raise ValueError("intervals cannot be empty.")
        for idx, (start, end) in enumerate(intervals):
            if start < 0:
                raise ValueError(f"intervals[{idx}] start must be >= 0.")
            if end <= start:
                raise ValueError(f"intervals[{idx}] must satisfy end > start.")
        return intervals

    @field_validator("left_context", "right_context")
    @classmethod
    def _normalize_context(cls, context: str) -> str:
        normalized = context.strip().upper()
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            raise ValueError(f"AlphaGenome context contains invalid DNA characters: {invalid}.")
        return normalized

    @field_validator("context_length")
    @classmethod
    def _validate_context_length(cls, context_length: int) -> int:
        if context_length not in SUPPORTED_CONTEXT_LENGTHS:
            supported = ", ".join(str(length) for length in sorted(SUPPORTED_CONTEXT_LENGTHS))
            raise ValueError(f"context_length must be one of the AlphaGenome-supported lengths: {supported}.")
        return context_length


def _window_to_rows(
    start: int,
    end: int,
    sequence_length: int,
    num_rows: int,
) -> tuple[int, int]:
    row_start = math.floor(start * num_rows / sequence_length)
    row_end = math.ceil(end * num_rows / sequence_length)
    row_start = max(0, min(row_start, num_rows - 1))
    row_end = max(row_start + 1, min(row_end, num_rows))
    return row_start, row_end


def _compose_context_sequence(sequence: str, config: AlphaGenomeIntervalTrackConfig) -> tuple[str, int]:
    target_length = len(sequence)
    if target_length > config.context_length:
        raise ValueError(f"target sequence length {target_length} exceeds context_length {config.context_length}.")

    if target_length == config.context_length and not config.left_context and not config.right_context:
        return sequence, 0

    needed_context = config.context_length - target_length
    if not config.left_context and not config.right_context:
        raise ValueError(
            "AlphaGenome interval scoring requires left_context/right_context when the target segment "
            f"is shorter than the supported context_length {config.context_length}."
        )

    left_len = min(len(config.left_context), needed_context // 2)
    right_len = min(len(config.right_context), needed_context - left_len)
    left_len = min(len(config.left_context), needed_context - right_len)
    right_len = needed_context - left_len
    if right_len > len(config.right_context):
        raise ValueError(
            f"Insufficient AlphaGenome context for length {target_length}: need {needed_context} total flanking bp, "
            f"got {len(config.left_context)} left and {len(config.right_context)} right."
        )

    context_sequence = config.left_context[-left_len:] + sequence + config.right_context[:right_len]
    if len(context_sequence) != config.context_length:
        raise RuntimeError(
            f"Composed AlphaGenome context length {len(context_sequence)} != requested {config.context_length}."
        )
    return context_sequence, left_len


def _safe_numeric_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0 or arr.size == 0:
        return None
    if not np.isfinite(arr).any():
        return None
    return arr


def _collect_value_arrays(node: Any, arrays: list[np.ndarray]) -> None:
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


def _normalize_output_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _extract_track_matrix(
    result_payload: dict[str, Any],
    requested_output: str,
) -> np.ndarray:
    predictions = result_payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("AlphaGenome result payload missing 'predictions' dictionary.")

    requested_key = _normalize_output_key(requested_output)
    track_payload = None
    for key, value in predictions.items():
        if _normalize_output_key(key) == requested_key:
            track_payload = value
            break
    if track_payload is None:
        raise ValueError(f"AlphaGenome prediction payload missing requested output '{requested_output}'.")

    arrays: list[np.ndarray] = []
    _collect_value_arrays(track_payload, arrays)
    if not arrays:
        raise ValueError(f"Unable to extract numeric values for requested output '{requested_output}'.")

    matrix = max(arrays, key=lambda a: (a.shape[0], a.size))
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return matrix


def _extract_track_payload(result_payload: dict[str, Any], requested_output: str) -> dict[str, Any]:
    """Return the serialized TrackData payload (``{"values":..., "metadata":...}``) for an output."""
    predictions = result_payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("AlphaGenome result payload missing 'predictions' dictionary.")
    requested_key = _normalize_output_key(requested_output)
    for key, value in predictions.items():
        if _normalize_output_key(key) == requested_key:
            return value if isinstance(value, dict) else {"values": value}
    raise ValueError(f"AlphaGenome prediction payload missing requested output '{requested_output}'.")


def _extract_track_metadata_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-track metadata records from a serialized TrackData payload, if present."""
    metadata = payload.get("metadata")
    if isinstance(metadata, list):
        return [row for row in metadata if isinstance(row, dict)]
    if isinstance(metadata, dict):
        records = metadata.get("records")
        if isinstance(records, list):
            return [row for row in records if isinstance(row, dict)]
    return []


def _matrix_from_payload(payload: dict[str, Any]) -> np.ndarray:
    """Parse the [bins x tracks] value matrix from a serialized TrackData payload."""
    arr = _safe_numeric_array(payload.get("values"))
    if arr is None:
        arrays: list[np.ndarray] = []
        _collect_value_arrays(payload, arrays)
        if not arrays:
            raise ValueError("Unable to extract numeric 'values' for track-keyword selection.")
        arr = max(arrays, key=lambda a: (a.shape[0], a.size))
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _select_track_columns_by_keywords(
    metadata_records: list[dict[str, Any]],
    num_columns: int,
    keywords: list[str],
) -> list[int]:
    """Return column indices whose per-track metadata matches any keyword (case-insensitive).

    Raises:
        ValueError: If no per-track metadata is available, or no track matches the keywords.
    """
    if not metadata_records:
        raise ValueError(
            "track_name_keywords requires per-track AlphaGenome metadata, but the payload "
            "exposed none; cannot select individual tracks (e.g. a single histone mark)."
        )
    lowered = [keyword.lower() for keyword in keywords]
    selected: list[int] = []
    for idx in range(num_columns):
        if idx >= len(metadata_records):
            break
        haystack = " ".join(str(value) for value in metadata_records[idx].values()).lower()
        if any(keyword in haystack for keyword in lowered):
            selected.append(idx)
    if not selected:
        raise ValueError(f"No AlphaGenome tracks matched track_name_keywords={keywords}.")
    return selected


@constraint(
    key="alphagenome-interval-track",
    label="AlphaGenome Interval Track",
    config=AlphaGenomeIntervalTrackConfig,
    description=("Score AlphaGenome track signal over one or more intervals by minimizing or maximizing mean value."),
    uses_gpu=True,
    tools_called=["alphagenome-predict-sequences"],
    category="sequence_annotation",
    supported_sequence_types=["dna"],
)
def alphagenome_interval_track_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: AlphaGenomeIntervalTrackConfig,
) -> list[ConstraintOutput]:
    """Score mean AlphaGenome track signal over one or more intervals."""
    if not input_sequences:
        return []

    sequences = [sequence for (sequence,) in input_sequences]
    target_sequences = [sequence.sequence for sequence in sequences]
    for target_sequence in target_sequences:
        for interval in config.intervals:
            if interval[1] > len(target_sequence):
                raise ValueError(f"interval {interval} exceeds target segment length {len(target_sequence)}.")

    composed_sequences: list[str] = []
    target_offsets: list[int] = []
    for target_sequence in target_sequences:
        composed_sequence, target_offset = _compose_context_sequence(target_sequence, config)
        composed_sequences.append(composed_sequence)
        target_offsets.append(target_offset)

    def _mean_signals(ontology_terms: list[str]) -> list[float]:
        """Run AlphaGenome for the given ontology terms and return per-sequence interval mean signal."""
        pred_cfg = AlphaGenomePredictSequencesConfig(
            model_version=config.model_version,
            requested_outputs=[config.requested_output],
            ontology_terms=ontology_terms,
            organism=config.organism,
            device=config.device,
            timeout=config.prediction_timeout,
        )
        outs = run_alphagenome_predict_sequences(
            AlphaGenomePredictSequencesInput(sequences=composed_sequences),
            pred_cfg,
        ).results
        signals: list[float] = []
        for composed_sequence, target_offset, output in zip(composed_sequences, target_offsets, outs, strict=True):
            sequence_length = len(composed_sequence)
            if config.track_name_keywords:
                # Select individual tracks (e.g. one histone mark) by per-track metadata, then
                # average over just those columns instead of all tracks in the bundled output.
                payload = _extract_track_payload(output.result, config.requested_output)
                matrix = _matrix_from_payload(payload)
                columns = _select_track_columns_by_keywords(
                    _extract_track_metadata_records(payload), matrix.shape[1], config.track_name_keywords
                )
                matrix = matrix[:, columns]
            else:
                matrix = _extract_track_matrix(output.result, config.requested_output)
            shifted = [(start + target_offset, end + target_offset) for start, end in config.intervals]
            interval_values: list[np.ndarray] = []
            for interval in shifted:
                row_start, row_end = _window_to_rows(
                    start=interval[0], end=interval[1], sequence_length=sequence_length, num_rows=matrix.shape[0]
                )
                interval_values.append(matrix[row_start:row_end, :].reshape(-1))
            if not interval_values:
                raise RuntimeError("No interval values were collected for scoring.")
            signals.append(float(np.mean(np.concatenate(interval_values, axis=0))))
        return signals

    target_signals = _mean_signals(config.ontology_terms)
    contrastive_signals = (
        _mean_signals(config.contrastive_ontology_terms) if config.contrastive_ontology_terms else None
    )

    results: list[ConstraintOutput] = []
    for idx, (composed_sequence, target_offset) in enumerate(zip(composed_sequences, target_offsets, strict=True)):
        sequence_length = len(composed_sequence)
        shifted_intervals = [(start + target_offset, end + target_offset) for start, end in config.intervals]
        mean_signal = target_signals[idx]
        maximize_sigmoid_value = math.nan
        minimize_clipped_signal = math.nan
        contrastive_signal = math.nan
        margin = math.nan

        if contrastive_signals is not None:
            # Contrastive: maximize the (target - off-target) signal margin via a smooth sigmoid.
            contrastive_signal = contrastive_signals[idx]
            margin = mean_signal - contrastive_signal
            scaled = (margin - config.margin_inflection_value) / config.margin_sigmoid_scale
            score = float(1.0 - 1.0 / (1.0 + math.exp(-scaled)))
        elif config.direction == "maximize":
            scaled = (mean_signal - config.maximize_inflection_value) / config.maximize_sigmoid_scale
            maximize_sigmoid_value = float(1.0 / (1.0 + math.exp(-scaled)))
            score = float(1.0 - maximize_sigmoid_value)
        else:
            minimize_clipped_signal = float(min(max(mean_signal, 0.0), config.minimize_threshold_value))
            score = float(minimize_clipped_signal / config.minimize_threshold_value)

        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "intervals": [list(interval) for interval in config.intervals],
                    "scored_intervals": [list(interval) for interval in shifted_intervals],
                    "target_offset": target_offset,
                    "context_length": sequence_length,
                    "ontology_terms": config.ontology_terms,
                    "contrastive_ontology_terms": config.contrastive_ontology_terms,
                    "requested_output": config.requested_output,
                    "track_name_keywords": config.track_name_keywords,
                    "direction": config.direction,
                    "interval_mean_signal": mean_signal,
                    "contrastive_mean_signal": contrastive_signal,
                    "contrastive_margin": margin,
                    "maximize_inflection_value": config.maximize_inflection_value,
                    "maximize_sigmoid_scale": config.maximize_sigmoid_scale,
                    "maximize_sigmoid_value": maximize_sigmoid_value,
                    "minimize_threshold_value": config.minimize_threshold_value,
                    "minimize_clipped_signal": minimize_clipped_signal,
                    "alphagenome_interval_track_score": score,
                },
            )
        )

    return results
