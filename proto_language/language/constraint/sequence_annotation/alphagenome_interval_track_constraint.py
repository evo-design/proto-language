"""proto_language/language/constraint/sequence_annotation/alphagenome_interval_track_constraint.py

AlphaGenome interval track-value constraint."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from proto_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)
from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    DEFAULT_ALPHAGENOME_MODEL_VERSION,
    OutputTypeName,
)
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence


class AlphaGenomeIntervalTrackConfig(BaseConfig):
    """Configuration for interval track-value scoring with AlphaGenome.

    Attributes:
        intervals (list[tuple[int, int]]): 0-based half-open intervals scored on the predicted track.
        ontology_terms (list[str]): AlphaGenome ontology term(s) to score.
        requested_output (OutputTypeName): AlphaGenome output type to score (e.g. RNA_SEQ).
        direction (Literal['maximize', 'minimize']): Whether to maximize or minimize interval mean signal.
        maximize_inflection_value (float): Signal value for maximize-mode sigmoid inflection.
        maximize_sigmoid_scale (float): Scale for maximize-mode sigmoid transform.
        minimize_threshold_value (float): Upper bound for minimize-mode linear score.
        model_version (str): AlphaGenome model version.
        organism (Literal['human', 'mouse']): Organism for AlphaGenome prediction.
        prediction_timeout (int): Timeout (seconds) for each prediction call.
        device (str): PyTorch device string for model inference (e.g. 'cpu', 'cuda').
    """

    intervals: List[Tuple[int, int]] = ConfigField(
        title="Intervals",
        description="0-based half-open intervals scored on the predicted track.",
    )
    ontology_terms: List[str] = ConfigField(
        title="Ontology Terms",
        description="AlphaGenome ontology term(s) to score.",
    )
    requested_output: OutputTypeName = ConfigField(
        title="Requested Output",
        default="RNA_SEQ",
        description="AlphaGenome output type to score.",
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
    model_version: str = ConfigField(
        title="Model Version",
        default=DEFAULT_ALPHAGENOME_MODEL_VERSION,
        description="AlphaGenome model version.",
        advanced=True,
    )
    organism: Literal["human", "mouse"] = ConfigField(
        title="Organism",
        default="human",
        description="Organism for AlphaGenome prediction.",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device for AlphaGenome prediction.",
        hidden=True,
    )
    prediction_timeout: int = ConfigField(
        title="Prediction Timeout",
        default=3600,
        ge=1,
        description="Timeout (seconds) for each AlphaGenome prediction call.",
        advanced=True,
    )

    @field_validator("ontology_terms", mode="before")
    @classmethod
    def _normalize_terms(cls, terms: List[str] | str) -> List[str]:
        if isinstance(terms, str):
            terms = [terms]
        normalized = [t.strip() for t in terms if t and t.strip()]
        if not normalized:
            raise ValueError("ontology_terms cannot be empty.")
        return normalized

    @field_validator("intervals")
    @classmethod
    def _validate_intervals(cls, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            raise ValueError("intervals cannot be empty.")
        for idx, (start, end) in enumerate(intervals):
            if start < 0:
                raise ValueError(f"intervals[{idx}] start must be >= 0.")
            if end <= start:
                raise ValueError(f"intervals[{idx}] must satisfy end > start.")
        return intervals


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


def _normalize_output_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _extract_track_matrix(
    result_payload: Dict[str, Any],
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
        raise ValueError(
            f"AlphaGenome prediction payload missing requested output '{requested_output}'."
        )

    arrays: List[np.ndarray] = []
    _collect_value_arrays(track_payload, arrays)
    if not arrays:
        raise ValueError(
            f"Unable to extract numeric values for requested output '{requested_output}'."
        )

    matrix = max(arrays, key=lambda a: (a.shape[0], a.size))
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return matrix


@constraint(
    key="alphagenome-interval-track",
    label="AlphaGenome Interval Track",
    config=AlphaGenomeIntervalTrackConfig,
    description=(
        "Score AlphaGenome track signal over one or more intervals by minimizing "
        "or maximizing mean value."
    ),
    uses_gpu=True,
    tools_called=["alphagenome-predict-sequences"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
    num_input_sequences_per_tuple=1,
)
def alphagenome_interval_track_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: AlphaGenomeIntervalTrackConfig,
) -> List[float]:
    """Score mean AlphaGenome track signal over one or more intervals."""
    if not input_sequences:
        return []

    sequences = [sequence for (sequence,) in input_sequences]
    sequence_lengths = [len(sequence.sequence) for sequence in sequences]
    for sequence_length in sequence_lengths:
        for interval in config.intervals:
            if interval[1] > sequence_length:
                raise ValueError(
                    f"interval {interval} exceeds sequence length {sequence_length}."
                )

    prediction_config = AlphaGenomePredictSequencesConfig(
        model_version=config.model_version,
        requested_outputs=[config.requested_output],
        ontology_terms=config.ontology_terms,
        organism=config.organism,
        device=config.device,
        timeout=config.prediction_timeout,
    )

    batch_output = run_alphagenome_predict_sequences(
        AlphaGenomePredictSequencesInput(
            sequences=[sequence.sequence for sequence in sequences]
        ),
        prediction_config,
    )
    outputs = batch_output.results

    scores: List[float] = []
    for sequence, output in zip(sequences, outputs, strict=True):
        sequence_length = len(sequence.sequence)
        matrix = _extract_track_matrix(output.result, config.requested_output)

        interval_values: List[np.ndarray] = []
        for interval in config.intervals:
            row_start, row_end = _window_to_rows(
                start=interval[0],
                end=interval[1],
                sequence_length=sequence_length,
                num_rows=matrix.shape[0],
            )
            interval_values.append(matrix[row_start:row_end, :].reshape(-1))

        if not interval_values:
            raise RuntimeError("No interval values were collected for scoring.")
        mean_signal = float(np.mean(np.concatenate(interval_values, axis=0)))

        if config.direction == "maximize":
            scaled = (mean_signal - config.maximize_inflection_value) / config.maximize_sigmoid_scale
            maximize_sigmoid_value = float(1.0 / (1.0 + math.exp(-scaled)))
            minimize_clipped_signal = math.nan
            score = float(1.0 - maximize_sigmoid_value)
        else:
            maximize_sigmoid_value = math.nan
            minimize_clipped_signal = float(
                min(max(mean_signal, 0.0), config.minimize_threshold_value)
            )
            score = float(minimize_clipped_signal / config.minimize_threshold_value)

        sequence._metadata.update(
            {
                "intervals": [list(interval) for interval in config.intervals],
                "ontology_terms": config.ontology_terms,
                "requested_output": config.requested_output,
                "direction": config.direction,
                "interval_mean_signal": mean_signal,
                "maximize_inflection_value": config.maximize_inflection_value,
                "maximize_sigmoid_scale": config.maximize_sigmoid_scale,
                "maximize_sigmoid_value": maximize_sigmoid_value,
                "minimize_threshold_value": config.minimize_threshold_value,
                "minimize_clipped_signal": minimize_clipped_signal,
                "alphagenome_interval_track_score": score,
            }
        )
        scores.append(score)

    return scores
