"""Puffin promoter activity and TSS sharpness constraint."""

from typing import Literal

import numpy as np
from proto_tools.tools.sequence_scoring.puffin.puffin_prediction import (
    PUFFIN_PADDING,
    TRACK_NAMES,
    PuffinPredictionConfig,
    PuffinPredictionInput,
    run_puffin_prediction,
)
from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField


class PuffinPromoterActivityConfig(BaseConfig):
    """Configuration for Puffin promoter activity scoring.

    Attributes:
        left_context (str): Upstream DNA padding supplied to Puffin.
        right_context (str): Downstream DNA padding supplied to Puffin.
        track_names (list[str]): Puffin output channels to score.
        score_interval (tuple[int, int] | None): Target-relative interval to score.
        direction (Literal['maximize', 'minimize']): Whether high or low signal is preferred.
        activity_threshold (float): Activity treated as saturating success.
        sharpness_threshold (float): Peak-fraction value treated as saturating success.
        activity_weight (float): Weight for activity loss.
        sharpness_weight (float): Weight for TSS sharpness loss.
        puffin_config (PuffinPredictionConfig): Runtime settings for Puffin.
    """

    left_context: str = ConfigField(
        default="N" * PUFFIN_PADDING,
        title="Left Context",
        description="Upstream DNA padding supplied to Puffin.",
    )
    right_context: str = ConfigField(
        default="N" * PUFFIN_PADDING,
        title="Right Context",
        description="Downstream DNA padding supplied to Puffin.",
    )
    track_names: list[str] = ConfigField(
        default=["ENCODE_CAGE+", "ENCODE_RAMPAGE+", "GRO_CAP+", "PRO_CAP+"],
        title="Track Names",
        description="Puffin output channels to average.",
    )
    score_interval: tuple[int, int] | None = ConfigField(
        default=None,
        title="Score Interval",
        description="Optional 0-based half-open interval relative to the target segment.",
    )
    direction: Literal["maximize", "minimize"] = ConfigField(
        default="maximize",
        title="Direction",
        description="Whether high or low Puffin promoter signal is preferred.",
    )
    activity_threshold: float = ConfigField(
        default=2.0,
        gt=0.0,
        title="Activity Threshold",
        description="Mean activity treated as saturating success.",
    )
    sharpness_threshold: float = ConfigField(
        default=0.2,
        gt=0.0,
        title="Sharpness Threshold",
        description="Peak fraction treated as saturating success.",
    )
    activity_weight: float = ConfigField(
        default=1.0,
        ge=0.0,
        title="Activity Weight",
        description="Weight for the activity component.",
    )
    sharpness_weight: float = ConfigField(
        default=1.0,
        ge=0.0,
        title="Sharpness Weight",
        description="Weight for the TSS sharpness component.",
    )
    puffin_config: PuffinPredictionConfig = ConfigField(
        default_factory=PuffinPredictionConfig,
        title="Puffin Config",
        description="Runtime settings for Puffin prediction.",
    )

    @field_validator("left_context", "right_context")
    @classmethod
    def _validate_context(cls, sequence: str) -> str:
        normalized = sequence.strip().upper()
        if len(normalized) < PUFFIN_PADDING:
            raise ValueError(f"context must be at least {PUFFIN_PADDING} bp.")
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            raise ValueError(f"context contains invalid DNA characters: {invalid}.")
        return normalized

    @field_validator("track_names", mode="before")
    @classmethod
    def _normalize_track_names(cls, names: list[str] | str) -> list[str]:
        if isinstance(names, str):
            names = [names]
        normalized = [name.strip() for name in names if name and name.strip()]
        if not normalized:
            raise ValueError("track_names cannot be empty.")
        invalid = sorted(set(normalized) - set(TRACK_NAMES))
        if invalid:
            raise ValueError(f"unknown Puffin track name(s): {invalid}.")
        return normalized

    @model_validator(mode="after")
    def _validate_weights(self) -> "PuffinPromoterActivityConfig":
        if self.activity_weight == 0.0 and self.sharpness_weight == 0.0:
            raise ValueError("at least one of activity_weight or sharpness_weight must be positive.")
        return self


def _component_score(value: float, threshold: float, direction: str) -> float:
    bounded = min(max(value, 0.0), threshold) / threshold
    if direction == "maximize":
        return 1.0 - bounded
    return bounded


@constraint(
    key="puffin-promoter-activity",
    label="Puffin Promoter Activity",
    config=PuffinPromoterActivityConfig,
    description="Score promoter activity and TSS sharpness from Puffin predictions.",
    uses_gpu=True,
    tools_called=["puffin-prediction"],
    category="sequence_annotation",
    supported_sequence_types=["dna"],
)
def puffin_promoter_activity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: PuffinPromoterActivityConfig,
) -> list[ConstraintOutput]:
    """Score Puffin promoter signal for target DNA segments."""
    if not input_sequences:
        return []

    target_sequences = [sequence.sequence for (sequence,) in input_sequences]
    for idx, target_sequence in enumerate(target_sequences):
        if config.score_interval is not None:
            start, end = config.score_interval
            if start < 0 or end <= start or end > len(target_sequence):
                raise ValueError(f"score_interval {config.score_interval} is invalid for target {idx}.")

    left = config.left_context[-PUFFIN_PADDING:]
    right = config.right_context[:PUFFIN_PADDING]
    full_sequences = [left + target + right for target in target_sequences]
    track_indices = [TRACK_NAMES.index(name) for name in config.track_names]

    output = run_puffin_prediction(
        PuffinPredictionInput(sequences=full_sequences),
        config.puffin_config,
    )

    results: list[ConstraintOutput] = []
    for target_sequence, prediction in zip(target_sequences, output.results, strict=True):
        matrix = np.asarray(prediction.predictions, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(TRACK_NAMES):
            raise ValueError(f"Unexpected Puffin prediction shape: {matrix.shape}.")
        if matrix.shape[0] != len(target_sequence):
            raise ValueError(f"Puffin output length {matrix.shape[0]} does not match target length {len(target_sequence)}.")

        start, end = config.score_interval or (0, len(target_sequence))
        values = matrix[start:end, :][:, track_indices]
        if values.size == 0:
            raise ValueError("Puffin score interval produced no values.")

        activity = float(np.mean(values))
        position_signal = np.mean(values, axis=1)
        peak = float(np.max(position_signal))
        peak_position = int(np.argmax(position_signal))
        total_signal = float(np.sum(np.maximum(position_signal, 0.0)))
        sharpness = 0.0 if total_signal <= 0.0 else float(np.max(position_signal) / total_signal)

        activity_score = _component_score(activity, config.activity_threshold, config.direction)
        sharpness_score = _component_score(sharpness, config.sharpness_threshold, config.direction)
        total_weight = config.activity_weight + config.sharpness_weight
        score = min(
            MAX_ENERGY,
            (
                config.activity_weight * activity_score
                + config.sharpness_weight * sharpness_score
            )
            / total_weight,
        )

        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "puffin_activity": activity,
                    "puffin_peak": peak,
                    "puffin_peak_position": peak_position,
                    "puffin_tss_sharpness": sharpness,
                    "puffin_activity_score": activity_score,
                    "puffin_sharpness_score": sharpness_score,
                    "puffin_score": score,
                    "puffin_direction": config.direction,
                    "puffin_tracks": list(config.track_names),
                    "puffin_score_interval": [start, end],
                },
            )
        )

    return results
