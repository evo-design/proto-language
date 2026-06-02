"""Borzoi track-activity constraint."""

from __future__ import annotations

from typing import Literal

import numpy as np
from proto_tools.tools.sequence_scoring.borzoi import (
    BORZOI_CONTEXT,
    BORZOI_OUTPUT_FLANK,
    BorzoiEnsembleConfig,
    BorzoiInput,
    run_borzoi_ensemble,
)
from proto_tools.tools.sequence_scoring.shared_data_models import SequenceTargetRange, SequenceWindow
from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.sequence_annotation.chromatin_accessibility_morse_utils import (
    ReduceMethod,
    prepare_context_padded_candidate,
    reduce_2d_by_method,
    slice_signal,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import BaseConfig, ConfigField

_Organism = Literal["human", "mouse"]
_Direction = Literal["maximize", "minimize"]


class BorzoiTrackInterval(BaseConfig):
    """Target-relative interval scored from Borzoi output."""

    start_bp: int = ConfigField(
        title="Start (bp)",
        description="0-based inclusive interval start relative to the target segment.",
        ge=0,
    )
    end_bp: int = ConfigField(
        title="End (bp)",
        description="0-based exclusive interval end relative to the target segment.",
        ge=0,
    )

    @model_validator(mode="after")
    def validate_interval(self) -> BorzoiTrackInterval:
        """Validate interval coordinates."""
        if self.end_bp <= self.start_bp:
            raise ValueError("end_bp must be greater than start_bp.")
        return self


class BorzoiTrackActivityConfig(BaseConfig):
    """Configuration for Borzoi track-activity scoring.

    Attributes:
        organism (_Organism): Borzoi species head to use.
        borzoi_output_tracks (list[int]): Output track IDs to score.
        score_intervals (list[BorzoiTrackInterval] | None): Target-relative intervals to score.
        direction (_Direction): Whether high or low activity is preferred.
        activity_threshold (float): Activity treated as saturating success.
        borzoi_ensemble_reduce_method (ReduceMethod): How to combine replicate signals.
        device (str): Device for Borzoi inference.
        batch_size (int): Candidate sequences per batch.
        trim_prefix_bp (int): Leading target bases to ignore.
    """

    organism: _Organism = ConfigField(
        default="human",
        title="Organism",
        description="Borzoi species head to use: human or mouse.",
    )
    borzoi_output_tracks: list[int] = ConfigField(
        default=[0],
        title="Borzoi Output Tracks",
        description="Borzoi output track IDs to average before scoring.",
    )
    score_intervals: list[BorzoiTrackInterval] | None = ConfigField(
        default=None,
        title="Score Intervals",
        description="Optional target-relative intervals to score. If omitted, scores the full target segment.",
    )
    direction: _Direction = ConfigField(
        default="maximize",
        title="Direction",
        description="Whether high activity or low activity is preferred.",
    )
    activity_threshold: float = ConfigField(
        default=200.0,
        title="Activity Threshold",
        description="Activity value treated as saturating success for the objective.",
        gt=0.0,
    )
    borzoi_ensemble_reduce_method: ReduceMethod = ConfigField(
        default="mean",
        title="Borzoi Ensemble Reduce",
        description="How to combine Borzoi replicate signals.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="CUDA device for Borzoi inference.",
    )
    batch_size: int = ConfigField(
        default=1,
        title="Batch Size",
        description="Candidate sequences per Borzoi model batch.",
        ge=1,
    )
    trim_prefix_bp: int = ConfigField(
        default=0,
        title="Trim Prefix (bp)",
        description="Leading target bases to ignore before Borzoi scoring.",
        ge=0,
    )

    @field_validator("organism", mode="before")
    @classmethod
    def normalize_organism(cls, organism: object) -> object:
        """Normalize the requested organism before literal validation."""
        if isinstance(organism, str):
            return organism.strip().lower()
        return organism

    @model_validator(mode="after")
    def validate_borzoi_settings(self) -> BorzoiTrackActivityConfig:
        """Validate Borzoi activity settings."""
        if not self.borzoi_output_tracks:
            raise ValueError("borzoi_output_tracks must be provided.")
        return self


def _score_activity(activity: float, config: BorzoiTrackActivityConfig) -> float:
    """Convert raw activity to a 0-1 energy where lower is better."""
    bounded_fraction = min(max(activity, 0.0), config.activity_threshold) / config.activity_threshold
    if config.direction == "maximize":
        return 1.0 - bounded_fraction
    return bounded_fraction


def _activity_signal_for_intervals(
    signal: np.ndarray,
    *,
    target_start: int,
    target_end: int,
    output_start: int,
    output_resolution: float,
    intervals: list[BorzoiTrackInterval] | None,
) -> np.ndarray:
    """Return Borzoi signal bins for configured target-relative intervals."""
    target_length = target_end - target_start
    selected = intervals or [BorzoiTrackInterval(start_bp=0, end_bp=target_length)]
    slices = []
    for interval in selected:
        if interval.end_bp > target_length:
            raise ValueError(
                f"Borzoi score interval {interval.start_bp}:{interval.end_bp} exceeds target length {target_length}."
            )
        sliced = slice_signal(
            signal,
            target_start + interval.start_bp,
            target_start + interval.end_bp,
            output_start,
            output_resolution,
        )
        if sliced.size:
            slices.append(sliced)
    if not slices:
        raise ValueError("Borzoi track activity found no output bins for the configured score intervals.")
    return np.concatenate(slices)


@constraint(
    key="borzoi-track-activity",
    label="Borzoi Track Activity",
    config=BorzoiTrackActivityConfig,
    description="Score DNA target activity on selected Borzoi output tracks.",
    uses_gpu=True,
    tools_called=["borzoi-ensemble"],
    category="sequence_annotation",
    supported_sequence_types=["dna"],
    input_labels=["Left Flank", "Target", "Right Flank"],
)
def borzoi_track_activity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: BorzoiTrackActivityConfig,
) -> list[ConstraintOutput]:
    """Score target proposals for selected Borzoi output tracks."""
    if not input_sequences:
        return []

    prepared_candidates = [
        prepare_context_padded_candidate(
            candidate,
            trim_prefix_bp=config.trim_prefix_bp,
            output_flank=BORZOI_OUTPUT_FLANK,
            context_length=BORZOI_CONTEXT,
            model_name="Borzoi",
        )
        for candidate in input_sequences
    ]

    result = run_borzoi_ensemble(
        BorzoiInput(
            sequences=[
                SequenceWindow(
                    sequence=full_sequence,
                    target_range=SequenceTargetRange(start=target_start, end=target_end),
                )
                for full_sequence, target_start, target_end in prepared_candidates
            ],
        ),
        BorzoiEnsembleConfig(
            output_tracks=config.borzoi_output_tracks,
            species=config.organism,
            avg_output_tracks=True,
            batch_size=config.batch_size,
            device=config.device,
        ),
    )

    outputs: list[ConstraintOutput] = []
    for (_, target_start, target_end), prediction_result in zip(prepared_candidates, result.results, strict=True):
        preds = np.asarray(prediction_result.predictions, dtype=np.float32)
        if preds.ndim != 3:
            raise ValueError(f"Unexpected Borzoi ensemble prediction shape: {preds.shape}")
        replicate_signals = preds.mean(axis=1)
        signal = reduce_2d_by_method(replicate_signals, axis=0, method=config.borzoi_ensemble_reduce_method)
        activity_signal = _activity_signal_for_intervals(
            signal,
            target_start=target_start,
            target_end=target_end,
            output_start=prediction_result.output_start,
            output_resolution=float(prediction_result.output_resolution),
            intervals=config.score_intervals,
        )
        activity = float(np.mean(activity_signal))
        score = _score_activity(activity, config)
        outputs.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "borzoi_track_activity": activity,
                    "borzoi_track_activity_score": score,
                    "borzoi_track_activity_direction": config.direction,
                    "borzoi_track_activity_threshold": config.activity_threshold,
                    "borzoi_output_tracks": config.borzoi_output_tracks,
                    "borzoi_ensemble_reduce_method": config.borzoi_ensemble_reduce_method,
                    "borzoi_target_bp": target_end - target_start,
                    "borzoi_output_start": prediction_result.output_start,
                    "borzoi_output_resolution": prediction_result.output_resolution,
                },
            )
        )

    return outputs
