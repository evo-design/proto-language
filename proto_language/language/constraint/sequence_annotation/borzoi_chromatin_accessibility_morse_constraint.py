"""Borzoi chromatin accessibility MORSE constraint."""

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
from proto_tools.tools.sequence_scoring.shared_data_models import SequenceTargetRange
from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.constraint.sequence_annotation.chromatin_accessibility_morse_utils import (
    PatternNormalization,
    ReduceMethod,
    WindowStatTransform,
    compute_morse_windows,
    prepare_context_padded_candidate,
    reduce_2d_by_method,
    score_morse_signal,
)
from proto_language.language.core import ConstraintOutput, Sequence

_Organism = Literal["human", "mouse"]

_DEFAULT_BORZOI_OUTPUT_TRACKS: dict[_Organism, list[int]] = {
    "human": [1901],
    "mouse": [741],
}


class BorzoiChromatinAccessibilityMorseConfig(BaseConfig):
    """Configuration for Borzoi chromatin accessibility MORSE scoring.

    Attributes:
        organism (_Organism): Borzoi species head and default-track selector.
        pattern (str): MORSE pattern using '.', '-', and spaces between letters.
        dot_bp (int): Base-pair length of each dot window.
        dash_bp (int): Base-pair length of each dash window.
        intra_symbol_gap_bp (int): Gap between symbols within a letter.
        inter_letter_gap_bp (int): Gap between letters in the MORSE pattern.
        pattern_start_bp (int): Target-relative start coordinate for the pattern.
        pattern_normalization (PatternNormalization): Signal scale used before pattern matching.
        contrast_margin (float): Minimum normalized high-minus-low margin to reward.
        contrast_weight (float): Penalty weight for missing the requested contrast margin.
        raw_amplitude_weight (float): Reward weight for larger raw target signal range.
        high_window_reward_weight (float): Reward weight for larger raw signal in high windows.
        low_window_penalty_weight (float): Penalty weight for larger raw signal in gap windows.
        window_stat_transform (WindowStatTransform): Transform for amplitude and window terms.
        batch_size (int): Candidate sequences per Borzoi model batch.
        trim_prefix_bp (int): Leading target bases ignored before scoring.
        borzoi_output_tracks (list[int]): Borzoi accessibility tracks to score.
        borzoi_ensemble_reduce_method (ReduceMethod): How to combine Borzoi replicate signals.
    """

    organism: _Organism = ConfigField(
        title="Organism",
        description="Borzoi species head to use: human or mouse.",
        default="human",
        advanced=True,
    )
    pattern: str = ConfigField(
        title="Morse Pattern",
        description="Morse pattern; use '.', '-', and spaces between letters.",
        default=".--. .-. --- - ---",
    )
    dot_bp: int = ConfigField(
        title="Dot Length (bp)",
        description="Length of each dot window.",
        ge=1,
        default=25,
    )
    dash_bp: int = ConfigField(
        title="Dash Length (bp)",
        description="Length of each dash window.",
        ge=1,
        default=75,
    )
    intra_symbol_gap_bp: int = ConfigField(
        title="Intra-symbol Gap (bp)",
        description="Gap between symbols within a letter.",
        ge=0,
        default=25,
    )
    inter_letter_gap_bp: int = ConfigField(
        title="Inter-letter Gap (bp)",
        description="Gap between letters in the Morse pattern.",
        ge=0,
        default=75,
    )
    pattern_start_bp: int = ConfigField(
        title="Pattern Start (bp)",
        description="Target-relative start coordinate where the Morse pattern begins.",
        ge=0,
        default=0,
    )
    pattern_normalization: PatternNormalization = ConfigField(
        title="Pattern Normalization",
        description="Scale signal by full-output max, target-region max, or not at all.",
        default="global_max",
        advanced=True,
    )
    contrast_margin: float = ConfigField(
        title="Contrast Margin",
        description="Minimum normalized high-minus-low signal margin to reward.",
        default=0.0,
        ge=0.0,
        advanced=True,
    )
    contrast_weight: float = ConfigField(
        title="Contrast Weight",
        description="Penalty weight when the normalized contrast margin is too small.",
        default=0.0,
        ge=0.0,
        advanced=True,
    )
    raw_amplitude_weight: float = ConfigField(
        title="Raw Amplitude Weight",
        description="Reward weight for larger raw target signal range.",
        default=0.0,
        ge=0.0,
        advanced=True,
    )
    high_window_reward_weight: float = ConfigField(
        title="High Window Reward",
        description="Reward weight for larger raw signal in dot and dash windows.",
        default=0.0,
        ge=0.0,
        advanced=True,
    )
    low_window_penalty_weight: float = ConfigField(
        title="Low Window Penalty",
        description="Penalty weight for larger raw signal in gap windows.",
        default=0.0,
        ge=0.0,
        advanced=True,
    )
    window_stat_transform: WindowStatTransform = ConfigField(
        title="Window Stat Transform",
        description="Transform for raw amplitude and window-mean reward terms.",
        default="log1p",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        description="CUDA device for Borzoi inference.",
        default="cuda",
        hidden=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        description="Candidate sequences per Borzoi model batch.",
        default=1,
        ge=1,
        advanced=True,
    )
    trim_prefix_bp: int = ConfigField(
        title="Trim Prefix (bp)",
        description="Leading target bases to ignore before accessibility scoring.",
        default=0,
        ge=0,
        advanced=True,
    )
    borzoi_output_tracks: list[int] = ConfigField(
        title="Borzoi Output Tracks",
        description="Borzoi chromatin-accessibility tracks; defaults by organism.",
        default=[1901],
        advanced=True,
    )
    borzoi_ensemble_reduce_method: ReduceMethod = ConfigField(
        title="Borzoi Ensemble Reduce",
        description="How to combine Borzoi replicate signals.",
        default="mean",
        advanced=True,
    )

    @model_validator(mode="before")
    @classmethod
    def set_default_borzoi_tracks(cls, data: object) -> object:
        """Use organism-specific chromatin accessibility tracks by default."""
        if not isinstance(data, dict):
            return data
        if data.get("borzoi_output_tracks") is not None:
            return data

        organism = str(data.get("organism", "human")).strip().lower()
        if organism == "human":
            return {**data, "borzoi_output_tracks": list(_DEFAULT_BORZOI_OUTPUT_TRACKS["human"])}
        if organism == "mouse":
            return {**data, "borzoi_output_tracks": list(_DEFAULT_BORZOI_OUTPUT_TRACKS["mouse"])}
        return data

    @field_validator("organism", mode="before")
    @classmethod
    def normalize_organism(cls, organism: object) -> object:
        """Normalize the requested organism before literal validation."""
        if isinstance(organism, str):
            return organism.strip().lower()
        return organism

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, pattern: str) -> str:
        """Validate the Morse pattern alphabet."""
        pattern = pattern.strip()
        if not pattern:
            raise ValueError("Morse pattern must be non-empty.")
        unsupported = sorted({symbol for token in pattern.split() for symbol in token if symbol not in {".", "-"}})
        if unsupported:
            raise ValueError(f"Unsupported Morse symbol(s): {', '.join(unsupported)}")
        return pattern

    @model_validator(mode="after")
    def validate_borzoi_settings(self) -> BorzoiChromatinAccessibilityMorseConfig:
        """Validate Borzoi MORSE settings."""
        if not self.borzoi_output_tracks:
            raise ValueError("borzoi_output_tracks must be provided.")
        return self


@constraint(
    key="borzoi-chromatin-accessibility-morse",
    label="Borzoi Chromatin Accessibility MORSE",
    config=BorzoiChromatinAccessibilityMorseConfig,
    description="Score a DNA target for a Morse-code chromatin accessibility pattern using Borzoi.",
    uses_gpu=True,
    tools_called=["borzoi-ensemble"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
    input_labels=["Left Flank", "Target", "Right Flank"],
)
def borzoi_chromatin_accessibility_morse_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: BorzoiChromatinAccessibilityMorseConfig,
) -> list[ConstraintOutput]:
    """Score target proposals against a Borzoi chromatin accessibility MORSE objective."""
    if not input_sequences:
        return []

    highs, lows = compute_morse_windows(
        pattern=config.pattern,
        pattern_start_bp=config.pattern_start_bp,
        dot_bp=config.dot_bp,
        dash_bp=config.dash_bp,
        intra_symbol_gap_bp=config.intra_symbol_gap_bp,
        inter_letter_gap_bp=config.inter_letter_gap_bp,
    )
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
            sequences=[full_sequence for full_sequence, _, _ in prepared_candidates],
            target_ranges=[
                SequenceTargetRange(start=target_start, end=target_end)
                for _, target_start, target_end in prepared_candidates
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
        outputs.append(
            score_morse_signal(
                model_name="borzoi",
                raw_signal=signal,
                target_start=target_start,
                target_end=target_end,
                pattern=config.pattern,
                pattern_start_bp=config.pattern_start_bp,
                pattern_normalization=config.pattern_normalization,
                contrast_margin=config.contrast_margin,
                contrast_weight=config.contrast_weight,
                raw_amplitude_weight=config.raw_amplitude_weight,
                high_window_reward_weight=config.high_window_reward_weight,
                low_window_penalty_weight=config.low_window_penalty_weight,
                window_stat_transform=config.window_stat_transform,
                highs=highs,
                lows=lows,
                resolution=float(prediction_result.output_resolution),
                output_start=prediction_result.output_start,
            )
        )

    return outputs


borzoi_chromatin_accessibility_morse_constraint._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
