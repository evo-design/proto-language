"""Shared helpers for chromatin accessibility MORSE constraints."""

from __future__ import annotations

from typing import Literal

import numpy as np

from proto_language.language.core import ConstraintOutput, Sequence

MorseWindow = tuple[int, int]
PatternNormalization = Literal["global_max", "region_max", "none"]
ReduceMethod = Literal["mean", "min", "std", "lcb"]
WindowStatTransform = Literal["log1p", "identity"]


def prepare_context_padded_candidate(
    candidate: tuple[Sequence, ...],
    *,
    trim_prefix_bp: int,
    output_flank: int,
    context_length: int,
    model_name: str,
) -> tuple[str, int, int]:
    """Clean three input segments and pad enough context for model output bins."""
    if len(candidate) != 3:
        raise ValueError(f"{model_name} MORSE scoring expects left flank, target, and right flank inputs.")

    clean_sequences = ["".join(sequence.sequence.upper().split()) for sequence in candidate]
    if trim_prefix_bp:
        clean_sequences[1] = clean_sequences[1][trim_prefix_bp:]
    if not clean_sequences[1]:
        raise ValueError(f"{model_name} MORSE scoring requires a non-empty target after trim_prefix_bp.")

    full_sequence = "".join(clean_sequences)
    target_start = len(clean_sequences[0])
    target_end = target_start + len(clean_sequences[1])

    # Use deterministic A padding to satisfy model context without introducing random scoring noise.
    synthetic_left_bp = max(0, output_flank - target_start)
    if synthetic_left_bp:
        full_sequence = ("A" * synthetic_left_bp) + full_sequence
        target_start += synthetic_left_bp
        target_end += synthetic_left_bp

    # Extend the right side when the full model context exceeds supplied sequence.
    context_start = target_start - output_flank
    synthetic_right_bp = max(0, context_start + context_length - len(full_sequence))
    if synthetic_right_bp:
        full_sequence += "A" * synthetic_right_bp

    return full_sequence, target_start, target_end


def compute_morse_windows(
    *,
    pattern: str,
    pattern_start_bp: int,
    dot_bp: int,
    dash_bp: int,
    intra_symbol_gap_bp: int,
    inter_letter_gap_bp: int,
) -> tuple[list[MorseWindow], list[MorseWindow]]:
    """Return target-relative high and low MORSE windows in base pairs."""
    highs: list[MorseWindow] = []
    lows: list[MorseWindow] = []
    cursor = pattern_start_bp
    tokens = pattern.split()

    for token_index, token in enumerate(tokens):
        for symbol_index, symbol in enumerate(token):
            length = dot_bp if symbol == "." else dash_bp
            highs.append((cursor, cursor + length))
            cursor += length
            if symbol_index < len(token) - 1:
                lows.append((cursor, cursor + intra_symbol_gap_bp))
                cursor += intra_symbol_gap_bp
        if token_index < len(tokens) - 1:
            lows.append((cursor, cursor + inter_letter_gap_bp))
            cursor += inter_letter_gap_bp
    return highs, lows


def reduce_2d_by_method(values: np.ndarray, axis: int, method: ReduceMethod) -> np.ndarray:
    """Reduce a 2D prediction array along one axis."""
    if method == "mean":
        reduced = values.mean(axis=axis)
    elif method == "min":
        reduced = values.min(axis=axis)
    elif method == "std":
        reduced = values.std(axis=axis)
    elif method == "lcb":
        reduced = values.mean(axis=axis) - values.std(axis=axis)
    else:
        raise ValueError(f"Unsupported reduce method: {method}")
    return np.asarray(reduced)


def slice_signal(
    signal: np.ndarray,
    start_bp: int,
    end_bp: int,
    output_start: int,
    resolution: float,
) -> np.ndarray:
    """Return signal bins overlapping a source-sequence interval."""
    start_idx = max(0, int(np.floor((start_bp - output_start) / resolution)))
    end_idx = min(len(signal), int(np.ceil((end_bp - output_start) / resolution)))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    return np.asarray(signal[start_idx:end_idx], dtype=np.float32)


def window_means(
    signal: np.ndarray,
    *,
    windows: list[MorseWindow],
    target_start: int,
    target_end: int,
    output_start: int,
    resolution: float,
) -> list[float]:
    """Return mean signal values for target-relative windows."""
    means: list[float] = []
    for rel_start, rel_end in windows:
        start_bp = target_start + rel_start
        end_bp = min(target_end, target_start + rel_end)
        if end_bp <= start_bp:
            continue
        sliced = slice_signal(signal, start_bp, end_bp, output_start, resolution)
        if sliced.size:
            means.append(float(np.mean(sliced)))
    return means


def build_binary_pattern_for_target(
    highs: list[MorseWindow],
    *,
    target_num_bins: int,
    resolution: float,
) -> np.ndarray:
    """Rasterize high-signal MORSE windows onto target output bins."""
    pattern = np.zeros(target_num_bins, dtype=np.float32)
    for start_bp, end_bp in highs:
        start_idx = max(0, int(np.floor(start_bp / resolution)))
        end_idx = max(start_idx + 1, int(np.ceil(end_bp / resolution)))
        if start_idx >= target_num_bins:
            continue
        pattern[start_idx : min(target_num_bins, end_idx)] = 1.0
    return pattern


def score_morse_signal(
    *,
    model_name: str,
    raw_signal: np.ndarray,
    target_start: int,
    target_end: int,
    pattern: str,
    pattern_start_bp: int,
    pattern_normalization: PatternNormalization,
    contrast_margin: float,
    contrast_weight: float,
    raw_amplitude_weight: float,
    high_window_reward_weight: float,
    low_window_penalty_weight: float,
    window_stat_transform: WindowStatTransform,
    highs: list[MorseWindow],
    lows: list[MorseWindow],
    resolution: float,
    output_start: int,
) -> ConstraintOutput:
    """Compute the MORSE objective score and metadata for one predicted signal."""
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    raw_target_signal = slice_signal(raw_signal, target_start, target_end, output_start, resolution)
    target_signal = raw_target_signal

    if pattern_normalization == "global_max":
        denom = float(np.max(raw_signal)) if raw_signal.size else 0.0
    elif pattern_normalization == "region_max":
        denom = float(np.max(target_signal)) if target_signal.size else 0.0
    else:
        denom = None

    if target_signal.size == 0:
        raise ValueError(f"{model_name.capitalize()} MORSE scoring found no model output bins for the target.")

    if denom is None:
        normalized_signal = target_signal.copy()
    elif denom > 0.0:
        normalized_signal = target_signal / denom
    else:
        normalized_signal = np.zeros_like(target_signal)

    pattern_bins = build_binary_pattern_for_target(highs, target_num_bins=len(normalized_signal), resolution=resolution)
    score = float(np.mean(np.abs(pattern_bins - normalized_signal)))

    if contrast_weight > 0.0:
        high_mask = pattern_bins >= 0.5
        low_mask = ~high_mask
        if np.any(high_mask) and np.any(low_mask):
            contrast_gap = float(np.mean(normalized_signal[high_mask]) - np.mean(normalized_signal[low_mask]))
            score += contrast_weight * max(0.0, contrast_margin - contrast_gap)

    high_window_means = window_means(
        raw_signal,
        windows=highs,
        target_start=target_start,
        target_end=target_end,
        output_start=output_start,
        resolution=resolution,
    )
    low_window_means = window_means(
        raw_signal,
        windows=lows,
        target_start=target_start,
        target_end=target_end,
        output_start=output_start,
        resolution=resolution,
    )
    high_mean = float(np.mean(high_window_means)) if high_window_means else None
    low_mean = float(np.mean(low_window_means)) if low_window_means else None
    raw_amplitude = (
        max(0.0, float(np.max(raw_target_signal)) - float(np.min(raw_target_signal)))
        if raw_target_signal.size
        else None
    )

    amplitude_term = raw_amplitude if raw_amplitude is not None else 0.0
    high_window_term = high_mean or 0.0
    low_window_term = low_mean or 0.0
    if window_stat_transform == "log1p":
        amplitude_term = float(np.log1p(max(0.0, amplitude_term)))
        high_window_term = float(np.log1p(max(0.0, high_window_term)))
        low_window_term = float(np.log1p(max(0.0, low_window_term)))

    score -= raw_amplitude_weight * amplitude_term
    score -= high_window_reward_weight * high_window_term
    score += low_window_penalty_weight * low_window_term

    prefix = "chromatin_accessibility_morse"
    pattern_end_bp = max((end for _, end in highs + lows), default=pattern_start_bp)
    normalization_denom = None if denom is None else float(denom)
    high_low_gap = high_mean - low_mean if high_mean is not None and low_mean is not None else None
    target_signal_min: float | None = None
    target_signal_max: float | None = None
    target_signal_mean: float | None = None
    if raw_target_signal.size:
        target_signal_min = float(np.min(raw_target_signal))
        target_signal_max = float(np.max(raw_target_signal))
        target_signal_mean = float(np.mean(raw_target_signal))

    metadata = {
        f"{prefix}_model": model_name.lower(),
        f"{prefix}_pattern": pattern,
        f"{prefix}_pattern_start_bp": pattern_start_bp,
        f"{prefix}_pattern_end_bp": pattern_end_bp,
        f"{prefix}_target_bp": target_end - target_start,
        f"{prefix}_output_start": output_start,
        f"{prefix}_output_resolution": resolution,
        f"{prefix}_normalization_denom": normalization_denom,
        f"{prefix}_target_signal_min": target_signal_min,
        f"{prefix}_target_signal_max": target_signal_max,
        f"{prefix}_target_signal_mean": target_signal_mean,
        f"{prefix}_high_window_mean": high_mean,
        f"{prefix}_low_window_mean": low_mean,
        f"{prefix}_high_low_gap": high_low_gap,
        f"{prefix}_raw_amplitude": raw_amplitude,
    }
    return ConstraintOutput(score=score, metadata=metadata, metadata_recipient="Target")
