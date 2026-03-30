"""
Plot Borzoi predictions across all 4 replicates for a designed DNA sequence.

This script takes a designed DNA sequence (e.g., from the CREB design program),
embeds it in genomic flanking context, runs Borzoi predictions across all 4
replicates, and plots the results as a line graph.

Usage:
    python plot_borzoi_predictions.py --sequence <DNA_SEQUENCE> --output <OUTPUT_PATH>
    python plot_borzoi_predictions.py --sequence_file <FASTA_FILE> --output <OUTPUT_PATH>
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from proto_tools import (
    BORZOI_CONTEXT,  # 524,288 bp
    BORZOI_OUTPUT,  # 6,144 output bins
    BorzoiEnsembleConfig,
    BorzoiInput,
    run_borzoi_ensemble,
)

# Constants from the design script
BORZOI_OUTPUT_RESOLUTION = 32  # bp per output bin (note: actually 524288/6144 ≈ 85.3)
BORZOI_FLANK = 163_840  # Flanking region not included in output
BORZOI_HUMAN_TARGETS = 'examples/data/borzoi_targets_human.txt'

# Default paths for flanking sequences (same as design script)
DEFAULT_LEFT_FLANK = 'examples/data/creb_dna_design_left_flank.fasta'
DEFAULT_RIGHT_FLANK = 'examples/data/creb_dna_design_right_flank.fasta'

# Slate gray color palette for replicates (light to dark)
SLATE_COLORS = [
    '#94a3b8',  # slate-400
    '#64748b',  # slate-500
    '#475569',  # slate-600
    '#334155',  # slate-700
]
SLATE_MEAN = '#1e293b'  # slate-800 for mean line
SLATE_STD = '#cbd5e1'   # slate-300 for std fill
SLATE_HIGHLIGHT = '#e2e8f0'  # slate-200 for design region highlight

# Figure dimensions (100 mm x 30 mm)
FIG_WIDTH_MM = 100
FIG_HEIGHT_MM = 30
MM_TO_INCHES = 1 / 25.4


def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    mpl.rcParams['font.family'] = 'Liberation Sans'
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['axes.titlesize'] = 9
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG


def load_flanking_sequences(
    left_flank_path: str = DEFAULT_LEFT_FLANK,
    right_flank_path: str = DEFAULT_RIGHT_FLANK,
) -> tuple[str, str]:
    """Load left and right flanking sequences from FASTA files."""
    left_flank_seq = str(SeqIO.read(left_flank_path, 'fasta').seq)
    right_flank_seq = str(SeqIO.read(right_flank_path, 'fasta').seq)
    return left_flank_seq, right_flank_seq


def embed_sequence_in_context(
    designed_seq: str,
    left_flank_seq: str,
    right_flank_seq: str,
) -> str:
    """
    Embed a designed sequence in flanking genomic context to create
    a full Borzoi input of exactly BORZOI_CONTEXT bp.

    Args:
        designed_seq: The designed DNA sequence
        left_flank_seq: Left flanking genomic sequence (must be >= BORZOI_CONTEXT // 2)
        right_flank_seq: Right flanking genomic sequence (must be >= BORZOI_CONTEXT // 2)

    Returns:
        Full sequence of length BORZOI_CONTEXT
    """
    design_len = len(designed_seq)

    # Calculate how much flanking sequence we need on each side
    len_left_flank = math.ceil((BORZOI_CONTEXT - design_len) / 2.)
    len_right_flank = math.floor((BORZOI_CONTEXT - design_len) / 2.)

    # Ensure we have enough flanking sequence
    assert len(left_flank_seq) >= len_left_flank, \
        f"Left flank too short: need {len_left_flank}, have {len(left_flank_seq)}"
    assert len(right_flank_seq) >= len_right_flank, \
        f"Right flank too short: need {len_right_flank}, have {len(right_flank_seq)}"

    # Build the full sequence
    full_sequence = (
        left_flank_seq[-len_left_flank:] +
        designed_seq +
        right_flank_seq[:len_right_flank]
    )

    assert len(full_sequence) == BORZOI_CONTEXT, \
        f"Full sequence length {len(full_sequence)} != {BORZOI_CONTEXT}"

    return full_sequence


def get_output_tracks(track_pattern: str, targets_file: str = BORZOI_HUMAN_TARGETS) -> List[int]:
    """
    Get Borzoi output track indices matching a pattern.

    Args:
        track_pattern: Pattern to match in track descriptions (e.g., 'CHIP:CREB1:HepG2')
        targets_file: Path to Borzoi targets file

    Returns:
        List of track indices
    """
    borzoi_target_df = pd.read_csv(targets_file, sep='\t')
    all_tracks = list(borzoi_target_df['description'])
    matching_tracks = [
        idx for idx, track in enumerate(all_tracks) if track_pattern in track
    ]
    return matching_tracks


def run_all_replicates(
    full_sequence: str,
    output_tracks: List[int],
    species: str = 'human',
    avg_tracks: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run Borzoi prediction across all 4 replicates.

    Args:
        full_sequence: Full BORZOI_CONTEXT length sequence
        output_tracks: List of track indices to predict
        species: 'human' or 'mouse'
        avg_tracks: Whether to average across output tracks
        verbose: Print progress

    Returns:
        numpy array of shape (4, BORZOI_OUTPUT) with predictions from each replicate
    """
    full_sequence = full_sequence.replace('N', 'A')

    borzoi_input = BorzoiInput(sequence=full_sequence)

    # Use the ensemble function for convenience
    ensemble_config = BorzoiEnsembleConfig(
        output_tracks=output_tracks,
        species=species,
        avg_output_tracks=avg_tracks,
        verbose=verbose,
    )

    ensemble_output = run_borzoi_ensemble(borzoi_input, ensemble_config)

    predictions = np.array(ensemble_output.predictions)  # (4, 1, BORZOI_OUTPUT).
    predictions = np.squeeze(predictions, axis=1)  # (4, BORZOI_OUTPUT).

    return predictions


def compute_design_region_mask(
    design_len: int,
) -> tuple[np.ndarray, int, int]:
    """
    Compute which output bins correspond to the designed region.

    Args:
        design_len: Length of the designed sequence

    Returns:
        Tuple of (boolean mask, start_bin, end_bin)
    """
    # Calculate flanking lengths used in embedding
    len_left_flank = math.ceil((BORZOI_CONTEXT - design_len) / 2.)

    # The Borzoi output excludes BORZOI_FLANK on each side
    # So the output covers positions [BORZOI_FLANK, BORZOI_CONTEXT - BORZOI_FLANK)
    output_start = BORZOI_FLANK
    output_end = BORZOI_CONTEXT - BORZOI_FLANK
    output_len = output_end - output_start  # = 196,608 bp

    # The designed region starts at len_left_flank and ends at len_left_flank + design_len
    design_start = len_left_flank
    design_end = len_left_flank + design_len

    # Convert to output coordinates (relative to output_start)
    design_start_in_output = design_start - output_start
    design_end_in_output = design_end - output_start

    # Convert to bins (each bin is output_len / BORZOI_OUTPUT bp)
    bp_per_bin = output_len / BORZOI_OUTPUT
    start_bin = int(design_start_in_output / bp_per_bin)
    end_bin = int(np.ceil(design_end_in_output / bp_per_bin))

    # Clamp to valid range
    start_bin = max(0, start_bin)
    end_bin = min(BORZOI_OUTPUT, end_bin)

    # Create mask
    mask = np.zeros(BORZOI_OUTPUT, dtype=bool)
    mask[start_bin:end_bin] = True

    return mask, start_bin, end_bin


def plot_borzoi_predictions(
    predictions: np.ndarray,
    design_len: int,
    output_path: str,
    track_name: str = "CREB1 ChIP-seq",
    title: Optional[str] = None,
    highlight_design: bool = True,
    zoom_window_bp: Optional[int] = None,
):
    """
    Plot Borzoi predictions from all 4 replicates.

    Args:
        predictions: Array of shape (4, BORZOI_OUTPUT) with replicate predictions
        design_len: Length of the designed sequence (for highlighting)
        output_path: Path to save the figure (will be saved as SVG)
        track_name: Name of the predicted track for y-axis label
        title: Plot title (optional)
        highlight_design: Whether to highlight the designed region
        zoom_window_bp: If provided, zoom to this window size (in bp) centered on the design
    """
    setup_plot_style()

    assert predictions.shape == (4, BORZOI_OUTPUT), \
        f"Expected shape (4, {BORZOI_OUTPUT}), got {predictions.shape}"

    # Compute x-axis in kb (relative to center of context)
    output_len_bp = BORZOI_CONTEXT - 2 * BORZOI_FLANK  # 196,608 bp
    bp_per_bin = output_len_bp / BORZOI_OUTPUT

    # X positions in bp relative to output start
    x_bp = np.arange(BORZOI_OUTPUT) * bp_per_bin + bp_per_bin / 2

    # Convert to position relative to center of context window
    center_offset = output_len_bp / 2
    x_centered = (x_bp - center_offset) / 1000  # in kb

    # Determine zoom range if specified
    if zoom_window_bp is not None:
        zoom_half_kb = (zoom_window_bp / 2) / 1000
        x_min, x_max = -zoom_half_kb, zoom_half_kb
    else:
        x_min, x_max = x_centered[0], x_centered[-1]

    # Create figure with specified dimensions
    fig_width = FIG_WIDTH_MM * MM_TO_INCHES
    fig_height = FIG_HEIGHT_MM * MM_TO_INCHES
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot each replicate with slate colors
    for i in range(4):
        ax.plot(
            x_centered,
            predictions[i],
            label=f'Rep {i}',
            color=SLATE_COLORS[i],
            alpha=0.9,
        )

    # Plot mean and std across replicates
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    ax.fill_between(
        x_centered,
        mean_pred - std_pred,
        mean_pred + std_pred,
        alpha=0.3,
        color=SLATE_STD,
        linewidth=0,
    )
    ax.plot(
        x_centered,
        mean_pred,
        label='Mean',
        color=SLATE_MEAN,
        linewidth=1.0,
        linestyle='--',
    )

    # Highlight the designed region
    if highlight_design:
        mask, start_bin, end_bin = compute_design_region_mask(design_len)
        design_start_kb = x_centered[start_bin]
        design_end_kb = x_centered[min(end_bin - 1, BORZOI_OUTPUT - 1)]

        ax.axvspan(
            design_start_kb,
            design_end_kb,
            alpha=0.4,
            color=SLATE_HIGHLIGHT,
            linewidth=0,
        )

    # Labels and formatting
    ax.set_xlabel('Position (kb)')
    ax.set_ylabel(f'{track_name}')

    if title is not None:
        ax.set_title(title)

    # Apply zoom
    ax.set_xlim(x_min, x_max)

    # Set reasonable y-axis limits
    ax.set_ylim(0, 65)

    # Minimal legend
    ax.legend(
        loc='upper right',
        frameon=False,
        ncol=5,
        handlelength=1.0,
        columnspacing=0.8,
    )

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.3)

    # Ensure output path has .svg extension
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.svg':
        output_path = output_path.with_suffix('.svg')

    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close()

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Borzoi predictions for a designed DNA sequence'
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--sequence',
        type=str,
        help='DNA sequence string to analyze',
    )
    input_group.add_argument(
        '--sequence_file',
        type=str,
        help='Path to FASTA file containing the designed sequence',
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for the plot (will be saved as .svg)',
    )

    # Flanking sequences
    parser.add_argument(
        '--left_flank',
        type=str,
        default=DEFAULT_LEFT_FLANK,
        help='Path to left flanking sequence FASTA',
    )
    parser.add_argument(
        '--right_flank',
        type=str,
        default=DEFAULT_RIGHT_FLANK,
        help='Path to right flanking sequence FASTA',
    )

    # Track specification
    parser.add_argument(
        '--track_pattern',
        type=str,
        default='CHIP:CREB1:HepG2',
        help='Pattern to match Borzoi output tracks (default: CHIP:CREB1:HepG2)',
    )
    parser.add_argument(
        '--track_indices',
        type=int,
        nargs='+',
        help='Explicit track indices (overrides --track_pattern)',
    )

    # Zoom option
    parser.add_argument(
        '--zoom',
        type=int,
        metavar='BP',
        help='Zoom to a centered window of this size in bp (e.g., --zoom 10000 for 10kb window)',
    )

    # Other options
    parser.add_argument(
        '--species',
        type=str,
        default='human',
        choices=['human', 'mouse'],
        help='Species for Borzoi model',
    )
    parser.add_argument(
        '--title',
        type=str,
        help='Custom plot title',
    )
    parser.add_argument(
        '--no_highlight',
        action='store_true',
        help='Do not highlight the designed region',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output',
    )

    args = parser.parse_args()

    # Load designed sequence
    if args.sequence:
        designed_seq = args.sequence.upper()
    else:
        designed_seq = str(SeqIO.read(args.sequence_file, 'fasta').seq).upper()

    print(f"Designed sequence length: {len(designed_seq)} bp")

    # Load flanking sequences
    print("Loading flanking sequences...")
    left_flank, right_flank = load_flanking_sequences(
        args.left_flank,
        args.right_flank,
    )

    # Embed in context
    print("Embedding sequence in genomic context...")
    full_sequence = embed_sequence_in_context(designed_seq, left_flank, right_flank)
    print(f"Full sequence length: {len(full_sequence)} bp")

    # Get output tracks
    if args.track_indices:
        output_tracks = args.track_indices
        track_name = f"Tracks {args.track_indices}"
    else:
        output_tracks = get_output_tracks(args.track_pattern)
        track_name = args.track_pattern
        print(f"Found {len(output_tracks)} tracks matching '{args.track_pattern}'")

    if not output_tracks:
        raise ValueError(f"No tracks found matching pattern '{args.track_pattern}'")

    # Run predictions
    print("Running Borzoi predictions across all 4 replicates...")
    predictions = run_all_replicates(
        full_sequence,
        output_tracks,
        species=args.species,
        avg_tracks=True,
        verbose=args.verbose,
    )

    print(f"Predictions shape: {predictions.shape}")

    # Plot
    print("Generating plot...")
    plot_borzoi_predictions(
        predictions,
        design_len=len(designed_seq),
        output_path=args.output,
        track_name=track_name,
        title=args.title,
        highlight_design=not args.no_highlight,
        zoom_window_bp=args.zoom,
    )

    # Print summary statistics
    print("\nPrediction summary (designed region):")
    mask, start_bin, end_bin = compute_design_region_mask(len(designed_seq))
    design_predictions = predictions[:, mask]

    for i in range(4):
        mean_val = design_predictions[i].mean()
        max_val = design_predictions[i].max()
        print(f"  Replicate {i}: mean={mean_val:.2f}, max={max_val:.2f}")

    ensemble_mean = design_predictions.mean()
    ensemble_std = design_predictions.std()
    print(f"  Ensemble: mean={ensemble_mean:.2f} ± {ensemble_std:.2f}")


if __name__ == '__main__':
    main()
