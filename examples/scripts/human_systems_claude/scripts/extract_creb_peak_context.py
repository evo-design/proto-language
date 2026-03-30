"""
Usage: python examples/bin/extract_creb_peak_context.py --output_dir peak_output

Extract Borzoi-length context flanks around the highest activity peak.

Defaults to HG38 reference genome and peaks corresponding to the Borzoi entry
``CHIP:CREB1:HepG2``.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from pyfaidx import Fasta

HG38_REF_FILE = "/large_storage/hielab/ykhao/datasets/humanCRE/hg38.fa"
DEFAULT_CREB_PEAK_FILE = "examples/data/ENCFF550TXR.bed.gz"
FLANK_LENGTH = 262_144  # Half of Borzoi input context.


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


def load_creb_peaks(creb_peak_file: str) -> pd.DataFrame:
    """
    Load CREB peaks from a BED file (supports gzipped files).

    Args:
        creb_peak_file: Path to the CREB peaks BED file

    Returns:
        DataFrame with CREB peaks
    """
    # Handle gzipped files
    if creb_peak_file.endswith('.gz'):
        creb_peaks = pd.read_csv(creb_peak_file, sep='\t', header=None, compression='gzip')
    else:
        creb_peaks = pd.read_csv(creb_peak_file, sep='\t', header=None)

    creb_peaks.rename(columns={
        0: 'chr', 1: 'start', 2: 'end', 3: 'name',
        4: 'score', 5: 'strand', 6: 'signalValue', 7: 'p-value',
        8: 'q-value', 9: 'peak'
    }, inplace=True)

    # Calculate summit position (peak column is 0-based offset from start)
    creb_peaks['summit'] = creb_peaks['start'] + creb_peaks['peak']

    return creb_peaks


def get_max_activity_peak(creb_peaks: pd.DataFrame) -> pd.Series:
    """
    Get the peak with maximum activity (signalValue).

    Ignore the mitochondrial chromosome.

    Args:
        creb_peaks: DataFrame with CREB peaks

    Returns:
        Series with the maximum activity peak
    """
    max_idx = creb_peaks[creb_peaks['chr'] != 'chrM']['signalValue'].idxmax()
    return creb_peaks.loc[max_idx]


def extract_flanking_sequences(
    chrom: str,
    start: int,
    end: int,
    strand: str,
    hg38: Fasta,
    flank_length: int = FLANK_LENGTH
) -> tuple[str, str]:
    """
    Extract flanking sequences to the left and right of a peak.

    Args:
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position (0-based)
        strand: '+' or '-'
        hg38: Fasta object for reference genome
        flank_length: Length of flanking sequences

    Returns:
        Tuple of (left_seq, right_seq)
    """
    # Get chromosome length for boundary checking
    chrom_length = len(hg38[chrom])

    # Calculate coordinates
    left_start = start - flank_length
    left_end = start
    right_start = end
    right_end = end + flank_length

    # Check boundaries
    if left_start < 0:
        print(f"Warning: Left flank extends past chromosome start. Adjusting from {left_start} to 0.")
        left_start = 0
    if right_end > chrom_length:
        print(f"Warning: Right flank extends past chromosome end. Adjusting from {right_end} to {chrom_length}.")
        right_end = chrom_length

    # Extract sequences
    left_seq_raw = hg38[chrom][left_start:left_end].seq.upper()
    right_seq_raw = hg38[chrom][right_start:right_end].seq.upper()

    if strand == '+':
        left_seq = left_seq_raw
        right_seq = right_seq_raw
    elif strand == '-':
        # For minus strand, swap and reverse complement
        # "Left" in biological sense (upstream) is actually downstream in genomic coords
        left_seq = reverse_complement(right_seq_raw)
        right_seq = reverse_complement(left_seq_raw)
    else:
        # If strand is not specified, assume '+'
        print(f"Warning: Unknown strand '{strand}', assuming '+'")
        left_seq = left_seq_raw
        right_seq = right_seq_raw

    return left_seq, right_seq


def main(
    creb_peak_file: str = DEFAULT_CREB_PEAK_FILE,
    hg38_ref_file: str = HG38_REF_FILE,
    output_dir: Optional[str] = None
) -> dict[str, dict | str]:
    """
    Main function to extract flanking sequences around max activity peak.

    Args:
        creb_peak_file: Path to CREB peaks BED file
        hg38_ref_file: Path to hg38 reference genome FASTA
        output_dir: Optional output directory for saving sequences

    Returns:
        Dictionary with 'peak' (metadata dict), 'left_sequence', and 'right_sequence'
    """
    # Load CREB peaks
    print(f"Loading CREB peaks from {creb_peak_file}...")
    creb_peaks = load_creb_peaks(creb_peak_file)
    print(f"Loaded {len(creb_peaks)} peaks")

    # Get max activity peak
    max_peak = get_max_activity_peak(creb_peaks)
    print(f"\n{'='*60}")
    print("Maximum activity peak:")
    print(f"{'='*60}")
    print(f"  Chromosome: {max_peak['chr']}")
    print(f"  Start: {max_peak['start']}")
    print(f"  End: {max_peak['end']}")
    print(f"  Summit: {max_peak['summit']}")
    print(f"  Strand: {max_peak['strand']}")
    print(f"  Signal Value: {max_peak['signalValue']}")
    print(f"  q-value: {max_peak['q-value']}")
    print(f"  Name: {max_peak['name']}")
    print(f"{'='*60}\n")

    # Load reference genome
    print(f"Loading reference genome from {hg38_ref_file}...")
    hg38 = Fasta(hg38_ref_file)

    # Extract flanking sequences
    print(f"Extracting {FLANK_LENGTH:,}-bp flanking sequences...")
    left_seq, right_seq = extract_flanking_sequences(
        chrom=max_peak['chr'],
        start=max_peak['start'],
        end=max_peak['end'],
        strand=max_peak['strand'],
        hg38=hg38,
        flank_length=FLANK_LENGTH
    )

    print(f"\nLeft sequence length: {len(left_seq):,} bp")
    print(f"Right sequence length: {len(right_seq):,} bp")

    # Print first/last 50 bp of each sequence
    print(f"\nLeft sequence (first 50 bp): {left_seq[:50]}")
    print(f"Left sequence (last 50 bp):  {left_seq[-50:]}")
    print(f"\nRight sequence (first 50 bp): {right_seq[:50]}")
    print(f"Right sequence (last 50 bp):  {right_seq[-50:]}")

    # Save sequences if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        strand_label = 'plus' if max_peak['strand'] == '+' else 'minus'

        # Save left sequence
        left_file = os.path.join(output_dir, f"max_peak_left_{FLANK_LENGTH}bp_{strand_label}.fa")
        with open(left_file, 'w') as f:
            f.write(f">{max_peak['chr']}:{max_peak['start']}-{max_peak['end']}_left_{FLANK_LENGTH}bp_strand{max_peak['strand']}\n")
            # Write in 80-character lines
            for i in range(0, len(left_seq), 80):
                f.write(left_seq[i:i+80] + '\n')
        print(f"\nLeft sequence saved to: {left_file}")

        # Save right sequence
        right_file = os.path.join(output_dir, f"max_peak_right_{FLANK_LENGTH}bp_{strand_label}.fa")
        with open(right_file, 'w') as f:
            f.write(f">{max_peak['chr']}:{max_peak['start']}-{max_peak['end']}_right_{FLANK_LENGTH}bp_strand{max_peak['strand']}\n")
            for i in range(0, len(right_seq), 80):
                f.write(right_seq[i:i+80] + '\n')
        print(f"Right sequence saved to: {right_file}")

    return {
        'peak': max_peak.to_dict(),
        'left_sequence': left_seq,
        'right_sequence': right_seq
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract flanking sequences around max activity CREB peak")
    parser.add_argument("--creb_peak_file", type=str, default=DEFAULT_CREB_PEAK_FILE,
                        help="Path to CREB peaks BED file (supports .gz)")
    parser.add_argument("--hg38_ref_file", type=str, default=HG38_REF_FILE,
                        help="Path to hg38 reference genome FASTA")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for saving sequences (optional)")
    args = parser.parse_args()

    result = main(
        creb_peak_file=args.creb_peak_file,
        hg38_ref_file=args.hg38_ref_file,
        output_dir=args.output_dir
    )
