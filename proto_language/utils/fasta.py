"""Shared FASTA parsing helpers."""

import gzip
from functools import lru_cache


@lru_cache(maxsize=16)
def load_reference_sequences(fasta_path: str) -> dict[str, str]:
    """Load a FASTA or FASTA.GZ file into an ID-to-sequence mapping.

    Args:
        fasta_path (str): Local FASTA path. Files ending in ``.gz`` are read
            as gzip-compressed text.

    Returns:
        dict[str, str]: Mapping from FASTA record ID to concatenated sequence.
    """
    opener = gzip.open if fasta_path.endswith(".gz") else open
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []
    with opener(fasta_path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
    return sequences
