"""Shared ORF selection helpers for protein-based constraints."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from proto_tools import ORF, OrfipyConfig, OrfipyInput, run_orfipy_prediction

from proto_language.storage import FileType, store_file

if TYPE_CHECKING:
    from proto_language.language.core import Sequence

CANONICAL_START_CODONS = ["ATG"]
CANONICAL_STOP_CODONS = ["TAA", "TAG", "TGA"]

logger = logging.getLogger(__name__)


def predict_longest_canonical_cds(dna_sequences: list[Sequence]) -> list[tuple[ORF | None, dict[str, Any]]]:
    """Find the longest ATG-to-stop ORF on either strand for each DNA sequence.

    Protein-based constraints that accept DNA proposals can use this to score one
    translated CDS per proposal. ORFipy is used as an explicit ORF scanner with a
    canonical ATG start, canonical stop codons, and both strands enabled; the
    longest translated ORF is selected.

    Args:
        dna_sequences (list[Sequence]): DNA sequences to scan for canonical ORFs.

    Returns:
        list[tuple[ORF | None, dict[str, Any]]]: Per-sequence selected ORF and
            ORFipy metadata. The ORF is ``None`` when no canonical ATG-to-stop ORF
            is found for that sequence.
    """
    orfipy_result = run_orfipy_prediction(
        inputs=OrfipyInput(sequences=[seq.sequence for seq in dna_sequences]),
        config=OrfipyConfig(
            start_codons=CANONICAL_START_CODONS,
            stop_codons=CANONICAL_STOP_CODONS,
            strand="b",
        ),
    )

    selections: list[tuple[ORF | None, dict[str, Any]]] = []
    for sequence_idx, (dna_sequence, orfs) in enumerate(zip(dna_sequences, orfipy_result.predicted_orfs, strict=True)):
        orf_dicts = [orf.model_dump() for orf in orfs]
        metadata: dict[str, Any] = {
            "orfipy_orfs": store_file(json.dumps(orf_dicts), FileType.JSON) if orf_dicts else None,
            "orfipy_orf_count": len(orfs),
            "orf_selection": {
                "caller": "orfipy",
                "start_codons": CANONICAL_START_CODONS,
                "stop_codons": CANONICAL_STOP_CODONS,
                "strand": "both",
                "selection": "longest_orf",
            },
        }

        if not orfs:
            logger.warning(
                "No canonical ATG-to-stop ORF found for DNA sequence %d (length=%d).",
                sequence_idx,
                len(dna_sequence.sequence),
            )
            selections.append((None, metadata))
            continue

        selected = max(orfs, key=lambda orf: (orf.amino_acid_length, orf.nucleotide_length))
        metadata["selected_cds"] = {
            "id": selected.id,
            "orf_id": selected.orf_id,
            "strand": selected.strand,
            "frame": selected.frame,
            "amino_acid_length": selected.amino_acid_length,
            "nucleotide_length": selected.nucleotide_length,
            "nucleotide_start": selected.nucleotide_start,
            "nucleotide_end": selected.nucleotide_end,
        }
        selections.append((selected, metadata))

    return selections
