"""Shared ORF selection helpers for protein-based constraints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from proto_tools import ORF, OrfipyConfig, OrfipyInput, run_orfipy_prediction

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
            "orfipy_orfs": orf_dicts or None,
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
            "parent_id": selected.parent_id,
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


def resolve_protein_complex_chains(
    input_sequences: list[tuple[Sequence, ...]],
) -> list[tuple[list[str] | None, dict[str, Any]]]:
    """Resolve protein/DNA proposal tuples into protein chain sequences.

    Protein chains are used directly. DNA chains are translated by selecting the
    longest canonical ATG-to-stop ORF on either strand. A proposal returns
    ``None`` for chain sequences when any DNA chain lacks a canonical ORF.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal protein/DNA
            chain tuples to resolve for protein-structure prediction.

    Returns:
        list[tuple[list[str] | None, dict[str, Any]]]: Per-proposal protein chain
            sequences and metadata. Chain sequences are ``None`` when one or more
            DNA chains in that proposal lacks a canonical ORF.
    """
    dna_sequences: list[Sequence] = []
    dna_locations: list[tuple[int, int]] = []
    for proposal_idx, proposal in enumerate(input_sequences):
        for chain_idx, sequence in enumerate(proposal):
            if sequence.sequence_type == "dna":
                dna_sequences.append(sequence)
                dna_locations.append((proposal_idx, chain_idx))

    dna_by_location: dict[tuple[int, int], tuple[ORF | None, dict[str, Any]]] = (
        dict(zip(dna_locations, predict_longest_canonical_cds(dna_sequences), strict=True)) if dna_sequences else {}
    )

    resolved: list[tuple[list[str] | None, dict[str, Any]]] = []
    for proposal_idx, proposal in enumerate(input_sequences):
        chain_sequences: list[str] = []
        dna_chain_orfs: list[dict[str, Any]] = []
        translated_cds_by_chain: list[dict[str, Any]] = []
        missing_orf_chain_indices: list[int] = []
        metadata: dict[str, Any] = {
            "chain_count": len(proposal),
            "input_chain_types": [sequence.sequence_type for sequence in proposal],
        }

        for chain_idx, sequence in enumerate(proposal):
            if sequence.sequence_type == "protein":
                chain_sequences.append(sequence.sequence)
                continue
            if sequence.sequence_type != "dna":
                raise ValueError(
                    "Protein complex chain resolution only supports protein and DNA sequences, "
                    f"got {sequence.sequence_type!r} at chain {chain_idx}."
                )

            selected_orf, orf_metadata = dna_by_location[(proposal_idx, chain_idx)]
            chain_orf_metadata = {"chain_index": chain_idx, **orf_metadata}
            dna_chain_orfs.append(chain_orf_metadata)

            if selected_orf is None:
                missing_orf_chain_indices.append(chain_idx)
                continue

            chain_sequences.append(selected_orf.amino_acid_sequence)
            translated_cds_by_chain.append({"chain_index": chain_idx, **chain_orf_metadata["selected_cds"]})

        if dna_chain_orfs:
            metadata["dna_chain_orfs"] = dna_chain_orfs
            metadata["translated_cds_by_chain"] = translated_cds_by_chain
            if len(dna_chain_orfs) == 1:
                metadata.update({k: v for k, v in dna_chain_orfs[0].items() if k != "chain_index"})

        if missing_orf_chain_indices:
            metadata["missing_orf_chain_indices"] = missing_orf_chain_indices
            metadata["chain_resolution_error"] = "No canonical ATG-to-stop ORF found for DNA chain(s): " + ", ".join(
                str(chain_idx) for chain_idx in missing_orf_chain_indices
            )
            resolved.append((None, metadata))
            continue

        metadata["resolved_protein_sequences"] = chain_sequences
        resolved.append((chain_sequences, metadata))

    return resolved
