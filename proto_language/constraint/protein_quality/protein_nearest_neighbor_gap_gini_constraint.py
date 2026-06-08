"""Nearest-neighbor protein gap Gini constraint."""

import logging
from typing import Any

from proto_tools import (
    MafftConfig,
    MafftInput,
    Mmseqs2SearchProteinsConfig,
    Mmseqs2SearchProteinsInput,
    run_mafft_align,
    run_mmseqs2_search_proteins,
)

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.sequence_alignment.gap_gini_constraint import gap_gini_single, trim_alignment
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY, load_fasta
from proto_language.utils.base import BaseConfig, ConfigField
from proto_language.utils.orf_selection import resolve_protein_complex_chains

logger = logging.getLogger(__name__)


class ProteinNearestNeighborGapGiniConfig(BaseConfig):
    """Configuration for gap Gini against nearest protein neighbors.

    Attributes:
        mmseqs_db (str): Path to a target database for MMseqs2 (FASTA file or
            MMseqs2 ``createdb`` output).
        reference_fasta (str): FASTA file used to recover top-hit sequences by
            target ID for pairwise MAFFT alignment.
        max_gap_gini (float): Maximum acceptable gap Gini score (0-1).
        pass_no_hits (bool): If True, proposals with no MMseqs2 hit pass.
        trim_alignment (bool): Whether to center-crop and trim pairwise
            alignments before computing gap Gini.
        mmseqs_config (Mmseqs2SearchProteinsConfig): Advanced MMseqs2 search
            configuration.
        mafft_config (MafftConfig): Advanced MAFFT pairwise alignment
            configuration.
    """

    mmseqs_db: str = ConfigField(
        title="MMseqs2 Protein Database",
        description="Path to MMseqs2 target database (FASTA file or MMseqs2 createdb output).",
    )
    reference_fasta: str = ConfigField(
        title="Reference FASTA",
        description="FASTA file used to recover top-hit sequences by target ID.",
    )
    max_gap_gini: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Max Gap Gini",
        description="Maximum acceptable gap Gini (0-1, inclusive) against the nearest reference protein.",
        examples=[0.1, 0.2, 0.3],
    )
    pass_no_hits: bool = ConfigField(
        default=True,
        title="Pass No Hits",
        description="If True, proposals with no MMseqs2 hit pass as novel sequences.",
    )
    trim_alignment: bool = ConfigField(
        default=True,
        title="Trim Alignment",
        description="Center-crop the pairwise alignment and strip end gaps before computing gap Gini.",
    )
    mmseqs_config: Mmseqs2SearchProteinsConfig = ConfigField(
        default_factory=Mmseqs2SearchProteinsConfig,
        title="MMseqs2 Config",
        description="Advanced MMseqs2 protein search configuration.",
    )
    mafft_config: MafftConfig = ConfigField(
        default_factory=MafftConfig,
        title="MAFFT Config",
        description="Advanced MAFFT pairwise alignment configuration.",
    )


@constraint(
    key="protein-nearest-neighbor-gap-gini",
    label="Protein Nearest-Neighbor Gap Gini",
    config=ProteinNearestNeighborGapGiniConfig,
    description="Align proteins (or longest ORFs from DNA) to the nearest reference hit and score gap concentration.",
    tools_called=["mmseqs2-search-proteins", "mafft-align", "orfipy-prediction"],
    category="protein_quality",
    supported_sequence_types=["dna", "protein"],
)
def protein_nearest_neighbor_gap_gini_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinNearestNeighborGapGiniConfig
) -> list[ConstraintOutput]:
    """Require low gap Gini against the nearest reference protein.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal sequence
            tuples. Each tuple must contain one DNA or protein sequence. DNA is
            translated using the longest canonical ORF rule.
        config (ProteinNearestNeighborGapGiniConfig): MMseqs2 reference,
            reference FASTA, and gap Gini threshold.

    Returns:
        list[ConstraintOutput]: One output per proposal. A score of 0.0 passes
            and 1.0 fails. Metadata contains the nearest target ID, nearest hit
            sequence, and computed gap Gini.

    Raises:
        RuntimeError: If MMseqs2 reports failure.
    """
    resolved_sequences = resolve_protein_complex_chains(input_sequences)
    proteins: list[str] = []
    valid_indices: list[int] = []
    metadata_by_idx: list[dict[str, Any]] = []
    for idx, (chain_sequences, resolution_metadata) in enumerate(resolved_sequences):
        metadata_by_idx.append(resolution_metadata)
        if chain_sequences is None:
            continue
        if len(chain_sequences) != 1:
            raise ValueError(
                f"protein_nearest_neighbor_gap_gini_constraint expects single-chain proposals; "
                f"got {len(chain_sequences)} chains at proposal index {idx}."
            )
        proteins.append(chain_sequences[0])
        valid_indices.append(idx)

    outputs = [ConstraintOutput(score=MAX_ENERGY, metadata=metadata) for metadata in metadata_by_idx]
    if not proteins:
        return outputs

    mmseqs_result = run_mmseqs2_search_proteins(
        Mmseqs2SearchProteinsInput(query_sequences=proteins),
        config.mmseqs_config.model_copy(update={"mmseqs_db": config.mmseqs_db}),
    )

    reference_sequences = load_fasta(config.reference_fasta)
    for protein_idx, original_idx in enumerate(valid_indices):
        result = mmseqs_result.results[protein_idx]
        top_hit = result.hits[0] if result.hits else None
        result_metadata: dict[str, Any] = {
            **metadata_by_idx[original_idx],
            "resolved_protein_sequence": proteins[protein_idx],
            "has_mmseqs_hit": top_hit is not None,
            "gap_gini": None,
            "top_hit_target_id": None,
            "top_hit_evalue": None,
            "nearest_hit_seq": None,
        }

        if top_hit is None:
            outputs[original_idx] = ConstraintOutput(
                score=MIN_ENERGY if config.pass_no_hits else MAX_ENERGY,
                metadata=result_metadata,
            )
            continue

        nearest_hit = reference_sequences.get(top_hit.target_id)
        result_metadata.update(
            {
                "top_hit_target_id": top_hit.target_id,
                "top_hit_evalue": top_hit.evalue,
                "nearest_hit_seq": nearest_hit,
            }
        )
        if nearest_hit is None:
            logger.warning(
                "Top hit target_id %r not found in reference FASTA; failing the proposal "
                "(reference_fasta may be out of sync with mmseqs_db).",
                top_hit.target_id,
            )
            result_metadata["nearest_hit_lookup_error"] = (
                f"top hit target_id {top_hit.target_id!r} not in reference_fasta"
            )
            outputs[original_idx] = ConstraintOutput(score=MAX_ENERGY, metadata=result_metadata)
            continue

        gap_gini = _compute_pairwise_gap_gini(
            proteins[protein_idx], nearest_hit, config.trim_alignment, config.mafft_config
        )
        result_metadata["gap_gini"] = gap_gini
        outputs[original_idx] = ConstraintOutput(
            score=MIN_ENERGY if gap_gini <= config.max_gap_gini else MAX_ENERGY,
            metadata=result_metadata,
        )

    n_pass = sum(1 for result in outputs if result.score == MIN_ENERGY)
    logger.info(
        "protein_nearest_neighbor_gap_gini_constraint: %d/%d have gap Gini <= %.3f",
        n_pass,
        len(outputs),
        config.max_gap_gini,
    )
    return outputs


def _compute_pairwise_gap_gini(
    query: str, reference: str, should_trim_alignment: bool, mafft_config: MafftConfig
) -> float:
    """Align two proteins with MAFFT and compute gap Gini."""
    try:
        align_result = run_mafft_align(MafftInput(sequences=[query, reference]), mafft_config)
    except Exception as e:
        logger.warning(
            "Pairwise MAFFT alignment failed for nearest-neighbor gap Gini (query_len=%d, reference_len=%d): %s",
            len(query),
            len(reference),
            e,
            exc_info=True,
        )
        return 1.0

    if not align_result.msa or len(align_result.msa) < 2:
        logger.warning(
            "Pairwise MAFFT alignment returned fewer than two aligned sequences for nearest-neighbor gap Gini "
            "(query_len=%d, reference_len=%d); returning worst score.",
            len(query),
            len(reference),
        )
        return 1.0

    al1, al2 = align_result.msa[0], align_result.msa[1]
    if should_trim_alignment:
        al1, al2 = trim_alignment(al1, al2)
        if al1 is None:
            logger.warning(
                "Pairwise MAFFT alignment had no overlapping residues after trimming for nearest-neighbor gap Gini "
                "(query_len=%d, reference_len=%d); returning worst score.",
                len(query),
                len(reference),
            )
            return 1.0
    return gap_gini_single(al1, al2)
