"""Maximum protein identity constraint using MMseqs2."""

import logging
from typing import Any

from proto_tools import Mmseqs2SearchProteinsConfig, Mmseqs2SearchProteinsInput, run_mmseqs2_search_proteins

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY, load_reference_sequences
from proto_language.utils.orf_selection import resolve_protein_complex_chains

logger = logging.getLogger(__name__)


class ProteinMaxIdentityConfig(BaseConfig):
    """Configuration for maximum identity to a protein reference set.

    Attributes:
        mmseqs_db (str): Path to a target database for MMseqs2 (FASTA file or
            MMseqs2 ``createdb`` output).
        max_identity (float): Maximum allowed percent identity (0-100) to the
            top hit.
        pass_no_hits (bool): If True, proposals with no MMseqs2 hits pass.
        reference_fasta (str | None): Optional FASTA file used to recover the top
            hit sequence for downstream reporting.
        mmseqs_config (Mmseqs2SearchProteinsConfig): Advanced MMseqs2 search
            configuration.
    """

    mmseqs_db: str = ConfigField(
        title="MMseqs2 Protein Database",
        description="Path to MMseqs2 target database (FASTA file or MMseqs2 createdb output).",
    )
    max_identity: float = ConfigField(
        default=90.0,
        ge=0.0,
        le=100.0,
        title="Maximum Percent Identity",
        description="Maximum allowed percent identity (0-100, inclusive) to the top reference hit.",
        examples=[70, 90, 95],
    )
    pass_no_hits: bool = ConfigField(
        default=True,
        title="Pass No Hits",
        description="If True, proposals with no MMseqs2 hit pass as novel sequences.",
    )
    reference_fasta: str | None = ConfigField(
        default=None,
        title="Reference FASTA",
        description="Optional FASTA file for recovering the top hit sequence by target ID.",
    )
    mmseqs_config: Mmseqs2SearchProteinsConfig = ConfigField(
        default_factory=Mmseqs2SearchProteinsConfig,
        title="MMseqs2 Config",
        description="Advanced MMseqs2 protein search configuration.",
    )


@constraint(
    key="protein-max-identity",
    label="Protein Max Identity",
    config=ProteinMaxIdentityConfig,
    description="Require proteins, or longest ORFs from DNA, to stay below a maximum identity to references.",
    tools_called=["mmseqs2-search-proteins", "orfipy-prediction"],
    category="protein quality",
    supported_sequence_types=["dna", "protein"],
)
def protein_max_identity_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinMaxIdentityConfig
) -> list[ConstraintOutput]:
    """Require top-hit MMseqs2 identity to remain below a maximum threshold.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal sequence
            tuples. Each tuple must contain one DNA or protein sequence. DNA is
            translated using the longest canonical ORF rule.
        config (ProteinMaxIdentityConfig): MMseqs2 database, maximum identity,
            no-hit handling, and optional reference FASTA.

    Returns:
        list[ConstraintOutput]: One output per proposal. A score of 0.0 passes
            and 1.0 fails. Metadata contains top-hit identity, target ID, target
            sequence when available, and selected ORF details for DNA.

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
                f"protein_max_identity_constraint expects single-chain proposals; "
                f"got {len(chain_sequences)} chains at proposal index {idx}."
            )
        proteins.append(chain_sequences[0])
        valid_indices.append(idx)

    outputs = [ConstraintOutput(score=MAX_ENERGY, metadata=metadata) for metadata in metadata_by_idx]
    if not proteins:
        return outputs

    mmseqs_result = run_mmseqs2_search_proteins(
        Mmseqs2SearchProteinsInput(query_sequences=proteins, mmseqs_db=config.mmseqs_db),
        config.mmseqs_config,
    )

    reference_sequences = load_reference_sequences(config.reference_fasta) if config.reference_fasta else {}

    for protein_idx, original_idx in enumerate(valid_indices):
        result = mmseqs_result.results[protein_idx]
        top_hit = result.hits[0] if result.hits else None
        result_metadata: dict[str, Any] = {
            **metadata_by_idx[original_idx],
            "resolved_protein_sequence": proteins[protein_idx],
            "has_mmseqs_hit": top_hit is not None,
            "identity": 0.0,
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

        result_metadata.update(
            {
                "identity": top_hit.pident,
                "top_hit_target_id": top_hit.target_id,
                "top_hit_evalue": top_hit.evalue,
                "nearest_hit_seq": reference_sequences.get(top_hit.target_id),
            }
        )
        passes = top_hit.pident <= config.max_identity
        outputs[original_idx] = ConstraintOutput(score=MIN_ENERGY if passes else MAX_ENERGY, metadata=result_metadata)

    n_pass = sum(1 for result in outputs if result.score == MIN_ENERGY)
    logger.info(
        "protein_max_identity_constraint: %d/%d have top-hit identity <= %.2f%%",
        n_pass,
        len(outputs),
        config.max_identity,
    )
    return outputs
