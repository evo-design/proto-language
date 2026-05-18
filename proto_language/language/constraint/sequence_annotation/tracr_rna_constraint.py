"""tracrRNA prediction constraint for CRISPR loci."""

import logging
from typing import Any

from proto_tools import CrisprTracrRNAConfig, CrisprTracrRNAInput, run_crispr_tracr_rna

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY

logger = logging.getLogger(__name__)


class CrisprTracrRNAConstraintConfig(BaseConfig):
    """Configuration for tracrRNA prediction in CRISPR loci.

    Attributes:
        require_intarna_interaction (bool): If True, require the top tracrRNA
            candidate to include an IntaRNA anti-repeat interaction.
        tracr_config (CrisprTracrRNAConfig): Advanced CRISPRtracrRNA tool
            configuration.
    """

    require_intarna_interaction: bool = ConfigField(
        default=True,
        title="Require IntaRNA Interaction",
        description="Require an IntaRNA anti-repeat interaction for the top tracrRNA candidate.",
    )
    tracr_config: CrisprTracrRNAConfig = ConfigField(
        default_factory=CrisprTracrRNAConfig,
        title="CRISPRtracrRNA Config",
        description="Advanced CRISPRtracrRNA tool configuration.",
    )


@constraint(
    key="crispr-tracr-rna",
    label="CRISPR tracrRNA",
    config=CrisprTracrRNAConstraintConfig,
    description="Predict tracrRNA candidates for CRISPR loci and optionally require IntaRNA support.",
    tools_called=["crispr-tracr-rna"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
)
def crispr_tracr_rna_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: CrisprTracrRNAConstraintConfig
) -> list[ConstraintOutput]:
    """Require a CRISPRtracrRNA tracrRNA candidate in each DNA sequence.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal DNA sequence
            tuples. Each tuple must contain one DNA sequence.
        config (CrisprTracrRNAConstraintConfig): tracrRNA prediction settings.

    Returns:
        list[ConstraintOutput]: One output per proposal. A score of 0.0 passes
            and 1.0 fails. Metadata contains the top tracrRNA sequence, IntaRNA
            interaction details, interaction energy, and all candidate calls.

    Raises:
        RuntimeError: If the CRISPRtracrRNA tool reports failure.
    """
    sequences = [seq.sequence for (seq,) in input_sequences]
    tracr_result = run_crispr_tracr_rna(CrisprTracrRNAInput(sequences=sequences), config.tracr_config)

    results: list[ConstraintOutput] = []
    for seq_idx in range(len(sequences)):
        seq_result = tracr_result.results[seq_idx] if seq_idx < len(tracr_result.results) else None
        candidates = list(seq_result.candidates) if seq_result is not None else []
        top = candidates[0] if candidates else None
        tracr_sequence = top.tracr_rna_sequence if top is not None else None
        intarna_interaction = top.intarna_anti_repeat_interaction if top is not None else None
        interaction_energy = top.interaction_energy if top is not None else None
        has_tracr = tracr_sequence is not None
        has_required_interaction = (not config.require_intarna_interaction) or intarna_interaction is not None
        passes = has_tracr and has_required_interaction

        candidate_dicts = [candidate.model_dump() for candidate in candidates]
        metadata: dict[str, Any] = {
            "has_tracr": has_tracr,
            "has_intarna_interaction": intarna_interaction is not None,
            "tracr_sequence": tracr_sequence,
            "interaction_energy": interaction_energy,
            "intarna_anti_repeat_interaction": intarna_interaction,
            "tracr_candidates": candidate_dicts or None,
        }
        results.append(ConstraintOutput(score=MIN_ENERGY if passes else MAX_ENERGY, metadata=metadata))

    n_pass = sum(1 for result in results if result.score == MIN_ENERGY)
    logger.info("crispr_tracr_rna_constraint: %d/%d have tracrRNA support", n_pass, len(results))
    return results
