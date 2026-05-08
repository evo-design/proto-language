"""CRISPR array detection constraint using MinCED."""

import logging
from typing import Any

from proto_tools import MincedConfig, MincedInput, run_minced

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY

logger = logging.getLogger(__name__)


class CrisprArrayConfig(BaseConfig):
    """Configuration for CRISPR array detection.

    Attributes:
        minced_config (MincedConfig): MinCED tool configuration. Tune
            ``min_num_repeats`` and ``min_repeat_length`` here to control what
            counts as a CRISPR array. Defaults to ``min_num_repeats=3``,
            ``min_repeat_length=23`` (more permissive than MinCED's stock
            ``min_repeat_length=27``).
    """

    minced_config: MincedConfig = ConfigField(
        default_factory=lambda: MincedConfig(min_num_repeats=3, min_repeat_length=23),
        title="MinCED Config",
        description="MinCED CRISPR detection configuration. Tune min_num_repeats and min_repeat_length here.",
    )


@constraint(
    key="crispr-array",
    label="CRISPR Array",
    config=CrisprArrayConfig,
    description="Detect CRISPR repeat-spacer arrays in DNA sequences using MinCED.",
    tools_called=["minced-crispr"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
)
def crispr_array_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: CrisprArrayConfig
) -> list[ConstraintOutput]:
    """Require a MinCED-detected CRISPR array in each DNA sequence.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal DNA sequence
            tuples. Each tuple must contain one DNA sequence.
        config (CrisprArrayConfig): MinCED thresholds and advanced tool config.

    Returns:
        list[ConstraintOutput]: One output per proposal. A score of 0.0 passes
            and 1.0 fails. Metadata contains array count, first repeat sequence,
            and a JSON file reference with all MinCED array calls.

    Raises:
        RuntimeError: If MinCED reports failure.
    """
    sequences = [seq.sequence for (seq,) in input_sequences]
    minced_result = run_minced(MincedInput(sequences=sequences), config.minced_config)
    if minced_result.success is False:
        raise RuntimeError(f"MinCED CRISPR detection failed: {minced_result.errors}")

    results: list[ConstraintOutput] = []
    for seq_idx in range(len(sequences)):
        seq_result = minced_result.results[seq_idx] if seq_idx < len(minced_result.results) else None
        arrays = list(seq_result.crispr_arrays) if seq_result is not None else []
        first_repeat = None
        if arrays and arrays[0].repeats_and_spacers:
            first_repeat = arrays[0].repeats_and_spacers[0].repeat

        array_dicts = [array.model_dump() for array in arrays]
        has_crispr = bool(arrays)
        metadata: dict[str, Any] = {
            "has_crispr_array": has_crispr,
            "crispr_array_count": len(arrays),
            "crispr_repeat": first_repeat,
            "minced_arrays": array_dicts or None,
        }
        results.append(ConstraintOutput(score=MIN_ENERGY if has_crispr else MAX_ENERGY, metadata=metadata))

    n_pass = sum(1 for result in results if result.score == MIN_ENERGY)
    logger.info("crispr_array_constraint: %d/%d have CRISPR arrays", n_pass, len(results))
    return results
