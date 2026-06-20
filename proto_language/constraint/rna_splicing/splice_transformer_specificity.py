"""Evaluate tissue-specific splicing with SpliceTransformer.

Accepts three segments (left_flank, intron_core, right_flank), concatenates
them, windows the result to the fixed SpliceTransformer target length centred
on the requested splice positions, and scores tissue-specific splice site usage.
"""

import logging
from typing import Literal

import numpy as np
from proto_tools import CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH
from proto_tools import (
    SPLICE_TISSUE_CHANNEL_INDEX,
    SpliceTransformerConfig,
    SpliceTransformerInput,
    SpliceTransformerTissue,
    run_splice_transformer,
)
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.rna_splicing.splice_transformer_target import (
    apply_target_window,
    remap_positions,
    splice_target_window_start,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)


class SpliceTransformerSpecificityConfig(BaseConfig):
    """Configuration for SpliceTransformer tissue-specific splicing constraint.

    This class defines configuration parameters for evaluating tissue-specific
    splicing patterns using SpliceTransformer, a deep learning model trained to
    predict splice site usage across different human tissues. The constraint can
    be used to either maximize splicing in a specific tissue (tissue-specific
    activation) or minimize it (tissue-specific repression), enabling design of
    sequences with controlled tissue-specific alternative splicing.

    Attributes:
        left_context (str): DNA sequence providing left (5') context for
            SpliceTransformer prediction. Must be exactly 4000 bp (CONTEXT_LENGTH).
            This flanking sequence provides upstream genomic context that influences
            tissue-specific splice site recognition. Should be the genomic sequence
            immediately 5' of the target sequence.

        right_context (str): DNA sequence providing right (3') context for
            SpliceTransformer prediction. Must be exactly 4000 bp (CONTEXT_LENGTH).
            This flanking sequence provides downstream genomic context. Should be
            the genomic sequence immediately 3' of the target sequence.

        splice_pos (list[int]): Zero-indexed position(s) within the concatenated
            target sequence (left_flank + intron_core + right_flank) to evaluate
            for tissue-specific splicing. These positions
            typically correspond to splice sites (donor or acceptor) where you
            want to assess or control tissue-specific usage. Can be a single
            integer (automatically converted to list) or list of integers for
            multiple positions.

        tissue (SpliceTransformerTissue): Target tissue for specificity evaluation.
            Options include "AVERAGE" (average across all tissues, default) or
            specific tissues like "BRAIN", "HEART", "LIVER", "MUSCLE", "STOMACH",
            etc. SpliceTransformer was trained on RNA-seq data from multiple human
            tissues and can predict tissue-specific splicing patterns. Use "AVERAGE"
            for general splice site quality or specific tissues for tissue-specific
            designs. Default: "AVERAGE".

        direction (Literal['max', 'min']): Optimization direction for the splice
            score. Use "max" to maximize splicing at the position (encourage splice
            site usage), or "min" to minimize splicing (discourage splice site usage).
            For example, use "max" to create tissue-specific splice sites that are
            active in the target tissue, or "min" to create sites that are silenced
            in the target tissue. Default: "max".

        splice_transformer_config (SpliceTransformerConfig): Advanced SpliceTransformer
            configuration including context length, device settings, and model
            parameters. Default: SpliceTransformerConfig().
    """

    # Required parameters
    left_context: str = ConfigField(
        title="Left Context",
        description="Sequence of the left context for SpliceTransformer",
    )
    right_context: str = ConfigField(
        title="Right Context",
        description="Sequence of the right context for SpliceTransformer",
    )
    splice_pos: list[int] = ConfigField(
        title="Splice Position(s)",
        description="0-indexed position(s) into the concatenated target sequence on which to compute the score",
    )
    tissue: SpliceTransformerTissue = ConfigField(
        title="Tissue to Evaluate",
        default="AVERAGE",
        description="Tissue on which to define the score. By default, averages across all tissues.",
    )
    direction: Literal["max", "min"] = ConfigField(
        title="Optimization Direction",
        default="max",
        description="Whether to maximize or minimize the value. Defaults to 'max'",
    )
    # Optional parameter
    splice_transformer_config: SpliceTransformerConfig = ConfigField(
        title="SpliceTransformer Config",
        default_factory=SpliceTransformerConfig,
        description="Advanced parameter configuration for SpliceTransformer.",
    )

    @field_validator("splice_pos", mode="before")
    @classmethod
    def convert_splice_pos_to_list(cls, v: int | list[int]) -> list[int]:
        """Convert single int to list of ints."""
        if isinstance(v, int):
            return [v]
        return v


@constraint(
    key="splice-transformer-specificity",
    label="SpliceTransformer tissue specificity score",
    config=SpliceTransformerSpecificityConfig,
    description="Score tissue-specific splicing with SpliceTransformer on three segments concatenated into one 1-kb target.",
    uses_gpu=True,
    tools_called=["splice-transformer-prediction"],
    category="rna_splicing",
    supported_sequence_types=["dna"],
    input_labels=["Left Flank", "Intron Core", "Right Flank"],
)
def splice_transformer_specificity(
    input_sequences: list[tuple[Sequence, ...]],
    config: SpliceTransformerSpecificityConfig,
) -> list[ConstraintOutput]:
    """Score tissue-specific splice site usage for three-segment intron boundaries.

    Accepts three segments (left_flank, intron_core, right_flank), concatenates
    them into a single 1-kb target sequence, and scores tissue-specific splice
    site usage.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Mapping of segment IDs to
            their current sequences.
        config (SpliceTransformerSpecificityConfig): Constraint configuration
            controlling evaluation parameters.

    Returns:
        list[ConstraintOutput]: One result per input. ``score`` is in ``[0.0, 1.0]``
            (interpretation depends on direction). ``metadata`` carries
            ``specificity_direction_<tissue>`` and ``specificity_score_<tissue>``.
    """
    if not input_sequences:
        return []

    if not (len(config.left_context) == len(config.right_context) == SPLICE_TRANSFORMER_CONTEXT_LENGTH):
        raise ValueError(
            f"splice-transformer: left/right context must each be {SPLICE_TRANSFORMER_CONTEXT_LENGTH} bp; "
            f"got left={len(config.left_context)}, right={len(config.right_context)}"
        )
    tissue_channel_index = SPLICE_TISSUE_CHANNEL_INDEX[config.tissue]

    # Concatenate 3-part tuples into target sequences for batched inference.
    target_seqs = []
    for left_flank, intron_core, right_flank in input_sequences:
        target_seqs.append(left_flank.sequence + intron_core.sequence + right_flank.sequence)

    target_lengths = {len(t) for t in target_seqs}
    if len(target_lengths) != 1:
        raise ValueError("SpliceTransformer specificity requires equal-length target sequences in a batch.")

    # Window the concatenated target down to the fixed SpliceTransformer target
    # length, centred on the requested splice positions, and remap them.
    window_start = splice_target_window_start(target_lengths.pop(), config.splice_pos)
    target_seqs = apply_target_window(target_seqs, window_start)
    splice_pos = remap_positions(config.splice_pos, window_start)

    splice_transformer_input = SpliceTransformerInput(
        target_seqs=target_seqs,
        left_contexts=[config.left_context] * len(target_seqs),
        right_contexts=[config.right_context] * len(target_seqs),
    )

    output = np.array(
        run_splice_transformer(
            splice_transformer_input,
            config.splice_transformer_config,
        ).prediction
    )

    if output.shape[0] != len(target_seqs):
        raise ValueError(
            f"SpliceTransformer batch size mismatch: {output.shape[0]} outputs for {len(target_seqs)} inputs."
        )

    results: list[ConstraintOutput] = []
    for batch_idx in range(len(input_sequences)):
        if output.shape[1] != len(target_seqs[batch_idx]):
            raise ValueError(
                f"SpliceTransformer output length mismatch: {output.shape[1]} != {len(target_seqs[batch_idx])}."
            )

        if tissue_channel_index is None:
            raw_score = float(output[batch_idx, splice_pos, 3:].mean())
        else:
            raw_score = float(output[batch_idx, splice_pos, tissue_channel_index].mean())

        if config.direction == "max":
            score = 1.0 - raw_score
        elif config.direction == "min":
            score = raw_score
        else:
            raise ValueError(
                f"Invalid SpliceTransformer specificity direction: {config.direction}, must be either 'max' or 'min'."
            )

        metadata = {
            f"specificity_direction_{config.tissue}": config.direction,
            f"specificity_score_{config.tissue}": score,
            "windowed_splice_pos": splice_pos,
            "target_window_start": window_start,
        }
        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results
