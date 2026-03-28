"""
proto_language/language/constraint/rna_splicing/splice_transformer_specificity.py

Evaluate tissue-specific splicing with SpliceTransformer.

Accepts three segments (left_flank, intron_core, right_flank), concatenates
them into a single 1-kb target sequence, and scores tissue-specific splice
site usage. Metadata is propagated back to all three input segments.
"""
from __future__ import annotations

import logging
from typing import List, Literal, Tuple

from proto_tools import (
    SpliceTransformerConfig,
    SpliceTransformerInput,
    run_splice_transformer,
)
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence

logger = logging.getLogger(__name__)

SpliceTransformerTissueName = Literal[
    "AVERAGE",
    "ADIPOSE_TISSUE",
    "BLOOD",
    "BLOOD_VESSEL",
    "BRAIN",
    "COLON",
    "HEART",
    "KIDNEY",
    "LIVER",
    "LUNG",
    "MUSCLE",
    "NERVE",
    "SMALL_INTESTINE",
    "SKIN",
    "SPLEEN",
    "STOMACH",
]

# SpliceTransformer prediction channels are:
# [0: neither, 1: acceptor, 2: donor, 3+: tissue logits].
SPLICE_TISSUE_CHANNEL_INDEX: dict[SpliceTransformerTissueName, int | None] = {
    "AVERAGE": None,
    "ADIPOSE_TISSUE": 3,
    "BLOOD": 4,
    "BLOOD_VESSEL": 5,
    "BRAIN": 6,
    "COLON": 7,
    "HEART": 8,
    "KIDNEY": 9,
    "LIVER": 10,
    "LUNG": 11,
    "MUSCLE": 12,
    "NERVE": 13,
    "SMALL_INTESTINE": 14,
    "SKIN": 15,
    "SPLEEN": 16,
    "STOMACH": 17,
}


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

        splice_pos (list[int]): Zero-indexed position(s) within the input
            sequence to evaluate for tissue-specific splicing. These positions
            typically correspond to splice sites (donor or acceptor) where you
            want to assess or control tissue-specific usage. Can be a single
            integer (automatically converted to list) or list of integers for
            multiple positions.

        tissue (SpliceTransformerTissueName): Target tissue for specificity evaluation.
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
    splice_pos: List[int] = ConfigField(
        title="Splice Position(s)",
        description="0-indexed position(s) into input_sequence on which to compute the score",
    )
    tissue: SpliceTransformerTissueName = ConfigField(
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
        advanced=True,
    )

    @field_validator('splice_pos', mode='before')
    @classmethod
    def convert_splice_pos_to_list(cls, v):
        """Convert single int to list of ints."""
        if isinstance(v, int):
            return [v]
        return v


@constraint(
    key="splice-transformer-specificity",
    label="SpliceTransformer tissue specificity score",
    config=SpliceTransformerSpecificityConfig,
    description=(
        "Evaluate tissue-specific splicing with SpliceTransformer. "
        "Takes three segments (left_flank, intron_core, right_flank), "
        "concatenates them into the 1-kb target, and scores tissue specificity."
    ),
    uses_gpu=True,
    tools_called=["splice-transformer-prediction"],
    category="rna splicing",
    supported_sequence_types=["dna"],
    num_input_sequences_per_tuple=3,
)
def splice_transformer_specificity(
    input_sequences: List[Tuple[Sequence, ...]],
    config: SpliceTransformerSpecificityConfig,
) -> List[float]:
    """Score tissue-specific splice site usage for three-segment intron boundaries.

    Accepts three segments (left_flank, intron_core, right_flank), concatenates
    them into a single 1-kb target sequence, and scores tissue-specific splice
    site usage. Metadata is propagated back to all three input segments.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Mapping of segment IDs to
            their current sequences.
        config (SpliceTransformerSpecificityConfig): Constraint configuration
            controlling evaluation parameters.
    """
    if not input_sequences:
        return []

    assert len(config.left_context) == len(config.right_context)
    context_length = len(config.left_context)
    tissue_channel_index = SPLICE_TISSUE_CHANNEL_INDEX[config.tissue]

    # Concatenate 3-part tuples into target sequences for batched inference.
    target_seqs = []
    for left_flank, intron_core, right_flank in input_sequences:
        target_seqs.append(
            left_flank.sequence + intron_core.sequence + right_flank.sequence
        )

    target_lengths = {len(t) for t in target_seqs}
    if len(target_lengths) != 1:
        raise ValueError(
            "SpliceTransformer specificity requires equal-length target sequences in a batch."
        )

    splice_transformer_input = SpliceTransformerInput(
        target_seqs=target_seqs,
        left_contexts=[config.left_context] * len(target_seqs),
        right_contexts=[config.right_context] * len(target_seqs),
    )
    splice_transformer_config = config.splice_transformer_config.model_copy(
        update={"context_length": context_length}
    )

    output = run_splice_transformer(
        splice_transformer_input,
        splice_transformer_config,
    ).prediction

    if output.shape[0] != len(target_seqs):
        raise ValueError(
            "SpliceTransformer batch size mismatch: "
            f"{output.shape[0]} outputs for {len(target_seqs)} inputs."
        )

    scores = []
    for batch_idx, (left_flank, intron_core, right_flank) in enumerate(input_sequences):
        if output.shape[1] != len(target_seqs[batch_idx]):
            raise ValueError(
                "SpliceTransformer output length mismatch: "
                f"{output.shape[1]} != {len(target_seqs[batch_idx])}."
            )

        if tissue_channel_index is None:
            raw_score = float(output[batch_idx, config.splice_pos, 3:].mean())
        else:
            raw_score = float(
                output[batch_idx, config.splice_pos, tissue_channel_index].mean()
            )

        if config.direction == "max":
            score = 1.0 - raw_score
        elif config.direction == "min":
            score = raw_score
        else:
            raise ValueError(
                f"Invalid SpliceTransformer specificity direction: {config.direction}, "
                "must be either 'max' or 'min'."
            )

        metadata = {
            f"specificity_direction_{config.tissue}": config.direction,
            f"specificity_score_{config.tissue}": score,
        }
        for seq in (left_flank, intron_core, right_flank):
            seq._metadata.update(metadata)

        scores.append(score)

    return scores
