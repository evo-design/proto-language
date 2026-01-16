"""
Evaluate tissue-specific splicing with SpliceTransformer.
"""
from __future__ import annotations
from typing import List, Optional, Literal

from pydantic import field_validator

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.rna_splicing.splice_transformer import (
    SpliceTransformerConfig,
    SpliceTransformerInput,
    SpliceTransformerTissue,
    TISSUE_INDEX_OFFSET,
    run_splice_transformer,
)


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

        splice_pos (List[int]): Zero-indexed position(s) within the input
            sequence to evaluate for tissue-specific splicing. These positions
            typically correspond to splice sites (donor or acceptor) where you
            want to assess or control tissue-specific usage. Can be a single
            integer (automatically converted to list) or list of integers for
            multiple positions.

        tissue (SpliceTransformerTissue): Target tissue for specificity evaluation.
            Options include "AVERAGE" (average across all tissues, default) or
            specific tissues like "BRAIN", "HEART", "LIVER", "MUSCLE", "TESTIS",
            etc. SpliceTransformer was trained on RNA-seq data from multiple human
            tissues and can predict tissue-specific splicing patterns. Use "AVERAGE"
            for general splice site quality or specific tissues for tissue-specific
            designs. Default: "AVERAGE".

        direction (Literal["max", "min"]): Optimization direction for the splice
            score. Use "max" to maximize splicing at the position (encourage splice
            site usage), or "min" to minimize splicing (discourage splice site usage).
            For example, use "max" to create tissue-specific splice sites that are
            active in the target tissue, or "min" to create sites that are silenced
            in the target tissue. Default: "max".

        splice_transformer_config (Optional[SpliceTransformerConfig]): Optional
            advanced SpliceTransformer configuration including context length,
            device settings, and model parameters. If None, uses default
            configuration with context_length=4000. Default: None.
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
    tissue: SpliceTransformerTissue.as_literal() = ConfigField(
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
    splice_transformer_config: Optional[SpliceTransformerConfig] = ConfigField(
        title="SpliceTransformer Config",
        default=None,
        description="Advanced parameter configuration for SpliceTransformer. If None, uses default configuration.",
        advanced=True,
    )

    @field_validator('splice_pos', mode='before')
    @classmethod
    def convert_splice_pos_to_list(cls, v):
        """Convert single int to list of ints."""
        if isinstance(v, int):
            return [v]
        return v


@ConstraintRegistry.register(
    key="splice-transformer-specificity",
    label="SpliceTransformer tissue specificity score",
    config=SpliceTransformerSpecificityConfig,
    description="Evaluate tissue-specific splicing with SpliceTransformer",
    batched=False,
    concatenate=True,
    gpu_required=True,
    tools_called=["splice_transformer"],
    category="rna splicing",
    supported_sequence_types=["dna"],
)
def splice_transformer_specificity(
    input_sequence: Sequence,
    config: SpliceTransformerSpecificityConfig,
) -> float:
    """Evaluate tissue-specific splicing with SpliceTransformer.
    
    This constraint function uses SpliceTransformer to predict tissue-specific
    splice site usage at specified positions in a DNA sequence. The model was
    trained on GTEx (Genotype-Tissue Expression) RNA-seq data from multiple human
    tissues and can predict how strongly a splice site will be used in different tissue
    contexts.
    
    The constraint enables design of sequences with controlled tissue-specific
    alternative splicing, such as brain-specific exon inclusion or liver-specific
    exon skipping. By setting the direction parameter, you can either maximize
    splicing in a target tissue (for tissue-specific activation) or minimize it
    (for tissue-specific repression).
    
    The function requires precisely sized inputs: the target sequence must be
    exactly 1000 bp, and both flanking contexts must be exactly 4000 bp each,
    for a total analyzed region of 9000 bp.

    Args:
        input_sequence (Sequence): DNA sequence to evaluate. Must be exactly
            1000 bp in length. This is the central region containing the positions
            to be evaluated for tissue-specific splicing.
            
        config (SpliceTransformerSpecificityConfig): Configuration object containing
            ``left_context`` (4000 bp), ``right_context`` (4000 bp), ``splice_pos``
            (position(s) to evaluate), ``tissue`` (target tissue, default: "AVERAGE"),
            ``direction`` (optimization direction, default: "max"), and optional
            ``splice_transformer_config`` for advanced settings.

    Returns:
        float: Constraint score ranging from 0.0 to 1.0. The interpretation depends
            on the ``direction`` parameter:
            
            - **direction="max"** (maximize splicing): Score = 1.0 - tissue_probability.
              Lower scores indicate stronger predicted splicing (0.0 = 100% probability,
              1.0 = 0% probability). Use this to encourage tissue-specific splice
              site usage.
            - **direction="min"** (minimize splicing): Score = tissue_probability.
              Lower scores indicate weaker predicted splicing (0.0 = 0% probability,
              1.0 = 100% probability). Use this to discourage tissue-specific splice
              site usage.
            
            When multiple positions are specified, the score uses the mean probability
            across all positions.

    Raises:
        AssertionError: If left_context and right_context lengths don't match, or
            if the output shape doesn't match the input sequence length.
        ValueError: If direction is not "max" or "min".
    
    Note:
        This function modifies the input sequence by adding metadata to the
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:
        
        - ``specificity_direction_{tissue}``: String indicating the optimization
          direction used ("max" or "min")
        - ``specificity_score_{tissue}``: Float constraint score for the specified
          tissue
        
        The metadata keys include the tissue name, so evaluating multiple tissues
        on the same sequence will create separate metadata entries for each.
    
    Examples:
        Maximizing brain-specific splicing at a splice site:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> # 1000 bp target sequence
        >>> target_seq = Sequence("ATCG" * 250, "dna")
        >>> # 4000 bp flanking contexts
        >>> left_ctx = "ATCG" * 1000
        >>> right_ctx = "GCTA" * 1000
        >>> 
        >>> config = SpliceTransformerSpecificityConfig(
        ...     left_context=left_ctx,
        ...     right_context=right_ctx,
        ...     splice_pos=500,  # Position to evaluate
        ...     tissue="BRAIN",
        ...     direction="max"  # Maximize brain-specific splicing
        ... )
        >>> score = splice_transformer_specificity(target_seq, config)
        >>> print(score)  # e.g., 0.15 (85% brain-specific probability)
        >>> print(target_seq._metadata["specificity_direction_BRAIN"])  # "max"
        >>> print(target_seq._metadata["specificity_score_BRAIN"])  # 0.15
    """
    assert len(config.left_context) == len(config.right_context)
    context_length = len(config.left_context)
    tissue = SpliceTransformerTissue[config.tissue]

    splice_transformer_input = SpliceTransformerInput(
        target_seqs=[input_sequence.sequence],
        left_contexts=[config.left_context],
        right_contexts=[config.right_context],
    )
    if config.splice_transformer_config is None:
        splice_transformer_config = SpliceTransformerConfig(context_length=context_length)
    else:
        splice_transformer_config = config.splice_transformer_config

    output = run_splice_transformer(
        splice_transformer_input,
        splice_transformer_config,
    ).prediction

    assert output.shape[1] == len(input_sequence.sequence)

    if tissue == SpliceTransformerTissue.AVERAGE:
        score = float(output[:, config.splice_pos, TISSUE_INDEX_OFFSET:].mean())
    else:
        score = float(output[:, config.splice_pos, TISSUE_INDEX_OFFSET + tissue.value].mean())

    if config.direction == "max":
        score = 1. - score
    elif config.direction == "min":
        pass
    else:
        raise ValueError(
            f"Invalid SpliceTransformer specificity direction: {config.direction}, "
            "must be either 'max' or 'min'."
        )

    input_sequence._metadata.update({
        f"specificity_direction_{config.tissue}": config.direction,
        f"specificity_score_{config.tissue}": score,
    })

    return score
