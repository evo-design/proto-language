"""
Evaluate intron boundary prediction with SpliceTransformer.
"""
from __future__ import annotations
from typing import List

from pydantic import field_validator

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.rna_splicing.splice_transformer import (
    run_splice_transformer,
    SpliceTransformerConfig,
    SpliceTransformerInput,
    SpliceTransformerType,
    CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH,
)


class SpliceTransformerIntronBoundaryConfig(BaseConfig):
    """Configuration for SpliceTransformer intron boundary constraint.
    
    This class defines configuration parameters for evaluating splice site quality
    using SpliceTransformer, a deep learning model trained to predict splice sites
    in pre-mRNA sequences. The constraint assesses whether specified positions
    in a sequence are likely to function as authentic splice sites, which is critical
    for proper intron removal and mRNA processing.
    
    Attributes:
        left_context (str): DNA sequence providing left (5') context for
            SpliceTransformer prediction. Must be exactly 4000 bp (CONTEXT_LENGTH).
            This flanking sequence provides upstream context that influences splice
            site recognition. Should be the genomic sequence immediately 5' of the
            target sequence.

        right_context (str): DNA sequence providing right (3') context for
            SpliceTransformer prediction. Must be exactly 4000 bp (CONTEXT_LENGTH).
            This flanking sequence provides downstream context. Should be the
            genomic sequence immediately 3' of the target sequence.

        donor_pos (List[int]): Zero-indexed position(s) of expected donor
            splice site(s) within the input sequence. The donor site marks the 5'
            end of an intron (exon-intron boundary), typically at a "GT" dinucleotide.
            SpliceTransformer predicts on the position immediately before the "GT".
            Can be a single integer (automatically converted to list) or list of
            integers for multiple donors.

        acceptor_pos (List[int]): Zero-indexed position(s) of expected
            acceptor splice site(s) within the input sequence. The acceptor site
            marks the 3' end of an intron (intron-exon boundary), typically at an
            "AG" dinucleotide. SpliceTransformer predicts on the position immediately
            after the "AG". Can be a single integer (automatically converted to list)
            or list of integers for multiple acceptors.

        splice_transformer_config (SpliceTransformerConfig): Advanced SpliceTransformer
            configuration including context length, device settings, and model
            parameters. Default: SpliceTransformerConfig().
    
    Note:
        SpliceTransformer requires sequences of specific lengths:
        - Target sequence (input_sequence): Must be exactly 1000 bp
        - Left context: Must be exactly 4000 bp
        - Right context: Must be exactly 4000 bp
        - Total sequence analyzed: 9000 bp (4000 + 1000 + 4000)
        
        The model outputs splice site probabilities for each position. Higher
        scores indicate stronger predicted splice sites. Authentic splice sites
        typically have scores > 0.5, while non-splice positions have scores near 0.
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
    donor_pos: List[int] = ConfigField(
        title="Donor Position(s)",
        description="0-indexed position(s) into input_sequence of expected donor",
    )
    acceptor_pos: List[int] = ConfigField(
        title="Acceptor Position(s)",
        description="0-indexed position(s) into input_sequence of expected acceptor",
    )

    @field_validator('donor_pos', 'acceptor_pos', mode='before')
    @classmethod
    def convert_pos_to_list(cls, v):
        """Convert single int to list of ints."""
        if isinstance(v, int):
            return [v]
        return v

    # Optional parameter
    splice_transformer_config: SpliceTransformerConfig = ConfigField(
        title="SpliceTransformer Config",
        default_factory=SpliceTransformerConfig,
        description="Advanced parameter configuration for SpliceTransformer.",
        advanced=True,
    )


@ConstraintRegistry.register(
    key="splice-transformer-intron-boundary",
    label="SpliceTransformer intron boundary score",
    config=SpliceTransformerIntronBoundaryConfig,
    description="Evaluate intron boundary prediction with SpliceTransformer",
    batched=False,
    multi_input=False,
    gpu_required=True,
    tools_called=["splice_transformer"],
    category="rna splicing",
    supported_sequence_types=["dna"],
)
def splice_transformer_intron_boundary(
    sequence: Sequence,
    config: SpliceTransformerIntronBoundaryConfig,
) -> float:
    """Evaluate intron boundary prediction with SpliceTransformer
    
    This constraint function uses SpliceTransformer, a deep learning model, to 
    predict the quality of donor and acceptor splice sites at specified positions
    in a DNA sequence. The model analyzes the sequence in its genomic context 
    (with 4 kb flanking regions on each side) to assess whether the specified
    positions are likely to function as authentic splice sites during pre-mRNA
    processing.
    
    SpliceTransformer outputs probabilities for donor and acceptor splice sites
    at each position. Higher probabilities indicate stronger predicted splice
    sites. The constraint score is calculated as 1 - (average of donor and
    acceptor probabilities), so lower scores indicate better splice sites.
    
    The function requires precisely sized inputs: the target sequence must be
    exactly 1000 bp, and both flanking contexts must be exactly 4000 bp each,
    for a total analyzed region of 9000 bp.

    Args:
        sequence (Sequence): DNA sequence to evaluate. Must be exactly
            1000 bp in length. This is the central region containing the splice
            sites to be evaluated.
            
        config (SpliceTransformerIntronBoundaryConfig): Configuration object
            containing ``left_context`` (4000 bp), ``right_context`` (4000 bp),
            ``donor_pos`` (position(s) of donor sites), ``acceptor_pos``
            (position(s) of acceptor sites), and optional
            ``splice_transformer_config`` for advanced settings.

    Returns:
        float: Constraint score ranging from 0.0 (perfect splice sites, both
            donor and acceptor probabilities = 1.0) to 1.0 (poor splice sites,
            probabilities = 0.0). The score is calculated as:
            1.0 - ((donor_probability + acceptor_probability) / 2).
            When multiple positions are specified, the score uses the mean
            probability across all donor and acceptor positions.

    Raises:
        AssertionError: If left_context or right_context are not exactly 4000 bp,
            or if the output shape doesn't match the input sequence length.
    
    Note:
        This function modifies the input sequence by adding metadata to the
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:
        
        - ``donor_pos``: Integer or list of integers indicating donor site
          position(s) evaluated
        - ``acceptor_pos``: Integer or list of integers indicating acceptor
          site position(s) evaluated
        - ``donor_score``: Float constraint score for donor site(s), calculated
          as 1.0 - mean(donor_probabilities). Lower is better.
        - ``acceptor_score``: Float constraint score for acceptor site(s),
          calculated as 1.0 - mean(acceptor_probabilities). Lower is better.
        - ``total_splice_score``: Float overall constraint score (average of
          donor_score and acceptor_score)
        
        **Important prediction positions:**
        - For donor sites: SpliceTransformer predicts on the nucleotide
          immediately **before** the "GT" dinucleotide
        - For acceptor sites: SpliceTransformer predicts on the nucleotide
          immediately **after** the "AG" dinucleotide
    
    Examples:
        Evaluating a single intron with one donor and one acceptor:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> # 1000 bp target sequence with GT at position 100, AG at position 900
        >>> target_seq = Sequence("ATCG" * 250, "dna")  # 1000 bp
        >>> # 4000 bp flanking contexts
        >>> left_ctx = "ATCG" * 1000  # 4000 bp
        >>> right_ctx = "GCTA" * 1000  # 4000 bp
        >>> 
        >>> config = SpliceTransformerIntronBoundaryConfig(
        ...     left_context=left_ctx,
        ...     right_context=right_ctx,
        ...     donor_pos=99,     # Position before GT
        ...     acceptor_pos=901  # Position after AG
        ... )
        >>> score = splice_transformer_intron_boundary(target_seq, config)
        >>> print(score)  # e.g., 0.15 (good splice sites, 85% probability)
        >>> print(target_seq._metadata["donor_score"])  # e.g., 0.12
        >>> print(target_seq._metadata["acceptor_score"])  # e.g., 0.18
    """
    assert len(config.left_context) == len(config.right_context) == SPLICE_TRANSFORMER_CONTEXT_LENGTH, \
        f"Context lengths must be {SPLICE_TRANSFORMER_CONTEXT_LENGTH}"
    context_length = len(config.left_context)


    splice_transformer_input = SpliceTransformerInput(
        target_seqs=[sequence.sequence],
        left_contexts=[config.left_context],
        right_contexts=[config.right_context],
    )
    splice_transformer_config = config.splice_transformer_config.model_copy(
        update={"context_length": context_length}
    )

    output = run_splice_transformer(
        splice_transformer_input,
        splice_transformer_config,
    ).prediction

    assert output.shape[1] == len(sequence.sequence)

    donor_score = float(output[:, config.donor_pos, SpliceTransformerType.DONOR.value].mean())
    acceptor_score = float(output[:, config.acceptor_pos, SpliceTransformerType.ACCEPTOR.value].mean())
    score = 1. - ((donor_score + acceptor_score) / 2)

    sequence._metadata.update({
        "donor_pos": config.donor_pos,
        "acceptor_pos": config.acceptor_pos,
        "donor_score": 1. - donor_score,
        "acceptor_score": 1. - acceptor_score,
        "total_splice_score": score,
    })

    return score
