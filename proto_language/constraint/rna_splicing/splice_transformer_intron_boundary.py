"""Evaluate intron boundary prediction with SpliceTransformer.

Accepts three segments (left_flank, intron_core, right_flank), concatenates
them, windows the result to the fixed SpliceTransformer target length centred
on the donor/acceptor sites, and scores donor/acceptor splice sites.
"""

import logging
from typing import Literal

import numpy as np
from proto_tools import CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH
from proto_tools import (
    SpliceTransformerConfig,
    SpliceTransformerInput,
    SpliceTransformerType,
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

        donor_pos (list[int]): Zero-indexed position(s) of expected donor
            splice site(s) within the concatenated target sequence (left_flank +
            intron_core + right_flank). The donor site marks the 5'
            end of an intron (exon-intron boundary), typically at a "GT" dinucleotide.
            SpliceTransformer scores the donor probability at the "GT" position.
            Can be a single integer (automatically converted to list) or list of
            integers for multiple donors.

        acceptor_pos (list[int]): Zero-indexed position(s) of expected
            acceptor splice site(s) within the concatenated target sequence
            (left_flank + intron_core + right_flank). The acceptor site
            marks the 3' end of an intron (intron-exon boundary), typically at an
            "AG" dinucleotide. SpliceTransformer scores the acceptor probability at
            the "AG" position. Can be a single integer (automatically converted to
            list) or list of integers for multiple acceptors.

        reduction (Literal['mean', 'min']): Combine donor/acceptor probabilities by
            'mean' (average) or 'min' (the weaker site must be strong).
        peak_search_radius (int): Max-probability search window (+/- positions) around
            each donor/acceptor; 0 scores exactly at the requested index.

        splice_transformer_config (SpliceTransformerConfig): Advanced SpliceTransformer
            configuration including context length, device settings, and model
            parameters. Default: SpliceTransformerConfig().

    Note:
        SpliceTransformer requires sequences of specific lengths:
        - Concatenated target (left_flank + intron_core + right_flank): scored
          over a 1000 bp window. If the concatenation is longer, it is windowed
          to 1000 bp centred on the donor/acceptor positions and the positions
          are remapped into that window.
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
    donor_pos: list[int] = ConfigField(
        title="Donor Position(s)",
        description="0-indexed position(s) into the concatenated target sequence of expected donor",
    )
    acceptor_pos: list[int] = ConfigField(
        title="Acceptor Position(s)",
        description="0-indexed position(s) into the concatenated target sequence of expected acceptor",
    )

    @field_validator("donor_pos", "acceptor_pos", mode="before")
    @classmethod
    def convert_pos_to_list(cls, v: int | list[int]) -> list[int]:
        """Convert single int to list of ints."""
        if isinstance(v, int):
            return [v]
        return v

    reduction: Literal["mean", "min"] = ConfigField(
        title="Donor/Acceptor Reduction",
        default="mean",
        description="Combine donor/acceptor probs: 'mean' (average) or 'min' (weaker site must be strong).",
    )
    peak_search_radius: int = ConfigField(
        title="Peak Search Radius",
        default=0,
        ge=0,
        description="Max donor/acceptor probability within +/- this many positions of the index; 0 = exact index.",
    )

    # Optional parameter
    splice_transformer_config: SpliceTransformerConfig = ConfigField(
        title="SpliceTransformer Config",
        default_factory=SpliceTransformerConfig,
        description="Advanced parameter configuration for SpliceTransformer.",
    )


@constraint(
    key="splice-transformer-intron-boundary",
    label="SpliceTransformer intron boundary score",
    config=SpliceTransformerIntronBoundaryConfig,
    description="Score intron-boundary prediction with SpliceTransformer on three segments concatenated into one 1-kb target.",
    uses_gpu=True,
    tools_called=["splice-transformer-prediction"],
    category="rna_splicing",
    supported_sequence_types=["dna"],
    input_labels=["Left Flank", "Intron Core", "Right Flank"],
)
def splice_transformer_intron_boundary(
    input_sequences: list[tuple[Sequence, ...]],
    config: SpliceTransformerIntronBoundaryConfig,
) -> list[ConstraintOutput]:
    """Score donor/acceptor splice sites for three-segment intron boundaries.

    Accepts three segments (left_flank, intron_core, right_flank), concatenates
    them into a single 1-kb target sequence, and scores donor/acceptor splice
    sites.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Mapping of segment IDs to
            their current sequences.
        config (SpliceTransformerIntronBoundaryConfig): Constraint configuration
            controlling evaluation parameters.

    Returns:
        list[ConstraintOutput]: One result per input. ``score`` is the combined
            boundary penalty in ``[0.0, 1.0]``. ``metadata`` carries ``donor_pos``,
            ``acceptor_pos``, ``donor_score``, ``acceptor_score``, and
            ``total_splice_score``.
    """
    if not input_sequences:
        return []

    if not (len(config.left_context) == len(config.right_context) == SPLICE_TRANSFORMER_CONTEXT_LENGTH):
        raise ValueError(
            f"splice-transformer: left/right context must each be {SPLICE_TRANSFORMER_CONTEXT_LENGTH} bp; "
            f"got left={len(config.left_context)}, right={len(config.right_context)}"
        )

    # Concatenate 3-part tuples into target sequences for batched inference.
    target_seqs = []
    for left_flank, intron_core, right_flank in input_sequences:
        target_seqs.append(left_flank.sequence + intron_core.sequence + right_flank.sequence)

    target_lengths = {len(t) for t in target_seqs}
    if len(target_lengths) != 1:
        raise ValueError("SpliceTransformer intron-boundary scoring requires equal-length target sequences in a batch.")

    # Window the concatenated target down to the fixed SpliceTransformer target
    # length, centred on the donor/acceptor sites, and remap the positions.
    window_start = splice_target_window_start(target_lengths.pop(), config.donor_pos, config.acceptor_pos)
    target_seqs = apply_target_window(target_seqs, window_start)
    donor_pos = remap_positions(config.donor_pos, window_start)
    acceptor_pos = remap_positions(config.acceptor_pos, window_start)

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

        radius = config.peak_search_radius
        seq_len = output.shape[1]

        def _peak_prob(
            positions: list[int], channel: int, radius: int = radius, seq_len: int = seq_len, batch_idx: int = batch_idx
        ) -> float:
            # Score each requested site as the max probability within +/- radius
            # (robust to the SpliceAI donor off-by-one), then average across sites.
            # radius/seq_len/batch_idx are bound as defaults so the closure captures
            # this iteration's values (avoids late-binding of the loop variables).
            site_probs = []
            for pos in positions:
                # Support Python-style negative indexing (e.g. -1 = last position).
                abs_pos = pos if pos >= 0 else seq_len + pos
                lo = max(0, abs_pos - radius)
                hi = max(lo + 1, min(seq_len, abs_pos + radius + 1))  # always a non-empty window
                site_probs.append(float(output[batch_idx, lo:hi, channel].max()))
            return sum(site_probs) / len(site_probs)

        donor_prob = _peak_prob(donor_pos, SpliceTransformerType.DONOR.value)
        acceptor_prob = _peak_prob(acceptor_pos, SpliceTransformerType.ACCEPTOR.value)
        if config.reduction == "min":
            # Reward the weaker of the two sites so a strong acceptor cannot mask a dead donor.
            combined_prob = min(donor_prob, acceptor_prob)
        else:
            combined_prob = (donor_prob + acceptor_prob) / 2.0
        score = 1.0 - combined_prob

        metadata = {
            "donor_pos": config.donor_pos,
            "acceptor_pos": config.acceptor_pos,
            "windowed_donor_pos": donor_pos,
            "windowed_acceptor_pos": acceptor_pos,
            "target_window_start": window_start,
            "reduction": config.reduction,
            "donor_score": 1.0 - donor_prob,
            "acceptor_score": 1.0 - acceptor_prob,
            "total_splice_score": score,
        }
        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results
