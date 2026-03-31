"""
proto_language/language/generator/msa_generator.py

MSAGenerator for sampling mutations from multiple sequence alignment distributions.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, final

from proto_tools import MSA
from pydantic import ConfigDict, field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator, Segment
from proto_language.language.generator.generator_registry import generator


class MSAGeneratorConfig(BaseConfig):
    """Configuration object for MSAGenerator.

    This class defines configuration parameters for the MSA generator, which samples
    mutations from position-specific probability distributions derived from a multiple
    sequence alignment.

    Attributes:
        msa (MSA): Multiple sequence alignment containing homologous sequences.
            Accepts either an MSA object or a list of aligned sequences (same length).
            The alignment defines position-specific amino acid/nucleotide distributions.

        num_mutations (int): Number of positions to randomly mutate per sample.
            Positions are selected from non-gap columns and mutated according to
            the empirical distribution at that position. Automatically capped at
            the number of mutable positions. Must be at least 1. Default: 1.

        include_gaps (bool): Whether to include gap characters ('-') when computing
            position probability distributions:

            - ``False``: Gaps are excluded; probabilities are computed only from
              non-gap characters (default)
            - ``True``: Gaps are included in distributions; sampled mutations may
              introduce gaps

            Default: ``False``.
    """

    msa: MSA = ConfigField(
        title="MSA",
        description="Multiple sequence alignment (list of aligned sequences).",
    )
    num_mutations: int = ConfigField(
        default=1,
        ge=1,
        title="Number of Mutations",
        description="Number of positions to mutate per sample",
        advanced=True,
    )
    include_gaps: bool = ConfigField(
        default=False,
        title="Include Gaps",
        description="Whether to include gaps when calculating position probabilities",
        advanced=True,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("msa", mode="before")
    @classmethod
    def validate_msa(cls, v):
        """Accept MSA object or list of aligned sequences."""
        if isinstance(v, MSA):
            return v
        if isinstance(v, list):
            return MSA(v)
        raise ValueError(
            f"msa must be an MSA object or list of aligned sequences, got {type(v)}"
        )


@generator(
    key="msa",
    label="MSA Generator",
    config=MSAGeneratorConfig,
    description="Sample mutations from MSA position-specific distributions",
    category="mutation",
    uses_gpu=False,
)
@final
class MSAGenerator(Generator):
    """Generator that samples mutations from MSA position-specific distributions.

    This generator computes empirical probability distributions for each position
    in a multiple sequence alignment, then mutates proposal sequences by sampling
    from these distributions.

    Attributes:
        msa: The input multiple sequence alignment.
        num_mutations: Number of positions to mutate per sample.
        include_gaps: Whether gaps are included in probability calculations.
        position_probs: Position-specific
            probability distributions. None for positions with no valid characters.
        mutable_positions: Indices of positions that can be mutated.
        batch_size (int): Number of sequences to generate per batch.

    Example:
        >>> from proto_language.language.generator import MSAGenerator, MSAGeneratorConfig
        >>> from proto_language.language.core import Segment
        >>> from proto_tools import MSA
        >>> config = MSAGeneratorConfig(
        ...     msa=MSA(["MVLS", "AVLS", "MVLS"]),
        ...     num_mutations=1,
        ... )
        >>> gen = MSAGenerator(config)
        >>> segment = Segment(sequence="MVLS", sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Position 0 has 2/3 chance of M, 1/3 chance of A
    """

    def __init__(self, config: MSAGeneratorConfig) -> None:
        """Initialize the MSA generator.

        Args:
            config (MSAGeneratorConfig): Configuration containing MSA and parameters.
        """
        super().__init__()
        self.config = config
        self.msa = config.msa
        self.num_mutations = config.num_mutations
        self.include_gaps = config.include_gaps

        # Compute position-specific probability distributions
        self.position_probs: List[Optional[Dict[str, float]]] = []
        self.mutable_positions: List[int] = []
        self._compute_position_probabilities()

    def _compute_position_probabilities(self) -> None:
        """Compute empirical probability distribution for each position in the MSA."""
        for position in range(self.msa.alignment_length):
            probs = self.msa.get_position_frequencies(
                position, include_gaps=self.include_gaps
            )

            if not probs:
                # All gaps at this position - cannot mutate
                self.position_probs.append(None)
            else:
                self.position_probs.append(probs)
                self.mutable_positions.append(position)

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a segment to this generator.

        Validates that the alignment length matches the segment's sequence length
        and that the MSA has at least one mutable position.

        Args:
            assigned_segment (Segment): The segment to assign.

        Raises:
            ValueError: If alignment/segment length mismatch, no mutable positions, or invalid segment.
        """
        super().assign(assigned_segment)

        if self.msa.alignment_length != assigned_segment.sequence_length:
            raise ValueError(
                f"MSA alignment length ({self.msa.alignment_length}) must match segment length ({assigned_segment.sequence_length})"
            )

        if not self.mutable_positions:
            raise ValueError("No mutable positions in MSA (all positions are gaps)")

    def sample(self) -> None:
        """Sample mutations for proposal sequences using MSA distributions.

        For each proposal sequence in the pool, randomly selects positions from
        mutable columns and replaces characters according to the empirical
        probability distribution at each position. The number of mutations is
        capped at the number of available mutable positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()
        for sequence in self._assigned_segment.proposal_sequences:
            seq_list = list(sequence.sequence)

            # Cap mutations at available mutable positions
            actual_num_mutations = min(self.num_mutations, len(self.mutable_positions))
            positions_to_mutate = random.sample(self.mutable_positions, actual_num_mutations)

            for pos in positions_to_mutate:
                # Sample a character according to the empirical probability distribution
                probs = self.position_probs[pos]
                chars = list(probs.keys())
                weights = list(probs.values())
                seq_list[pos] = random.choices(chars, weights=weights, k=1)[0]

            sequence.sequence = "".join(seq_list)
