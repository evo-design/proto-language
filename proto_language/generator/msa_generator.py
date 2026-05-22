"""MSAGenerator for sampling mutations from multiple sequence alignment distributions."""

import random
from collections.abc import Iterable
from typing import Any, final

from proto_tools import MSA
from pydantic import ConfigDict, field_validator

from proto_language.core import Generator, GeneratorInputType, Segment
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField


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
        title="Input MSA",
        description="Multiple sequence alignment (list of aligned sequences).",
    )
    num_mutations: int = ConfigField(
        default=1,
        ge=1,
        title="Number of Mutations",
        description="Number of positions to mutate per sample",
    )
    include_gaps: bool = ConfigField(
        default=False,
        title="Include Gaps",
        description="Whether to include gaps when calculating position probabilities",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("msa", mode="before")
    @classmethod
    def validate_msa(cls, v: Any) -> Any:
        """Accept MSA object or list of aligned sequences."""
        if isinstance(v, MSA):
            return v
        if isinstance(v, list):
            return MSA(aligned_sequences=v)
        raise ValueError(f"msa must be an MSA object or list of aligned sequences, got {type(v)}")


@generator(
    key="msa",
    label="MSA Generator",
    config=MSAGeneratorConfig,
    description="Sample mutations from MSA position-specific distributions",
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
        >>> from proto_language.generator import MSAGenerator, MSAGeneratorConfig
        >>> from proto_language.core import Segment
        >>> from proto_tools import MSA
        >>> config = MSAGeneratorConfig(
        ...     msa=MSA(aligned_sequences=["MVLS", "AVLS", "MVLS"]),
        ...     num_mutations=1,
        ... )
        >>> gen = MSAGenerator(config)
        >>> segment = Segment(sequence="MVLS", sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Position 0 has 2/3 chance of M, 1/3 chance of A
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE

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
        self.position_probs: list[dict[str, float] | None] = []
        self.mutable_positions: list[int] = []
        self._compute_position_probabilities()

    def _compute_position_probabilities(self) -> None:
        """Compute empirical probability distribution for each position in the MSA."""
        for position in range(self.msa.alignment_length):
            probs = self.msa.get_position_frequencies(position, include_gaps=self.include_gaps)

            if not probs:
                # All gaps at this position - cannot mutate
                self.position_probs.append(None)
            else:
                self.position_probs.append(probs)
                self.mutable_positions.append(position)

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        """Assign segment(s) to this generator.

        Validates that the alignment length matches the segment's sequence length
        and that the MSA has at least one mutable position.

        Args:
            segments (Segment | Iterable[Segment]): Segment or tied segments to assign.

        Raises:
            ValueError: If alignment/segment length mismatch, no mutable positions, or invalid segment.
        """
        super().assign(segments)

        if self.msa.alignment_length != self.segment.sequence_length:
            raise ValueError(
                f"MSA alignment length ({self.msa.alignment_length}) must match segment length ({self.segment.sequence_length})"
            )

        if not self.mutable_positions:
            raise ValueError("No mutable positions in MSA (all positions are gaps)")

    def _sample(self) -> None:
        """Sample mutations for proposal sequences using MSA distributions.

        For each proposal sequence in the pool, randomly selects positions from
        mutable columns and replaces characters according to the empirical
        probability distribution at each position. The number of mutations is
        capped at the number of available mutable positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()
        rng = random.Random(self._next_seed())  # noqa: S311 -- not cryptographic
        for sequence in self.segment.proposal_sequences:
            seq_list = list(sequence.sequence)

            # Cap mutations at available mutable positions
            actual_num_mutations = min(self.num_mutations, len(self.mutable_positions))
            positions_to_mutate = rng.sample(self.mutable_positions, actual_num_mutations)

            for pos in positions_to_mutate:
                # Sample a character according to the empirical probability distribution
                probs = self.position_probs[pos]
                assert probs is not None  # noqa: S101 -- mypy type narrowing; mutable_positions only includes non-None entries
                chars = list(probs.keys())
                weights = list(probs.values())
                seq_list[pos] = rng.choices(chars, weights=weights, k=1)[0]

            sequence.sequence = "".join(seq_list)
