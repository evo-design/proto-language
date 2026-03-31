"""
proto_language/language/generator/random_protein_generator.py

RandomProteinGenerator for CPU-based random protein mutagenesis.
"""

from __future__ import annotations

from typing import Optional, final

from proto_tools import (
    RandomProteinSampleConfig,
    RandomProteinSampleInput,
    run_random_protein_sample,
)
from proto_tools.tools.masked_models.masking import MaskingStrategy
from proto_tools.tools.mutagenesis.random_protein.random_protein_sample import (
    CodonScheme,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import generator


class RandomProteinGeneratorConfig(BaseConfig):
    """Configuration object for RandomProteinGenerator.

    This class defines configuration parameters for the random protein generator,
    which introduces random amino acid mutations at masked positions using
    codon scheme-biased sampling.

    Attributes:
        masking_strategy (MaskingStrategy): Controls which positions to mask and
            how many. Supports exact count (``num_mutations``), fractional
            (``mask_fraction``), or default random 30%.

        codon_scheme (CodonScheme): Codon scheme controlling amino acid sampling
            probabilities. Each scheme defines which amino acids can be
            sampled and their relative weights (based on the number of
            codons encoding each amino acid). Available schemes:

            - ``"UNIFORM"``: Equal weight to all 20 standard amino acids. Default.
            - ``"NNN"``: All 64 codons; all 20 amino acids reachable, weighted
              by natural codon frequency (e.g., Leu has 6 codons, Trp has 1).
            - ``"NNK"``: 32 codons (K = G/T at position 3); covers all 20
              amino acids with reduced stop codon frequency. Common in
              directed evolution libraries.
            - ``"NNS"``: 32 codons (S = G/C at position 3); similar coverage
              to NNK with slightly different codon bias.
            - ``"NDT"``: 12 codons (D = A/G/T, T at position 3); encodes
              12 amino acids (F, L, I, V, Y, H, N, D, C, R, S, G) with
              equal representation. Good for small, balanced libraries.
            - ``"DBK"``: 18 codons; encodes 12 amino acids. Compact library
              design with broad chemical diversity.
            - ``"NRT"``: 8 codons (R = A/G at position 2); encodes 8 amino
              acids. Very compact library for focused mutagenesis.

        seed (int | None): Random seed for reproducibility.
    """

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )

    # Advanced parameters
    codon_scheme: CodonScheme = ConfigField(
        default="UNIFORM",
        title="Codon Scheme",
        description="Codon scheme for amino acid sampling probabilities.",
        advanced=True,
    )
    seed: Optional[int] = ConfigField(
        default=None,
        title="Random Seed",
        description="Random seed for reproducible sampling.",
        advanced=True,
    )


@generator(
    key="random-protein",
    label="Random Protein Mutation",
    config=RandomProteinGeneratorConfig,
    description="Random amino acid mutations using codon scheme-biased sampling",
    uses_gpu=False,
    tools_called=["random-protein-sample"],
    category="mutation",
    supported_sequence_types=["protein"],
)
@final
class RandomProteinGenerator(Generator):
    """Protein sequence generator that introduces random amino acid mutations.

    This generator creates sequence diversity by randomly mutating masked positions
    in protein sequences. Amino acid selection is biased by the configured codon
    scheme, allowing simulation of library diversity achievable through degenerate
    codon synthesis.

    The generator category is ``"mutation"``, indicating it modifies existing
    sequences rather than generating from scratch.

    Attributes:
        masking_strategy (MaskingStrategy): Strategy for selecting positions to mutate.
        codon_scheme (CodonScheme): Codon scheme for amino acid sampling.
        seed (int | None): Random seed for reproducibility.

    Example:
        >>> from proto_language.language.generator import RandomProteinGenerator, RandomProteinGeneratorConfig
        >>> from proto_language.language.core import Segment
        >>> config = RandomProteinGeneratorConfig(
        ...     masking_strategy=MaskingStrategy(num_mutations=2),
        ... )
        >>> gen = RandomProteinGenerator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Introduces 2 random amino acid mutations
    """

    def __init__(self, config: RandomProteinGeneratorConfig) -> None:
        """
        Initialize the random protein generator.

        Args:
            config (RandomProteinGeneratorConfig): Configuration object
                containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.masking_strategy = config.masking_strategy
        self.codon_scheme = config.codon_scheme
        self.seed = config.seed

    def sample(self) -> None:
        """
        Introduce random amino acid mutations at masked positions.

        Applies the masking strategy to select positions, then samples random
        amino acids from the configured codon scheme at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        sequences = [seq.sequence for seq in self._assigned_segment.proposal_sequences]
        tool_input = RandomProteinSampleInput(sequences=sequences)
        tool_config = RandomProteinSampleConfig(
            masking_strategy=self.masking_strategy,
            codon_scheme=self.codon_scheme,
            seed=self.seed,
        )
        result = run_random_protein_sample(inputs=tool_input, config=tool_config)

        for proposal, sequence in zip(
            self._assigned_segment.proposal_sequences, result.sequences, strict=True
        ):
            proposal.sequence = sequence
