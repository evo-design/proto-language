"""
proto_language/language/generator/random_nucleotide_generator.py

RandomNucleotideGenerator for CPU-based random nucleotide mutagenesis.
"""

from __future__ import annotations

from typing import Optional, final

from proto_tools import (
    RandomNucleotideSampleConfig,
    RandomNucleotideSampleInput,
    run_random_nucleotide_sample,
)
from proto_tools.tools.masked_models.masking import MaskingStrategy
from proto_tools.tools.mutagenesis.random_nucleotide.random_nucleotide_sample import (
    SubstitutionScheme,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import generator


class RandomNucleotideGeneratorConfig(BaseConfig):
    """Configuration object for RandomNucleotideGenerator.

    This class defines configuration parameters for the random nucleotide generator,
    which introduces random nucleotide mutations at masked positions using
    IUPAC ambiguity code-based sampling.

    Attributes:
        masking_strategy (MaskingStrategy): Controls which positions to mask and
            how many. Supports exact count (``num_mutations``), fractional
            (``mask_fraction``), or default random 30%.

        substitution_scheme (SubstitutionScheme): IUPAC ambiguity code defining
            the nucleotide pool for substitutions. Available schemes:

            - ``"N"``: Any base (A, C, G, T). Default.
            - ``"R"``: Purines only (A, G)
            - ``"Y"``: Pyrimidines only (C, T)
            - ``"S"``: Strong bases (G, C)
            - ``"W"``: Weak bases (A, T)
            - ``"K"``: Keto bases (G, T)
            - ``"M"``: Amino bases (A, C)
            - ``"B"``: Not A (C, G, T)
            - ``"D"``: Not C (A, G, T)
            - ``"H"``: Not G (A, C, T)
            - ``"V"``: Not T (A, C, G)

        seed (int | None): Random seed for reproducibility.
    """

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )

    # Advanced parameters
    substitution_scheme: SubstitutionScheme = ConfigField(
        default="N",
        title="Substitution Scheme",
        description="IUPAC code defining the nucleotide substitution pool.",
        advanced=True,
    )
    seed: Optional[int] = ConfigField(
        default=None,
        title="Random Seed",
        description="Random seed for reproducible sampling.",
        advanced=True,
    )


@generator(
    key="random-nucleotide",
    label="Random Nucleotide Mutation",
    config=RandomNucleotideGeneratorConfig,
    description="Random nucleotide mutations using IUPAC substitution schemes",
    uses_gpu=False,
    tools_called=["random-nucleotide-sample"],
    category="mutation",
    supported_sequence_types=["dna", "rna"],
)
@final
class RandomNucleotideGenerator(Generator):
    """Nucleotide sequence generator that introduces random mutations.

    This generator creates sequence diversity by randomly mutating masked positions
    in DNA or RNA sequences. Nucleotide selection is controlled by the IUPAC
    substitution scheme, allowing targeted mutation strategies (e.g., transitions
    only with ``"R"``/``"Y"``, or any base with ``"N"``).

    The generator category is ``"mutation"``, indicating it modifies existing
    sequences rather than generating from scratch.

    Attributes:
        masking_strategy (MaskingStrategy): Strategy for selecting positions to mutate.
        substitution_scheme (SubstitutionScheme): IUPAC code for nucleotide sampling.
        seed (int | None): Random seed for reproducibility.

    Example:
        >>> from proto_language.language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
        >>> from proto_language.language.core import Segment
        >>> config = RandomNucleotideGeneratorConfig(
        ...     masking_strategy=MaskingStrategy(num_mutations=2),
        ... )
        >>> gen = RandomNucleotideGenerator(config)
        >>> segment = Segment(length=100, sequence_type="dna")
        >>> gen.assign(segment)
        >>> gen.sample()  # Introduces 2 random nucleotide mutations
    """

    def __init__(self, config: RandomNucleotideGeneratorConfig) -> None:
        """
        Initialize the random nucleotide generator.

        Args:
            config (RandomNucleotideGeneratorConfig): Configuration object
                containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.masking_strategy = config.masking_strategy
        self.substitution_scheme = config.substitution_scheme
        self.seed = config.seed

    def sample(self) -> None:
        """
        Introduce random nucleotide mutations at masked positions.

        Applies the masking strategy to select positions, then samples random
        nucleotides from the configured IUPAC substitution scheme at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        sequences = [seq.sequence for seq in self._assigned_segment.proposal_sequences]
        tool_input = RandomNucleotideSampleInput(sequences=sequences)
        tool_config = RandomNucleotideSampleConfig(
            masking_strategy=self.masking_strategy,
            substitution_scheme=self.substitution_scheme,
            sequence_type=self._assigned_segment.sequence_type,
            seed=self.seed,
        )
        result = run_random_nucleotide_sample(inputs=tool_input, config=tool_config)

        for proposal, sequence in zip(
            self._assigned_segment.proposal_sequences, result.sequences, strict=True
        ):
            proposal.sequence = sequence
