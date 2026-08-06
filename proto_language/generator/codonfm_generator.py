"""CodonFM (Encodon) generator for coding-sequence mutation via masked-codon resampling."""

from typing import final

from proto_tools import (
    CodonFMCheckpoint,
    CodonFMSampleConfig,
    CodonFMSampleInput,
    run_codonfm_sample,
)
from proto_tools.transforms.masking import RandomMaskingStrategy

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField


class CodonFMGeneratorConfig(BaseConfig):
    """Configuration object for CodonFMGenerator.

    CodonFM (Encodon) is a codon-level masked language model. As a Proto Language generator it
    refines an existing coding sequence by masking a subset of codons and resampling them from
    the model — a mutation-category generator, so the segment must already carry a sequence
    (directly or from a prior optimizer stage).

    Attributes:
        model_checkpoint (CodonFMCheckpoint): Encodon checkpoint to sample from.
        masking_strategy (RandomMaskingStrategy): Which codons to resample, counted in codons.
            ``num_mutations`` sets an exact count, ``mask_fraction`` a fraction, and
            ``fixed_positions`` pins codons that must survive (e.g. the start or stop codon).
        temperature (float): Softmax temperature for codon sampling; higher is more diverse.
        device (str): GPU device to run CodonFM on, e.g. ``"cuda"`` or ``"cuda:0"``.
        batch_size (int): Number of same-length sequences to process per GPU batch.
    """

    model_checkpoint: CodonFMCheckpoint = ConfigField(
        default="encodon_80m",
        title="Model Checkpoint",
        description="Encodon checkpoint: encodon_80m | encodon_600m | encodon_1b | encodon_1b_cdwt.",
    )
    masking_strategy: RandomMaskingStrategy = ConfigField(
        default_factory=RandomMaskingStrategy,
        title="Masking Strategy",
        description="Which codons to resample (counted in codons); fixed_positions pins codons that must survive.",
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Sampling temperature. Below 1 sharpens toward the likely codon; above 1 adds diversity.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="GPU device to run CodonFM on (e.g. 'cuda' or 'cuda:0').",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of same-length sequences to process simultaneously on GPU.",
    )


@generator(
    key="codonfm",
    label="CodonFM Codon Language Model",
    config=CodonFMGeneratorConfig,
    description="CodonFM/Encodon masked codon language model for local coding-sequence mutation/refinement",
    uses_gpu=True,
    tools_called=["codonfm-sample"],
    supported_sequence_types=["dna", "rna"],
)
@final
class CodonFMGenerator(Generator):
    """Coding-sequence mutation generator using the CodonFM (Encodon) codon language model.

    Refines existing coding sequences by masking a subset of codon positions and resampling
    them from Encodon's per-codon distribution. Sequence length is preserved. The generator
    category is ``"mutation"``.

    Attributes:
        model_checkpoint (str): Encodon checkpoint name.
        masking_strategy (RandomMaskingStrategy): Which codons to resample; ``num_mutations`` /
            ``mask_fraction`` set how many and ``fixed_positions`` pins codons that must survive.
        temperature (float): Sampling temperature for diversity control.
        device (str): GPU device.
        batch_size (int): Number of sequences to process simultaneously on GPU.

    Example:
        >>> from proto_language.generator import CodonFMGenerator, CodonFMGeneratorConfig
        >>> from proto_language.core import Segment
        >>> from proto_tools.transforms.masking import RandomMaskingStrategy
        >>> gen = CodonFMGenerator(CodonFMGeneratorConfig(masking_strategy=RandomMaskingStrategy(num_mutations=3)))
        >>> segment = Segment(sequence="ATGGTGAGCAAGGGC", sequence_type="dna")  # 5 codons
        >>> gen.assign(segment)
        >>> gen.sample()  # resamples 3 randomly masked codons
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self, config: CodonFMGeneratorConfig) -> None:
        """Initialize the CodonFM generator with model and sampling configuration.

        Args:
            config (CodonFMGeneratorConfig): Configuration object with all generator parameters.
        """
        super().__init__()
        self.config = config
        self.model_checkpoint = config.model_checkpoint
        self.masking_strategy = config.masking_strategy
        self.temperature = config.temperature
        self.device = config.device
        self.batch_size = config.batch_size

    def _sample(self) -> None:
        """Resample masked codons for every proposal sequence in the assigned segment.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        sequences = [seq.sequence for seq in self.segment.proposal_sequences]
        result = run_codonfm_sample(
            CodonFMSampleInput(sequences=sequences),
            CodonFMSampleConfig(
                model_checkpoint=self.model_checkpoint,
                masking_strategy=self.masking_strategy,
                temperature=self.temperature,
                device=self.device,
                batch_size=self.batch_size,
                verbose=False,
                seed=self._next_seed(),
            ),
        )

        for proposal, sequence in zip(self.segment.proposal_sequences, result.sequences, strict=True):
            proposal.sequence = sequence
