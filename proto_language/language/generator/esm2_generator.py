"""
proto_language/language/generator/esm2_generator.py

ESM2 Generator for protein sequence generation
"""

from __future__ import annotations

from typing import final

from proto_tools import ESM2SampleConfig, ESM2SampleInput, run_esm2_sample
from proto_tools.tools.masked_models.esm2.esm2_sample import (
    ESM2_MODEL_CHECKPOINTS,
)
from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import generator


class ESM2GeneratorConfig(BaseConfig):
    """Configuration object for ESM2Generator.

    This class defines configuration parameters for the ESM2 generator, which uses
    a protein language model to generate and refine protein sequences through
    iterative mutation of masked positions.

    Attributes:
        model_checkpoint (ESM2_MODEL_CHECKPOINTS): ESM2 model checkpoint to use. Options:

            - ``"esm2_t6_8M_UR50D"``: 8M parameters, 6 layers (fastest)
            - ``"esm2_t12_35M_UR50D"``: 35M parameters, 12 layers
            - ``"esm2_t30_150M_UR50D"``: 150M parameters, 30 layers
            - ``"esm2_t33_650M_UR50D"``: 650M parameters, 33 layers (default, balanced)
            - ``"esm2_t36_3B_UR50D"``: 3B parameters, 36 layers
            - ``"esm2_t48_15B_UR50D"``: 15B parameters, 48 layers (best quality)

            Default: ``"esm2_t33_650M_UR50D"``.

        temperature (float): Scales randomness of amino acid sampling by adjusting
            probability distribution sharpness:

            - ``< 1.0``: More deterministic, favors high-probability amino acids
            - ``1.0``: Standard sampling from model distribution (default)
            - ``> 1.0``: More diverse, explores lower-probability amino acids

            Must be greater than 0. Default: 1.0.

        masking_strategy (MaskingStrategy): Controls which positions to mask and
            how many. Supports exact count (``num_mutations``), fractional
            (``mask_fraction``), or default random 30%. Model-based strategies
            like ``MaskingStrategy(method="entropy")`` and
            ``MaskingStrategy(method="max-logit")`` use model logits to select
            high-uncertainty positions.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.
    """

    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        default="esm2_t33_650M_UR50D",
        title="Model Checkpoint",
        description="ESM2 model checkpoint to use",
    )

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )

    # Advanced parameters
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling by adjusting probability distribution sharpness.",
        advanced=True,
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )


@generator(
    key="esm2",
    label="ESM2 Protein Language Model",
    config=ESM2GeneratorConfig,
    description="ESM-2 protein language model for protein sequence generation",
    uses_gpu=True,
    tools_called=["esm2-sample"],
    category="mutation",
    supported_sequence_types=["protein"],
)
@final
class ESM2Generator(Generator):
    """Protein sequence generator using ESM2 language model.

    This generator uses the ESM2 protein language model to generate and refine
    protein sequences through iterative mutation. It masks positions according
    to the configured masking strategy and samples biologically plausible amino
    acids at those positions.

    The generator category is ``"mutation"``, indicating it refines sequences
    through targeted mutations rather than generating from scratch.

    Attributes:
        model_checkpoint (str): ESM2 model checkpoint name.
        temperature (float): Sampling temperature for diversity control.
        masking_strategy (MaskingStrategy): Strategy for selecting positions to mutate.
        batch_size (int): Number of sequences to process simultaneously on GPU.

    Example:
        >>> from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig
        >>> from proto_language.language.core import Segment, SequenceType
        >>> config = ESM2GeneratorConfig(
        ...     temperature=1.0,
        ...     masking_strategy=MaskingStrategy(num_mutations=5),
        ... )
        >>> gen = ESM2Generator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Refines 5 highest-uncertainty positions
    """

    def __init__(self, config: ESM2GeneratorConfig) -> None:
        """
        Initialize the ESM-2 generator with model and sampling configuration.

        Args:
            config (ESM2GeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.model_checkpoint = config.model_checkpoint
        self.temperature = config.temperature
        self.masking_strategy = config.masking_strategy
        self.batch_size = config.batch_size

    def sample(self) -> None:
        """
        Sample new amino acids at masked positions for all sequences in the batch.

        For each sequence in the batch, applies the masking strategy to select
        positions, then uses ESM2 to sample new amino acids at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        sequences = [seq.sequence for seq in self._assigned_segment.proposal_sequences]
        esm2_input = ESM2SampleInput(sequences=sequences)
        config = ESM2SampleConfig(
            model_checkpoint=self.model_checkpoint,
            temperature=self.temperature,
            masking_strategy=self.masking_strategy,
            batch_size=self.batch_size,
            verbose=False,
        )
        result = run_esm2_sample(inputs=esm2_input, config=config)
        mutated_sequences = result.sequences

        for proposal, sequence in zip(
            self._assigned_segment.proposal_sequences, mutated_sequences, strict=True
        ):
            proposal.sequence = sequence
