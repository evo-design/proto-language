"""
proto_language/language/generator/esm3_generator.py

ESM3 Generator for protein sequence generation
"""

from __future__ import annotations

from typing import final

from proto_tools import ESM3SampleConfig, ESM3SampleInput, run_esm3_sample
from proto_tools.tools.masked_models.esm3.esm3_sample import (
    ESM3_MODEL_CHECKPOINTS,
)
from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import generator


class ESM3GeneratorConfig(BaseConfig):
    """Configuration object for ESM3Generator.

    This class defines configuration parameters for the ESM3 generator, which uses
    the open-source ESM3 protein language model to generate and refine protein
    sequences through iterative mutation of masked positions.

    Attributes:
        model_checkpoint (ESM3_MODEL_CHECKPOINTS): ESM3 model checkpoint to use. Currently available:

            - ``"esm3_sm_open_v1"``: Small open-source ESM3 model (default)

            Future versions may include additional model variants.
            Default: ``"esm3_sm_open_v1"``.

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

    Note:
        ESM3 is the open-source version of EvolutionaryScale's protein language model.
    """

    model_checkpoint: ESM3_MODEL_CHECKPOINTS = ConfigField(
        default="esm3_sm_open_v1",
        title="Model Checkpoint",
        description="ESM3 model checkpoint to use",
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
    key="esm3",
    label="ESM3 Protein Language Model",
    config=ESM3GeneratorConfig,
    description="ESM-3 open protein language model for protein sequence generation",
    uses_gpu=True,
    tools_called=["esm3-sample"],
    category="mutation",
    supported_sequence_types=["protein"],
)
@final
class ESM3Generator(Generator):
    """Protein sequence generator using ESM3 open language model.

    This generator uses the open-source ESM3 protein language model to generate
    and refine protein sequences through iterative mutation. It masks positions
    according to the configured masking strategy and samples biologically
    plausible amino acids at those positions.

    The generator category is ``"mutation"``, indicating it refines sequences
    through targeted mutations rather than generating from scratch.

    Attributes:
        model_checkpoint (str): ESM3 model checkpoint name.
        temperature (float): Sampling temperature for diversity control.
        masking_strategy (MaskingStrategy): Strategy for selecting positions to mutate.
        batch_size (int): Number of sequences to process simultaneously on GPU.

    Example:
        >>> from proto_language.language.generator import ESM3Generator, ESM3GeneratorConfig
        >>> from proto_language.language.core import Segment, SequenceType
        >>> config = ESM3GeneratorConfig(
        ...     temperature=1.0,
        ...     masking_strategy=MaskingStrategy(num_mutations=5),
        ... )
        >>> gen = ESM3Generator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Refines 5 highest-uncertainty positions
    """

    def __init__(self, config: ESM3GeneratorConfig) -> None:
        """
        Initialize the ESM3 generator with model and sampling configuration.

        Args:
            config (ESM3GeneratorConfig): Configuration object containing all generator parameters.
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
        positions, then uses ESM3 to sample new amino acids at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        sequences = [seq.sequence for seq in self._assigned_segment.proposal_sequences]
        esm3_input = ESM3SampleInput(sequences=sequences)
        config = ESM3SampleConfig(
            model_checkpoint=self.model_checkpoint,
            temperature=self.temperature,
            masking_strategy=self.masking_strategy,
            batch_size=self.batch_size,
            verbose=False,
        )

        result = run_esm3_sample(inputs=esm3_input, config=config)
        mutated_sequences = result.sequences

        for proposal, sequence in zip(
            self._assigned_segment.proposal_sequences, mutated_sequences, strict=True
        ):
            proposal.sequence = sequence
