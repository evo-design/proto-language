"""ESM2 generator for protein sequence mutation and refinement."""

from typing import Literal, final

from proto_tools import ESM2SampleConfig, ESM2SampleInput, run_esm2_sample
from proto_tools.tools.masked_models.esm2.esm2_sample import (
    ESM2_MODEL_CHECKPOINTS,
)
from proto_tools.transforms.masking import MaskingStrategy

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField


class ESM2GeneratorConfig(BaseConfig):
    """Configuration object for ESM2Generator.

    This class defines configuration parameters for the ESM2 generator, which uses
    a protein language model to refine existing protein sequences through iterative
    mutation of masked positions. In Proto Language, ESM2 is registered as a
    mutation-category generator that edits the supplied starting sequence; the
    segment must carry a sequence (directly or from a prior optimizer stage).

    Attributes:
        model_checkpoint (ESM2_MODEL_CHECKPOINTS): ESM2 model checkpoint to use. Options:

            - ``"esm2_t6_8M_UR50D"``: 8M parameters, 6 layers (fastest)
            - ``"esm2_t12_35M_UR50D"``: 35M parameters, 12 layers
            - ``"esm2_t30_150M_UR50D"``: 150M parameters, 30 layers
            - ``"esm2_t33_650M_UR50D"``: 650M parameters, 33 layers (default, balanced)
            - ``"esm2_t36_3B_UR50D"``: 3B parameters, 36 layers
            - ``"esm2_t48_15B_UR50D"``: 15B parameters, 48 layers (best quality)

            Default: ``"esm2_t33_650M_UR50D"``.

        masking_strategy (MaskingStrategy): Controls which positions to mask and
            how many. Supports exact count (``num_mutations``), fractional
            (``mask_fraction``), or default random 30%. Model-based strategies
            like ``MaskingStrategy(method="entropy")`` and
            ``MaskingStrategy(method="max-logit")`` use model logits to select
            high-uncertainty positions.

        sampling_method (Literal["single_pass", "iterative_refinement"]):
            ``"single_pass"`` fills every mask in one forward pass.
            ``"iterative_refinement"`` runs a MaskGIT-style loop driven by the
            five iterative knobs below — slower (~``num_steps``x compute),
            but commits positions in rounds rather than independently, which
            improves coherence when many positions are masked.
            Default: ``"single_pass"``.

        temperature (float): Scales randomness of amino acid sampling by adjusting
            probability distribution sharpness:

            - ``< 1.0``: More deterministic, favors high-probability amino acids
            - ``1.0``: Standard sampling from model distribution (default)
            - ``> 1.0``: More diverse, explores lower-probability amino acids

            Applied in both ``single_pass`` and ``iterative_refinement`` modes.
            Must be greater than 0. Default: 1.0.

        top_p (float): Nucleus sampling threshold for ``iterative_refinement``;
            ``1.0`` disables. Default: 1.0.

        num_steps (int): Number of refinement steps for
            ``iterative_refinement``. Diminishing returns above 20.
            Default: 20.

        schedule (Literal["cosine", "linear"]): Unmask schedule across rounds
            for ``iterative_refinement``. ``"cosine"`` fronts more commits
            late; ``"linear"`` is uniform. Default: ``"cosine"``.

        strategy (Literal["random", "entropy"]): Per-round commit selection for
            ``iterative_refinement``. ``"entropy"`` commits the most-confident
            positions first. Default: ``"random"``.

        temperature_annealing (bool): For ``iterative_refinement``, anneal
            ``temperature`` toward 0 across rounds. Default: ``True``.

        device (str): GPU device to run ESM2 on, e.g. ``"cuda"`` or
            ``"cuda:0"``. Default: ``"cuda"``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.
    """

    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        default="esm2_t33_650M_UR50D",
        title="Model Checkpoint",
        description="ESM-2 model variant to load (e.g. esm2_t33_650M_UR50D).",
    )

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )

    sampling_method: Literal["single_pass", "iterative_refinement"] = ConfigField(
        default="single_pass",
        title="Sampling Method",
        description=("'single_pass' fills all masks in one forward; 'iterative_refinement' runs a MaskGIT-style loop."),
    )

    # Advanced parameters
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Sharpness of sampling. Below 1 sharpens toward the likely amino acid; above 1 increases diversity.",
    )
    top_p: float = ConfigField(
        default=1.0,
        gt=0.0,
        le=1.0,
        title="Top-p",
        description="Nucleus sampling cumulative probability cutoff used in iterative refinement. 1.0 disables it.",
    )
    num_steps: int = ConfigField(
        default=20,
        ge=1,
        title="Refinement Steps",
        description="Number of iterative-refinement rounds. Returns diminish above 20.",
    )
    schedule: Literal["cosine", "linear"] = ConfigField(
        default="cosine",
        title="Unmask Schedule",
        description="Per-round unmask rate. 'cosine' commits more positions late; 'linear' commits the same each round.",
    )
    strategy: Literal["random", "entropy"] = ConfigField(
        default="random",
        title="Unmask Strategy",
        description="How positions are picked each round. 'entropy' takes most-confident first; 'random' is uniform.",
    )
    temperature_annealing: bool = ConfigField(
        default=True,
        title="Temperature Annealing",
        description="Anneal temperature toward 0 across rounds",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="GPU device to run ESM2 on (e.g. 'cuda' or 'cuda:0').",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences to process simultaneously on GPU",
    )


@generator(
    key="esm2",
    label="ESM2 Protein Language Model",
    config=ESM2GeneratorConfig,
    description="ESM-2 masked protein language model for local sequence mutation/refinement",
    uses_gpu=True,
    tools_called=["esm2-sample"],
    supported_sequence_types=["protein"],
)
@final
class ESM2Generator(Generator):
    """Protein sequence mutation/refinement generator using ESM2 language model.

    This generator uses the ESM2 protein language model to refine existing
    protein sequences through iterative mutation. It masks positions according
    to the configured masking strategy and samples biologically plausible amino
    acids at those positions.

    The generator category is ``"mutation"``, indicating it refines proposal
    sequences through targeted mutations.

    Attributes:
        model_checkpoint (str): ESM2 model checkpoint name.
        masking_strategy (MaskingStrategy): Strategy for selecting positions to mutate.
        sampling_method (str): ``"single_pass"`` (default) or
            ``"iterative_refinement"`` (MaskGIT-style loop).
        temperature (float): Sampling temperature for diversity control.
        top_p (float): Nucleus threshold; iterative-refinement only.
        num_steps (int): Refinement steps; iterative-refinement only.
        schedule (str): Unmask schedule; iterative-refinement only.
        strategy (str): Per-round commit selection; iterative-refinement only.
        temperature_annealing (bool): Anneal toward 0 across rounds; iterative only.
        batch_size (int): Number of sequences to process simultaneously on GPU.

    Example:
        >>> from proto_language.generator import ESM2Generator, ESM2GeneratorConfig
        >>> from proto_language.core import Segment
        >>> from proto_tools.transforms.masking import MaskingStrategy
        >>> config = ESM2GeneratorConfig(
        ...     temperature=1.0,
        ...     masking_strategy=MaskingStrategy(num_mutations=5),
        ... )
        >>> gen = ESM2Generator(config)
        >>> segment = Segment(sequence="M" * 100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Re-samples 5 randomly masked positions
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self, config: ESM2GeneratorConfig) -> None:
        """Initialize the ESM-2 generator with model and sampling configuration.

        Args:
            config (ESM2GeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.model_checkpoint = config.model_checkpoint
        self.masking_strategy = config.masking_strategy
        self.sampling_method = config.sampling_method
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.num_steps = config.num_steps
        self.schedule = config.schedule
        self.strategy = config.strategy
        self.temperature_annealing = config.temperature_annealing
        self.device = config.device
        self.batch_size = config.batch_size

    def _sample(self) -> None:
        """Sample new amino acids at masked positions for all sequences in the batch.

        For each sequence in the batch, applies the masking strategy to select
        positions, then uses ESM2 to sample new amino acids at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        sequences = [seq.sequence for seq in self.segment.proposal_sequences]
        esm2_input = ESM2SampleInput(sequences=sequences)
        config = ESM2SampleConfig(
            model_checkpoint=self.model_checkpoint,
            temperature=self.temperature,
            masking_strategy=self.masking_strategy,
            sampling_method=self.sampling_method,
            top_p=self.top_p,
            num_steps=self.num_steps,
            schedule=self.schedule,
            strategy=self.strategy,
            temperature_annealing=self.temperature_annealing,
            device=self.device,
            batch_size=self.batch_size,
            verbose=False,
            seed=self._next_seed(),
        )
        result = run_esm2_sample(inputs=esm2_input, config=config)
        mutated_sequences = result.sequences

        for proposal, sequence in zip(self.segment.proposal_sequences, mutated_sequences, strict=True):
            proposal.sequence = sequence
