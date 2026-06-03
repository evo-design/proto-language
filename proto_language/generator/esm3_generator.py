"""ESM3 generator for protein sequence mutation and refinement."""

from typing import Literal, final

from proto_tools import ESM3SampleConfig, ESM3SampleInput, run_esm3_sample
from proto_tools.tools.masked_models.esm3.esm3_sample import (
    ESM3_MODEL_CHECKPOINTS,
)
from proto_tools.transforms.masking import MaskingStrategy

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField


class ESM3GeneratorConfig(BaseConfig):
    """Configuration object for ESM3Generator.

    This class defines configuration parameters for the ESM3 generator, which uses
    the open-source ESM3 protein language model to refine existing protein
    sequences through iterative mutation of masked positions. In Proto Language,
    ESM3 is registered as a mutation-category generator that edits the supplied
    starting sequence; the segment must carry a sequence (directly or from a
    prior optimizer stage).

    Attributes:
        model_checkpoint (ESM3_MODEL_CHECKPOINTS): ESM3 model checkpoint to use. Currently available:

            - ``"esm3_sm_open_v1"``: Small open-source ESM3 model (default)

            Future versions may include additional model variants.
            Default: ``"esm3_sm_open_v1"``.

        masking_strategy (MaskingStrategy): Controls which positions to mask and
            how many. Supports exact count (``num_mutations``), fractional
            (``mask_fraction``), or default random 30%. Model-based strategies
            like ``MaskingStrategy(method="entropy")`` and
            ``MaskingStrategy(method="max-logit")`` use model logits to select
            high-uncertainty positions.

        sampling_method (Literal["single_pass", "iterative_refinement"]):
            ``"single_pass"`` fills every mask in one forward pass.
            ``"iterative_refinement"`` dispatches to ESM3's
            ``model.batch_generate`` and uses the five iterative knobs below —
            slower (~``num_steps``x compute) but produces more coherent
            outputs when many positions are masked. Default: ``"single_pass"``.

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

        device (str): GPU device to run ESM3 on, e.g. ``"cuda"`` or
            ``"cuda:0"``. Default: ``"cuda"``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.

    Note:
        ESM3 is the open-source version of EvolutionaryScale's protein language model.
    """

    model_checkpoint: ESM3_MODEL_CHECKPOINTS = ConfigField(
        default="esm3_sm_open_v1",
        title="Model Checkpoint",
        description="ESM-3 model variant to load.",
    )

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )

    sampling_method: Literal["single_pass", "iterative_refinement"] = ConfigField(
        default="single_pass",
        title="Sampling Method",
        description=(
            "'single_pass' fills all masks in one forward; 'iterative_refinement' uses ESM-3 batch generation."
        ),
    )

    # Advanced parameters
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling by adjusting probability distribution sharpness.",
    )
    top_p: float = ConfigField(
        default=1.0,
        gt=0.0,
        le=1.0,
        title="Top P",
        description="Nucleus sampling threshold; 1.0 disables",
    )
    num_steps: int = ConfigField(
        default=20,
        ge=1,
        title="Num Steps",
        description="Iterative-refinement decoding steps; diminishing returns above 20",
    )
    schedule: Literal["cosine", "linear"] = ConfigField(
        default="cosine",
        title="Unmask Schedule",
        description="Unmask schedule across rounds; 'cosine' fronts more commits late",
    )
    strategy: Literal["random", "entropy"] = ConfigField(
        default="random",
        title="Unmask Strategy",
        description="Position-selection per round; 'entropy' commits the most-confident first",
    )
    temperature_annealing: bool = ConfigField(
        default=True,
        title="Temperature Annealing",
        description="Anneal temperature toward 0 across rounds",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="GPU device to run ESM3 on (e.g. 'cuda' or 'cuda:0').",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences to process simultaneously on GPU",
    )


@generator(
    key="esm3",
    label="ESM3 Protein Language Model",
    config=ESM3GeneratorConfig,
    description="ESM-3 open masked protein language model for local sequence mutation/refinement",
    uses_gpu=True,
    tools_called=["esm3-sample"],
    supported_sequence_types=["protein"],
)
@final
class ESM3Generator(Generator):
    """Protein sequence mutation/refinement generator using ESM3 open language model.

    This generator uses the open-source ESM3 protein language model to refine
    existing protein sequences through iterative mutation. It masks positions
    according to the configured masking strategy and samples biologically
    plausible amino acids at those positions.

    The generator category is ``"mutation"``, indicating it refines proposal
    sequences through targeted mutations.

    Attributes:
        model_checkpoint (str): ESM3 model checkpoint name.
        masking_strategy (MaskingStrategy): Strategy for selecting positions to mutate.
        sampling_method (str): ``"single_pass"`` (default) or
            ``"iterative_refinement"`` (delegates to ``model.batch_generate``).
        temperature (float): Sampling temperature for diversity control.
        top_p (float): Nucleus threshold; iterative-refinement only.
        num_steps (int): Refinement steps; iterative-refinement only.
        schedule (str): Unmask schedule; iterative-refinement only.
        strategy (str): Per-round commit selection; iterative-refinement only.
        temperature_annealing (bool): Anneal toward 0 across rounds; iterative only.
        batch_size (int): Number of sequences to process simultaneously on GPU.

    Example:
        >>> from proto_language.generator import ESM3Generator, ESM3GeneratorConfig
        >>> from proto_language.core import Segment
        >>> from proto_tools.transforms.masking import MaskingStrategy
        >>> config = ESM3GeneratorConfig(
        ...     temperature=1.0,
        ...     masking_strategy=MaskingStrategy(num_mutations=5),
        ... )
        >>> gen = ESM3Generator(config)
        >>> segment = Segment(sequence="M" * 100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Re-samples 5 randomly masked positions
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self, config: ESM3GeneratorConfig) -> None:
        """Initialize the ESM3 generator with model and sampling configuration.

        Args:
            config (ESM3GeneratorConfig): Configuration object containing all generator parameters.
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
        positions, then uses ESM3 to sample new amino acids at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        sequences = [seq.sequence for seq in self.segment.proposal_sequences]
        esm3_input = ESM3SampleInput(sequences=sequences)
        config = ESM3SampleConfig(
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

        result = run_esm3_sample(inputs=esm3_input, config=config)
        mutated_sequences = result.sequences

        for proposal, sequence in zip(self.segment.proposal_sequences, mutated_sequences, strict=True):
            proposal.sequence = sequence
