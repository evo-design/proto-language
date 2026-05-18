"""ProteinMPNN Generator for structure-conditioned protein sequence design."""

from typing import Any, Literal, final

from proto_tools import (
    InverseFoldingInput,
    InverseFoldingStructureInput,
    ProteinMPNNSampleConfig,
    Structure,
    run_proteinmpnn_sample,
)
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator, GeneratorInputType
from proto_language.language.generator.generator_registry import generator


class ProteinMPNNGeneratorConfig(BaseConfig):
    """Configuration object for ProteinMPNNGenerator.

    This class defines configuration parameters for the ProteinMPNN generator, which
    uses the ProteinMPNN inverse folding model to design protein sequences conditioned
    on a given 3D backbone structure.

    ProteinMPNN is a message-passing neural network that predicts amino acid sequences
    likely to fold into a specified protein backbone structure. It excels at redesigning
    existing proteins while maintaining structural compatibility.

    Attributes:
        model_choice (Literal["proteinmpnn", "abmpnn", "soluble"]): Model weights to use.
            ``"proteinmpnn"`` for the general-purpose model, ``"abmpnn"`` for
            antibody-optimized weights, ``"soluble"`` for soluble-protein-trained
            weights (same architecture, different training data).
        structure_inputs (list[InverseFoldingStructureInput] | None): Structure(s) with per-structure
            design constraints. Each ``InverseFoldingStructureInput`` bundles a structure with optional
            ``chains_to_redesign`` and ``fixed_positions`` specific to that structure.

            This field is optional (defaults to ``None``) primarily to support ``CyclingOptimizer``
            workflows, where the structure is provided dynamically from a previous step (e.g.,
            structure prediction) rather than being specified upfront in the config.

            **InverseFoldingStructureInput fields:**

            - ``structure``: File path, PDB content string, or ``Structure`` object
            - ``chains_to_redesign``: Optional chains to redesign (e.g., ``["A", "B"]``).
              If None, all chains in the structure are redesigned.
            - ``fixed_positions``: Optional per-chain residue positions to keep fixed
              (e.g., ``{"A": [1, 2, 3]}``, 1-indexed)

            **Accepts flexible input formats:**

            - A single string (file path or PDB content) - auto-converted to ``InverseFoldingStructureInput``
            - A single ``InverseFoldingStructureInput`` object
            - A list of strings or ``InverseFoldingStructureInput`` objects
            - A list of dicts with ``structure``, ``chains_to_redesign``, ``fixed_positions`` keys
        output_chain_id (str | None): For multi-chain sampling, write only this
            chain's generated sequence to the assigned segment. If unset, preserve
            ProteinMPNN's full output sequence.

        temperature (float): Controls randomness in amino acid sampling from the
            model's predicted probability distribution:

            - ``< 0.1``: Nearly deterministic, strongly favors most likely residues
            - ``0.1``: Low diversity, high confidence predictions (default)
            - ``0.5``: Moderate diversity
            - ``1.0``: High diversity, samples proportionally to probabilities

            Lower temperatures produce more consensus-like sequences; higher
            temperatures explore more sequence diversity. Must be in range [0, 1].
            Default: ``0.1``.

        excluded_amino_acids (list[str] | None): List of amino acids to exclude
            from designed sequences, specified as single-letter codes. Common uses:

            - ``["C"]``: Exclude cysteine to avoid disulfide complications
            - ``["M"]``: Exclude methionine to simplify expression
            - ``["C", "M", "W"]``: Exclude multiple residues

            Default: ``None`` (all amino acids allowed).

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.

        device (str): GPU device for model inference, e.g. ``"cuda"`` or
            ``"cuda:0"``. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during model loading
            and sequence generation. Default: ``False``.

    Example:
        Simple usage with just a file path:

        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure_inputs="/path/to/backbone.pdb",
        ...     temperature=0.1,
        ... )

        With per-structure chain selection and fixed positions:

        >>> from proto_tools import InverseFoldingStructureInput
        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure_inputs=InverseFoldingStructureInput(
        ...         structure="/path/to/backbone.pdb",
        ...         chains_to_redesign=["A"],  # Only design chain A
        ...         fixed_positions={"A": [1, 2, 3]},  # Keep positions 1-3 fixed
        ...     ),
        ...     temperature=0.1,
        ... )

        Multiple structures with different constraints:

        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure_inputs=[
        ...         InverseFoldingStructureInput(
        ...             structure="/path/to/struct1.pdb",
        ...             chains_to_redesign=["A"],
        ...             fixed_positions={"A": [1, 2, 3]},
        ...         ),
        ...         InverseFoldingStructureInput(
        ...             structure="/path/to/struct2.pdb",
        ...             chains_to_redesign=["A", "B"],
        ...         ),
        ...     ],
        ...     temperature=0.1,
        ... )
    """

    model_choice: Literal["proteinmpnn", "abmpnn", "soluble"] = ConfigField(
        default="proteinmpnn",
        title="Model Choice",
        description="Model weights: 'proteinmpnn' (general), 'abmpnn' (antibody), or 'soluble' (soluble proteins).",
    )

    # Structure parameters - bundles structure, chains_to_redesign, and fixed_positions per structure.
    structure_inputs: list[InverseFoldingStructureInput] | None = ConfigField(
        default=None,
        title="Structure Inputs",
        description="Structure(s) with optional chains_to_redesign and fixed_positions constraints.",
    )
    output_chain_id: str | None = ConfigField(
        default=None,
        title="Output Chain",
        description="When sampling a multi-chain structure, write only this chain's sequence to the target segment.",
    )

    # Optional parameters.
    temperature: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Temperature",
        description="Controls randomness in sampling. Lower values produce more deterministic sequences.",
    )
    excluded_amino_acids: list[str] | None = ConfigField(
        default=None,
        title="Unallowed Amino Acids",
        description="List of amino acids (single-letter codes) to exclude from designed sequences.",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences to process simultaneously on GPU",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="GPU device for inference (e.g. 'cuda' or 'cuda:0').",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print status messages during execution.",
    )

    @field_validator("structure_inputs", mode="before")
    @classmethod
    def normalize_structure_inputs(cls, v: Any) -> Any:
        """Convert various input formats to List[InverseFoldingStructureInput]."""
        if v is None:
            return None

        if not isinstance(v, list):
            v = [v]

        result = []
        for item in v:
            if isinstance(item, InverseFoldingStructureInput):
                result.append(item)
            elif isinstance(item, (str, Structure)):
                # Simple path/content/object -> InverseFoldingStructureInput with no constraints
                result.append(InverseFoldingStructureInput(structure=item))
            elif isinstance(item, dict):
                # Dict -> InverseFoldingStructureInput
                result.append(InverseFoldingStructureInput(**item))
            else:
                raise ValueError(f"Unsupported structure_inputs item type: {type(item)}")
        return result


@generator(
    key="proteinmpnn",
    label="ProteinMPNN Inverse Folding",
    config=ProteinMPNNGeneratorConfig,
    description="ProteinMPNN structure-conditioned protein sequence design",
    uses_gpu=True,
    tools_called=["proteinmpnn-sample"],
    supported_sequence_types=["protein"],
)
@final
class ProteinMPNNGenerator(Generator):
    """Protein sequence generator using ProteinMPNN inverse folding model.

    This generator uses ProteinMPNN to design protein sequences that are predicted
    to fold into a given 3D backbone structure. Unlike mutation-based generators
    that refine existing sequences, ProteinMPNN generates sequences directly from
    structural information.

    ProteinMPNN is particularly effective for:

    - Redesigning existing proteins while maintaining fold
    - Designing sequences for computationally generated backbones
    - Creating sequence diversity for experimental screening
    - Stabilizing protein structures through sequence optimization

    Attributes:
        batch_size (int): Number of sequences to generate per batch.

    Example:
        >>> from proto_language.language.generator import ProteinMPNNGenerator, ProteinMPNNGeneratorConfig
        >>> from proto_language.language.core import Segment
        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure_inputs="/path/to/backbone.pdb",
        ...     temperature=0.1,
        ... )
        >>> gen = ProteinMPNNGenerator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates num_proposals sequences from the backbone
    """

    input_type = GeneratorInputType.STRUCTURE

    def __init__(self, config: ProteinMPNNGeneratorConfig) -> None:
        """Initialize the ProteinMPNN generator with structure and sampling configuration.

        Args:
            config (ProteinMPNNGeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config

        self.model_choice = config.model_choice
        self.structure_inputs = config.structure_inputs
        self.output_chain_id = config.output_chain_id
        self.temperature = config.temperature
        self.excluded_amino_acids = config.excluded_amino_acids
        self.batch_size = config.batch_size
        self.device = config.device
        self.verbose = config.verbose

    def _sample(self, structure_inputs: list[InverseFoldingStructureInput] | None = None) -> None:
        """Generate protein sequences using ProteinMPNN and update proposal sequences.

        Args:
            structure_inputs (list[InverseFoldingStructureInput] | None): Optional structure inputs to use instead of config.
                Accepts flexible formats (same as config): single structure, list of structures,
                ``Structure`` objects, file paths, or ``InverseFoldingStructureInput`` objects.
                If provided, generates one sequence per structure. If None, uses
                config structure_inputs (single structure generates num_proposals
                sequences, multiple structures generate one sequence each).

        Raises:
            ValueError: If no structure_inputs provided and none configured.
        """
        self._validate_generator()
        num_proposals = self.segment.num_proposals

        # Normalize and use provided structure_inputs, or fall back to config inputs
        sampling_structure_inputs = (
            ProteinMPNNGeneratorConfig.normalize_structure_inputs(structure_inputs)
            if structure_inputs is not None
            else self.structure_inputs
        )

        if sampling_structure_inputs is None:
            raise ValueError(
                "No structure_inputs provided. Either pass structure_inputs to sample() or configure structure_inputs in the generator config."
            )

        generated_sequences: list[str] = []
        perplexities: list[float] = []
        sequence_recoveries: list[float] = []

        if len(sampling_structure_inputs) == 1:
            # Single structure: generate num_proposals sequences in chunks of batch_size
            num_seqs = num_proposals
            bs = self.batch_size
        else:
            # N structures: one sequence per structure
            if len(sampling_structure_inputs) != num_proposals:
                raise ValueError(
                    f"Number of structure_inputs ({len(sampling_structure_inputs)}) must either be 1 or match num_proposals ({num_proposals})"
                )
            num_seqs = 1
            bs = 1

        tool_config = ProteinMPNNSampleConfig(
            num_sequences_per_structure=num_seqs,
            batch_size=bs,
            temperature=self.temperature,
            excluded_amino_acids=self.excluded_amino_acids,
            seed=self._next_seed(),
            model_choice=self.model_choice,
            device=self.device,
            verbose=self.verbose,
        )

        result = run_proteinmpnn_sample(
            inputs=InverseFoldingInput(inputs=sampling_structure_inputs),
            config=tool_config,
        )
        full_sequences: list[str] = []
        for designed, struct_input in zip(result.designed_sequences, sampling_structure_inputs, strict=True):
            full_sequences.extend(designed.sequences)
            generated_sequences.extend(
                self._select_output_sequence(sequence, struct_input) for sequence in designed.sequences
            )
            perplexities.extend(designed.perplexity)
            sequence_recoveries.extend(designed.sequence_recovery)

        key = self._spec.key
        for proposal, sequence, full_sequence, perplexity, recovery in zip(
            self.segment.proposal_sequences,
            generated_sequences,
            full_sequences,
            perplexities,
            sequence_recoveries,
            strict=True,
        ):
            proposal.sequence = sequence
            proposal._generator_metadata[key] = {
                "perplexity": perplexity,
                "sequence_recovery": recovery,
                "full_sequence": full_sequence,
            }

        # Write the generating structure onto each proposal sequence
        if len(sampling_structure_inputs) == 1:
            for proposal in self.segment.proposal_sequences:
                proposal.structure = sampling_structure_inputs[0].structure
        else:
            for proposal, struct_input in zip(self.segment.proposal_sequences, sampling_structure_inputs, strict=True):
                proposal.structure = struct_input.structure

    def _select_output_sequence(self, sequence: str, struct_input: InverseFoldingStructureInput) -> str:
        """Return the configured output chain from a ProteinMPNN sequence."""
        if self.output_chain_id is None:
            return sequence
        chain_ids_to_redesign: list[str] = struct_input.chain_ids_to_redesign
        if self.output_chain_id not in chain_ids_to_redesign:
            raise ValueError(
                f"output_chain_id {self.output_chain_id!r} not found in chain_ids_to_redesign {chain_ids_to_redesign}"
            )
        parts = sequence.split("/")
        if len(parts) != len(chain_ids_to_redesign):
            raise ValueError(
                f"Expected {len(chain_ids_to_redesign)} slash-separated chains in ProteinMPNN output "
                f"(for chain_ids_to_redesign {chain_ids_to_redesign}), got {len(parts)}: {sequence!r}"
            )
        return parts[chain_ids_to_redesign.index(self.output_chain_id)]
