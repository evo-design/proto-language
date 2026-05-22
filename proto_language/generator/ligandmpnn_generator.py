"""LigandMPNN extends ProteinMPNN to consider ligand context when designing.

protein sequences, making it particularly effective for enzyme design
and binding site optimization.
"""

from typing import Any, final

from proto_tools import (
    InverseFoldingInput,
    InverseFoldingStructureInput,
    LigandMPNNSampleConfig,
    Structure,
    run_ligandmpnn_sample,
)
from pydantic import field_validator

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField


class LigandMPNNGeneratorConfig(BaseConfig):
    """Configuration object for LigandMPNNGenerator.

    This class defines configuration parameters for the LigandMPNN generator, which
    uses the LigandMPNN inverse folding model to design protein sequences conditioned
    on a given 3D backbone structure and ligand context.

    LigandMPNN extends ProteinMPNN to be aware of non-protein atoms (ligands, cofactors,
    metal ions), making it particularly effective for:

    - Enzyme active site design
    - Binding pocket optimization
    - Cofactor-dependent protein design

    Attributes:
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

        >>> config = LigandMPNNGeneratorConfig(
        ...     structure_inputs="/path/to/enzyme_with_ligand.pdb",
        ...     temperature=0.1,
        ... )

        With per-structure chain selection and fixed positions (e.g., preserve catalytic residues):

        >>> from proto_tools import InverseFoldingStructureInput
        >>> config = LigandMPNNGeneratorConfig(
        ...     structure_inputs=InverseFoldingStructureInput(
        ...         structure="/path/to/enzyme.pdb",
        ...         chains_to_redesign=["A"],
        ...         fixed_positions={"A": [45, 67, 89]},  # Catalytic triad
        ...     ),
        ...     temperature=0.1,
        ... )

        Multiple structures with different constraints:

        >>> config = LigandMPNNGeneratorConfig(
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

    # Structure parameters - bundles structure, chains_to_redesign, and fixed_positions per structure.
    structure_inputs: list[InverseFoldingStructureInput] | None = ConfigField(
        default=None,
        title="Structure Inputs",
        description="Structure(s) with optional chains_to_redesign and fixed_positions constraints.",
    )

    # Optional parameters.
    temperature: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Temperature",
        description="Randomness of sampling (0-1). Near 0 is deterministic; near 1 is proportional to model probs.",
    )
    excluded_amino_acids: list[str] | None = ConfigField(
        default=None,
        title="Excluded Amino Acids",
        description="Single-letter amino-acid codes to forbid in the designed sequence.",
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
    key="ligandmpnn",
    label="LigandMPNN Inverse Folding",
    config=LigandMPNNGeneratorConfig,
    description="LigandMPNN structure-conditioned protein sequence design with ligand awareness",
    uses_gpu=True,
    tools_called=["ligandmpnn-sample"],
    supported_sequence_types=["protein"],
)
@final
class LigandMPNNGenerator(Generator):
    """Protein sequence generator using LigandMPNN inverse folding model.

    This generator uses LigandMPNN to design protein sequences that are predicted
    to fold into a given 3D backbone structure while considering ligand context.
    Unlike ProteinMPNN, LigandMPNN is aware of non-protein atoms (ligands, cofactors,
    metal ions) in the structure.

    LigandMPNN is particularly effective for:

    - Designing enzymes with specific active site geometries
    - Optimizing binding pockets around ligands
    - Creating sequences for cofactor-dependent proteins
    - Redesigning protein-ligand interfaces

    Attributes:
        batch_size (int): Number of sequences to generate per batch.

    Example:
        >>> from proto_language.generator import LigandMPNNGenerator, LigandMPNNGeneratorConfig
        >>> from proto_language.core import Segment
        >>> config = LigandMPNNGeneratorConfig(
        ...     structure_inputs="/path/to/enzyme_with_ligand.pdb",
        ...     temperature=0.1,
        ... )
        >>> gen = LigandMPNNGenerator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates num_proposals sequences from the backbone
    """

    input_type = GeneratorInputType.STRUCTURE

    def __init__(self, config: LigandMPNNGeneratorConfig) -> None:
        """Initialize the LigandMPNN generator with structure and sampling configuration.

        Args:
            config (LigandMPNNGeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config

        self.structure_inputs = config.structure_inputs
        self.temperature = config.temperature
        self.excluded_amino_acids = config.excluded_amino_acids
        self.batch_size = config.batch_size
        self.device = config.device
        self.verbose = config.verbose

    def _sample(self, structure_inputs: list[InverseFoldingStructureInput] | None = None) -> None:
        """Generate protein sequences using LigandMPNN and update proposal sequences.

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

        # Normalize and use provided structure_inputs, or fall back to config
        sampling_structure_inputs = (
            LigandMPNNGeneratorConfig.normalize_structure_inputs(structure_inputs)
            if structure_inputs is not None
            else self.structure_inputs
        )

        if sampling_structure_inputs is None:
            raise ValueError(
                "No structure_inputs provided. Either pass structure_inputs to sample() or configure structure_inputs in the generator config."
            )

        generated_sequences: list[str] = []
        all_recovery: list[float] = []
        all_interface_recovery: list[float | None] = []

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

        tool_config = LigandMPNNSampleConfig(
            num_sequences_per_structure=num_seqs,
            batch_size=bs,
            temperature=self.temperature,
            excluded_amino_acids=self.excluded_amino_acids,
            seed=self._next_seed(),
            device=self.device,
            verbose=self.verbose,
        )

        result = run_ligandmpnn_sample(
            inputs=InverseFoldingInput(inputs=sampling_structure_inputs),
            config=tool_config,
        )
        for design in result.designed_sequences:
            generated_sequences.extend(design.sequences)
            all_recovery.extend(design.sequence_recovery)
            # ligand_interface_sequence_recovery is None when the input structure has no ligand;
            # pad with None so the per-proposal alignment via zip() below holds.
            interface = design.ligand_interface_sequence_recovery
            all_interface_recovery.extend([None] * len(design.sequences) if interface is None else interface)

        key = self._spec.key
        for proposal, sequence, recovery, interface_recovery in zip(
            self.segment.proposal_sequences,
            generated_sequences,
            all_recovery,
            all_interface_recovery,
            strict=True,
        ):
            proposal.sequence = sequence
            proposal._generator_metadata[key] = {
                "sequence_recovery": recovery,
                "ligand_interface_sequence_recovery": interface_recovery,
            }

        # Write the generating structure onto each proposal sequence
        if len(sampling_structure_inputs) == 1:
            for proposal in self.segment.proposal_sequences:
                proposal.structure = sampling_structure_inputs[0].structure
        else:
            for proposal, struct_input in zip(self.segment.proposal_sequences, sampling_structure_inputs, strict=True):
                proposal.structure = struct_input.structure
