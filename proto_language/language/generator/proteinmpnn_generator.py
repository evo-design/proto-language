"""
proto_language/language/generator/proteinmpnn_generator.py

ProteinMPNN Generator for structure-conditioned protein sequence design.
"""

from __future__ import annotations

from typing import List, Optional, final

from proto_tools import (
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingStructureInput,
    Structure,
    run_proteinmpnn_sample,
)
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
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
        structure_inputs (list[InverseFoldingStructureInput] | None): Structure(s) with per-structure
            design constraints. Each ``InverseFoldingStructureInput`` bundles a structure with optional
            ``chain_ids`` and ``fixed_positions`` specific to that structure.

            This field is optional (defaults to ``None``) primarily to support ``CyclingOptimizer``
            workflows, where the structure is provided dynamically from a previous step (e.g.,
            structure prediction) rather than being specified upfront in the config.

            **InverseFoldingStructureInput fields:**

            - ``structure``: File path, PDB content string, or ``Structure`` object
            - ``chain_ids``: Optional list of chain IDs to design (e.g., ``["A", "B"]``).
              If None, all chains in the structure are designed.
            - ``fixed_positions``: Optional dict mapping chain IDs to residue positions
              to keep fixed (e.g., ``{"A": [1, 2, 3]}``)

            **Accepts flexible input formats:**

            - A single string (file path or PDB content) - auto-converted to ``InverseFoldingStructureInput``
            - A single ``InverseFoldingStructureInput`` object
            - A list of strings or ``InverseFoldingStructureInput`` objects
            - A list of dicts with ``structure``, ``chain_ids``, ``fixed_positions`` keys

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

        seed (int): Random seed for reproducible sequence generation. Using the
            same seed with identical inputs produces identical outputs.
            Default: ``1337``.

        device (str): Compute device for model inference. Options:

            - ``"cuda"``: NVIDIA GPU (recommended, default)
            - ``"cpu"``: CPU execution (slower)

            Default: ``"cuda"``.

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
        ...         chain_ids=["A"],  # Only design chain A
        ...         fixed_positions={"A": [1, 2, 3]},  # Keep positions 1-3 fixed
        ...     ),
        ...     temperature=0.1,
        ... )

        Multiple structures with different constraints:

        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure_inputs=[
        ...         InverseFoldingStructureInput(
        ...             structure="/path/to/struct1.pdb",
        ...             chain_ids=["A"],
        ...             fixed_positions={"A": [1, 2, 3]},
        ...         ),
        ...         InverseFoldingStructureInput(
        ...             structure="/path/to/struct2.pdb",
        ...             chain_ids=["A", "B"],
        ...         ),
        ...     ],
        ...     temperature=0.1,
        ... )
    """

    # Structure parameters - bundles structure, chain_ids, and fixed_positions per structure.
    structure_inputs: Optional[List[InverseFoldingStructureInput]] = ConfigField(
        default=None,
        title="Structure Inputs",
        description="Structure(s) with optional chain_ids and fixed_positions constraints.",
    )

    # Optional parameters.
    temperature: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Temperature",
        description="Controls randomness in sampling. Lower values produce more deterministic sequences.",
        advanced=True,
    )
    excluded_amino_acids: Optional[List[str]] = ConfigField(
        default=None,
        title="Unallowed Amino Acids",
        description="List of amino acids (single-letter codes) to exclude from designed sequences.",
        advanced=True,
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )
    seed: int = ConfigField(
        default=1337,
        title="Random Seed",
        description="Random seed for reproducible sequence generation.",
        advanced=True,
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Compute device for inference: 'cuda' or 'cpu'.",
        hidden=True,
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print status messages during execution.",
        hidden=True,
    )

    @field_validator("structure_inputs", mode="before")
    @classmethod
    def normalize_structure_inputs(cls, v):
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
                raise ValueError(
                    f"Unsupported structure_inputs item type: {type(item)}"
                )
        return result


@generator(
    key="proteinmpnn",
    label="ProteinMPNN Inverse Folding",
    config=ProteinMPNNGeneratorConfig,
    description="ProteinMPNN structure-conditioned protein sequence design",
    uses_gpu=True,
    tools_called=["proteinmpnn-sample"],
    category="inverse_folding",
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

    def __init__(self, config: ProteinMPNNGeneratorConfig) -> None:
        """Initialize the ProteinMPNN generator with structure and sampling configuration.

        Args:
            config (ProteinMPNNGeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config

        self.structure_inputs = config.structure_inputs
        self.temperature = config.temperature
        self.excluded_amino_acids = config.excluded_amino_acids
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.device = config.device
        self.verbose = config.verbose

    def sample(
        self, structure_inputs: Optional[List[InverseFoldingStructureInput]] = None
    ) -> None:
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
        num_proposals = self._assigned_segment.num_proposals

        # Normalize and use provided structure_inputs, or fall back to config inputs
        sampling_structure_inputs = (
            ProteinMPNNGeneratorConfig.normalize_structure_inputs(structure_inputs)
            if structure_inputs is not None
            else self.structure_inputs
        )

        if sampling_structure_inputs is None:
            raise ValueError("No structure_inputs provided. Either pass structure_inputs to sample() or configure structure_inputs in the generator config.")

        generated_sequences = []
        perplexities = []
        sequence_identities = []

        if len(sampling_structure_inputs) == 1:
            # Single structure: generate num_proposals sequences in chunks of batch_size
            num_seqs = num_proposals
            bs = self.batch_size
        else:
            # N structures: one sequence per structure
            if len(sampling_structure_inputs) != num_proposals:
                raise ValueError(f"Number of structure_inputs ({len(sampling_structure_inputs)}) must either be 1 or match num_proposals ({num_proposals})")
            num_seqs = 1
            bs = 1

        tool_config = InverseFoldingConfig(
            num_sequences_per_structure=num_seqs,
            batch_size=bs,
            temperature=self.temperature,
            excluded_amino_acids=self.excluded_amino_acids,
            seed=self.seed,
            device=self.device,
            verbose=self.verbose,
        )

        result = run_proteinmpnn_sample(
            inputs=InverseFoldingInput(inputs=sampling_structure_inputs),
            config=tool_config,
        )
        for designed in result.designed_sequences:
            generated_sequences.extend(designed.sequences)
            perplexities.extend(designed.perplexity)
            sequence_identities.extend(designed.sequence_identity)

        for proposal, sequence, perplexity, identity in zip(
            self._assigned_segment.proposal_sequences,
            generated_sequences,
            perplexities,
            sequence_identities,
            strict=True,
        ):
            proposal.sequence = sequence
            proposal._metadata.update(
                {
                    "proteinmpnn_perplexity": perplexity,
                    "proteinmpnn_sequence_identity": identity,
                }
            )
