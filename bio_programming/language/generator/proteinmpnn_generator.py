"""
ProteinMPNN Generator for structure-conditioned protein sequence design.
"""
from __future__ import annotations

from typing import final, List, Optional

from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.tools.inverse_folding.proteinmpnn import run_proteinmpnn_sample
from proto_language.tools.inverse_folding.schemas import (
    InverseFoldingStructure,
    InverseFoldingInput,
    InverseFoldingConfig,
)
from proto_language.tools.structures import ProteinStructure


class ProteinMPNNGeneratorConfig(BaseConfig):
    """Configuration object for ProteinMPNNGenerator.

    This class defines configuration parameters for the ProteinMPNN generator, which
    uses the ProteinMPNN inverse folding model to design protein sequences conditioned
    on a given 3D backbone structure.

    ProteinMPNN is a message-passing neural network that predicts amino acid sequences
    likely to fold into a specified protein backbone structure. It excels at redesigning
    existing proteins while maintaining structural compatibility.

    Attributes:
        structure_inputs (List[InverseFoldingStructure]): Structure(s) with per-structure design
            constraints. Each ``InverseFoldingStructure`` bundles a structure with optional
            ``chain_ids`` and ``fixed_positions`` specific to that structure.

            **InverseFoldingStructure fields:**

            - ``structure``: File path, PDB content string, or ``ProteinStructure`` object
            - ``chain_ids``: Optional list of chain IDs to design (e.g., ``["A", "B"]``).
              If None, all chains in the structure are designed.
            - ``fixed_positions``: Optional dict mapping chain IDs to residue positions
              to keep fixed (e.g., ``{"A": [1, 2, 3]}``)

            **Accepts flexible input formats:**

            - A single string (file path or PDB content) - auto-converted to ``InverseFoldingStructure``
            - A single ``InverseFoldingStructure`` object
            - A list of strings or ``InverseFoldingStructure`` objects
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

        excluded_amino_acids (Optional[List[str]]): List of amino acids to exclude
            from designed sequences, specified as single-letter codes. Common uses:

            - ``["C"]``: Exclude cysteine to avoid disulfide complications
            - ``["M"]``: Exclude methionine to simplify expression
            - ``["C", "M", "W"]``: Exclude multiple residues

            Default: ``None`` (all amino acids allowed).

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

        >>> from proto_language.tools.inverse_folding.schemas import InverseFoldingStructure
        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure_inputs=InverseFoldingStructure(
        ...         structure="/path/to/backbone.pdb",
        ...         chain_ids=["A"],  # Only design chain A
        ...         fixed_positions={"A": [1, 2, 3]},  # Keep positions 1-3 fixed
        ...     ),
        ...     temperature=0.1,
        ... )

        Multiple structures with different constraints:

        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure_inputs=[
        ...         InverseFoldingStructure(
        ...             structure="/path/to/struct1.pdb",
        ...             chain_ids=["A"],
        ...             fixed_positions={"A": [1, 2, 3]},
        ...         ),
        ...         InverseFoldingStructure(
        ...             structure="/path/to/struct2.pdb",
        ...             chain_ids=["A", "B"],
        ...         ),
        ...     ],
        ...     temperature=0.1,
        ... )
    """

    # Structure parameters - bundles structure, chain_ids, and fixed_positions per structure.
    structure_inputs: List[InverseFoldingStructure] = ConfigField(
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
        """Convert various input formats to List[InverseFoldingStructure]."""
        if not isinstance(v, list):
            v = [v]

        result = []
        for item in v:
            if isinstance(item, InverseFoldingStructure):
                result.append(item)
            elif isinstance(item, (str, ProteinStructure)):
                # Simple path/content/object -> InverseFoldingStructure with no constraints
                result.append(InverseFoldingStructure(structure=item))
            elif isinstance(item, dict):
                # Dict -> InverseFoldingStructure
                result.append(InverseFoldingStructure(**item))
            else:
                raise ValueError(f"Unsupported structure_inputs item type: {type(item)}")
        return result


@GeneratorRegistry.register(
    key="proteinmpnn",
    label="ProteinMPNN Inverse Folding",
    config=ProteinMPNNGeneratorConfig,
    description="ProteinMPNN structure-conditioned protein sequence design",
    requires_gpu=True,
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
        >>> gen.sample()  # Generates num_candidates sequences from the backbone
    """

    def __init__(self, config: ProteinMPNNGeneratorConfig) -> None:
        """Initialize the ProteinMPNN generator with structure and sampling configuration.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()

        self.structure_inputs = config.structure_inputs
        self.temperature = config.temperature
        self.excluded_amino_acids = config.excluded_amino_acids
        self.seed = config.seed
        self.device = config.device
        self.verbose = config.verbose

    def sample(self, structure_inputs: Optional[List[InverseFoldingStructure]] = None) -> None:
        """Generate protein sequences using ProteinMPNN and update candidate sequences.

        Args:
            structure_inputs: Optional list of InverseFoldingStructure to use instead of config.
                If provided, generates one sequence per structure. If None, uses
                config structure_inputs (single structure generates batch_size sequences,
                multiple structures generate one sequence each).
        """
        num_candidates = self._assigned_segment.num_candidates

        # Use provided structure_inputs or fall back to config
        sampling_structure_inputs = structure_inputs if structure_inputs is not None else self.structure_inputs

        # Determine batch size based on number of structures
        if len(sampling_structure_inputs) == 1:
            batch_size = num_candidates
        else:
            if len(sampling_structure_inputs) != num_candidates:
                raise ValueError(f"Number of structure_inputs({len(sampling_structure_inputs)}) must either be 1 or match number of candidates ({num_candidates})")
            batch_size = 1

        tool_config = InverseFoldingConfig(
            batch_size=batch_size,
            temperature=self.temperature,
            excluded_amino_acids=self.excluded_amino_acids,
            seed=self.seed,
            device=self.device,
            keep_on_gpu=True,
            verbose=self.verbose,
        )

        # Run sampling
        result = run_proteinmpnn_sample(inputs=InverseFoldingInput(inputs=sampling_structure_inputs), config=tool_config)

        # Collect sequences and metrics from all structure results
        generated_sequences = []
        perplexities = []
        sequence_identities = []
        for designed in result.designed_sequences:
            generated_sequences.extend(designed.sequences)
            perplexities.extend(designed.mpnn_perplexity)
            sequence_identities.extend(designed.sequence_identity)

        if len(generated_sequences) != num_candidates:
            raise RuntimeError(f"Expected generator to generate {num_candidates} sequences but got {len(generated_sequences)}")

        # Update candidate sequences
        for candidate, sequence, perplexity, identity in zip(
            self._assigned_segment.candidate_sequences,
            generated_sequences,
            perplexities,
            sequence_identities,
            strict=True,
        ):
            candidate.sequence = sequence
            candidate._metadata.update({
                "proteinmpnn_perplexity": perplexity,
                "proteinmpnn_sequence_identity": identity,
            })
