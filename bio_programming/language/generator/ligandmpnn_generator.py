"""
LigandMPNN Generator for ligand-aware protein sequence design.

LigandMPNN extends ProteinMPNN to consider ligand context when designing
protein sequences, making it particularly effective for enzyme design
and binding site optimization.
"""

from __future__ import annotations

from typing import List, Optional, final

from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.tools.inverse_folding.ligandmpnn import (
    LigandMPNNConfig,
    run_ligandmpnn_sample,
)
from proto_language.tools.inverse_folding.schemas import (
    InverseFoldingInput,
    InverseFoldingStructureInput,
)
from proto_language.tools.structures import ProteinStructure


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
        structure_inputs (Optional[List[InverseFoldingStructureInput]]): Structure(s) with per-structure
            design constraints. Each ``InverseFoldingStructureInput`` bundles a structure with optional
            ``chain_ids`` and ``fixed_positions`` specific to that structure.

            This field is optional (defaults to ``None``) primarily to support ``CyclicalOptimizer``
            workflows, where the structure is provided dynamically from a previous step (e.g.,
            structure prediction) rather than being specified upfront in the config.

            **InverseFoldingStructureInput fields:**

            - ``structure``: File path, PDB content string, or ``ProteinStructure`` object
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

        >>> config = LigandMPNNGeneratorConfig(
        ...     structure_inputs="/path/to/enzyme_with_ligand.pdb",
        ...     temperature=0.1,
        ... )

        With per-structure chain selection and fixed positions (e.g., preserve catalytic residues):

        >>> from proto_language.tools.inverse_folding.schemas import InverseFoldingStructureInput
        >>> config = LigandMPNNGeneratorConfig(
        ...     structure_inputs=InverseFoldingStructureInput(
        ...         structure="/path/to/enzyme.pdb",
        ...         chain_ids=["A"],
        ...         fixed_positions={"A": [45, 67, 89]},  # Catalytic triad
        ...     ),
        ...     temperature=0.1,
        ... )

        Multiple structures with different constraints:

        >>> config = LigandMPNNGeneratorConfig(
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
            elif isinstance(item, (str, ProteinStructure)):
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


@GeneratorRegistry.register(
    key="ligandmpnn",
    label="LigandMPNN Inverse Folding",
    config=LigandMPNNGeneratorConfig,
    description="LigandMPNN structure-conditioned protein sequence design with ligand awareness",
    requires_gpu=True,
    tools_called=["ligandmpnn-sample"],
    category="inverse_folding",
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

    Example:
        >>> from proto_language.language.generator import LigandMPNNGenerator, LigandMPNNGeneratorConfig
        >>> from proto_language.language.core import Segment
        >>> config = LigandMPNNGeneratorConfig(
        ...     structure_inputs="/path/to/enzyme_with_ligand.pdb",
        ...     temperature=0.1,
        ... )
        >>> gen = LigandMPNNGenerator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates num_candidates sequences from the backbone
    """

    def __init__(self, config: LigandMPNNGeneratorConfig) -> None:
        """Initialize the LigandMPNN generator with structure and sampling configuration.

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

    def sample(
        self, structure_inputs: Optional[List[InverseFoldingStructureInput]] = None
    ) -> None:
        """Generate protein sequences using LigandMPNN and update candidate sequences.

        Args:
            structure_inputs: Optional structure inputs to use instead of config.
                Accepts flexible formats (same as config): single structure, list of structures,
                ``ProteinStructure`` objects, file paths, or ``InverseFoldingStructureInput`` objects.
                If provided, generates one sequence per structure. If None, uses
                config structure_inputs (single structure generates batch_size sequences,
                multiple structures generate one sequence each).

        Raises:
            ValueError: If no structure_inputs provided and none configured.
        """
        self._validate_generator()
        num_candidates = self._assigned_segment.num_candidates

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

        # Determine batch size based on number of structures
        if len(sampling_structure_inputs) == 1:
            batch_size = num_candidates
        else:
            if len(sampling_structure_inputs) != num_candidates:
                raise ValueError(
                    f"Number of structure_inputs({len(sampling_structure_inputs)}) must either be 1 or match number of candidates ({num_candidates})"
                )
            batch_size = 1

        tool_config = LigandMPNNConfig(
            batch_size=batch_size,
            temperature=self.temperature,
            excluded_amino_acids=self.excluded_amino_acids,
            seed=self.seed,
            device=self.device,
            verbose=self.verbose,
        )

        # Run sampling
        result = run_ligandmpnn_sample(
            inputs=InverseFoldingInput(inputs=sampling_structure_inputs),
            config=tool_config,
        )

        # Collect sequences and metrics from all structure results
        generated_sequences = []
        all_scores = []
        for design in result.designed_sequences:
            generated_sequences.extend(design.sequences)
            all_scores.extend(design.ligandmpnn_scores)

        if len(generated_sequences) != num_candidates:
            raise RuntimeError(
                f"Expected generator to generate {num_candidates} sequences but got {len(generated_sequences)}"
            )

        # Update candidate sequences
        for candidate, sequence, score in zip(
            self._assigned_segment.candidate_sequences,
            generated_sequences,
            all_scores,
            strict=True,
        ):
            candidate.sequence = sequence
            candidate._metadata.update({"ligandmpnn_scores": score})
