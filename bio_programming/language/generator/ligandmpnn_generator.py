"""
LigandMPNN Generator for ligand-aware protein sequence design.

LigandMPNN extends ProteinMPNN to consider ligand context when designing
protein sequences, making it particularly effective for enzyme design
and binding site optimization.
"""
from __future__ import annotations

import os
from typing import Dict, final, List, Optional

from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import Generator
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.tools.inverse_folding.ligandmpnn import (
    LigandMPNNConfig,
    run_ligandmpnn_sample,
)
from proto_language.tools.inverse_folding.schemas import InverseFoldingInput
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
        structure (str | ProteinStructure): Protein structure to condition
            sequence design on. Accepts multiple formats:

            - Path to a PDB/CIF file (e.g., ``"/path/to/protein.pdb"``)
            - PDB content as a string
            - ``ProteinStructure`` instance

            The structure should include ligand atoms for ligand-aware design.

        chain_ids (Optional[List[str]]): Chain identifiers to design sequences for.
            If ``None``, automatically detects and uses all chains in the structure.
            Use this to target specific chains in multi-chain complexes.
            Example: ``["A", "B"]`` to design only chains A and B.
            Default: ``None``.

        dynamic_structure_path (bool): If true, and ``structure`` is set to a valid
            path, then this configures ``LigandMPNNGenerator`` to dynamically load
            the PDB from the path on each call to ``sample()``, which is useful for
            optimization loops that continuously change the protein structure.
            Default: ``False``.

        temperature (float): Controls randomness in amino acid sampling from the
            model's predicted probability distribution:

            - ``< 0.1``: Nearly deterministic, strongly favors most likely residues
            - ``0.1``: Low diversity, high confidence predictions (default)
            - ``0.5``: Moderate diversity
            - ``1.0``: High diversity, samples proportionally to probabilities

            Lower temperatures produce more consensus-like sequences; higher
            temperatures explore more sequence diversity. Must be in range [0, 1].
            Default: ``0.1``.

        fixed_positions (Optional[Dict[str, List[int]]]): Dictionary mapping chain
            IDs to residue positions that should remain fixed (not redesigned).
            Positions use the numbering from the input PDB structure (typically
            1-indexed). Useful for:

            - Preserving catalytic residues in enzymes
            - Maintaining binding interface residues
            - Keeping known functional motifs

            Example: ``{"A": [1, 2, 3, 45, 46], "B": [10, 11, 12]}`` fixes
            positions 1-3 and 45-46 on chain A, and 10-12 on chain B.
            Default: ``None`` (redesign all positions).

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

    Note:
        For detailed information on LigandMPNN, see:

        - Paper: Dauparas et al. "Atomic context-conditioned protein sequence
          design using LigandMPNN" (2023)
        - The model considers ligand atoms within ~4Å of protein residues

    Example:
        >>> config = LigandMPNNGeneratorConfig(
        ...     structure="/path/to/enzyme_with_ligand.pdb",
        ...     temperature=0.1,
        ...     fixed_positions={"A": [1, 2, 3]},  # Keep N-terminal residues
        ...     excluded_amino_acids=["C"],  # No cysteines
        ... )
    """

    # Required parameters.
    structure: str | ProteinStructure = ConfigField(
        title="Structure",
        description="PDB path, PDB content, or ProteinStructure object to condition design",
    )

    # Optional parameters.
    chain_ids: Optional[List[str]] = ConfigField(
        default=None,
        title="Chain IDs",
        description="Chain identifiers to design sequences for. If None, uses all chains in structure.",
    )
    dynamic_structure_path: bool = ConfigField(
        default=False,
        title="Dynamic Structure Path",
        description="Whether to reload the structure from a PDB file on each call to sample()",
    )
    temperature: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Temperature",
        description="Controls randomness in sampling. Lower values produce more deterministic sequences.",
        advanced=True,
    )
    fixed_positions: Optional[Dict[str, List[int]]] = ConfigField(
        default=None,
        title="Fixed Positions",
        description="Dictionary mapping chain IDs to residue positions to keep fixed during design.",
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

    @model_validator(mode="after")
    def validate_dynamic_structure_config(self):
        """Validate that dynamic structures have been set correctly."""
        if self.dynamic_structure_path:
            if not os.path.exists(self.structure):
                raise ValueError(
                    f"Dynamic structure configuration requires a valid structure path, found: {self.structure}"
                )
        return self


@GeneratorRegistry.register(
    key="ligandmpnn",
    label="LigandMPNN Inverse Folding",
    config=LigandMPNNGeneratorConfig,
    description="LigandMPNN structure-conditioned protein sequence design with ligand awareness",
    requires_gpu=True,
    tools_called=["ligandmpnn-sample"],
    category="autoregressive",
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
        ...     structure="/path/to/enzyme_with_ligand.pdb",
        ...     temperature=0.1,
        ... )
        >>> gen = LigandMPNNGenerator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates sequences compatible with the backbone and ligand
    """

    def __init__(self, config: LigandMPNNGeneratorConfig) -> None:
        """Initialize the LigandMPNN generator with structure and sampling configuration.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()

        self.config_structure = config.structure
        self.dynamic_structure_path = config.dynamic_structure_path
        self.chain_ids = config.chain_ids
        self.temperature = config.temperature
        self.fixed_positions = config.fixed_positions
        self.excluded_amino_acids = config.excluded_amino_acids
        self.seed = config.seed
        self.device = config.device
        self.verbose = config.verbose

        # Structure configuration.
        if self.dynamic_structure_path:
            # Initialize the structure to `None` and load on each call to
            # `sample()`, allowing for dynamically changing structures during an
            # optimization loop.
            self.structure = None
        else:
            # Just load the structure now.
            self._load_and_validate_structure()

        # Store metrics from last sample call
        self._last_scores: Optional[List[Dict[str, object]]] = None

    def _load_and_validate_structure(self) -> None:
        """
        Helper function for loading and validating the input structure.
        Called before each `self.sample()` to allow for dynamically changing structures.
        """
        # Load and convert structure input to ProteinStructure if needed.
        if isinstance(self.config_structure, ProteinStructure):
            self.structure = self.config_structure
        else:
            self.structure = ProteinStructure(
                structure_filepath_or_content=self.config_structure
            )

        # Auto-detect chain IDs if not provided.
        if self.chain_ids is None:
            self.chain_ids = self.structure.get_chain_ids()

        # Validate that specified chain IDs exist in structure.
        available_chains = set(self.structure.get_chain_ids())
        requested_chains = set(self.chain_ids)
        if not requested_chains.issubset(available_chains):
            missing = requested_chains - available_chains
            raise ValueError(
                f"Chain IDs {missing} not found in structure. "
                f"Available chains: {available_chains}"
            )

        # Validate fixed_positions chain IDs if provided.
        if self.fixed_positions is not None:
            fixed_chains = set(self.fixed_positions.keys())
            if not fixed_chains.issubset(available_chains):
                missing = fixed_chains - available_chains
                raise ValueError(
                    f"Fixed position chain IDs {missing} not found in structure. "
                    f"Available chains: {available_chains}"
                )

    def sample(self) -> None:
        """Generate protein sequences conditioned on the assigned structure and ligand context.

        Uses LigandMPNN to design sequences for all candidates in the batch.
        The number of sequences generated equals the number of candidate
        sequences in the assigned segment.

        After sampling, per-sequence metrics are available via ``self.last_scores``.

        Raises:
            RuntimeError: If called before assign().
        """
        if self.dynamic_structure_path:
            # Load the structure in case it has changed.
            self._load_and_validate_structure()

        num_candidates = len(self._assigned_segment.candidate_sequences)

        tool_input = InverseFoldingInput(
            structures=[self.structure],
            all_chain_ids=[self.chain_ids],
        )
        tool_config = LigandMPNNConfig(
            batch_size=num_candidates,
            temperature=self.temperature,
            fixed_positions=self.fixed_positions,
            excluded_amino_acids=self.excluded_amino_acids,
            seed=self.seed,
            device=self.device,
            verbose=self.verbose,
        )

        result = run_ligandmpnn_sample(inputs=tool_input, config=tool_config)

        # Extract sequences and metrics from first (only) structure result.
        designed = result.designed_sequences[0]
        generated_sequences = designed.sequences
        self._last_scores = designed.ligandmpnn_scores

        # Update candidate sequences.
        for idx, sequence in enumerate(generated_sequences):
            if idx < len(self._assigned_segment.candidate_sequences):
                candidate = self._assigned_segment.candidate_sequences[idx]
                candidate.sequence = sequence
                if self._last_scores and idx < len(self._last_scores):
                    candidate._metadata.update(
                        {"ligandmpnn_scores": self._last_scores[idx]}
                    )

    @property
    def last_scores(self) -> Optional[List[Dict[str, object]]]:
        """Get LigandMPNN scores from the last sample() call.

        Returns:
            List of score dictionaries, one per generated sequence, or None if
            sample() has not been called.
        """
        return self._last_scores
