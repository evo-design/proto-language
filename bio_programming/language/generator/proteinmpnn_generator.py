"""
ProteinMPNN Generator for structure-conditioned protein sequence design.
"""
from __future__ import annotations

import os
from pydantic import model_validator
from typing import final, Optional, Dict, List, Union

from proto_language.language.core import Generator, Segment
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.tools.inverse_folding.proteinmpnn import run_proteinmpnn_sample
from proto_language.tools.inverse_folding.schemas import (
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
        structure (Union[str, ProteinStructure]): Protein structure to condition
            sequence design on. Accepts multiple formats:

            - Path to a PDB file (e.g., ``"/path/to/protein.pdb"``)
            - PDB content as a string
            - ``ProteinStructure`` instance

            The structure defines the backbone geometry that designed sequences
            should be compatible with.

        chain_ids (Optional[List[str]]): Chain identifiers to design sequences for.
            If ``None``, automatically detects and uses all chains in the structure.
            Use this to target specific chains in multi-chain complexes.
            Example: ``["A", "B"]`` to design only chains A and B.
            Default: ``None``.

        dynamic_structure_path (bool): If true, and ``structure`` is set to a valid
            path, then this configures ``ProteinMPNNGenerator `` to dynamically load
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

        unallowed_amino_acids (Optional[List[str]]): List of amino acids to exclude
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
        For detailed information on ProteinMPNN, see:

        - Paper: Dauparas et al. "Robust deep learning-based protein sequence
          design using ProteinMPNN" Science (2022)
        - GitHub: https://github.com/dauparas/ProteinMPNN

    Example:
        >>> config = ProteinMPNNGeneratorConfig(
        ...     structure="/path/to/backbone.pdb",
        ...     temperature=0.1,
        ...     fixed_positions={"A": [1, 2, 3]},  # Keep N-terminal residues
        ...     unallowed_amino_acids=["C"],  # No cysteines
        ... )
    """

    # Required parameters.
    structure: Union[str, ProteinStructure] = ConfigField(
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
        description="Whether to reload the structure from a PDB file on each call to sample()"
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
    unallowed_amino_acids: Optional[List[str]] = ConfigField(
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

    @model_validator(mode='after')
    def validate_dynamic_structure_config(self):
        """Validate that dynamic structures have been set correctly."""
        if self.dynamic_structure_path:
            if not os.path.exists(self.structure):
                raise ValueError(f"Dynamic structure configuration requires a valid structure path, found: {self.structure}")
        return self


@GeneratorRegistry.register(
    key="proteinmpnn",
    label="ProteinMPNN Inverse Folding",
    config=ProteinMPNNGeneratorConfig,
    description="ProteinMPNN structure-conditioned protein sequence design",
    requires_gpu=True,
    tools_called=["proteinmpnn-sample"],
    category="autoregressive",
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
        ...     structure="/path/to/backbone.pdb",
        ...     temperature=0.1,
        ... )
        >>> gen = ProteinMPNNGenerator(config)
        >>> segment = Segment(length=100, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates sequences compatible with the backbone
    """

    supported_sequence_types = ["protein"]

    def __init__(self, config: ProteinMPNNGeneratorConfig) -> None:
        """Initialize the ProteinMPNN generator with structure and sampling configuration.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()

        self.config_structure = config.structure
        self.dynamic_structure_path = config.dynamic_structure_path
        self.chain_ids = config.chain_ids
        self.temperature = config.temperature
        self.fixed_positions = config.fixed_positions
        self.unallowed_amino_acids = config.unallowed_amino_acids
        self.seed = config.seed
        self.device = config.device
        self.verbose = config.verbose
        self.category = "autoregressive"

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
        self._last_perplexities: Optional[List[float]] = None
        self._last_sequence_identities: Optional[List[float]] = None

    def _load_and_validate_structure(self) -> None:
        """
        Helper function for loading and validating the input structure.
        Called before each `self.sample()` to allow for dynamically changing structures.
        """
        # Load and convert structure input to ProteinStructure if needed.
        if isinstance(self.config_structure, ProteinStructure):
            self.structure = self.config_structure
        else:
            self.structure = ProteinStructure(structure_filepath_or_content=self.config_structure)

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

    def assign(self, assigned_segment: Segment) -> None:
        """Assign a Segment to this generator.

        Validates that the structure is compatible with the segment and sets up
        chain information. Unlike mutation generators, does not initialize a
        random starting sequence since ProteinMPNN generates from structure.

        Args:
            assigned_segment: The Segment to assign to this generator.

        Raises:
            ValueError: If chain_ids contains chains not present in the structure.
        """
        super().assign(assigned_segment)
        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True

    def sample(self) -> None:
        """Generate protein sequences conditioned on the assigned structure.

        Uses ProteinMPNN to design sequences for all candidates in the batch.
        The number of sequences generated equals the number of candidate
        sequences in the assigned segment.

        After sampling, per-sequence metrics are available via:

        - ``self._last_perplexities``: ProteinMPNN perplexity scores
        - ``self._last_sequence_identities``: Sequence identity to original

        The above are also added to the sequence metadata.

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
        tool_config = InverseFoldingConfig(
            num_sequences_to_generate=num_candidates,
            temperature=self.temperature,
            fixed_positions=self.fixed_positions,
            unallowed_amino_acids=self.unallowed_amino_acids,
            seed=self.seed,
            device=self.device,
            keep_on_gpu=True,  # Keep for repeated calls.
            verbose=self.verbose,
        )

        result = run_proteinmpnn_sample(inputs=tool_input, config=tool_config)

        # Extract sequences and metrics from first (only) structure result.
        designed = result.designed_sequences[0]
        generated_sequences = designed.sequences

        # Store metrics for potential downstream use.
        self._last_perplexities = designed.mpnn_perplexity
        self._last_sequence_identities = designed.sequence_identity

        # Update candidate sequences.
        for idx, sequence in enumerate(generated_sequences):
            if idx < len(self._assigned_segment.candidate_sequences):
                candidate = self._assigned_segment.candidate_sequences[idx]
                candidate.sequence = sequence
                candidate._metadata.update({
                    "proteinmpnn_perplexity": self._last_perplexities[idx],
                    "proteinmpnn_sequence_identity": self._last_sequence_identities[idx],
                })

    @property
    def last_perplexities(self) -> Optional[List[float]]:
        """Get ProteinMPNN perplexity scores from the last sample() call.

        Lower perplexity indicates higher model confidence in the designed sequence.

        Returns:
            List of perplexity values, one per generated sequence, or None if
            sample() has not been called.
        """
        return self._last_perplexities

    @property
    def last_sequence_identities(self) -> Optional[List[float]]:
        """Get sequence identity to original structure from the last sample() call.

        Measures similarity between designed sequences and the sequence in the
        input PDB structure.

        Returns:
            List of sequence identity values (0-1), one per generated sequence,
            or None if sample() has not been called.
        """
        return self._last_sequence_identities
