"""Overall protein quality constraint function."""

from typing import Any

import numpy as np
from proto_tools import ProdigalConfig, ProdigalInput, run_prodigal_prediction
from pydantic import model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_quality.balanced_aa_constraint import (
    BalancedAaConfig,
    balanced_aa_constraint,
)
from proto_language.constraint.protein_quality.protein_complexity_constraint import (
    ProteinComplexityConfig,
    protein_complexity_constraint,
)
from proto_language.constraint.protein_quality.protein_diversity_constraint import (
    ProteinDiversityConfig,
    protein_diversity_constraint,
)
from proto_language.constraint.protein_quality.protein_repetitiveness_constraint import (
    ProteinRepetitivenessConfig,
    protein_repetitiveness_constraint,
)
from proto_language.constraint.sequence_composition.sequence_length_constraint import (
    SequenceLengthConfig,
    sequence_length_constraint,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import BaseConfig, ConfigField


class ProteinQualitySubConfig(BaseConfig):
    """Configuration for individual protein quality sub-constraints.

    This configuration class consolidates all parameters for the various protein
    quality sub-constraints into a single, flat structure.

    Each sub-constraint can be independently enabled or disabled using its
    corresponding ``enable_*`` toggle. When enabled, the appropriate parameters
    must be provided. The configuration includes helper methods (``get_*_config``)
    that build the underlying constraint-specific config objects (e.g.,
    ``SequenceLengthConfig``) from the flat parameter structure.

    All sub-constraints evaluate sequences on a 0.0-1.0 scale where 0.0 indicates
    perfect satisfaction and higher values indicate increasing violation. The
    final score is the average across all enabled sub-constraints, clipped to
    [0.0, 1.0]. Use the native ``Constraint(threshold=...)`` parameter for
    pass/fail filtering.

    Attributes:
        enable_length (bool): Toggle to include sequence length constraint. When
            True, you must specify either a length range (``length_min_length`` +
            ``length_max_length``) or a target length (``length_target_length``).
            Default: False.
        length_min_length (int | None): Minimum acceptable protein length in
            amino acids. Must be used with ``length_max_length`` for range-based
            validation. Cannot be combined with ``length_target_length``. Must be
            greater than 0. Default: None. Advanced parameter.
        length_max_length (int | None): Maximum acceptable protein length in
            amino acids. Must be used with ``length_min_length`` for range-based
            validation. Cannot be combined with ``length_target_length``. Must be
            greater than 0 and should be >= ``length_min_length``. Default: None.
            Advanced parameter.
        length_target_length (int | None): Exact target protein length in amino
            acids. Alternative to range mode; cannot be combined with
            ``length_min_length`` or ``length_max_length``. Sequences are penalized
            based on their distance from this target. Must be greater than 0.
            Default: None. Advanced parameter.
        enable_complexity (bool): Toggle to include segmasker-based low-complexity
            detection. When True, uses segmasker to identify low-complexity regions
            (e.g., homopolymeric runs, simple repeats) and penalizes sequences
            exceeding the complexity threshold. Requires segmasker to be installed
            and accessible. Default: False.
        complexity_max_low_complexity (float): Maximum acceptable fraction of
            residues identified as low-complexity by segmasker. Valid range: 0.0-1.0.
            Lower values enforce stricter complexity requirements. Default: 0.2.
            Advanced parameter.
        enable_repetitiveness (bool): Toggle to include k-mer repetitiveness
            constraint. When True, analyzes the sequence for repeated k-mer
            patterns and penalizes sequences with excessive repetition. Checks
            k-mers of sizes from ``repetitiveness_min_repeat_length`` up to
            ``repetitiveness_min_repeat_length + 7``. Default: False.
        repetitiveness_max_repetitiveness (float): Maximum allowed fraction of
            sequence covered by repeated k-mers. Valid range: 0.0-1.0. Lower values
            enforce stricter anti-repetition requirements. Default: 0.1.
            Advanced parameter.
        repetitiveness_min_repeat_length (int): Smallest k-mer size to consider
            as a potential repeat. The analysis examines k-mers from this size
            up to this size + 7. Must be >= 1. Lower values detect shorter repeats
            but are more computationally intensive. Default: 1. Advanced parameter.
        enable_diversity (bool): Toggle to include amino acid diversity constraint.
            When True, requires the sequence to contain a minimum fraction of the
            20 standard amino acid types. Penalizes sequences with low amino acid
            alphabet usage. Default: False.
        diversity_min_diversity (float): Minimum acceptable fraction of unique
            amino acid types, calculated as (unique amino acids / 20). Valid range:
            0.0-1.0. Higher values enforce greater amino acid diversity. Default: 0.7.
            Advanced parameter.
        enable_balanced_aas (bool): Toggle to include balanced amino acid
            representation constraint. When True, requires each amino acid type
            to appear above a minimum frequency threshold, with allowance for a
            limited number of underrepresented amino acids. Complements the
            diversity constraint by checking frequency in addition to presence.
            Default: False.
        balanced_min_aa_frequency (float): Minimum acceptable relative frequency
            for any amino acid type in the sequence. Valid range: 0.0-1.0.
            Amino acids below this threshold are considered "underrepresented."
            Default: 0.02. Advanced parameter.
        balanced_max_underrepresented_count (int): Maximum acceptable number of
            amino acid types that can fall below ``balanced_min_aa_frequency``
            before the sequence is penalized. Valid range: 0-20. Default: 3.
            Advanced parameter.
    """

    enable_length: bool = ConfigField(
        default=False,
        title="Enable Length",
        description="Include the sequence length constraint; provide min/max or target values below.",
    )
    length_min_length: int | None = ConfigField(
        default=None,
        gt=0,
        title="Length Minimum",
        description="Minimum acceptable protein length (amino acids). Used with length_max_length.",
    )
    length_max_length: int | None = ConfigField(
        default=None,
        gt=0,
        title="Length Maximum",
        description="Maximum acceptable protein length (amino acids). Used with length_min_length.",
    )
    length_target_length: int | None = ConfigField(
        default=None,
        gt=0,
        title="Length Target",
        description="Exact target protein length (amino acids). Alternative to range mode.",
    )

    enable_complexity: bool = ConfigField(
        default=False,
        title="Enable Complexity",
        description="Include the segmasker-based low-complexity sub-constraint.",
    )
    complexity_max_low_complexity: float = ConfigField(
        default=0.2,
        ge=0.0,
        le=1.0,
        title="Max Low-Complexity Fraction",
        description="Maximum acceptable fraction of low-complexity residues.",
    )
    enable_repetitiveness: bool = ConfigField(
        default=False,
        title="Enable Repetitiveness",
        description="Include the k-mer repetitiveness constraint.",
    )
    repetitiveness_max_repetitiveness: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Max Repetitiveness",
        description="Maximum allowed fraction of sequence covered by repeated k-mers.",
    )
    repetitiveness_min_repeat_length: int = ConfigField(
        default=1,
        ge=1,
        title="Minimum Repeat Length",
        description="Smallest k-mer size to treat as a repeat (analyzes up to +7 beyond this).",
    )

    enable_diversity: bool = ConfigField(
        default=False,
        title="Enable Diversity",
        description="Include the amino acid diversity sub-constraint.",
    )
    diversity_min_diversity: float = ConfigField(
        default=0.7,
        ge=0.0,
        le=1.0,
        title="Minimum Diversity",
        description="Minimum acceptable fraction of unique amino acid types (unique AAs / 20).",
    )

    enable_balanced_aas: bool = ConfigField(
        default=False,
        title="Enable Balanced AAs",
        description="Include the balanced amino acid representation constraint.",
    )
    balanced_min_aa_frequency: float = ConfigField(
        default=0.02,
        ge=0.0,
        le=1.0,
        title="Minimum AA Frequency",
        description="Minimum acceptable relative frequency for any amino acid type.",
    )
    balanced_max_underrepresented_count: int = ConfigField(
        default=3,
        ge=0,
        le=20,
        title="Max Underrepresented Count",
        description="Maximum acceptable number of amino acid types falling below the frequency threshold.",
    )

    def get_length_config(self) -> SequenceLengthConfig | None:
        """Build the SequenceLengthConfig if enabled."""
        if not self.enable_length:
            return None
        params = {
            "min_length": self.length_min_length,
            "max_length": self.length_max_length,
            "target_length": self.length_target_length,
        }
        filtered = {k: v for k, v in params.items() if v is not None}
        if not filtered:
            raise ValueError("Sequence length constraint enabled but no min/max or target values were provided.")
        return SequenceLengthConfig(**filtered)

    def get_complexity_config(self) -> ProteinComplexityConfig | None:
        """Build the ProteinComplexityConfig if enabled."""
        if not self.enable_complexity:
            return None
        return ProteinComplexityConfig(max_low_complexity=self.complexity_max_low_complexity)

    def get_repetitiveness_config(self) -> ProteinRepetitivenessConfig | None:
        """Build the ProteinRepetitivenessConfig if enabled."""
        if not self.enable_repetitiveness:
            return None
        return ProteinRepetitivenessConfig(
            max_repetitiveness=self.repetitiveness_max_repetitiveness,
            min_repeat_length=self.repetitiveness_min_repeat_length,
        )

    def get_diversity_config(self) -> ProteinDiversityConfig | None:
        """Build the ProteinDiversityConfig if enabled."""
        if not self.enable_diversity:
            return None
        return ProteinDiversityConfig(min_diversity=self.diversity_min_diversity)

    def get_balanced_config(self) -> BalancedAaConfig | None:
        """Build the BalancedAaConfig if enabled."""
        if not self.enable_balanced_aas:
            return None
        return BalancedAaConfig(
            min_aa_frequency=self.balanced_min_aa_frequency,
            max_underrepresented_count=self.balanced_max_underrepresented_count,
        )


class OverallProteinQualityConfig(BaseConfig):
    """Configuration for the overall protein quality constraint.

    This configuration class orchestrates multiple protein quality sub-constraints
    that can be enabled or disabled individually. It provides a flexible framework
    for comprehensive protein quality assessment by combining various metrics
    including sequence length, structural complexity, repetitiveness, amino acid
    diversity, and balanced amino acid representation.

    The configuration uses a nested structure where all sub-constraint parameters
    are exposed through a single ``protein_quality_config`` attribute of type
    ``ProteinQualitySubConfig``. This design allows for easy serialization in
    UI/API schemas while maintaining clear organization of constraint-specific
    parameters.

    At least one sub-constraint must be enabled for the configuration to be valid.
    This is enforced through a model validator that runs after initialization.

    Attributes:
        protein_quality_config (ProteinQualitySubConfig): Nested configuration
            object containing all parameters for individual protein quality checks.
            See ``ProteinQualitySubConfig`` for detailed parameter descriptions.
            This includes toggles for each sub-constraint and
            constraint-specific parameters.

    Raises:
        ValueError: If no sub-constraints are enabled (i.e., all ``enable_*``
            flags in ``protein_quality_config`` are False). At least one
            sub-constraint must be specified for meaningful quality assessment.

    Note:
        The nested ``protein_quality_config`` provides access to:

        - **Length constraint**: Validates protein length against min/max range
          or target value
        - **Complexity constraint**: Detects low-complexity regions using segmasker
        - **Repetitiveness constraint**: Identifies repeated k-mer patterns
        - **Diversity constraint**: Ensures adequate amino acid type diversity
        - **Balanced amino acids constraint**: Checks for underrepresented amino
          acid types

        Each sub-constraint can be independently enabled/disabled and configured
        with specific parameters. See ``ProteinQualitySubConfig`` documentation
        for complete parameter details.

        For more details, see:
            - ``ProteinQualitySubConfig``: Detailed documentation of all
              sub-constraint parameters and configuration options
            - ``overall_protein_quality_constraint``: The constraint function
              that uses this configuration
            - ``SequenceLengthConfig``: Configuration for length constraint
            - ``ProteinComplexityConfig``: Configuration for complexity constraint
            - ``ProteinRepetitivenessConfig``: Configuration for repetitiveness
              constraint
            - ``ProteinDiversityConfig``: Configuration for diversity constraint
            - ``BalancedAaConfig``: Configuration for balanced amino acids
              constraint
    """

    protein_quality_config: ProteinQualitySubConfig = ConfigField(
        title="Protein Quality Config",
        description="Nested configuration for protein quality checks",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "OverallProteinQualityConfig":
        """Validate that at least one sub-constraint is specified."""
        sub_config = self.protein_quality_config
        if not any(
            [
                sub_config.enable_length,
                sub_config.enable_complexity,
                sub_config.enable_repetitiveness,
                sub_config.enable_diversity,
                sub_config.enable_balanced_aas,
            ]
        ):
            raise ValueError("At least one protein quality sub-constraint must be specified")
        return self


@constraint(
    key="overall-protein-quality",
    label="Overall Protein Quality",
    config=OverallProteinQualityConfig,
    description="Evaluate overall protein quality using multiple sub-constraints",
    tools_called=["prodigal-prediction", "segmasker-score"],
    category="protein quality",
    supported_sequence_types=["dna", "protein"],
)
def overall_protein_quality_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: OverallProteinQualityConfig
) -> list[ConstraintOutput]:
    """Evaluate overall protein quality using multiple configurable sub-constraints.

    This constraint function provides a comprehensive assessment of protein quality
    by evaluating multiple aspects including sequence length, structural complexity,
    repetitiveness, amino acid diversity, and balanced amino acid representation.
    For DNA sequences, it first predicts protein-coding regions using Prodigal,
    then evaluates all predicted proteins. For protein sequences, it evaluates
    them directly.

    The function aggregates scores from enabled sub-constraints by averaging them
    and clipping to [0.0, 1.0]. Use the native ``Constraint(threshold=...)``
    parameter for pass/fail filtering.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one DNA or protein sequence.
            For DNA sequences, ORF prediction is performed automatically using
            Prodigal before quality assessment.

        config (OverallProteinQualityConfig): Configuration object containing a
            ``protein_quality_config`` attribute of type ``ProteinQualitySubConfig``,
            which exposes the following parameters:

            **Length constraint (optional):**
            - ``enable_length`` (bool): Toggle for sequence length constraint.
            - ``length_min_length`` (int): Minimum acceptable protein length in
              amino acids (used with ``length_max_length``).
            - ``length_max_length`` (int): Maximum acceptable protein length in
              amino acids (used with ``length_min_length``).
            - ``length_target_length`` (int): Exact target protein length in amino
              acids (alternative to range mode).

            **Complexity constraint (optional):**
            - ``enable_complexity`` (bool): Toggle for segmasker-based low-complexity
              detection.
            - ``complexity_max_low_complexity`` (float): Maximum acceptable fraction
              of low-complexity residues (0.0-1.0, default: 0.2).
            **Repetitiveness constraint (optional):**
            - ``enable_repetitiveness`` (bool): Toggle for k-mer repetitiveness
              constraint.
            - ``repetitiveness_max_repetitiveness`` (float): Maximum allowed fraction
              of sequence covered by repeated k-mers (0.0-1.0, default: 0.1).
            - ``repetitiveness_min_repeat_length`` (int): Smallest k-mer size to
              treat as a repeat, analyzes up to +7 beyond this (default: 1).

            **Diversity constraint (optional):**
            - ``enable_diversity`` (bool): Toggle for amino acid diversity constraint.
            - ``diversity_min_diversity`` (float): Minimum acceptable fraction of
              unique amino acid types, calculated as unique_AAs / 20 (0.0-1.0,
              default: 0.7).

            **Balanced amino acids constraint (optional):**
            - ``enable_balanced_aas`` (bool): Toggle for balanced amino acid
              representation constraint.
            - ``balanced_min_aa_frequency`` (float): Minimum acceptable relative
              frequency for any amino acid type (0.0-1.0, default: 0.02).
            - ``balanced_max_underrepresented_count`` (int): Maximum acceptable
              number of amino acid types falling below frequency threshold
              (0-20, default: 3).

            At least one sub-constraint must be enabled, or a ``ValueError`` is raised
            during configuration validation.

    Returns:
        list[ConstraintOutput]: One result per sequence. Scores range from 0.0 (best)
            to 1.0 (worst) and represent the average of all enabled sub-constraint
            scores, clipped to [0.0, 1.0]. For DNA sequences, the score reflects
            the average quality across all predicted proteins. ``metadata`` carries:

            **For DNA sequences:**

            - ``prodigal_proteins``: DataFrame of predicted proteins from Prodigal,
              containing columns for protein ID, sequence, length, etc.
            - ``prodigal_protein_count``: Integer count of predicted ORFs
            - ``predicted_protein_count``: Integer count of proteins (same as
              prodigal_protein_count)
            - ``avg_constraint_score``: Float average quality score across all
              predicted proteins
            - ``protein_quality_details``: List of dictionaries, one per predicted
              protein, each containing:

              - ``protein_id``: String identifier from Prodigal
              - ``length``: Integer protein length in amino acids
              - ``avg_constraint_score``: Float average across enabled constraints
              - ``quality_scores``: Dictionary mapping constraint names to scores
              - ``metadata``: Dictionary of additional constraint-specific metadata

            **For protein sequences:**

            - ``protein_quality_scores``: Dictionary mapping constraint names (e.g.,
              "length", "complexity", "repetitiveness", "diversity", "balanced_aas")
              to their individual scores
            - ``avg_constraint_score``: Float average across all enabled constraints

    Raises:
        ValueError: If no sub-constraints are enabled in the configuration, or if
            length constraint is enabled but no min/max or target values are provided.
        AssertionError: If any sequence in the input list is not a DNA or PROTEIN
            sequence type.

    Examples:
        Using all available constraints with custom thresholds:

        >>> quality_config = ProteinQualitySubConfig(
        ...     enable_length=True,
        ...     length_target_length=300,
        ...     enable_complexity=True,
        ...     complexity_max_low_complexity=0.25,
        ...     enable_repetitiveness=True,
        ...     repetitiveness_max_repetitiveness=0.08,
        ...     repetitiveness_min_repeat_length=3,
        ...     enable_diversity=True,
        ...     diversity_min_diversity=0.75,
        ...     enable_balanced_aas=True,
        ...     balanced_min_aa_frequency=0.03,
        ...     balanced_max_underrepresented_count=2,
        ... )
        >>> overall_cfg = OverallProteinQualityConfig(protein_quality_config=quality_config)
        >>> protein_seq = Sequence("MKYIVAVAG...", "protein")
        >>> results = overall_protein_quality_constraint([(protein_seq,)], overall_cfg)
    """
    protein_quality_config = config.protein_quality_config
    length_config = protein_quality_config.get_length_config()
    complexity_config = protein_quality_config.get_complexity_config()
    repetitiveness_config = protein_quality_config.get_repetitiveness_config()
    diversity_config = protein_quality_config.get_diversity_config()
    balanced_config = protein_quality_config.get_balanced_config()

    # Build per-index result placeholders to preserve original order
    final_results: list[ConstraintOutput | None] = [None] * len(input_sequences)

    dna_indices = [i for i, (seq,) in enumerate(input_sequences) if seq.sequence_type == "dna"]
    protein_indices = [i for i, (seq,) in enumerate(input_sequences) if seq.sequence_type == "protein"]

    if dna_indices:
        dna_sequences = [input_sequences[i][0] for i in dna_indices]
        prodigal_input = ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences])
        prodigal_config = ProdigalConfig()
        batch_result = run_prodigal_prediction(inputs=prodigal_input, config=prodigal_config)

        for original_idx, proteins_list, num_genes in zip(
            dna_indices, batch_result.predicted_orfs, batch_result.num_orfs_per_sequence, strict=False
        ):
            orf_dicts = [orf.model_dump() for orf in proteins_list]
            metadata: dict[str, Any] = {
                "prodigal_proteins": orf_dicts or None,
                "prodigal_protein_count": num_genes,
            }

            if len(proteins_list) == 0:
                metadata["predicted_protein_count"] = 0
                metadata["protein_quality_details"] = []
                final_results[original_idx] = ConstraintOutput(score=1.0, metadata=metadata)
                continue

            predicted_protein_seqs = [Sequence(orf.amino_acid_sequence, "protein") for orf in proteins_list]
            predicted_protein_input_seqs: list[tuple[Sequence, ...]] = [(seq,) for seq in predicted_protein_seqs]

            sub_results = _run_sub_constraints(
                predicted_protein_input_seqs,
                length_config,
                complexity_config,
                repetitiveness_config,
                diversity_config,
                balanced_config,
            )

            if sub_results:
                constraint_score_matrix = np.array([[r.score for r in rs] for rs in sub_results.values()])
                avg_scores = constraint_score_matrix.mean(axis=0)
            else:
                avg_scores = np.zeros(len(predicted_protein_seqs))

            protein_quality_details = []
            for prot_idx, orf in enumerate(proteins_list):
                individual_scores = {name: rs[prot_idx].score for name, rs in sub_results.items()}
                protein_metadata: dict[str, Any] = {}
                for rs in sub_results.values():
                    protein_metadata.update(rs[prot_idx].metadata)
                protein_quality_details.append(
                    {
                        "protein_id": orf.id,
                        "length": orf.amino_acid_length,
                        "avg_constraint_score": float(avg_scores[prot_idx]),
                        "quality_scores": individual_scores,
                        "metadata": protein_metadata,
                    }
                )

            overall_avg_protein_score = float(avg_scores.mean())
            metadata["predicted_protein_count"] = len(proteins_list)
            metadata["avg_constraint_score"] = overall_avg_protein_score
            metadata["protein_quality_details"] = protein_quality_details

            final_results[original_idx] = ConstraintOutput(
                score=float(np.clip(overall_avg_protein_score, 0.0, 1.0)),
                metadata=metadata,
            )

    if protein_indices:
        protein_input_seqs = [input_sequences[i] for i in protein_indices]
        sub_results = _run_sub_constraints(
            protein_input_seqs,
            length_config,
            complexity_config,
            repetitiveness_config,
            diversity_config,
            balanced_config,
        )

        if sub_results:
            constraint_score_matrix = np.array([[r.score for r in rs] for rs in sub_results.values()])
            avg_scores = constraint_score_matrix.mean(axis=0)
        else:
            avg_scores = np.zeros(len(protein_input_seqs))

        clipped_scores = np.clip(avg_scores, 0.0, 1.0)
        for local_idx, original_idx in enumerate(protein_indices):
            individual_scores = {name: rs[local_idx].score for name, rs in sub_results.items()}
            final_results[original_idx] = ConstraintOutput(
                score=float(clipped_scores[local_idx]),
                metadata={
                    "protein_quality_scores": individual_scores,
                    "avg_constraint_score": float(avg_scores[local_idx]),
                },
            )

    assert all(r is not None for r in final_results)  # noqa: S101 -- mypy narrowing
    return [r for r in final_results if r is not None]


def _run_sub_constraints(
    input_seqs: list[tuple[Sequence, ...]],
    length_config: SequenceLengthConfig | None,
    complexity_config: ProteinComplexityConfig | None,
    repetitiveness_config: ProteinRepetitivenessConfig | None,
    diversity_config: ProteinDiversityConfig | None,
    balanced_config: BalancedAaConfig | None,
) -> dict[str, list[ConstraintOutput]]:
    """Invoke enabled sub-constraints on the given inputs and return results per name."""
    sub_results: dict[str, list[ConstraintOutput]] = {}
    if length_config:
        sub_results["length"] = sequence_length_constraint(input_seqs, config=length_config)
    if complexity_config:
        sub_results["complexity"] = protein_complexity_constraint(input_seqs, config=complexity_config)
    if repetitiveness_config:
        sub_results["repetitiveness"] = protein_repetitiveness_constraint(input_seqs, config=repetitiveness_config)
    if diversity_config:
        sub_results["diversity"] = protein_diversity_constraint(input_seqs, config=diversity_config)
    if balanced_config:
        sub_results["balanced_aas"] = balanced_aa_constraint(input_seqs, config=balanced_config)
    return sub_results
