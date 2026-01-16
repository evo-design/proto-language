"""Overall protein quality constraint function."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import model_validator

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import sequence_length_constraint, SequenceLengthConfig  
from proto_language.language.constraint.protein_quality.protein_complexity_constraint import protein_complexity_constraint, ProteinComplexityConfig
from proto_language.language.constraint.protein_quality.protein_repetitiveness_constraint import protein_repetitiveness_constraint, ProteinRepetitivenessConfig
from proto_language.language.constraint.protein_quality.protein_diversity_constraint import protein_diversity_constraint, ProteinDiversityConfig
from proto_language.language.constraint.protein_quality.balanced_aa_constraint import balanced_aa_constraint, BalancedAaConfig

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
    ``quality_threshold`` parameter determines the maximum acceptable average
    score across all enabled constraints for a protein to be classified as
    "high quality."
    
    Attributes:
        quality_threshold (float): Maximum acceptable average constraint score for
            high-quality classification. Sequences with average scores ≤ this value
            receive a final score of 0.0, while those exceeding it receive their
            actual average score (capped at 1.0). Valid range: 0.0-1.0. Lower
            thresholds enforce stricter quality requirements. Example: 0.2 means
            proteins must satisfy constraints with an average score ≤ 0.2 to be
            considered high quality.
        
        **Length Constraint Parameters:**
        
        enable_length (bool): Toggle to include sequence length constraint. When
            True, you must specify either a length range (``length_min_length`` +
            ``length_max_length``) or a target length (``length_target_length``).
            Default: False.
        
        length_min_length (Optional[int]): Minimum acceptable protein length in
            amino acids. Must be used with ``length_max_length`` for range-based
            validation. Cannot be combined with ``length_target_length``. Must be
            greater than 0.
            Default: None. Advanced parameter.
        
        length_max_length (Optional[int]): Maximum acceptable protein length in
            amino acids. Must be used with ``length_min_length`` for range-based
            validation. Cannot be combined with ``length_target_length``. Must be
            greater than 0 and should be ≥ ``length_min_length``. Default: None. 
            Advanced parameter.
        
        length_target_length (Optional[int]): Exact target protein length in amino
            acids. Alternative to range mode; cannot be combined with
            ``length_min_length`` or ``length_max_length``. Sequences are penalized
            based on their distance from this target. Must be greater than 0.
            Default: None. Advanced parameter.
        
        **Complexity Constraint Parameters:**
        
        enable_complexity (bool): Toggle to include segmasker-based low-complexity
            detection. When True, uses segmasker to identify low-complexity regions
            (e.g., homopolymeric runs, simple repeats) and penalizes sequences
            exceeding the complexity threshold. Requires segmasker to be installed
            and accessible. Default: False.
        
        complexity_max_low_complexity (float): Maximum acceptable fraction of
            residues identified as low-complexity by segmasker. Valid range: 0.0-1.0.
            Lower values enforce stricter complexity requirements. Default: 0.2. 
            Advanced parameter.
        
        complexity_segmasker_path (str): Path to the segmasker executable for
            low-complexity analysis. Can be an absolute path or a command name
            if segmasker is in PATH. Default: "segmasker" (assumes it's in PATH).
            Hidden parameter.
        
        **Repetitiveness Constraint Parameters:**
        
        enable_repetitiveness (bool): Toggle to include k-mer repetitiveness
            constraint. When True, analyzes the sequence for repeated k-mer
            patterns and penalizes sequences with excessive repetition. Checks
            k-mers of sizes from ``repetitiveness_min_repeat_length`` up to
            ``repetitiveness_min_repeat_length + 7``. Default: False.
        
        repetitiveness_max_repetitiveness (float): Maximum allowed fraction of
            sequence covered by repeated k-mers. Valid range: 0.0-1.0. Lower values
            enforce stricter anti-repetition requirements. Default: 0.1. Advanced parameter.
        
        repetitiveness_min_repeat_length (int): Smallest k-mer size to consider
            as a potential repeat. The analysis examines k-mers from this size
            up to this size + 7. Must be ≥ 1. Lower values detect shorter repeats
            but are more computationally intensive. Default: 1. Advanced parameter.
        
        **Diversity Constraint Parameters:**
        
        enable_diversity (bool): Toggle to include amino acid diversity constraint.
            When True, requires the sequence to contain a minimum fraction of the
            20 standard amino acid types. Penalizes sequences with low amino acid
            alphabet usage. Default: False.
        
        diversity_min_diversity (float): Minimum acceptable fraction of unique
            amino acid types, calculated as (unique amino acids / 20). Valid range:
            0.0-1.0. Higher values enforce greater amino acid diversity. Default: 0.7.
            Advanced parameter.
        
        **Balanced Amino Acids Constraint Parameters:**
        
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

    quality_threshold: float = ConfigField(
        ge=0.0,
        le=1.0,
        title="Quality Threshold",
        description="Maximum acceptable average constraint score for high-quality classification. "
        "Sequences with scores ≤ threshold receive 0.0; sequences exceeding threshold receive their average score.",
    )

    enable_length: bool = ConfigField(
        default=False,
        title="Enable Sequence Length Constraint",
        description="Toggle to include the sequence length constraint. Provide min/max or target values below.",
    )
    length_min_length: Optional[int] = ConfigField(
        default=None,
        gt=0,
        title="Length Minimum",
        description="Minimum acceptable protein length (amino acids). Used with length_max_length.",
        advanced=True,
    )
    length_max_length: Optional[int] = ConfigField(
        default=None,
        gt=0,
        title="Length Maximum",
        description="Maximum acceptable protein length (amino acids). Used with length_min_length.",
        advanced=True,
    )
    length_target_length: Optional[int] = ConfigField(
        default=None,
        gt=0,
        title="Length Target",
        description="Exact target protein length (amino acids). Alternative to range mode.",
        advanced=True,
    )

    enable_complexity: bool = ConfigField(
        default=False,
        title="Enable Complexity Constraint",
        description="Toggle to include segmasker-based low-complexity detection.",
    )
    complexity_max_low_complexity: float = ConfigField(
        default=0.2,
        ge=0.0,
        le=1.0,
        title="Max Low-Complexity Fraction",
        description="Maximum acceptable fraction of low-complexity residues.",
        advanced=True,
    )
    complexity_segmasker_path: str = ConfigField(
        default="segmasker",
        title="Segmasker Path",
        description="Path to the segmasker executable for the complexity check.",
        hidden=True,
    )

    enable_repetitiveness: bool = ConfigField(
        default=False,
        title="Enable Repetitiveness Constraint",
        description="Toggle to include the k-mer repetitiveness constraint.",
    )
    repetitiveness_max_repetitiveness: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Max Repetitiveness",
        description="Maximum allowed fraction of sequence covered by repeated k-mers.",
        advanced=True,
    )
    repetitiveness_min_repeat_length: int = ConfigField(
        default=1,
        ge=1,
        title="Minimum Repeat Length",
        description="Smallest k-mer size to treat as a repeat (analyzes up to +7 beyond this).",
        advanced=True,
    )

    enable_diversity: bool = ConfigField(
        default=False,
        title="Enable Diversity Constraint",
        description="Toggle to include the amino acid diversity constraint.",
    )
    diversity_min_diversity: float = ConfigField(
        default=0.7,
        ge=0.0,
        le=1.0,
        title="Minimum Diversity",
        description="Minimum acceptable fraction of unique amino acid types (unique AAs / 20).",
        advanced=True,
    )

    enable_balanced_aas: bool = ConfigField(
        default=False,
        title="Enable Balanced Amino Acids Constraint",
        description="Toggle to include the balanced amino acid representation constraint.",
    )
    balanced_min_aa_frequency: float = ConfigField(
        default=0.02,
        ge=0.0,
        le=1.0,
        title="Minimum AA Frequency",
        description="Minimum acceptable relative frequency for any amino acid type.",
        advanced=True,
    )
    balanced_max_underrepresented_count: int = ConfigField(
        default=3,
        ge=0,
        le=20,
        title="Max Underrepresented Count",
        description="Maximum acceptable number of amino acid types falling below the frequency threshold.",
        advanced=True,
    )

    def get_length_config(self) -> Optional[SequenceLengthConfig]:
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
            raise ValueError(
                "Sequence length constraint enabled but no min/max or target values were provided."
            )
        return SequenceLengthConfig(**filtered)

    def get_complexity_config(self) -> Optional[ProteinComplexityConfig]:
        """Build the ProteinComplexityConfig if enabled."""
        if not self.enable_complexity:
            return None
        return ProteinComplexityConfig(
            max_low_complexity=self.complexity_max_low_complexity,
            segmasker_path=self.complexity_segmasker_path,
        )

    def get_repetitiveness_config(self) -> Optional[ProteinRepetitivenessConfig]:
        """Build the ProteinRepetitivenessConfig if enabled."""
        if not self.enable_repetitiveness:
            return None
        return ProteinRepetitivenessConfig(
            max_repetitiveness=self.repetitiveness_max_repetitiveness,
            min_repeat_length=self.repetitiveness_min_repeat_length,
        )

    def get_diversity_config(self) -> Optional[ProteinDiversityConfig]:
        """Build the ProteinDiversityConfig if enabled."""
        if not self.enable_diversity:
            return None
        return ProteinDiversityConfig(min_diversity=self.diversity_min_diversity)

    def get_balanced_config(self) -> Optional[BalancedAaConfig]:
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
            This includes the quality threshold, toggles for each sub-constraint,
            and constraint-specific parameters.
    
    Raises:
        ValueError: If no sub-constraints are enabled (i.e., all ``enable_*``
            flags in ``protein_quality_config`` are False). At least one
            sub-constraint must be specified for meaningful quality assessment.
    
    Note:
        The nested ``protein_quality_config`` provides access to:
        
        - **Quality threshold**: Maximum acceptable average constraint score
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
            - ``ProteinQualitySubConfig``: Detailed documentation of all sub-constraint
            parameters and configuration options
            - ``overall_protein_quality_constraint``: The constraint function that
            uses this configuration
            - ``SequenceLengthConfig``: Configuration for length constraint
            - ``ProteinComplexityConfig``: Configuration for complexity constraint
            - ``ProteinRepetitivenessConfig``: Configuration for repetitiveness constraint
            - ``ProteinDiversityConfig``: Configuration for diversity constraint
            - ``BalancedAaConfig``: Configuration for balanced amino acids constraint
    """
    protein_quality_config: ProteinQualitySubConfig = ConfigField(
        title="Protein Quality Config",
        description="Nested configuration for protein quality checks",
    )

    @model_validator(mode='after')
    def validate_config(self):
        """Validate that at least one sub-constraint is specified."""
        sub_config = self.protein_quality_config
        if not any([
            sub_config.enable_length,
            sub_config.enable_complexity,
            sub_config.enable_repetitiveness,
            sub_config.enable_diversity,
            sub_config.enable_balanced_aas,
        ]):
            raise ValueError("At least one protein quality sub-constraint must be specified")
        return self


@ConstraintRegistry.register(
    key="overall-protein-quality",
    label="Overall Protein Quality",
    config=OverallProteinQualityConfig,
    description="Evaluate overall protein quality using multiple sub-constraints",
    batched=True,
    concatenate=True,
    tools_called=["prodigal", "segmasker"],
    category="protein quality",
    supported_sequence_types=["dna", "protein"],
)
def overall_protein_quality_constraint(sequences: List[Sequence], config: OverallProteinQualityConfig) -> List[float]:
    """Evaluate overall protein quality using multiple configurable sub-constraints.
    
    This constraint function provides a comprehensive assessment of protein quality
    by evaluating multiple aspects including sequence length, structural complexity,
    repetitiveness, amino acid diversity, and balanced amino acid representation.
    For DNA sequences, it first predicts protein-coding regions using Prodigal,
    then evaluates all predicted proteins. For protein sequences, it evaluates
    them directly.
    
    The function aggregates scores from enabled sub-constraints by averaging them,
    then applies a quality threshold to determine if sequences are "high quality."
    Sequences meeting the threshold (average score ≤ threshold) receive a score
    of 0.0, while those exceeding it receive their average constraint score,
    capped at 1.0.
    
    Args:
        sequences (List[Sequence]): List of DNA or protein sequences to evaluate.
            All sequences in the list must be the same type (all DNA or all PROTEIN).
            For DNA sequences, ORF prediction is performed automatically using
            Prodigal before quality assessment.
            
        config (OverallProteinQualityConfig): Configuration object containing a
            ``protein_quality_config`` attribute of type ``ProteinQualitySubConfig``,
            which exposes the following parameters:
            
            - ``quality_threshold`` (float): Maximum acceptable average constraint
              score for high-quality classification (0.0-1.0, default: varies).
              Sequences with average scores ≤ this threshold receive 0.0; those
              exceeding it receive their actual average score.
            
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
            - ``complexity_segmasker_path`` (str): Path to segmasker executable
              (default: "segmasker").
            
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
        List[float]: Constraint scores for each sequence, ranging from 0.0 (best,
            high quality) to 1.0 (worst, poor quality). Scores are calculated as:
            
            - 0.0: Sequence meets quality threshold (average score ≤ threshold)
            - 0.0 < score ≤ 1.0: Sequence exceeds quality threshold, score equals
              the average of all enabled sub-constraint scores, capped at 1.0
            
            For DNA sequences, the score reflects the average quality of all
            predicted proteins. For protein sequences, the score reflects the
            direct quality assessment.

    Raises:
        ValueError: If no sub-constraints are enabled in the configuration, or if
            length constraint is enabled but no min/max or target values are provided.
        AssertionError: If any sequence in the input list is not a DNA or PROTEIN
            sequence type.

    Note:
        This function modifies the input sequences by adding metadata to
        each ``Sequence`` object's ``_metadata`` dictionary. Metadata varies by
        sequence type:
        
        **For DNA sequences:**
        - ``prodigal_proteins``: DataFrame of predicted proteins from Prodigal,
          containing columns for protein ID, sequence, length, etc.
        - ``prodigal_protein_count``: Integer count of predicted ORFs
        - ``predicted_protein_count``: Integer count of proteins (same as
          prodigal_protein_count)
        - ``avg_constraint_score``: Float average quality score across all
          predicted proteins
        - ``is_high_quality``: Boolean indicating if average score ≤ threshold
        - ``protein_quality_details``: List of dictionaries, one per predicted
          protein, each containing:
          
          - ``protein_id``: String identifier from Prodigal
          - ``length``: Integer protein length in amino acids
          - ``is_high_quality``: Boolean for this specific protein
          - ``avg_constraint_score``: Float average across enabled constraints
          - ``quality_scores``: Dictionary mapping constraint names to scores
          - ``metadata``: Dictionary of additional constraint-specific metadata
        
        - ``protein_quality_threshold``: Float threshold value used
        
        **For protein sequences:**
        - ``protein_quality_scores``: Dictionary mapping constraint names (e.g.,
          "length", "complexity", "repetitiveness", "diversity", "balanced_aas")
          to their individual scores
        - ``avg_constraint_score``: Float average across all enabled constraints
        - ``is_high_quality``: Boolean indicating if average score ≤ threshold
        - ``protein_quality_threshold``: Float threshold value used
        
        Each enabled sub-constraint may also add its own specific metadata fields
        to individual proteins, such as amino acid counts, low-complexity regions,
        repeat patterns, etc.

    Examples:        
        Using all available constraints with custom thresholds:
        
        >>> quality_config = ProteinQualitySubConfig(
        ...     quality_threshold=0.15,
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
        >>> scores = overall_protein_quality_constraint([protein_seq], overall_cfg)
    """
    # Extract config parameters
    protein_quality_config = config.protein_quality_config
    length_config = protein_quality_config.get_length_config()
    complexity_config = protein_quality_config.get_complexity_config()
    repetitiveness_config = protein_quality_config.get_repetitiveness_config()
    diversity_config = protein_quality_config.get_diversity_config()
    balanced_config = protein_quality_config.get_balanced_config()

    # Separate DNA and protein sequences (validated by Constraint._validate_sequence_types)
    dna_sequences = [seq for seq in sequences if seq.sequence_type == "dna"]
    protein_sequences = [seq for seq in sequences if seq.sequence_type == "protein"]

    dna_scores = []
    protein_scores = []
    threshold = protein_quality_config.quality_threshold

    if dna_sequences:
        # For DNA sequences: predict proteins first, get predicted proteins using Prodigal
        prodigal_input = ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences])
        prodigal_config = ProdigalConfig()
        batch_result = run_prodigal_prediction(inputs=prodigal_input, config=prodigal_config)

        # Process each DNA sequence's results
        for input_sequence, proteins_list, num_genes in zip(
            dna_sequences,
            batch_result.predicted_orfs,
            batch_result.num_orfs_per_sequence
        ):
            input_sequence._metadata["prodigal_proteins"] = [orf.model_dump() for orf in proteins_list]
            input_sequence._metadata["prodigal_protein_count"] = num_genes

            if len(proteins_list) == 0:
                input_sequence._metadata["predicted_protein_count"] = 0
                input_sequence._metadata["high_quality_protein_count"] = 0
                input_sequence._metadata["high_quality_protein_fraction"] = 0.0
                input_sequence._metadata["protein_quality_details"] = []
                dna_scores.append(1.0)
                continue

            # Convert to Sequence objects for batch constraint evaluation
            predicted_protein_seqs = [
                Sequence(orf.amino_acid_sequence, "protein")
                for orf in proteins_list
            ]

            quality_scores = {}

            if length_config:
                quality_scores["length"] = sequence_length_constraint(
                    predicted_protein_seqs, config=length_config
                )

            if complexity_config:
                quality_scores["complexity"] = protein_complexity_constraint(
                    predicted_protein_seqs, config=complexity_config
                )

            if repetitiveness_config:
                quality_scores["repetitiveness"] = protein_repetitiveness_constraint(
                    predicted_protein_seqs, config=repetitiveness_config
                )

            if diversity_config:
                quality_scores["diversity"] = protein_diversity_constraint(
                    predicted_protein_seqs, config=diversity_config
                )

            if balanced_config:
                quality_scores["balanced_aas"] = balanced_aa_constraint(
                    predicted_protein_seqs, config=balanced_config
                )

            # batched averaging
            if quality_scores:
                constraint_score_matrix = np.array(list(quality_scores.values()))
                avg_scores = constraint_score_matrix.mean(axis=0)
            else:
                avg_scores = np.zeros(len(predicted_protein_seqs))

            # batched quality determination
            is_high_quality = avg_scores <= threshold

            # Build details
            protein_quality_details = []
            for prot_idx, (orf, protein_seq) in enumerate(zip(proteins_list, predicted_protein_seqs)):
                individual_scores = {
                    name: scores[prot_idx] 
                    for name, scores in quality_scores.items()
                }

                protein_quality_details.append({
                    "protein_id": orf.id,
                    "length": orf.amino_acid_length,
                    "is_high_quality": bool(is_high_quality[prot_idx]),
                    "avg_constraint_score": float(avg_scores[prot_idx]),
                    "quality_scores": individual_scores,
                    "metadata": protein_seq._metadata.copy(),
                })

            overall_avg_protein_score = float(avg_scores.mean())
            is_dna_high_quality = overall_avg_protein_score <= threshold

            # Store metadata
            input_sequence._metadata["predicted_protein_count"] = len(proteins_list)
            input_sequence._metadata["avg_constraint_score"] = overall_avg_protein_score
            input_sequence._metadata["is_high_quality"] = is_dna_high_quality
            input_sequence._metadata["protein_quality_details"] = protein_quality_details
            input_sequence._metadata["protein_quality_threshold"] = threshold

            # Calculate score
            if is_dna_high_quality:
                score = 0.0
            else:
                score = min(1.0, max(0.0, overall_avg_protein_score))

            dna_scores.append(score)

    if protein_sequences:
        quality_scores = {}

        if length_config:
            quality_scores["length"] = sequence_length_constraint(
                protein_sequences, config=length_config
            )

        if complexity_config:
            quality_scores["complexity"] = protein_complexity_constraint(
                protein_sequences, config=complexity_config
            )

        if repetitiveness_config:
            quality_scores["repetitiveness"] = protein_repetitiveness_constraint(
                protein_sequences, config=repetitiveness_config
            )

        if diversity_config:
            quality_scores["diversity"] = protein_diversity_constraint(
                protein_sequences, config=diversity_config
            )

        if balanced_config:
            quality_scores["balanced_aas"] = balanced_aa_constraint(
                protein_sequences, config=balanced_config
            )

        if quality_scores:
            constraint_score_matrix = np.array(list(quality_scores.values()))
            avg_scores = constraint_score_matrix.mean(axis=0)
        else:
            avg_scores = np.zeros(len(protein_sequences))

        # batched quality determination
        is_high_quality = avg_scores <= threshold

        # batched score calculation
        protein_scores = np.where(
            is_high_quality,
            0.0,
            np.clip(avg_scores, 0.0, 1.0)
        ).tolist()

        # Store metadata
        for seq_idx, input_sequence in enumerate(protein_sequences):
            individual_scores = {
                name: scores[seq_idx] 
                for name, scores in quality_scores.items()
            }

            input_sequence._metadata["protein_quality_scores"] = individual_scores
            input_sequence._metadata["avg_constraint_score"] = float(avg_scores[seq_idx])
            input_sequence._metadata["is_high_quality"] = bool(is_high_quality[seq_idx])
            input_sequence._metadata["protein_quality_threshold"] = threshold

    final_scores = []
    dna_idx = 0
    protein_idx = 0

    for seq in sequences:
        if seq.sequence_type == "dna":
            final_scores.append(dna_scores[dna_idx])
            dna_idx += 1
        else:
            final_scores.append(protein_scores[protein_idx])
            protein_idx += 1

    return final_scores
