"""Overall protein quality constraint function."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from pydantic import Field, model_validator

from ...base import Sequence, SequenceType
from ...base.config import BaseConfig
from ..registry import ConstraintRegistry
from ....tools.orf_prediction.prodigal import run_prodigal_prediction, ProdigalConfig
from .protein_length_constraint import protein_length_constraint
from .protein_complexity_constraint import protein_complexity_constraint
from .protein_repetitiveness_constraint import protein_repetitiveness_constraint
from .protein_diversity_constraint import protein_diversity_constraint
from .balanced_aa_constraint import balanced_aa_constraint

if TYPE_CHECKING:
    from .protein_length_constraint import ProteinLengthConfig
    from .protein_complexity_constraint import ProteinComplexityConfig
    from .protein_repetitiveness_constraint import ProteinRepetitivenessConfig
    from .protein_diversity_constraint import ProteinDiversityConfig
    from .balanced_aa_constraint import BalancedAaConfig
else:
    # Runtime imports to avoid circular dependency issues
    from .protein_length_constraint import ProteinLengthConfig
    from .protein_complexity_constraint import ProteinComplexityConfig
    from .protein_repetitiveness_constraint import ProteinRepetitivenessConfig
    from .protein_diversity_constraint import ProteinDiversityConfig
    from .balanced_aa_constraint import BalancedAaConfig


class ProteinQualitySubConfig(BaseConfig):
    """Nested configuration for individual protein quality checks."""
    length: Optional[ProteinLengthConfig] = Field(default=None, description="Protein length constraints")
    complexity: Optional[ProteinComplexityConfig] = Field(default=None, description="Protein complexity constraints")
    repetitiveness: Optional[ProteinRepetitivenessConfig] = Field(default=None, description="Protein repetitiveness constraints")
    diversity: Optional[ProteinDiversityConfig] = Field(default=None, description="Amino acid diversity constraints")
    balanced_aas: Optional[BalancedAaConfig] = Field(default=None, description="Balanced amino acid constraints")
    quality_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Maximum acceptable constraint score for high quality")


class OverallProteinQualityConfig(BaseConfig):
    """Configuration for overall protein quality constraint."""
    min_high_quality_fraction: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Minimum fraction of predicted proteins that must be high quality (required for DNA input)"
    )
    protein_quality_config: ProteinQualitySubConfig = Field(description="Nested configuration for protein quality checks")
    
    @model_validator(mode='after')
    def validate_config(self):
        """Validate that at least one sub-constraint is specified."""
        sub_config = self.protein_quality_config
        if not any([sub_config.length, sub_config.complexity, sub_config.repetitiveness, 
                    sub_config.diversity, sub_config.balanced_aas]):
            raise ValueError("At least one protein quality sub-constraint must be specified")
        return self


@ConstraintRegistry.register(
    key="overall-protein-quality",
    config=OverallProteinQualityConfig,
    description="Evaluate overall protein quality using multiple sub-constraints",
    vectorized=False,
    concatenate=True
)
def overall_protein_quality_constraint(
    input_sequence: Sequence,
    config: OverallProteinQualityConfig
) -> float:
    """
    Evaluate protein quality either from predicted proteins (DNA input) or directly (protein input).

    For DNA sequences, runs Prodigal first to predict proteins, then checks all predicted
    proteins. For protein sequences, checks the sequence directly.

    Args:
        input_sequence: The DNA or protein sequence to analyze.
        config: Configuration dictionary containing:
            For DNA input:
                - min_high_quality_fraction (float): Minimum fraction of predicted proteins that must be high quality.
                - protein_quality_config (dict): Configuration dictionary with the following structure:
                {
                    "min_high_quality_fraction": 0.8,  # Minimum fraction of predicted proteins that must be high quality (0.0-1.0)
                    "protein_quality_config": {
                        "quality_threshold": 0.1,  # Maximum acceptable constraint score for a protein to be considered "high quality"

                        # Individual protein quality constraints (all optional):
                        "length": {
                            "min_length": 50,     # Minimum acceptable protein length (amino acids)
                            "max_length": 2000    # Maximum acceptable protein length (amino acids)
                        },
                        "complexity": {
                            "max_low_complexity": 0.3,              # Maximum fraction of low-complexity regions (0.0-1.0)
                            "segmasker_path": "segmasker"           # Path to segmasker executable (optional)
                        },
                        "repetitiveness": {
                            "max_repetitiveness": 0.4,              # Maximum acceptable repetitiveness fraction (0.0-1.0)
                            "min_repeat_length": 3                  # Minimum repeat length to consider (optional, default: 3)
                        },
                        "diversity": {
                            "min_diversity": 0.3                    # Minimum acceptable amino acid diversity (0.0-1.0, where 1.0 = all 20 amino acids)
                        },
                        "balanced_aas": {
                            "max_underrepresented": 0.2             # Maximum acceptable fraction of underrepresented amino acids (0.0-1.0)
                        }
                    }
                }

            For protein input:
                - protein_quality_config (dict): Configuration for protein quality checks with the following structure:
                {
                    "protein_quality_config": {
                        "quality_threshold": 0.1,  # Maximum acceptable constraint score for overall quality assessment

                        # Same individual constraints as above (all optional)
                        "length": { ... },
                        "complexity": { ... },
                        "repetitiveness": { ... },
                        "diversity": { ... },
                        "balanced_aas": { ... }
                    }
                }

    Returns:
        Constraint score between 0.0 and 1.0 where:
        - 0.0 indicates perfect/optimal protein quality (all constraints satisfied)
        - Values closer to 0.0 indicate better constraint satisfaction
        - 1.0 indicates worst possible protein quality (maximum constraint violation)

    Examples:
        DNA input with multiple quality checks:

        >>> from proto_language.language.constraint import ProteinLengthConfig, ProteinComplexityConfig
        >>> dna_seq = Sequence("ATGAAACGTATTGCGTCG...", SequenceType.DNA)
        >>> quality_config = ProteinQualitySubConfig(
        ...     quality_threshold=0.2,
        ...     length=ProteinLengthConfig(min_length=100, max_length=800),
        ...     complexity=ProteinComplexityConfig(max_low_complexity=0.3)
        ... )
        >>> score = overall_protein_quality_constraint(dna_seq, quality_config, min_high_quality_fraction=0.5)

        Protein input with diversity check:

        >>> protein_seq = Sequence("MVLSPADKTNVKAAW...", SequenceType.PROTEIN)
        >>> quality_config = ProteinQualitySubConfig(
        ...     quality_threshold=0.1,
        ...     diversity=ProteinDiversityConfig(min_diversity=0.3)
        ... )
        >>> score = overall_protein_quality_constraint(protein_seq, quality_config)
    """
    # Extract config parameters
    protein_quality_config = config.protein_quality_config
    min_high_quality_fraction = config.min_high_quality_fraction

    if input_sequence.sequence_type == SequenceType.DNA:
        # For DNA sequences: predict proteins first
        if min_high_quality_fraction is None:
            raise ValueError("min_high_quality_fraction is required for DNA sequences")

        # Get predicted proteins using Prodigal
        config = ProdigalConfig(input_sequence=input_sequence.sequence)
        result = run_prodigal_prediction(config)
        proteins_df = result.results_df

        input_sequence._metadata["prodigal_proteins"] = proteins_df
        input_sequence._metadata["prodigal_protein_count"] = result.num_genes

        if len(proteins_df) == 0:
            input_sequence._metadata["predicted_protein_count"] = 0
            input_sequence._metadata["high_quality_protein_count"] = 0
            input_sequence._metadata["high_quality_protein_fraction"] = 0.0
            input_sequence._metadata["protein_quality_details"] = []
            return 1.0  # Maximum penalty for no proteins found

        # Evaluate each predicted protein
        high_quality_count = 0
        protein_quality_details = []
        all_protein_avg_scores = []

        for idx, protein_row in proteins_df.iterrows():
            protein_seq = Sequence(protein_row["sequence"], SequenceType.PROTEIN)

            # Apply all protein quality constraints
            quality_scores = {}
            overall_scores = []

            if protein_quality_config.length:
                score = protein_length_constraint(protein_seq, config=protein_quality_config.length)
                quality_scores["length"] = score
                overall_scores.append(quality_scores["length"])

            if protein_quality_config.complexity:
                score = protein_complexity_constraint(
                    protein_seq, config=protein_quality_config.complexity
                )
                quality_scores["complexity"] = score
                overall_scores.append(quality_scores["complexity"])

            if protein_quality_config.repetitiveness:
                score = protein_repetitiveness_constraint(
                    protein_seq, config=protein_quality_config.repetitiveness
                )
                quality_scores["repetitiveness"] = score
                overall_scores.append(quality_scores["repetitiveness"])

            if protein_quality_config.diversity:
                score = protein_diversity_constraint(
                    protein_seq, config=protein_quality_config.diversity
                )
                quality_scores["diversity"] = score
                overall_scores.append(quality_scores["diversity"])

            if protein_quality_config.balanced_aas:
                score = balanced_aa_constraint(
                    protein_seq, config=protein_quality_config.balanced_aas
                )
                quality_scores["balanced_aas"] = score
                overall_scores.append(quality_scores["balanced_aas"])

            # Calculate average score for this protein
            avg_score = (
                sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            )
            all_protein_avg_scores.append(avg_score)

            # Consider protein high quality if average score is below threshold
            threshold = protein_quality_config.quality_threshold
            is_high_quality = avg_score <= threshold

            if is_high_quality:
                high_quality_count += 1

            protein_quality_details.append(
                {
                    "protein_id": protein_row["id"],
                    "length": len(protein_row["sequence"]),
                    "is_high_quality": is_high_quality,
                    "avg_constraint_score": avg_score,
                    "quality_scores": quality_scores,
                    "metadata": protein_seq._metadata.copy(),
                }
            )

        high_quality_fraction = high_quality_count / len(proteins_df)

        # Store comprehensive metadata
        input_sequence._metadata["predicted_protein_count"] = len(proteins_df)
        input_sequence._metadata["high_quality_protein_count"] = high_quality_count
        input_sequence._metadata["high_quality_protein_fraction"] = (
            high_quality_fraction
        )
        input_sequence._metadata["protein_quality_details"] = protein_quality_details
        input_sequence._metadata["protein_quality_threshold"] = protein_quality_config.quality_threshold

        # If we high quality fraction requirement is met, return 0
        if high_quality_fraction >= min_high_quality_fraction:
            return 0.0

        # Otherwise, return a score based on how far we are from meeting the requirement
        overall_avg_protein_score = sum(all_protein_avg_scores) / len(
            all_protein_avg_scores
        )
        fraction_deficit = (
            min_high_quality_fraction - high_quality_fraction
        ) / min_high_quality_fraction

        # Combine the average protein quality with the fraction deficit
        combined_score = (overall_avg_protein_score + fraction_deficit) / 2.0
        return min(1.0, max(0.0, combined_score))

    elif input_sequence.sequence_type == SequenceType.PROTEIN:
        # For protein sequences: evaluate quality directly on input sequence
        quality_scores = {}
        overall_scores = []

        if protein_quality_config.length:
            score = protein_length_constraint(input_sequence, config=protein_quality_config.length)
            quality_scores["length"] = score
            overall_scores.append(quality_scores["length"])

        if protein_quality_config.complexity:
            score = protein_complexity_constraint(
                input_sequence, config=protein_quality_config.complexity
            )
            quality_scores["complexity"] = score
            overall_scores.append(quality_scores["complexity"])

        if protein_quality_config.repetitiveness:
            score = protein_repetitiveness_constraint(
                input_sequence, config=protein_quality_config.repetitiveness
            )
            quality_scores["repetitiveness"] = score
            overall_scores.append(quality_scores["repetitiveness"])

        if protein_quality_config.diversity:
            score = protein_diversity_constraint(
                input_sequence, config=protein_quality_config.diversity
            )
            quality_scores["diversity"] = score
            overall_scores.append(quality_scores["diversity"])

        if protein_quality_config.balanced_aas:
            score = balanced_aa_constraint(
                input_sequence, config=protein_quality_config.balanced_aas
            )
            quality_scores["balanced_aas"] = score
            overall_scores.append(quality_scores["balanced_aas"])

        # Calculate overall quality score as average of individual constraint scores
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        threshold = protein_quality_config.quality_threshold
        is_high_quality = avg_score <= threshold

        # Store metadata for protein input
        input_sequence._metadata["protein_quality_scores"] = quality_scores
        input_sequence._metadata["avg_constraint_score"] = avg_score
        input_sequence._metadata["is_high_quality"] = is_high_quality
        input_sequence._metadata["protein_quality_threshold"] = threshold

        # If protein meets quality threshold, return 0, otherwise return the average score
        if is_high_quality:
            return 0.0
        else:
            return min(1.0, max(0.0, avg_score))

    else:
        raise ValueError("Input sequence must be either DNA or PROTEIN type")
