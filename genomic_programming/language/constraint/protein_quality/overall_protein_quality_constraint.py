"""Overall protein quality constraint function."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import Field, model_validator

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import (
    sequence_length_constraint, SequenceLengthConfig  
)
from proto_language.language.constraint.protein_quality.protein_complexity_constraint import (
    protein_complexity_constraint,ProteinComplexityConfig
)
from proto_language.language.constraint.protein_quality.protein_repetitiveness_constraint import (
    protein_repetitiveness_constraint,ProteinRepetitivenessConfig
)
from proto_language.language.constraint.protein_quality.protein_diversity_constraint import (
    protein_diversity_constraint,ProteinDiversityConfig
)
from proto_language.language.constraint.protein_quality.balanced_aa_constraint import (
    balanced_aa_constraint,BalancedAaConfig
)

class ProteinQualitySubConfig(BaseConfig):
    """Nested configuration for individual protein quality checks."""
    length: Optional[SequenceLengthConfig] = Field(default=None, description="Sequence length constraints")
    complexity: Optional[ProteinComplexityConfig] = Field(default=None, description="Protein complexity constraints")
    repetitiveness: Optional[ProteinRepetitivenessConfig] = Field(default=None, description="Protein repetitiveness constraints")
    diversity: Optional[ProteinDiversityConfig] = Field(default=None, description="Amino acid diversity constraints")
    balanced_aas: Optional[BalancedAaConfig] = Field(default=None, description="Balanced amino acid constraints")
    quality_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Maximum acceptable constraint score for high quality")


class OverallProteinQualityConfig(BaseConfig):
    """Configuration for overall protein quality constraint."""
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
    label="Overall Protein Quality",
    config=OverallProteinQualityConfig,
    description="Evaluate overall protein quality using multiple sub-constraints",
    vectorized=True,
    concatenate=True
)
def overall_protein_quality_constraint(sequences: List[Sequence], config: OverallProteinQualityConfig) -> List[float]:
    """
    Evaluate protein quality either from predicted proteins (DNA input) or directly (protein input).

    For DNA sequences, runs Prodigal first to predict proteins, then checks all predicted
    proteins. For protein sequences, checks the sequence directly.

    Args:
        input_sequences: The DNA or protein sequences to analyze.
        config: Configuration dictionary containing:
            For DNA input:
                - protein_quality_config (dict): Configuration dictionary with the following structure:
                {
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
        Constraint scores between 0.0 and 1.0 where:
        - 0.0 indicates perfect/optimal protein quality (all constraints satisfied)
        - Values closer to 0.0 indicate better constraint satisfaction
        - 1.0 indicates worst possible protein quality (maximum constraint violation)

    Examples:
        DNA input with multiple quality checks:

        >>> from proto_language.language.constraint import SequenceLengthConfig, ProteinComplexityConfig
        >>> dna_seq = [Sequence("ATGAAACGTATTGCGTCG...", SequenceType.DNA)]
        >>> quality_config = ProteinQualitySubConfig(
        ...     quality_threshold=0.2,
        ...     length=SequenceLengthConfig(min_length=100, max_length=800),
        ...     complexity=ProteinComplexityConfig(max_low_complexity=0.3)
        ... )
        >>> score = overall_protein_quality_constraint(dna_seq, quality_config)

        Protein input with diversity check:

        >>> protein_seq = [Sequence("MVLSPADKTNVKAAW...", SequenceType.PROTEIN)]
        >>> quality_config = ProteinQualitySubConfig(
        ...     quality_threshold=0.1,
        ...     diversity=ProteinDiversityConfig(min_diversity=0.3)
        ... )
        >>> score = overall_protein_quality_constraint(protein_seq, quality_config)
    """
    # Extract config parameters
    protein_quality_config = config.protein_quality_config

    # Separate DNA and protein sequences
    dna_sequences = [seq for seq in sequences if seq.sequence_type == SequenceType.DNA]
    protein_sequences = [seq for seq in sequences if seq.sequence_type == SequenceType.PROTEIN]

    if len(dna_sequences) + len(protein_sequences) != len(sequences):
        raise ValueError("All sequences must be either DNA or PROTEIN type")

    dna_scores = []
    protein_scores = []
    threshold = protein_quality_config.quality_threshold

    if dna_sequences:
        # For DNA sequences: predict proteins first, get predicted proteins using Prodigal
        prodigal_input = ProdigalInput(input_sequences=[seq.sequence for seq in dna_sequences])
        prodigal_config = ProdigalConfig()
        batch_result = run_prodigal_prediction(inputs=prodigal_input, config=prodigal_config)

        # Process each DNA sequence's results
        for input_sequence, proteins_df, num_genes in zip(
            dna_sequences,
            batch_result.results_per_sequence,
            batch_result.total_num_genes_per_sequence
        ):
            input_sequence._metadata["prodigal_proteins"] = proteins_df
            input_sequence._metadata["prodigal_protein_count"] = num_genes

            if len(proteins_df) == 0:
                input_sequence._metadata["predicted_protein_count"] = 0
                input_sequence._metadata["high_quality_protein_count"] = 0
                input_sequence._metadata["high_quality_protein_fraction"] = 0.0
                input_sequence._metadata["protein_quality_details"] = []
                dna_scores.append(1.0)
                continue

            # Convert to Sequence objects for batch constraint evaluation
            predicted_protein_seqs = [
                Sequence(row["protein_sequence"], SequenceType.PROTEIN)
                for _, row in proteins_df.iterrows()
            ]

            quality_scores = {}
            
            if protein_quality_config.length:
                quality_scores["length"] = sequence_length_constraint(
                    predicted_protein_seqs, config=protein_quality_config.length
                )

            if protein_quality_config.complexity:
                quality_scores["complexity"] = protein_complexity_constraint(
                    predicted_protein_seqs, config=protein_quality_config.complexity
                )

            if protein_quality_config.repetitiveness:
                quality_scores["repetitiveness"] = protein_repetitiveness_constraint(
                    predicted_protein_seqs, config=protein_quality_config.repetitiveness
                )

            if protein_quality_config.diversity:
                quality_scores["diversity"] = protein_diversity_constraint(
                    predicted_protein_seqs, config=protein_quality_config.diversity
                )

            if protein_quality_config.balanced_aas:
                quality_scores["balanced_aas"] = balanced_aa_constraint(
                    predicted_protein_seqs, config=protein_quality_config.balanced_aas
                )

            # Vectorized averaging
            if quality_scores:
                constraint_score_matrix = np.array(list(quality_scores.values()))
                avg_scores = constraint_score_matrix.mean(axis=0)
            else:
                avg_scores = np.zeros(len(predicted_protein_seqs))

            # Vectorized quality determination
            is_high_quality = avg_scores <= threshold

            # Build details
            protein_quality_details = []
            for prot_idx, (protein_row, protein_seq) in enumerate(zip(proteins_df.iterrows(), predicted_protein_seqs)):
                idx, row = protein_row
                individual_scores = {
                    name: scores[prot_idx] 
                    for name, scores in quality_scores.items()
                }
                
                protein_quality_details.append({
                    "protein_id": row["id"],
                    "length": row["protein_length"],
                    "is_high_quality": bool(is_high_quality[prot_idx]),
                    "avg_constraint_score": float(avg_scores[prot_idx]),
                    "quality_scores": individual_scores,
                    "metadata": protein_seq._metadata.copy(),
                })

            overall_avg_protein_score = float(avg_scores.mean())
            is_dna_high_quality = overall_avg_protein_score <= threshold

            # Store metadata
            input_sequence._metadata["predicted_protein_count"] = len(proteins_df)
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

        if protein_quality_config.length:
            quality_scores["length"] = sequence_length_constraint(
                protein_sequences, config=protein_quality_config.length
            )

        if protein_quality_config.complexity:
            quality_scores["complexity"] = protein_complexity_constraint(
                protein_sequences, config=protein_quality_config.complexity
            )

        if protein_quality_config.repetitiveness:
            quality_scores["repetitiveness"] = protein_repetitiveness_constraint(
                protein_sequences, config=protein_quality_config.repetitiveness
            )

        if protein_quality_config.diversity:
            quality_scores["diversity"] = protein_diversity_constraint(
                protein_sequences, config=protein_quality_config.diversity
            )

        if protein_quality_config.balanced_aas:
            quality_scores["balanced_aas"] = balanced_aa_constraint(
                protein_sequences, config=protein_quality_config.balanced_aas
            )

        if quality_scores:
            constraint_score_matrix = np.array(list(quality_scores.values()))
            avg_scores = constraint_score_matrix.mean(axis=0)
        else:
            avg_scores = np.zeros(len(protein_sequences))

        # Vectorized quality determination
        is_high_quality = avg_scores <= threshold
        
        # Vectorized score calculation
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
        if seq.sequence_type == SequenceType.DNA:
            final_scores.append(dna_scores[dna_idx])
            dna_idx += 1
        else:
            final_scores.append(protein_scores[protein_idx])
            protein_idx += 1
    
    return final_scores
