"""Protein complexity constraint function."""

from proto_tools import (
    SegmaskerConfig,
    SegmaskerInput,
    run_segmasker,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY


class ProteinComplexityConfig(BaseConfig):
    """Configuration for protein complexity constraint.

    This class defines configuration parameters for evaluating protein sequence
    complexity using NCBI's segmasker tool. The constraint detects and penalizes
    low-complexity regions, which contain repetitive or biased amino acid
    compositions that may indicate poor protein quality or non-functional sequences.

    Attributes:
        max_low_complexity (float): Maximum acceptable fraction of low-complexity
            regions (0.0-1.0). Low-complexity regions contain repetitive or biased
            amino acid compositions. Typical values range from 0.1 (strict, allows
            up to 20% low-complexity) to 0.3 (lenient, allows up to 30%). Default: 0.3.

    """

    # Required parameter
    max_low_complexity: float = ConfigField(
        title="Max Low Complexity Fraction",
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable fraction of low-complexity regions (repetitive/biased amino acid compositions)",
        examples=[0.1, 0.3],
    )


@constraint(
    key="protein-complexity",
    label="Protein Complexity",
    config=ProteinComplexityConfig,
    description="Evaluate protein sequence complexity using segmasker to detect low-complexity regions",
    tools_called=["segmasker-score"],
    category="protein quality",
    supported_sequence_types=["protein"],
)
def protein_complexity_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinComplexityConfig
) -> list[ConstraintOutput]:
    """Evaluate protein sequence complexity using segmasker to detect low-complexity regions.

    This constraint function uses NCBI's segmasker tool to identify low-complexity
    regions in protein sequences. Low-complexity regions contain repetitive or
    compositionally biased amino acid sequences that may indicate poor protein
    quality, tandem repeats, or non-functional segments. The constraint penalizes
    sequences where the fraction of low-complexity regions exceeds a specified
    threshold.

    The function processes multiple sequences simultaneously. Segmasker marks
    low-complexity regions by replacing them with lowercase characters, and
    the constraint calculates the fraction of masked positions.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one protein sequence.

        config (ProteinComplexityConfig): Configuration object containing
            ``max_low_complexity`` (maximum acceptable low-complexity fraction,
            default: 0.3).

    Returns:
        list[ConstraintOutput]: One result per sequence. A score of 0.0 indicates
            acceptable complexity (low-complexity fraction at or below threshold)
            and higher values indicate excessive low-complexity content. Scores
            scale linearly with excess low-complexity beyond the threshold, capped
            at 1.0. ``metadata`` carries:

            - ``low_complexity_fraction``: Float fraction of sequence identified as
              low-complexity (0.0-1.0)
            - ``segmasker_lowercase_count``: Integer count of positions masked as low-complexity
            - ``segmasker_error``: Boolean indicating if segmasker execution failed

    Raises:
        AssertionError: If any sequence in the input list is not a protein sequence.
        ValueError: If segmasker execution fails (e.g., segmasker not found in PATH,
            invalid sequence format, or tool error).

    Examples:
        Evaluating protein complexity:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> config = ProteinComplexityConfig(max_low_complexity=0.3)
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> results = protein_complexity_constraint([(seq,)], config)
        >>> print(results[0].score)  # 0.0 if low-complexity < 30%
        >>> print(results[0].metadata["low_complexity_fraction"])  # e.g., 0.15
        >>> print(results[0].metadata["segmasker_lowercase_count"])  # e.g., 5
    """
    segmasker_inputs = SegmaskerInput(sequences=[seq.sequence for (seq,) in input_sequences])
    segmasker_config = SegmaskerConfig()

    result = run_segmasker(inputs=segmasker_inputs, config=segmasker_config)

    results = []
    for (seq,), metrics in zip(input_sequences, result.results, strict=False):
        low_complexity_fraction = metrics.low_complexity_fraction

        if low_complexity_fraction <= config.max_low_complexity:
            score = MIN_ENERGY
        else:
            excess = low_complexity_fraction - config.max_low_complexity
            score = min(MAX_ENERGY, excess / (1.0 - config.max_low_complexity))

        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "low_complexity_fraction": low_complexity_fraction,
                    "segmasker_lowercase_count": int(low_complexity_fraction * len(seq)),
                    "segmasker_error": False,
                },
            )
        )

    return results
