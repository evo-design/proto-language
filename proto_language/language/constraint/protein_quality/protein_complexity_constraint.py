"""
proto_language/language/constraint/protein_quality/protein_complexity_constraint.py

Protein complexity constraint function.
"""

from __future__ import annotations

from typing import List, Tuple

from proto_tools import (
    SegmaskerConfig,
    SegmaskerInput,
    run_segmasker,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
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

        segmasker_path (str): Path to NCBI segmasker executable for detecting
            low-complexity regions. Must be installed separately from NCBI BLAST+
            toolkit. If segmasker is in your system PATH, use default "segmasker".
            Otherwise, provide full path like "/usr/local/bin/segmasker".
            Default: "segmasker".

    Note:
        To-do: Currently Segmasker must be installed separately. For client
        applicability might be good to incorporate this into a venv.
        Install via NCBI BLAST+ toolkit:
        https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
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

    # Optional parameter
    segmasker_path: str = ConfigField(
        title="Segmasker Path",
        default="segmasker",
        description="Path to NCBI segmasker executable for detecting low-complexity regions.",  # Must be installed separately.
        hidden=True,
    )


@constraint(
    key="protein-complexity",
    label="Protein Complexity",
    config=ProteinComplexityConfig,
    description="Evaluate protein sequence complexity using segmasker to detect low-complexity regions",
    tools_called=["segmasker-score"],
    category="protein quality",
    supported_sequence_types=["protein"],
    num_input_sequences_per_tuple=1,
)
def protein_complexity_constraint(input_sequences: List[Tuple[Sequence, ...]], config: ProteinComplexityConfig) -> List[float]:
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
            default: 0.3) and ``segmasker_path`` (path to segmasker executable,
            default: "segmasker").

    Returns:
        List[float]: Constraint scores for each sequence, where 0.0 indicates
            acceptable complexity (low-complexity fraction at or below threshold)
            and higher values indicate excessive low-complexity content. Scores
            scale linearly with excess low-complexity beyond the threshold, capped
            at 1.0.

    Raises:
        AssertionError: If any sequence in the input list is not a protein sequence.
        ValueError: If segmasker execution fails (e.g., segmasker not found in PATH,
            invalid sequence format, or tool error).

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:

        - ``low_complexity_fraction``: Float fraction of sequence identified as
          low-complexity (0.0-1.0)
        - ``segmasker_lowercase_count``: Integer count of positions masked as low-complexity
        - ``segmasker_error``: Boolean indicating if segmasker execution failed
        - ``segmasker_error_message``: Error message if execution failed (only
          present when segmasker_error is True)

    Examples:
        Evaluating protein complexity:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> config = ProteinComplexityConfig(max_low_complexity=0.3, segmasker_path="segmasker")
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> scores = protein_complexity_constraint([(seq,)], config)
        >>> print(scores[0])  # 0.0 if low-complexity < 30%
        >>> print(seq._metadata["low_complexity_fraction"])  # e.g., 0.15
        >>> print(seq._metadata["segmasker_lowercase_count"])  # e.g., 5
    """
    # Extract sequence strings from tuples
    segmasker_inputs = SegmaskerInput(sequences=[seq.sequence for (seq,) in input_sequences])
    segmasker_config = SegmaskerConfig(segmasker_path=config.segmasker_path)

    result = run_segmasker(inputs=segmasker_inputs, config=segmasker_config)

    if not result.success:
        # Tool failed - store error metadata in all sequences and raise
        scores = []
        error_msg = result.errors[0] if result.errors else "Unknown segmasker error"

        for (seq,) in input_sequences:
            seq._metadata["low_complexity_fraction"] = 0.0
            seq._metadata["segmasker_lowercase_count"] = 0
            seq._metadata["segmasker_error"] = True
            seq._metadata["segmasker_error_message"] = error_msg
            scores.append(MAX_ENERGY)

        raise ValueError(f"Segmasker analysis failed: {error_msg}")

    scores = []
    for (seq,), low_complexity_fraction in zip(input_sequences, result.low_complexity_fractions):
        seq._metadata["low_complexity_fraction"] = low_complexity_fraction
        seq._metadata["segmasker_lowercase_count"] = int(
            low_complexity_fraction * len(seq)
        )
        seq._metadata["segmasker_error"] = False

        if low_complexity_fraction <= config.max_low_complexity:
            score = MIN_ENERGY
        else:
            excess = low_complexity_fraction - config.max_low_complexity
            score = min(MAX_ENERGY, excess / (1.0 - config.max_low_complexity))

        scores.append(score)

    return scores
