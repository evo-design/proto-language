"""
proto_language/language/constraint/sequence_annotation/promoter_strength_constraint.py

Promoter strength constraint using Salis Lab Promoter Calculator.
"""

from __future__ import annotations

from typing import List, Literal, Tuple

from promoter_calculator.wrapper import promoter_calculator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence


class PromoterStrengthConfig(BaseConfig):
    """Configuration for promoter strength constraint using Salis Lab Promoter Calculator.

    This class defines configuration parameters for evaluating bacterial promoter
    strength using the Salis Lab Promoter Calculator, a biophysical model that
    predicts RNA polymerase binding affinity and transcription initiation
    rates for sigma-70 promoters in E. coli. The calculator identifies promoter elements
    (-10 and -35 boxes, spacer regions) and computes binding free energy (dG) and
    predicted transcription rates.

    Attributes:
        add_context (bool): If True, adds flanking 'A' nucleotides to sequences
            shorter than the calculator's minimum length requirement. This ensures
            short sequences can still be evaluated by providing neutral flanking
            context. The added context does not affect promoter predictions within
            the original sequence. Default: False.

        context_length (int): Number of 'A' nucleotides to add on each end (5' and 3')
            when ``add_context=True``. For example, 10 adds "AAAAAAAAAA" to both
            ends. Must be a positive integer. Default: 10.

        threads (int): Number of CPU threads for parallel processing of promoter
            predictions. Higher values speed up batch processing but use more CPU
            resources. Default: 8.

        verbosity (int): Verbosity level for promoter calculator output. 0 is quiet
            (no output). Default: 0.

        circular (bool): If True, treats sequences as circular DNA for promoter
            detection across sequence ends. Useful for plasmids where promoters
            may span the origin. If False, treats sequences as linear. Default: False.

        scoring_type (Literal['dG', 'tx_rate']): Metric to use for promoter strength
            scoring. Options:
            - "dG": Binding free energy in kcal/mol. More negative values indicate
              stronger RNAP binding. Range typically -5 to 2 kcal/mol. Use for
              biophysical interpretation, with values < -2.0 indicating a likely
              promoter.
            - "tx_rate": Predicted transcription initiation rate in arbitrary units.
              Higher values indicate stronger promoters. Range typically 0-30,000+.
              Use for relative promoter strength comparison.
            Default: "dG".

    Note:
        The Salis Lab Promoter Calculator specifically models E. coli sigma-70 promoters.

        Penalty scores are mapped from raw predictions:
        - **For dG scoring**: Strong promoters (dG < -3.0) get low penalties (0.0-0.5),
          moderate promoters (-3.0 to -1.5) get medium penalties (0.5-1.0), weak or unlikely
          promoters (> -1.5) get maximum penalty (1.0).
        - **For tx_rate scoring**: Strong promoters (>10000) get low penalties (0.0-0.5),
          moderate promoters (3000-10000) get medium penalties (0.5-1.0), weak
          promoters (<3000) get maximum penalty (1.0).
    """
    # Advanced parameters
    add_context: bool = ConfigField(
        title="Add Context",
        default=False,
        description="If True, adds flanking nucleotides to short sequences to meet calculator length minimums",
        advanced=True,
    )
    context_length: int = ConfigField(
        title="Added Context Length",
        default=10,
        description="Number of 'A' nucleotides to add on each end when add_context=True",
        advanced=True,
    )
    threads: int = ConfigField(
        title="Number of Threads",
        default=8,
        ge=1,
        description="Number of threads for parallel processing of promoter calculations",
        hidden=True,
    )
    verbosity: int = ConfigField(
        title="Verbosity Level",
        default=0,
        ge=0,
        description="Verbosity level for promoter calculator output (0=quiet, higher=more verbose)",
        hidden=True,
    )
    circular: bool = ConfigField(
        title="Circular Sequences",
        default=False,
        description="If True, treat sequences as circular for promoter detection across ends",
        advanced=True,
    )
    scoring_type: Literal["dG", "tx_rate"] = ConfigField(
        title="Scoring Type",
        default="dG",
        description="Score type to use: 'dG' (binding free energy) or 'tx_rate' (transcription rate). Defaults to 'dG'.",
        advanced=True,
    )


@constraint(
    key="promoter-strength",
    label="Promoter Strength",
    config=PromoterStrengthConfig,
    description="Evaluate promoter strength using Salis Lab Promoter Calculator",
    tools_called=["promoter_calculator"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
    num_input_sequences_per_tuple=1,
)
def promoter_strength_constraint(input_sequences: List[Tuple[Sequence, ...]], config: PromoterStrengthConfig) -> List[float]:
    """Evaluate bacterial promoter strength using Salis Lab Promoter Calculator.

    This constraint function uses the Salis Lab Promoter Calculator to predict
    E. coli sigma-70 promoter strength. The calculator scans sequences for canonical
    promoter elements (-10 and -35 boxes) and computes either binding free energy (dG)
    or predicted transcription initiation rate (tx_rate).

    The constraint returns penalty scores where lower values indicate stronger
    promoters. The penalty mapping differs based on scoring type:
    - **dG scoring**: Promoters with dG < -3.0 kcal/mol are strong (penalty 0.0-0.5)
    - **tx_rate scoring**: Promoters with tx_rate > 10000 are strong (penalty 0.0-0.5)

    The calculator can identify multiple promoters in a single sequence and returns
    predictions for the strongest promoter on the forward (+) strand.

    Args:
        sequences: List of DNA sequences to evaluate. Sequences
            should contain potential promoter regions (typically 70-200+ bp).
            The calculator scans for sigma-70 promoter elements throughout each sequence.

        config (PromoterStrengthConfig): Configuration object containing ``scoring_type``
            (default: "dG"), ``threads`` (default: 8), ``add_context`` (default: False),
            and other processing parameters.
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.

    Returns:
        List[float]: Penalty scores for each sequence, ranging from 0.0 (strong
            promoter) to 1.0 (weak/no promoter). The scoring scheme depends on
            ``scoring_type``:

            **For dG scoring:**
            - penalty = 1.0: dG > -1.5 (weak/no promoter)
            - penalty = 1.0 → 0.5: dG from -1.5 to -3.0 (moderate)
            - penalty = 0.5 → 0.0: dG from -3.0 to -5.0+ (strong)

            **For tx_rate scoring:**
            - penalty = 1.0: tx_rate < 3000 (weak/no promoter)
            - penalty = 1.0 → 0.5: tx_rate from 3000 to 10000 (moderate)
            - penalty = 0.5 → 0.0: tx_rate from 10000 to 20,000+ (strong)

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:

        **When promoter is found:**
        - ``promoter_strength``: Dictionary containing:
          - ``penalty``: Float penalty score (0.0-1.0)
          - ``tx_rate`` OR ``dG_rate``: Float best promoter strength value
            (depending on scoring_type)
          - ``raw_output``: List of dictionaries with detailed promoter predictions
            including -10/-35 box positions, sequences, spacer length, and
            individual energy terms

        **When no promoter is found:**
        - ``promoter_strength``: Dictionary containing:
          - ``penalty``: Float 1.0 (maximum penalty)
          - ``reason``: String "no_promoter_found"
          - ``raw_output``: Empty list []

    Examples:
        Evaluating promoter strength using dG scoring:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> # Sequence with strong constitutive promoter
        >>> promoter_seq = Sequence(
        ...     "TTGACAATGATACTTAGATTCACTTATAATACTAGTAGGAGGAACTTTATGAAA",
        ...     "dna"
        ... )
        >>> config = PromoterStrengthConfig(scoring_type="dG")
        >>> scores = promoter_strength_constraint([promoter_seq], config)
        >>> print(scores[0])  # e.g., 0.15 (strong promoter, dG ≈ -4.5)
        >>> print(promoter_seq._metadata["promoter_strength"]["dG_rate"])  # e.g., -4.5
    """

    # Extract and clean sequences from tuples
    processed_sequences = []
    for (seq_obj,) in input_sequences:
        s = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        if config.add_context:
            s = ("A" * config.context_length) + s + ("A" * config.context_length)
        processed_sequences.append(s)

    penalties: List[float] = []

    all_results = []
    for seq in processed_sequences:
        res = (
            promoter_calculator(
                seq,
                threads=config.threads,
                verbosity=config.verbosity,
                circular=config.circular,
            )
            or []
        )
        all_results.append(res)

    # Process results for each sequence
    for (seq_obj,), res in zip(input_sequences, all_results):
        # Keep only + strand
        res = [r for r in res if getattr(r, "strand", "+") == "+"]

        if not res:
            penalty = 1.0
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "reason": "no_promoter_found",
                "raw_output": [],
            }
            penalties.append(penalty)
            continue

        if config.scoring_type == "tx_rate":
            # Extract tx_rate
            tx_rate = max(float(r.Tx_rate) for r in res if hasattr(r, "Tx_rate"))

            # Penalty mapping
            if tx_rate < 3000.0:
                penalty = 1.0
            elif tx_rate <= 10000.0:
                penalty = 1.0 - 0.5 * ((tx_rate - 1000.0) / (10000.0 - 3000.0))
            else:
                penalty = 0.5 - min((tx_rate - 20000.0) / 20000.0 * 0.5, 0.5)
                penalty = max(0.0, penalty)

            # Store metadata
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "tx_rate": tx_rate,
                "raw_output": [r.__dict__ for r in res],
            }
            penalties.append(penalty)
        else:
            dG = min(float(r.dG_total) for r in res if hasattr(r, "dG_total"))
            if dG >= 0:
                penalty = 1.0
            elif dG > -1.5:
                penalty = 1.0
            elif dG >= -3.0:
                penalty = 1.0 - 0.5 * ((dG + 1.5) / -1.5)
            else:
                normalized = (dG - (-3.0)) / (-5.0 - (-3.0))
                penalty = 0.5 * (1 - normalized**2)
                penalty = max(0.0, min(0.5, penalty))

            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "dG_rate": dG,
                "raw_output": [r.__dict__ for r in res],
            }
            penalties.append(penalty)

    return penalties
