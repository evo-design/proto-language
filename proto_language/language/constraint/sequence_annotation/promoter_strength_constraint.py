"""Promoter strength constraint using Salis Lab Promoter Calculator."""

from typing import Literal

from proto_tools import (
    PromoterCalculatorConfig,
    PromoterCalculatorInput,
    run_promoter_calculator,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence


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


def _tx_rate_penalty(tx_rate: float) -> float:
    if tx_rate < 3000.0:
        return 1.0
    if tx_rate <= 10000.0:
        return 1.0 - 0.5 * ((tx_rate - 3000.0) / (10000.0 - 3000.0))
    penalty = 0.5 - 0.5 * min((tx_rate - 10000.0) / (20000.0 - 10000.0), 1.0)
    return max(0.0, penalty)


def _dG_penalty(dG: float) -> float:
    if dG >= 0 or dG > -1.5:
        return 1.0
    if dG >= -3.0:
        return 1.0 - 0.5 * ((dG + 1.5) / -1.5)
    normalized = (dG - (-3.0)) / (-5.0 - (-3.0))
    return max(0.0, min(0.5, 0.5 * (1 - normalized**2)))


@constraint(
    key="promoter-strength",
    label="Promoter Strength",
    config=PromoterStrengthConfig,
    description="Evaluate promoter strength using Salis Lab Promoter Calculator",
    tools_called=["promoter_calculator"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
)
def promoter_strength_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: PromoterStrengthConfig
) -> list[ConstraintOutput]:
    """Evaluate bacterial promoter strength using Salis Lab Promoter Calculator.

    This constraint function uses the Salis Lab Promoter Calculator to predict
    E. coli sigma-70 promoter strength. The calculator scans sequences for canonical
    promoter elements (-10 and -35 boxes) and computes either binding free energy (dG)
    or predicted transcription initiation rate (tx_rate).

    The constraint returns penalty scores where lower values indicate stronger
    promoters. The penalty mapping differs based on scoring type:
    - **dG scoring**: Promoters with dG < -3.0 kcal/mol are strong (penalty 0.0-0.5)
    - **tx_rate scoring**: Promoters with tx_rate > 10000 are strong (penalty 0.0-0.5)

    The calculator can identify multiple promoters in a single sequence; only the
    strongest forward-strand candidate contributes to the penalty.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-element tuple per
            target segment, each carrying the current sequence to evaluate.
        config (PromoterStrengthConfig): Configuration object containing ``scoring_type``
            (default: "dG"), ``threads`` (default: 8), ``add_context`` (default: False),
            and other processing parameters.

    Returns:
        list[ConstraintOutput]: One result per sequence. Score ranges from 0.0 (strong
            promoter) to 1.0 (weak/no promoter). ``metadata`` carries a single
            ``promoter_strength`` dict:

            **When promoter is found:**

            - ``penalty``: Float penalty score (0.0-1.0)
            - ``tx_rate`` OR ``dG_rate``: Float best promoter strength value
              (depending on scoring_type)
            - ``raw_output``: List of dictionaries with detailed promoter predictions
              including -10/-35 box positions, sequences, spacer length, and
              individual energy terms

            **When no promoter is found:**

            - ``penalty``: Float 1.0 (maximum penalty)
            - ``reason``: String "no_promoter_found"
            - ``raw_output``: Empty list []

    Examples:
        Evaluating promoter strength using dG scoring:

        >>> from proto_language.language.core import Sequence
        >>> # lacUV5 promoter padded with 20 nt of A on each side (calculator
        >>> # needs ~20 nt of flanking sequence to score the promoter elements)
        >>> seq = Sequence("A" * 20 + "AAAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACC" + "A" * 20, "dna")
        >>> config = PromoterStrengthConfig(scoring_type="dG")
        >>> results = promoter_strength_constraint([(seq,)], config)
        >>> print(results[0].score)
    """
    sequences = []
    for (seq_obj,) in input_sequences:
        s = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        if config.add_context:
            pad = "A" * config.context_length
            s = pad + s + pad
        sequences.append(s)

    output = run_promoter_calculator(
        PromoterCalculatorInput(sequences=sequences),
        PromoterCalculatorConfig(
            threads=config.threads,
            circular=config.circular,
        ),
    )

    results: list[ConstraintOutput] = []
    for seq_result in output.results:
        fwd = [p for p in seq_result.predictions if p.strand == "+"]
        if not fwd:
            results.append(
                ConstraintOutput(
                    score=1.0,
                    metadata={"promoter_strength": {"penalty": 1.0, "reason": "no_promoter_found", "raw_output": []}},
                )
            )
            continue

        raw_output = [p.model_dump() for p in fwd]
        if config.scoring_type == "tx_rate":
            tx_rate = max(p.Tx_rate for p in fwd)
            penalty = _tx_rate_penalty(tx_rate)
            results.append(
                ConstraintOutput(
                    score=penalty,
                    metadata={"promoter_strength": {"penalty": penalty, "tx_rate": tx_rate, "raw_output": raw_output}},
                )
            )
        else:
            dG = min(p.dG_total for p in fwd)
            penalty = _dG_penalty(dG)
            results.append(
                ConstraintOutput(
                    score=penalty,
                    metadata={"promoter_strength": {"penalty": penalty, "dG_rate": dG, "raw_output": raw_output}},
                )
            )

    return results
