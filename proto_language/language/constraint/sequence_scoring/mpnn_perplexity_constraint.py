"""Filter or score proposals by ProteinMPNN perplexity."""

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence

GENERATOR_KEY = "proteinmpnn"
PERPLEXITY_FIELD = "perplexity"


class MpnnPerplexityConfig(BaseConfig):
    """Filter or score proposals by ProteinMPNN perplexity.

    Attributes:
        top_k (int | None): Keep only the top-k proposals by perplexity. None returns raw scores.
    """

    top_k: int | None = ConfigField(
        default=None,
        title="Top K",
        description="Keep only the top-k proposals by perplexity. None returns raw scores.",
        ge=1,
    )


@constraint(
    key="mpnn-perplexity",
    label="MPNN Perplexity",
    config=MpnnPerplexityConfig,
    description="Filter or score proposals by ProteinMPNN perplexity. Requires ProteinMPNNGenerator upstream.",
    tools_called=[],
    category="sequence scoring",
    supported_sequence_types=["protein"],
)
def mpnn_perplexity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: MpnnPerplexityConfig,
) -> list[ConstraintOutput]:
    """Filter or score proposals by ProteinMPNN perplexity from generator metadata."""
    perplexities: list[float] = []
    for (seq,) in input_sequences:
        gen_meta = seq._generator_metadata.get(GENERATOR_KEY)
        if gen_meta is None or PERPLEXITY_FIELD not in gen_meta:
            raise ValueError(
                f"'{GENERATOR_KEY}.{PERPLEXITY_FIELD}' not found — "
                "attach a ProteinMPNN generator to the optimization stage"
            )
        perplexities.append(float(gen_meta[PERPLEXITY_FIELD]))

    if config.top_k is None:
        return [ConstraintOutput(score=p, metadata={"perplexity": p}) for p in perplexities]

    k = config.top_k
    if len(perplexities) <= k:
        return [ConstraintOutput(score=0.0, metadata={"perplexity": p}) for p in perplexities]

    cutoff = sorted(perplexities)[k - 1]
    accepted = 0
    results: list[ConstraintOutput] = []
    for p in perplexities:
        if p <= cutoff and accepted < k:
            results.append(ConstraintOutput(score=0.0, metadata={"perplexity": p}))
            accepted += 1
        else:
            results.append(ConstraintOutput(score=float("inf"), metadata={"perplexity": p}))
    return results


mpnn_perplexity_constraint._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
