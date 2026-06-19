"""microRNA repression specificity constraint using miRanda."""

from typing import Literal

from proto_tools.tools.gene_annotation.miranda.miranda_scan import (
    MirandaConfig,
    MirandaInput,
    run_miranda_scan,
)
from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField


class MiRNASpecificityConfig(BaseConfig):
    """Configuration for miRNA target-site specificity scoring.

    Attributes:
        mirna_queries (list[str]): miRNA query sequences used for target scanning.
        mirna_ids (list[str] | None): Optional identifiers matching ``mirna_queries``.
        mirna_weights (list[float] | None): Optional per-miRNA expression weights matching ``mirna_queries``.
        direction (Literal['maximize', 'minimize']): Whether predicted repression should be high or low.
        repression_threshold (float): Repression score treated as saturating success.
        site_score_reference (float): Alignment score used to normalize each hit.
        energy_reference (float): Absolute duplex energy used to normalize each hit.
        miranda_config (MirandaConfig): miRanda runtime and reporting thresholds.
    """

    mirna_queries: list[str] = ConfigField(
        title="miRNA Queries",
        description="miRNA query sequences to scan against each target.",
    )
    mirna_ids: list[str] | None = ConfigField(
        default=None,
        title="miRNA IDs",
        description="Optional identifiers for miRNA queries.",
    )
    mirna_weights: list[float] | None = ConfigField(
        default=None,
        title="miRNA Weights",
        description="Optional nonnegative per-miRNA expression weights matching the miRNA query order.",
    )
    direction: Literal["maximize", "minimize"] = ConfigField(
        default="maximize",
        title="Direction",
        description="Whether predicted miRNA repression should be high or low.",
    )
    repression_threshold: float = ConfigField(
        default=2.0,
        gt=0.0,
        title="Repression Threshold",
        description="Aggregated repression score treated as saturating success.",
    )
    site_score_reference: float = ConfigField(
        default=150.0,
        gt=0.0,
        title="Site Score Reference",
        description="Alignment score used to normalize each miRanda hit.",
    )
    energy_reference: float = ConfigField(
        default=20.0,
        gt=0.0,
        title="Energy Reference",
        description="Absolute duplex energy used to normalize each miRanda hit.",
    )
    miranda_config: MirandaConfig = ConfigField(
        default_factory=MirandaConfig,
        title="miRanda Config",
        description="miRanda scan configuration.",
    )

    @field_validator("mirna_queries", mode="before")
    @classmethod
    def _normalize_queries(cls, queries: list[str] | str) -> list[str]:
        if isinstance(queries, str):
            queries = [queries]
        normalized = [query.strip().upper().replace("T", "U") for query in queries if query and query.strip()]
        if not normalized:
            raise ValueError("mirna_queries cannot be empty.")
        invalid = {base for query in normalized for base in query if base not in "ACGUN"}
        if invalid:
            raise ValueError(f"miRNA queries contain invalid RNA characters: {sorted(invalid)}.")
        return normalized

    @field_validator("mirna_ids")
    @classmethod
    def _validate_ids(cls, ids: list[str] | None) -> list[str] | None:
        if ids is None:
            return None
        normalized = [value.strip() for value in ids]
        if any(not value for value in normalized):
            raise ValueError("mirna_ids cannot contain empty values.")
        return normalized

    @field_validator("mirna_weights")
    @classmethod
    def _validate_weights(cls, weights: list[float] | None) -> list[float] | None:
        if weights is None:
            return None
        normalized = [float(value) for value in weights]
        if any(value < 0.0 for value in normalized):
            raise ValueError("mirna_weights cannot contain negative values.")
        return normalized

    @model_validator(mode="after")
    def _validate_parallel_lengths(self) -> "MiRNASpecificityConfig":
        if self.mirna_ids is not None and len(self.mirna_ids) != len(self.mirna_queries):
            raise ValueError("mirna_ids must match mirna_queries length.")
        if self.mirna_weights is not None and len(self.mirna_weights) != len(self.mirna_queries):
            raise ValueError("mirna_weights must match mirna_queries length.")
        return self


def _site_strength(score: float, energy: float, config: MiRNASpecificityConfig) -> float:
    score_component = max(score, 0.0) / config.site_score_reference
    energy_component = abs(min(energy, 0.0)) / config.energy_reference
    return 0.5 * (score_component + energy_component)


@constraint(
    key="mirna-specificity",
    label="miRNA Specificity",
    config=MiRNASpecificityConfig,
    description="Score predicted miRNA-mediated repression with miRanda target-site calls.",
    uses_gpu=False,
    tools_called=["miranda-scan"],
    category="sequence_annotation",
    supported_sequence_types=["dna", "rna"],
)
def mirna_specificity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: MiRNASpecificityConfig,
) -> list[ConstraintOutput]:
    """Score miRNA target-site burden in target DNA/RNA sequences."""
    if not input_sequences:
        return []

    target_sequences = [sequence.sequence.replace("T", "U") for (sequence,) in input_sequences]
    output = run_miranda_scan(
        MirandaInput(
            target_sequences=target_sequences,
            mirna_queries=config.mirna_queries,
            mirna_ids=config.mirna_ids,
        ),
        config.miranda_config,
    )

    weight_ids = config.mirna_ids or [f"seq_{idx}" for idx in range(len(config.mirna_queries))]
    mirna_weights = dict(zip(weight_ids, config.mirna_weights or [], strict=False))
    default_weight = 1.0

    results: list[ConstraintOutput] = []
    for result in output.results:
        site_strengths = [
            _site_strength(site.score, site.energy, config) * mirna_weights.get(site.mirna_id, default_weight)
            for site in result.target_sites
        ]
        repression = float(sum(site_strengths))
        bounded = min(max(repression, 0.0), config.repression_threshold) / config.repression_threshold
        score = 1.0 - bounded if config.direction == "maximize" else bounded
        score = min(MAX_ENERGY, score)

        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "mirna_repression_score": repression,
                    "mirna_specificity_score": score,
                    "mirna_direction": config.direction,
                    "mirna_num_sites": result.num_sites,
                    "mirna_ids": config.mirna_ids,
                    "mirna_weights": config.mirna_weights,
                    "mirna_sites": [
                        {
                            "mirna_id": site.mirna_id,
                            "score": site.score,
                            "energy": site.energy,
                            "target_start": site.target_start,
                            "target_end": site.target_end,
                            "weight": mirna_weights.get(site.mirna_id, default_weight),
                            "strength": strength,
                        }
                        for site, strength in zip(result.target_sites, site_strengths, strict=True)
                    ],
                },
            )
        )

    return results
