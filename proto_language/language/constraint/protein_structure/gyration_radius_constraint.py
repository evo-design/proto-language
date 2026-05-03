"""Radius of gyration constraint using structure_metrics tool."""

import tempfile
from contextlib import ExitStack

from proto_tools import (
    StructureMetricsConfig,
    StructureMetricsInput,
    run_structure_metrics,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY


class GyrationRadiusConfig(BaseConfig):
    """Configuration for gyration radius constraint.

    Attributes:
        max_gyration_radius (float): Maximum acceptable gyration radius in Angstroms.
            Structures at or below score 0.0; larger radii are penalized linearly,
            clamped to 1.0.
    """

    max_gyration_radius: float = ConfigField(
        title="Max Gyration Radius",
        default=45.0,
        gt=0.0,
        description="Maximum acceptable gyration radius in Angstroms",
    )


@constraint(
    key="gyration-radius",
    label="Gyration Radius",
    config=GyrationRadiusConfig,
    description="Filter structures by radius of gyration (compactness)",
    uses_gpu=False,
    tools_called=["structure_metrics"],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
)
def gyration_radius_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: GyrationRadiusConfig,
) -> list[ConstraintOutput]:
    """Filter structures by radius of gyration.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Single-sequence tuples to evaluate.
            Each sequence must carry a predicted ``Sequence.structure``.
        config (GyrationRadiusConfig): Configuration with max_gyration_radius threshold.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0.0, 1.0]`` with
            ``gyration_radius`` and ``longest_alpha_helix`` metadata. Sequences
            without a structure receive ``MAX_ENERGY`` and no metadata.
    """
    sequences = [seq for (seq,) in input_sequences]

    with ExitStack() as stack:
        indexed_paths: list[tuple[int, str]] = []
        for i, seq in enumerate(sequences):
            if seq.structure is None:
                continue
            tmp = stack.enter_context(tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=True))
            tmp.write(seq.structure.structure_pdb)
            tmp.flush()
            indexed_paths.append((i, tmp.name))

        metrics_by_idx = {}
        if indexed_paths:
            metrics_result = run_structure_metrics(
                StructureMetricsInput(pdb_paths=[p for _, p in indexed_paths]),
                StructureMetricsConfig(),
            )
            metrics_by_idx = {idx: m for (idx, _), m in zip(indexed_paths, metrics_result.metrics, strict=False)}

    threshold = config.max_gyration_radius
    results: list[ConstraintOutput] = []
    for i in range(len(sequences)):
        m = metrics_by_idx.get(i)
        if m is None:
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"gyration_radius_error": f"structure_metrics returned no result for sequence {i}"},
                )
            )
            continue
        score = min(1.0, max(0.0, (m.gyration_radius - threshold) / threshold))
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "gyration_radius": m.gyration_radius,
                    "longest_alpha_helix": m.longest_alpha_helix,
                },
            )
        )
    return results
