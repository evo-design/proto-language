"""
proto_language/language/constraint/protein_structure/gyration_radius_constraint.py

Uses the structure_metrics tool to compute radius of gyration from PDB files,
then scores based on deviation from a maximum acceptable radius.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from proto_tools import (
    StructureMetricsConfig,
    StructureMetricsInput,
    run_structure_metrics,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.utils import MAX_ENERGY


class GyrationRadiusConfig(BaseConfig):
    """Configuration for gyration radius constraint.

    Attributes:
        max_gyration_radius (float): Maximum acceptable gyration radius in Angstroms.
            Structures at or below this threshold receive a score of 0.0 (perfect).
            Structures above are penalized proportionally.
        pdb_paths (list[str] | None): Optional list of PDB file paths. If not provided, the constraint
            reads PDB paths from sequence metadata (key ``pdb_path``).
    """

    max_gyration_radius: float = ConfigField(
        title="Max Gyration Radius",
        default=45.0,
        gt=0.0,
        description="Maximum acceptable gyration radius in Angstroms",
    )
    pdb_paths: Optional[List[str]] = ConfigField(
        title="PDB Paths",
        default=None,
        description="Optional explicit PDB paths (otherwise reads from sequence metadata)",
        hidden=True,
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
    num_input_sequences_per_tuple=1,
)
def gyration_radius_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: GyrationRadiusConfig,
) -> List[float]:
    """Filter structures by radius of gyration.

    Computes the radius of gyration for each input structure and returns
    a penalty score. Structures with gyration radius <= max_gyration_radius
    score 0.0, with penalty scaling linearly for larger radii, clamped to [0, 1].

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples to evaluate.
            Each sequence should have ``pdb_path`` in its metadata, or
            PDB paths should be provided via config.
        config (GyrationRadiusConfig): Configuration with max_gyration_radius threshold.

    Returns:
        list[float]: List of float scores in [0.0, 1.0]. 0.0 = within threshold, 1.0 = worst.
    """
    sequences = [seq for (seq,) in input_sequences]

    # Resolve PDB paths from config or sequence metadata
    if config.pdb_paths is not None:
        pdb_paths = config.pdb_paths
    else:
        pdb_paths = []
        for seq in sequences:
            pdb_path = seq._metadata.get("pdb_path") or seq._metadata.get("pdb_output")
            if pdb_path is None:
                pdb_paths.append(None)
            else:
                pdb_paths.append(str(pdb_path))

    # Compute structure metrics for sequences that have PDB paths
    valid_paths = [p for p in pdb_paths if p is not None]
    valid_indices = [i for i, p in enumerate(pdb_paths) if p is not None]

    metrics_map = {}
    if valid_paths:
        metrics_result = run_structure_metrics(
            StructureMetricsInput(pdb_paths=valid_paths),
            StructureMetricsConfig(),
        )
        for idx, metrics in zip(valid_indices, metrics_result.metrics):
            metrics_map[idx] = metrics

    # Compute scores
    scores = []
    for i, seq in enumerate(sequences):
        if i not in metrics_map:
            scores.append(MAX_ENERGY)
            continue

        metrics = metrics_map[i]
        radius = metrics.gyration_radius

        seq._metadata["gyration_radius"] = radius
        seq._metadata["longest_alpha_helix"] = metrics.longest_alpha_helix

        if radius <= config.max_gyration_radius:
            scores.append(0.0)
        else:
            # Linear penalty, clamped to [0, 1]
            deviation = (radius - config.max_gyration_radius) / config.max_gyration_radius
            scores.append(min(1.0, deviation))

    return scores
