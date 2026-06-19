"""PyRosetta interface-quality constraint for predicted complexes."""

from typing import Literal

from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_interface_analyzer import (
    InterfaceStructureInput,
    PyRosettaInterfaceAnalyzerConfig,
    PyRosettaInterfaceAnalyzerInput,
    run_pyrosetta_interface_analyzer,
)
from pydantic import model_validator

from proto_language.constraint.constraint_registry import InputSlot, constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField

_MetricName = Literal[
    "interface_sc",
    "interface_hbonds",
    "interface_dG",
    "interface_dSASA",
    "interface_packstat",
    "interface_hydrophobicity",
    "surface_hydrophobicity",
    "delta_unsat_hbonds",
]
_Direction = Literal["higher", "lower"]


class PyRosettaInterfaceConfig(BaseConfig):
    """Configuration for PyRosetta interface metric scoring.

    Attributes:
        target_chains (list[str]): Target-side chains in the attached complex structure.
        binder_chain (str): Binder chain in the attached complex structure.
        metric (str): PyRosetta interface metric to score.
        direction (Literal['higher', 'lower']): Whether larger or smaller metric values are better.
        desired_value (float): Metric value treated as a satisfied objective.
        tolerance (float): Distance from desired value at which score reaches 1.
        pyrosetta_config (PyRosettaInterfaceAnalyzerConfig): PyRosetta runtime settings.
    """

    target_chains: list[str] = ConfigField(
        default_factory=lambda: ["A"],
        title="Target Chains",
        description="Target-side chain labels in the attached complex structure.",
    )
    binder_chain: str = ConfigField(
        default="B",
        title="Binder Chain",
        description="Binder chain label in the attached complex structure.",
    )
    metric: _MetricName = ConfigField(
        default="interface_dG",
        title="Metric",
        description="PyRosetta interface metric to score.",
    )
    direction: _Direction = ConfigField(
        default="lower",
        title="Direction",
        description="Whether higher or lower metric values are better.",
    )
    desired_value: float = ConfigField(
        default=-10.0,
        title="Desired Value",
        description="Metric value treated as satisfied.",
    )
    tolerance: float = ConfigField(
        default=10.0,
        gt=0.0,
        title="Tolerance",
        description="Metric distance from desired value at which the score reaches 1.",
    )
    pyrosetta_config: PyRosettaInterfaceAnalyzerConfig = ConfigField(
        default_factory=PyRosettaInterfaceAnalyzerConfig,
        title="PyRosetta Config",
        description="PyRosetta interface analyzer configuration.",
    )

    @model_validator(mode="after")
    def _validate_chains(self) -> "PyRosettaInterfaceConfig":
        if not self.target_chains:
            raise ValueError("target_chains cannot be empty.")
        if self.binder_chain in self.target_chains:
            raise ValueError("binder_chain cannot also be a target chain.")
        return self


def _score_metric(value: float, config: PyRosettaInterfaceConfig) -> float:
    if config.direction == "higher":
        if value >= config.desired_value:
            return MIN_ENERGY
        return min(MAX_ENERGY, (config.desired_value - value) / config.tolerance)
    if value <= config.desired_value:
        return MIN_ENERGY
    return min(MAX_ENERGY, (value - config.desired_value) / config.tolerance)


@constraint(
    key="pyrosetta-interface",
    label="PyRosetta Interface",
    config=PyRosettaInterfaceConfig,
    description="Score interface metrics on an attached predicted complex structure.",
    uses_gpu=False,
    tools_called=["pyrosetta-interface-analyzer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=[InputSlot(label="Complex Carrier", requires_structure=True)],
)
def pyrosetta_interface_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: PyRosettaInterfaceConfig,
) -> list[ConstraintOutput]:
    """Score PyRosetta interface metrics from attached complex structures."""
    if not input_sequences:
        return []

    interface_inputs = []
    for (sequence,) in input_sequences:
        if sequence.structure is None:
            raise ValueError("pyrosetta-interface requires the input sequence to carry an attached Structure.")
        interface_inputs.append(
            InterfaceStructureInput(
                structure=sequence.structure,
                target_chains=config.target_chains,
                binder_chain=config.binder_chain,
            )
        )

    output = run_pyrosetta_interface_analyzer(
        PyRosettaInterfaceAnalyzerInput(inputs=interface_inputs),
        config.pyrosetta_config,
    )

    results: list[ConstraintOutput] = []
    for metrics in output.results:
        value = getattr(metrics, config.metric)
        if value is None:
            raise ValueError(f"PyRosetta metric {config.metric!r} is unavailable for this structure.")
        metric_value = float(value)
        score = _score_metric(metric_value, config)
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "pyrosetta_interface_metric": config.metric,
                    "pyrosetta_interface_value": metric_value,
                    "pyrosetta_interface_score": score,
                    "pyrosetta_interface_direction": config.direction,
                    "pyrosetta_interface_desired_value": config.desired_value,
                    "pyrosetta_interface_tolerance": config.tolerance,
                    "pyrosetta_interface_metrics": metrics.model_dump(mode="json"),
                },
            )
        )

    return results
