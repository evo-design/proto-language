"""Score ViennaRNA-predicted minimum free energy (MFE).

Folds each sequence with ViennaRNA's nearest-neighbor thermodynamic model and scores the predicted
minimum free energy in kcal/mol. A more-negative MFE means the model assigns a more energetically
favorable minimum-energy fold under the selected conditions; it does not by itself establish in-vivo
stability, compactness, expression, or a unique solution structure. ``direction="min"`` drives the
sequence toward more-negative MFE, ``direction="max"`` toward less-negative.

MFE is length-dependent, so ``sigmoid_center`` should be chosen for the sequence length under
optimization (during an optimizer run the segment length is fixed).
"""

import math
from typing import Any, Literal

from proto_tools import ViennaRNAConfig, ViennaRNAInput, run_viennarna

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY, sigmoid_score
from proto_language.utils.base import BaseConfig, ConfigField

_MIN_FLOAT32_NORMAL = 1.1754943508222875e-38


class MFEConfig(BaseConfig):
    """Configuration for the minimum-free-energy (MFE) constraint.

    Attributes:
        direction (Literal['min', 'max']): ``"min"`` rewards more-negative MFE (more stable
            structure), ``"max"`` rewards less-negative MFE (less structure).
        sigmoid_center (float): MFE (kcal/mol) mapped to energy 0.5. Length-dependent — set it
            for the sequence length being optimized.
        sigmoid_scale (float): Positive width of the sigmoid transform in kcal/mol.
        temperature (float): Folding temperature in Celsius passed to ViennaRNA.
    """

    direction: Literal["min", "max"] = ConfigField(
        default="min",
        title="Direction",
        description="'min' rewards more-negative MFE (more stable); 'max' rewards less structure.",
    )
    sigmoid_center: float = ConfigField(
        default=-30.0,
        allow_inf_nan=False,
        title="Sigmoid Center",
        description="MFE (kcal/mol) mapped to energy 0.5; length-dependent, tune per sequence length.",
    )
    sigmoid_scale: float = ConfigField(
        default=10.0,
        ge=_MIN_FLOAT32_NORMAL,
        allow_inf_nan=False,
        title="Sigmoid Scale",
        description="Positive float32-safe width of the sigmoid transform in kcal/mol; larger is gentler.",
    )
    temperature: float = ConfigField(
        default=37.0,
        title="Folding Temperature",
        description="ViennaRNA folding temperature in Celsius.",
    )


@constraint(
    key="mfe",
    label="Minimum Free Energy",
    config=MFEConfig,
    description="Drive a sequence toward more-negative (or less-negative) ViennaRNA minimum free energy.",
    uses_gpu=False,
    tools_called=["viennarna"],
    category="rna_secondary_structure",
    supported_sequence_types=["dna", "rna"],
)
def mfe_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: MFEConfig
) -> list[ConstraintOutput]:
    """Score sequences by ViennaRNA minimum free energy.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): One single-sequence tuple per proposal.
        config (MFEConfig): Validated MFE configuration.

    Returns:
        list[ConstraintOutput]: One result per sequence. ``score`` is in ``[0.0, 1.0]`` (lower is
            better in the requested direction); ``metadata`` carries the raw ``mfe`` (kcal/mol)
            and the ViennaRNA dot-bracket ``structure``.
    """
    if not input_sequences:
        return []

    sequences = [seq.sequence for (seq,) in input_sequences]
    output = run_viennarna(ViennaRNAInput(sequences=sequences), ViennaRNAConfig(temperature=config.temperature))

    results: list[ConstraintOutput] = []
    for result in output.results:
        mfe = result.mfe
        if mfe is None or not math.isfinite(mfe):
            raise ValueError(f"ViennaRNA returned a non-finite MFE ({mfe!r}); check the input sequence.")
        # sigmoid increases with MFE (less negative). direction="min": less-negative MFE -> higher
        # energy, so minimizing energy drives MFE more negative. direction="max": the reverse.
        sigmoid = sigmoid_score(mfe, config.sigmoid_center, slope=1.0 / config.sigmoid_scale)
        score = sigmoid if config.direction == "min" else (1.0 - sigmoid)
        metadata: dict[str, Any] = {"mfe": float(mfe), "structure": result.structure, "direction": config.direction}
        results.append(ConstraintOutput(score=min(MAX_ENERGY, max(MIN_ENERGY, score)), metadata=metadata))
    return results
