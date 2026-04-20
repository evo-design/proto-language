"""Two-stage gradient protein hallucination (Germinal-like pipeline).

Demonstrates the GradientOptimizer in a two-stage pipeline:
  Stage 1 (logit phase): soft ramps 0→1, temperature fixed, logits evolve
  Stage 2 (softmax phase): soft=1, temperature anneals 1→0.01, sequences sharpen

This example uses mock backward functions that push toward alanine,
so it runs on CPU without real models. To use real constraints, replace
the mock backward with e.g. ``af2_binder_backward`` and ``ablang_naturalness_gradient_backward``.
"""

import numpy as np
from pydantic import BaseModel

from proto_language.language.core import Constraint, Construct, Program, Segment
from proto_language.language.core.constraint import GradientResult
from proto_language.language.generator import PositionWeightGenerator, PositionWeightGeneratorConfig
from proto_language.language.optimizer import GradientOptimizer, GradientOptimizerConfig

# --- Mock backward (replace with real differentiable constraints for production) ---


class _EmptyCfg(BaseModel):
    """Empty config for mock backward."""


def mock_structure_backward(  # noqa: ARG001 -- params required by backward protocol
    inputs: tuple, *, config: BaseModel, temperature: float = 1.0, soft: float = 1.0, **kwargs: object
) -> GradientResult:
    """Mock structural constraint: pushes logits toward a target distribution."""
    logits = inputs[0].logits
    target = np.zeros_like(logits)
    target[:, 0] = 1.0  # Prefer alanine
    grad = logits - target
    return GradientResult(gradient=(grad,), loss=float(np.mean(grad**2)), metrics={})


# --- Pipeline setup ---

# Design a 20-residue protein starting from a VHH seed
segment = Segment(sequence="EVQLVESGGGLVQPGGSLRL", sequence_type="protein", label="binder")
construct = Construct([segment])

# Stage 1: Logit phase — soft ramps 0→1, no temperature annealing
gen1 = PositionWeightGenerator(PositionWeightGeneratorConfig())
con1 = Constraint(
    inputs=[segment], backward=mock_structure_backward, backward_config=_EmptyCfg(), label="structure_s1",
)
# For real pipelines, add a naturalness constraint:
# con1_nat = Constraint(inputs=[segment], backward=ablang_naturalness_gradient_backward,
#     backward_config=AbLangConstraintConfig(temperature=0.6), label="ablang_s1", weight=0.2)

stage1 = GradientOptimizer(
    target_segment=segment,
    constructs=[construct],
    generators=[gen1],
    constraints=[con1],
    config=GradientOptimizerConfig.germinal_logit_preset(),
)

# Stage 2: Softmax phase — soft=1, temperature anneals 1→0.01
gen2 = PositionWeightGenerator(PositionWeightGeneratorConfig())
con2 = Constraint(
    inputs=[segment], backward=mock_structure_backward, backward_config=_EmptyCfg(), label="structure_s2",
)

stage2 = GradientOptimizer(
    target_segment=segment,
    constructs=[construct],
    generators=[gen2],
    constraints=[con2],
    config=GradientOptimizerConfig.germinal_softmax_preset(),
)

# --- Run ---

program = Program(optimizers=[stage1, stage2], num_results=1)
program.run()

# --- Results ---

result = segment.result_sequences[0]
print(f"Input:    {segment.original_sequence.sequence}")
print(f"Designed: {result.sequence}")
print(f"Logits shape: {result.logits.shape if result.logits is not None else 'None'}")
print(f"Final energy: {program.energy_scores}")
