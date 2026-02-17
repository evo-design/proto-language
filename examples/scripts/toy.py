from __future__ import annotations

from typing import Tuple

from proto_language.language.constraint import gc_content_constraint
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig

# Construct Segment
seq1 = Segment(length=20, sequence_type="dna")

# Construct
construct = Construct([seq1])

# Generator
uniform_gen_config = UniformMutationGeneratorConfig()
uniform_gen = UniformMutationGenerator(uniform_gen_config)

# Assign
uniform_gen.assign(seq1)

# Contraint
gc_constraint = Constraint(
    inputs=[seq1],
    function=gc_content_constraint,
    function_config={"min_gc": 80, "max_gc": 90},
)

def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
    output_sequence: Sequence = outputs[0].candidate_sequences[0]
    gc_content = output_sequence._metadata["constraints"]["gc_content_constraint"]["data"].get("gc_content", "N/A")
    print(
        f"Custom Log - Step {step} | "
        f"sequence: {output_sequence.sequence}, "
        f"gc_content: {gc_content}"
    )

# Optimizer config
optimizer_config = MCMCOptimizerConfig(
    num_selected=1,
    mcmc_width=20,
    num_steps=10,
    max_temperature=2.0,
)

# Optimizer
optimizer = MCMCOptimizer(
    constructs=[construct],
    generators=[uniform_gen],
    constraints=[gc_constraint],
    config=optimizer_config,
    custom_logging=custom_logging,
)

# Program
program = Program(
    optimizers=[optimizer],
)

program.run()

# Outputs
last_construct: Construct = program.constructs[0]
last_sequence_batch: Tuple[Sequence, ...] = last_construct.joined_sequences
last_sequence: Sequence = last_sequence_batch[0]
print("---------FINAL SEQUENCE------------")
print(last_sequence)
