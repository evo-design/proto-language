from __future__ import annotations
from typing import Tuple

from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    SequenceType,
    Sequence,
)
from proto_language.language.core import Program
from proto_language.language.constraint import gc_content_constraint

# Construct Segment
seq1 = Segment(starting_sequence_or_desired_length=20, sequence_type=SequenceType.DNA)

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
    gc_content = output_sequence.metadata.get('gc_content', 'N/A')
    print(
        f"Custom Log - Step {step} | "
        f"sequence: {output_sequence.sequence}, "
        f"gc_content: {gc_content}"
    )

# Optimizer config
optimizer_config = MCMCOptimizerConfig(
    num_steps=10,
    track_step_size=1,
    temperature=2.0,
)

# Program
program = Program(
    optimizer_type=MCMCOptimizer,
    optimizer_config=optimizer_config,
    constructs=[construct],
    generators=[uniform_gen],
    constraints=[gc_constraint],
    custom_logging=custom_logging,
)

program.run()

# Outputs
last_construct: Construct = program.constructs[0]
last_sequence_batch: Tuple[Sequence, ...] = last_construct.joined_sequences
last_sequence: Sequence = last_sequence_batch[0]
print("---------FINAL SEQUENCE------------")
print(last_sequence)
