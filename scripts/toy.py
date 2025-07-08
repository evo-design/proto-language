from typing import Tuple

import sys

sys.path.append(".")

from language.generator import UniformMutationGenerator, MCMCGenerator
from language.base import (
    Constraint,
    Construct,
    ConstructSegment,
    SequenceType,
    Sequence,
)
from language.program import Program
from language.constraint import gc_content_constraint

# Construct Segment
seq1 = ConstructSegment(sequence_type=SequenceType.DNA)

# Construct

construct = Construct([seq1])

# Generator
uniform_gen = UniformMutationGenerator(
    sequence_length=20,
    batch_size=1,
)

# Assign
uniform_gen.assign(seq1)


# Contraint
def analyze_gc_content(sequence: str) -> float:
    """Calculate the GC content of a DNA sequence."""
    gc_count = 0
    for nucleotide in sequence:
        if nucleotide in "GC":
            gc_count += 1

    return (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0.0


gc_constraint = Constraint(
    inputs=[seq1],
    scoring_function=gc_content_constraint,
    scoring_function_config={"min_gc": 80, "max_gc": 90},
)


def custom_logging(step: int, outputs: Tuple[ConstructSegment]) -> None:
    output_sequence: Sequence = outputs[0].batch_sequences[0]
    print(
        f"Iteration {step} | "
        f"time_step: {output_sequence._metadata['time_step']}, "
        f"sequence: {output_sequence._sequence}, "
        f"metadata_sequence: {output_sequence._metadata['sequence']}, "
        f"gc_content: {output_sequence._metadata['gc_content']}, "
        # TODO: the temperature key is not found
        # f"temperature: {output_sequence._metadata['temperature']}"
    )


# Program
program = Program(
    iterative_generator_type=MCMCGenerator,
    constructs=[construct],
    generators=[uniform_gen],
    constraints=[gc_constraint],
    num_steps=10,
    track_step_size=1,
    custom_logging=custom_logging,
    temperature=2.0,
)

sequence_history = program.run()

# Outputs
last_sequence_history: Tuple[Construct, ...] = sequence_history[-1]
last_construct: Construct = last_sequence_history[0]
last_sequence_batch: Tuple[Sequence, ...] = last_construct.batch_sequences
last_sequence: Sequence = last_sequence_batch[0]
print("---------FINAL SEQUENCE------------")
print(last_sequence)
