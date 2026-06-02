"""Run the smallest end-to-end DNA optimization example.

This script shows the canonical proto-language flow: define one variable DNA
segment, assign a random nucleotide generator, add GC-content and homopolymer
constraints, run MCMC, and inspect the final joined sequence. The toy objective
drives a 100 bp synthetic DNA sequence to balanced 40-60% GC content with no long
homopolymers. It mirrors ``examples/jsons/toy.json`` and the ``write-program``
skill template, and is the minimal pattern for copying Program, Construct,
Segment, Generator, Constraint, and Optimizer wiring.
"""

from proto_language.constraint import gc_content_constraint, max_homopolymer_constraint
from proto_language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig

# Construct Segment
seq1 = Segment(length=100, sequence_type="dna")

# Construct
construct = Construct([seq1])

# Generator
uniform_gen_config = RandomNucleotideGeneratorConfig()
uniform_gen = RandomNucleotideGenerator(uniform_gen_config)

# Assign
uniform_gen.assign(seq1)

# Constraints
gc_constraint = Constraint(
    inputs=[seq1],
    function=gc_content_constraint,
    function_config={"min_gc": 40, "max_gc": 60},
)
homopolymer_constraint = Constraint(
    inputs=[seq1],
    function=max_homopolymer_constraint,
    function_config={"max_length": 5},
)


def custom_logging(step: int, outputs: tuple[Segment]) -> None:
    output_sequence: Sequence = outputs[0].proposal_sequences[0]
    gc_content = output_sequence._constraints_metadata["gc_content_constraint"]["data"]["gc_content"]
    print(f"Custom Log - Step {step} | sequence: {output_sequence.sequence}, gc_content: {gc_content}")


# Optimizer config
optimizer_config = MCMCOptimizerConfig(
    num_results=1,
    proposals_per_result=1,
    num_steps=100,
    max_temperature=1.0,
)

# Optimizer
optimizer = MCMCOptimizer(
    constructs=[construct],
    generators=[uniform_gen],
    constraints=[gc_constraint, homopolymer_constraint],
    config=optimizer_config,
    custom_logging=custom_logging,
)

# Program
program = Program(
    optimizers=[optimizer],
    num_results=1,
)

program.run()

# Outputs
last_construct: Construct = program.constructs[0]
last_sequence_batch: tuple[Sequence, ...] = last_construct.joined_sequences
last_sequence: Sequence = last_sequence_batch[0]
print("---------FINAL SEQUENCE------------")
print(last_sequence)
