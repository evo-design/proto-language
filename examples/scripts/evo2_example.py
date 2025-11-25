from __future__ import annotations
from typing import Tuple

from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig
from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    SequenceType,
    Sequence,
)
from proto_language.language.core import Program
from proto_language.language.constraint import gc_content_constraint

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 10
PROMPTS = ["ATG"] * BATCH_SIZE # 200 prompts of 3 tokens each
NUM_TOKENS: int = 1000 # Number of tokens to generate for each prompt
MIN_GC: float = 80 # Minimum GC content
MAX_GC: float = 90 # Maximum GC content


# Construct Segment
expected_length = len(PROMPTS[0]) + NUM_TOKENS
seq1 = Segment(starting_sequence_or_desired_length=expected_length, sequence_type=SequenceType.DNA)

# Construct
construct = Construct([seq1])

# Generator
evo2_gen_config = Evo2GeneratorConfig(
    prompts=PROMPTS,
    prepend_prompt=True,  # Include prompt in output
)
evo2_gen = Evo2Generator(evo2_gen_config)

# Assign
evo2_gen.assign(seq1)

# Contraint
gc_constraint = Constraint(
    inputs=[seq1],
    function=gc_content_constraint,
    function_config={"min_gc": MIN_GC, "max_gc": MAX_GC},
)

# Optimizer config
config = TopKOptimizerConfig(
    min_num_samples=BATCH_SIZE,
    k=2,                     
    max_num_samples=BATCH_SIZE * 2,       
    verbose=True,
    batch_size=BATCH_SIZE
)

# Create optimizer
optimizer = TopKOptimizer(
    constructs=[construct],
    generators=[evo2_gen],
    constraints=[gc_constraint],
    config=config,
)

# Create program with optimizer
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
