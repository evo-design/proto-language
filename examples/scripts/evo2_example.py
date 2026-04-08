from proto_language.language.constraint import gc_content_constraint
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 10
PROMPTS = ["ATG"] * BATCH_SIZE  # 200 prompts of 3 tokens each
NUM_TOKENS: int = 1000  # Number of tokens to generate for each prompt
MIN_GC: float = 80  # Minimum GC content
MAX_GC: float = 90  # Maximum GC content


# Construct Segment
expected_length = len(PROMPTS[0]) + NUM_TOKENS
seq1 = Segment(length=expected_length, sequence_type="dna")

# Construct
construct = Construct([seq1])

# Generator
evo2_gen_config = Evo2GeneratorConfig(
    prompts=PROMPTS,
    prepend_prompt=True,  # Include prompt in output
    batch_size=BATCH_SIZE,
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

# Optimizer config (threshold mode)
config = RejectionSamplingOptimizerConfig(
    num_samples=BATCH_SIZE * 2,
    energy_threshold=0.0,
    samples_per_round=BATCH_SIZE,
    verbose=True,
)

# Create optimizer
optimizer = RejectionSamplingOptimizer(
    constructs=[construct],
    generators=[evo2_gen],
    constraints=[gc_constraint],
    config=config,
)

# Create program with optimizer
program = Program(
    optimizers=[optimizer],
    num_results=2,
)

program.run()

# Outputs
last_construct: Construct = program.constructs[0]
last_sequence_batch: tuple[Sequence, ...] = last_construct.joined_sequences
last_sequence: Sequence = last_sequence_batch[0]
print("---------FINAL SEQUENCE------------")
print(last_sequence)
