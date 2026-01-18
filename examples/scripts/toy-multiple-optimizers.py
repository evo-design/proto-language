from __future__ import annotations
from typing import Tuple

from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig, MCMCOptimizer, MCMCOptimizerConfig
from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    Sequence,
)
from proto_language.language.core import Program
from proto_language.language.constraint import gc_content_constraint

# Construct Segment
seq1 = Segment(length=20, sequence_type="dna")

# Construct
construct = Construct([seq1])


# OPTIMIZATION STAGE 1 

# Generator
uniform_gen_config_1 = UniformMutationGeneratorConfig(
    num_mutations=10
)
uniform_gen_1 = UniformMutationGenerator(uniform_gen_config_1)

# Assign
uniform_gen_1.assign(seq1)

# Contraint
gc_constraint_1 = Constraint(
    inputs=[seq1],
    function=gc_content_constraint,
    function_config={"min_gc": 70, "max_gc": 100},
)

def topk_custom_logger(round_idx, segments):
    print(f"After round {round_idx + 1}:")
    for i, segment in enumerate(segments):
        print(f"Selected sequences for Segment {i + 1}:")
        # show metadata of each sequence in the segment
        for j, seq in enumerate(segment.selected_sequences):
            print(seq._metadata)

# Optimizer 1: TopK optimizer (standard mode)
topk_config = TopKOptimizerConfig(
    num_samples=10, 
    k=3,            
    batch_size=2,
    verbose=True, 
)

optimizer_1 = TopKOptimizer(
    constructs=[construct],
    generators=[uniform_gen_1],
    constraints=[gc_constraint_1],
    config=topk_config,
    custom_logging=topk_custom_logger

)

# OPTIMIZATION STAGE 2

# Generator
uniform_gen_config_2 = UniformMutationGeneratorConfig(
    num_mutations=1
)
uniform_gen_2 = UniformMutationGenerator(uniform_gen_config_2)

# Assign
uniform_gen_2.assign(seq1)

# Contraint
gc_constraint_2 = Constraint(
    inputs=[seq1],
    function=gc_content_constraint,
    function_config={"min_gc": 80, "max_gc": 90},
)

def mcmc_custom_logger(round_idx, segments):
    print(f"After round {round_idx + 1}:")
    for i, segment in enumerate(segments):
        print(f"Selected sequences for Segment {i + 1}:")
        # show metadata of each sequence in the segment
        for j, seq in enumerate(segment.selected_sequences):
            print(seq._metadata)

mcmc_config = MCMCOptimizerConfig(
    num_selected=1,
    mcmc_width=20,
    num_steps=10,
    track_step_size=1,
    max_temperature=2.0,
)

def mcmc_custom_logger(step: int, outputs: Tuple[Segment]) -> None:
    output_sequence: Sequence = outputs[0].candidate_sequences[0]
    gc_content = output_sequence.metadata.get('gc_content', 'N/A')
    print(
        f"Custom Log - Step {step} | "
        f"sequence: {output_sequence.sequence}, "
        f"gc_content: {gc_content}"
    )

optimizer_2 = MCMCOptimizer(
    constructs=[construct],
    generators=[uniform_gen_2],
    constraints=[gc_constraint_2],
    config=mcmc_config,
    custom_logging=mcmc_custom_logger

)
# ============================================================================
# MODE 1: Run all stages at once (existing behavior)
# ============================================================================
print("=" * 70)
print("MODE 1: All-at-once execution")
print("=" * 70)

program = Program(
    optimizers=[optimizer_1, optimizer_2],
)

program.run()

last_construct: Construct = program.constructs[0]
last_sequence: Sequence = last_construct.joined_sequences[0]
print("---------FINAL SEQUENCE (MODE 1)------------")
print(last_sequence)


# ============================================================================
# MODE 2: Run stages incrementally with inspection between stages
# ============================================================================
print("\n\n" + "=" * 70)
print("MODE 2: Incremental stage-by-stage execution")
print("=" * 70)

# Create fresh segment and construct for independent run
seq1_incremental = Segment(length=20, sequence_type="dna")
construct_incremental = Construct([seq1_incremental])

# Stage 1: TopK
uniform_gen_1_inc = UniformMutationGenerator(uniform_gen_config_1)
uniform_gen_1_inc.assign(seq1_incremental)

gc_constraint_1_inc = Constraint(
    inputs=[seq1_incremental],
    function=gc_content_constraint,
    function_config={"min_gc": 70, "max_gc": 100},
)

optimizer_1_inc = TopKOptimizer(
    constructs=[construct_incremental],
    generators=[uniform_gen_1_inc],
    constraints=[gc_constraint_1_inc],
    config=topk_config,
    custom_logging=topk_custom_logger
)

# Stage 2: MCMC
uniform_gen_2_inc = UniformMutationGenerator(uniform_gen_config_2)
uniform_gen_2_inc.assign(seq1_incremental)

gc_constraint_2_inc = Constraint(
    inputs=[seq1_incremental],
    function=gc_content_constraint,
    function_config={"min_gc": 80, "max_gc": 90},
)

optimizer_2_inc = MCMCOptimizer(
    constructs=[construct_incremental],
    generators=[uniform_gen_2_inc],
    constraints=[gc_constraint_2_inc],
    config=mcmc_config,
    custom_logging=mcmc_custom_logger
)

program_incremental = Program(
    optimizers=[optimizer_1_inc, optimizer_2_inc],
    verbose=True,
)

