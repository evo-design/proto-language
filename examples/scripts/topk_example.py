"""
Simple example demonstrating the TopK Optimizer.
"""

from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig
from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    SequenceType,
)
from proto_language.language.constraint import gc_content_constraint

# Create a DNA segment
dna_segment = Segment(sequence_type=SequenceType.DNA)

# Create construct
construct = Construct([dna_segment])

# Configure generator
mutation_config = UniformMutationGeneratorConfig(
    sequence_length=100, 
    num_mutations=100,    
)
mutation_generator = UniformMutationGenerator(mutation_config)

# Assign generator to segment
mutation_generator.assign(dna_segment)

# Define GC content constraint
gc_constraint = Constraint(
inputs=[dna_segment],
    scoring_function=gc_content_constraint,
    scoring_function_config={"min_gc": 80, "max_gc": 100},
)

# Configure TopK optimizer
topk_config = TopKOptimizerConfig(
    rounds=1000,      # Run 1000 sampling rounds
    k=10,            # Keep top 10 candidates
    verbose=True,    # Show progress
)

# Create and run optimizer
optimizer = TopKOptimizer(
    constructs=[construct],
    generators=[mutation_generator],
    constraints=[gc_constraint],
    config=topk_config,
)

# Run optimization
optimizer.run()

# Access results
print("\nTop 10 sequences found:")
for i in range(optimizer.k):
    sequence = dna_segment.batch_sequences[i]
    energy = optimizer.energy_scores[i]

    # Calculate actual GC content
    gc_count = sequence.sequence.count('G') + sequence.sequence.count('C')
    gc_percent = (gc_count / len(sequence.sequence)) * 100

    print(f"  {i+1}. Energy: {energy:.6f}, GC: {gc_percent:.1f}%")