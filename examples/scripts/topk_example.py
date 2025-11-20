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
dna_segment = Segment(starting_sequence_or_desired_length=100, sequence_type=SequenceType.DNA)

# Create construct
construct = Construct([dna_segment])

# Configure generator
mutation_config = UniformMutationGeneratorConfig(
    num_mutations=100,    
)
mutation_generator = UniformMutationGenerator(mutation_config)

# Assign generator to segment
mutation_generator.assign(dna_segment)

# Define GC content constraint
gc_constraint = Constraint(
inputs=[dna_segment],
    function=gc_content_constraint,
    function_config={"min_gc": 80, "max_gc": 100},
)


# Define custom logging function
def custom_logger(round_idx, segments):
    print(f"After round {round_idx + 1}:")
    for i, segment in enumerate(segments):
        print(f"Selected sequences for Segment {i + 1}:")
        # show metadata of each sequence in the segment
        for j, seq in enumerate(segment.selected_sequences):
            print(seq._metadata)

# Configure TopK optimizer
topk_config = TopKOptimizerConfig(
    min_num_samples=100, 
    k=3,            
    batch_size=20,
    verbose=True, 
)

# Create and run optimizer
optimizer = TopKOptimizer(
    constructs=[construct],
    generators=[mutation_generator],
    constraints=[gc_constraint],
    config=topk_config,
    custom_logging=custom_logger

)

# Run optimization
optimizer.run()

# Access results
print("\nTop 10 sequences found:")
for i in range(optimizer.k):
    sequence = dna_segment.selected_sequences[i]
    energy = optimizer.energy_scores[i]

    # Calculate actual GC content
    gc_count = sequence.sequence.count('G') + sequence.sequence.count('C')
    gc_percent = (gc_count / len(sequence.sequence)) * 100

    print(f"  {i+1}. Energy: {energy:.6f}, GC: {gc_percent:.1f}%")