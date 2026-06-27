from proto_language.core import Segment, Construct, Constraint, Program
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.constraint import gc_content_constraint, max_homopolymer_constraint
from proto_tools.transforms.masking import MaskingStrategy

# Define a 200bp DNA sequence to optimize.
# RandomNucleotideGenerator in mutation mode requires a valid starting sequence
# to mutate; provide a 50% GC random seed (here: repeating ATCG as a neutral start).
_SEED = "ATCG" * 50  # 200 nt, exactly 50% GC, no homopolymers
dna = Segment(sequence=_SEED, sequence_type="dna")
construct = Construct(segments=[dna])

# Generator: random point mutations to explore sequence space
gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3)))
gen.assign(dna)

# Constraints: what the sequence must satisfy
constraints = [
    Constraint(inputs=[dna], function=gc_content_constraint,
               function_config={"min_gc": 45, "max_gc": 55}, weight=1.0),
    Constraint(inputs=[dna], function=max_homopolymer_constraint,
               function_config={"max_length": 5}, threshold=0.0),
]

# Optimize with MCMC
optimizer = MCMCOptimizer(
    constructs=[construct], generators=[gen], constraints=constraints,
    config=MCMCOptimizerConfig(num_steps=500, num_results=5),
)

program = Program(optimizers=[optimizer], num_results=5)
program.run()

# Results: 5 optimized sequences ranked by quality
for seq in construct.joined_sequences:
    print(seq.sequence)