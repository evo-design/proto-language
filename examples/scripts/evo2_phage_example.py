## NOTE: THIS SCRIPT ONLY WORKS ON CHIMERA DUE TO PHROGS DATABASE PATH

from typing import Tuple

import sys
import os

from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    SequenceType,
    Sequence,
)
from proto_language.language.core import Program
from proto_language.language.constraint import sequence_length_constraint, gc_content_constraint, max_homopolymer_constraint, dinucleotide_frequency_constraint, tetranucleotide_usage_constraint, orfipy_mmseqs_gene_hit_count_constraint, orfipy_mmseqs_gene_homology_constraint

NUM_MCMC_STEPS = 3 # Number of MCMC steps to run
MIN_GC = 35  # Min target for high GC content (%)
MAX_GC = 65  # Max target for high GC content (%)
MAX_HOMOPOLYMER = 10
MIN_FREQ = 0.0  # Min dinucleotide frequency (proportion 0-1)
MAX_FREQ = 0.20  # Max dinucleotide frequency (proportion 0-1, 20% = 0.20)
TETRANUCLEOTIDE = "GATC"
MIN_TUD = 0.0  # Min tetranucleotide usage deviation
MAX_TUD = 1.0  # Max tetranucleotide usage deviation
MIN_GENE_HITS = 10  # Min target for gene hits
MAX_GENE_HITS = 12  # Max target for gene hits
TRACK_EVERY = 1

# Add custom model to evo2 registry
from evo2.utils import MODEL_NAMES, CONFIG_MAP
MODEL_NAMES.append("evo2_7b_phage")
# Use absolute path to config file based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'configs', 'config.yaml')
CONFIG_MAP["evo2_7b_phage"] = config_path

#######################
## Segments ##
#######################

segment = Segment(
    sequence_type=SequenceType.DNA,
    valid_chars=set("ATCG+~ "),
)

###############
## Generator ##
###############

# Initialize ProgramGenerator
evo2_config = Evo2GeneratorConfig(
    prompts=["+~GAGTTTTA"],
    model_name="evo2_7b_phage",
    local_path="/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_phage/iter_12000.pt",
    num_tokens=5500,
    temperature=0.9,
    prepend_prompt=True,
)
evo2_generator = Evo2Generator(evo2_config)

evo2_generator.assign(segment)

################
## Construct ##
################

construct = Construct([segment])

#################
## Constraints ##
#################

sequence_length = Constraint(
    inputs=[segment],
    scoring_function=sequence_length_constraint,
    scoring_function_config={"target_length": 5386}
)

gc_content = Constraint(
    inputs=[segment],
    scoring_function=gc_content_constraint,
    scoring_function_config={"min_gc": MIN_GC, "max_gc": MAX_GC}
)

max_homopolymer = Constraint(
    inputs=[segment],
    scoring_function=max_homopolymer_constraint,
    scoring_function_config={"max_length": MAX_HOMOPOLYMER}
)

dinucleotide_frequency = Constraint(
    inputs=[segment],
    scoring_function=dinucleotide_frequency_constraint,
    scoring_function_config={"min_freq": MIN_FREQ, "max_freq": MAX_FREQ}
)

tetranucleotide_usage = Constraint(
    inputs=[segment],
    scoring_function=tetranucleotide_usage_constraint,
    scoring_function_config={"tetranucleotide": TETRANUCLEOTIDE, "min_tud": MIN_TUD, "max_tud": MAX_TUD}
)

gene_hit_count_config = {
    "min_hits": MIN_GENE_HITS,
    "max_hits": MAX_GENE_HITS,
    "orfipy_config": {
        "input_fasta": "",  # Placeholder - filled at runtime
        "output_dir": "",   # Placeholder - filled at runtime
        "threads": 96,
        "min_len": 30,
        "max_len": 5500
    },
    "mmseqs_config": {
        "query_fasta": "",  # Placeholder - filled at runtime
        "mmseqs_db": "/large_storage/hielab/samuelking/phrogs/phrogs_mmseqs_db/phrogs_mmseqs_db",
        "results_dir": "",  # Placeholder - filled at runtime
        "threads": 96,
        "sensitivity": 4.0
    }
}
gene_hit_count = Constraint(
    inputs=[segment],
    scoring_function=orfipy_mmseqs_gene_hit_count_constraint,
    scoring_function_config=gene_hit_count_config
)

gene_homology_config = {
    "min_homology": 80.0,  # Minimum 80% homology
    "max_homology": 100.0,  # Maximum 100% homology
    "orfipy_config": {
        "input_fasta": "",  # Placeholder - filled at runtime
        "output_dir": "",   # Placeholder - filled at runtime
        "threads": 96,
        "min_len": 30,
        "max_len": 5500
    },
    "mmseqs_config": {
        "query_fasta": "",  # Placeholder - filled at runtime
        "mmseqs_db": "/large_storage/hielab/samuelking/phrogs/phrogs_mmseqs_db/phrogs_mmseqs_db",
        "results_dir": "",  # Placeholder - filled at runtime
        "threads": 96,
        "sensitivity": 4.0
    }
}
gene_homology = Constraint(
    inputs=[segment],
    scoring_function=orfipy_mmseqs_gene_homology_constraint,
    scoring_function_config=gene_homology_config
)

#############
## Program ##
#############

# Optimizer config
optimizer_config = MCMCOptimizerConfig(
    num_steps=NUM_MCMC_STEPS,
    track_step_size=TRACK_EVERY,
)

# Initialize Program with correct sequence_order type
program = Program(
    optimizer_type=MCMCOptimizer,
    optimizer_config=optimizer_config,
    constructs=[construct],
    generators=[evo2_generator],
    constraints=[
        sequence_length,
        gc_content,
        max_homopolymer,
        dinucleotide_frequency,
        tetranucleotide_usage,
        gene_hit_count,
        gene_homology,
    ],
)

program.run()

# Outputs
final_construct: Construct = program.constructs[0]
final_sequence_batch: Tuple[Sequence, ...] = final_construct.joined_sequences
final_sequence: Sequence = final_sequence_batch[0]
print("---------FINAL SEQUENCE------------")
print(final_sequence)