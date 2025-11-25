from __future__ import annotations
from typing import Tuple

from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import (
    esmfold_plddt_constraint,
    esmfold_ptm_constraint,
    protein_globularity_constraint,
    protein_symmetry_ring_constraint,
)
from proto_language.language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
)
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.core import Program


MONOMER_LENGTH = 50
N_SYMMETRIC_UNITS = 3
N_STEPS = 30_000


#######################
## Segments ##
#######################

protomer = Segment(
    starting_sequence_or_desired_length=MONOMER_LENGTH,
    sequence_type=SequenceType.PROTEIN,
)

################
## Constructs ##
################

protomer_construct = Construct([protomer])

################
## Generators ##
################

uniform_gen_config = UniformMutationGeneratorConfig()
uniform_gen = UniformMutationGenerator(uniform_gen_config)

uniform_gen.assign(protomer)

#################
## Constraints ##
#################

esmfold_plddt = Constraint(
    inputs=[protomer],
    function=esmfold_plddt_constraint,
    function_config={"n_replications": N_SYMMETRIC_UNITS},
)

esmfold_ptm = Constraint(
    inputs=[protomer],
    function=esmfold_ptm_constraint,
    function_config={"n_replications": N_SYMMETRIC_UNITS},
)

symmetry = Constraint(
    inputs=[protomer],
    function=protein_symmetry_ring_constraint,
    function_config={
        "n_replications": N_SYMMETRIC_UNITS,
        "all_to_all_protomer_symmetry": True,
    },
)

globularity = Constraint(
    inputs=[protomer],
    function=protein_globularity_constraint,
    function_config={"n_replications": N_SYMMETRIC_UNITS},
)

#############
## Program ##
#############

def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
    output_sequence: Sequence = outputs[0].selected_sequences[0]
    metakeys = list(output_sequence._metadata.keys())
    folded_sequence = output_sequence._metadata[
        next(key for key in metakeys if key.endswith('esmfolded_sequence'))
    ]
    plddt = output_sequence._metadata[
        next(key for key in metakeys if key.endswith('avg_plddt'))
    ]
    ptm = output_sequence._metadata[
        next(key for key in metakeys if key.endswith('ptm'))
    ]
    print(
        f"Iteration {step} | \n"
        f"\tsequence (monomer): {output_sequence._sequence}, \n"
        f"\tsequence (duplicated): {folded_sequence}, \n"
        f"\tpLDDT: {plddt}, \n"
        f"\tpTM: {ptm}"
    )


mcmc_optimizer_config = MCMCOptimizerConfig(
    mcmc_width=1,
    num_steps=N_STEPS,
    max_temperature=1.,
    min_temperature=0.0001,
    track_step_size=1,
    verbose=True,
)


# Create optimizer
optimizer = MCMCOptimizer(
    constructs=[protomer_construct],
    generators=[uniform_gen],
    constraints=[
        esmfold_plddt,
        esmfold_ptm,
        symmetry,
        globularity,
    ],
    config=mcmc_optimizer_config,
    constraint_weights=[
        20.0,
        20.0,
        1.0,
        1.0,
    ],
    custom_logging=custom_logging,
)

# Create program with optimizer
program = Program(
    optimizers=[optimizer],
)

program.run()

with open("design.pdb", "w") as f:
    # Outputs
    final_construct: Construct = program.constructs[0]
    final_sequence_batch: Tuple[Sequence, ...] = final_construct.joined_sequences
    final_sequence: Sequence = final_sequence_batch[0]
    f.write(final_sequence._metadata["pdb_output"] + "\n")
