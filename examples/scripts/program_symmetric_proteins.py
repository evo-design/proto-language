from typing import Tuple

import sys
import os
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
from proto_language.language.generator import MCMCGenerator, UniformMutationGenerator
from proto_language.language.core import Program


MONOMER_LENGTH = 150
N_SYMMETRIC_UNITS = 3
N_STEPS = 30_000


#######################
## Segments ##
#######################

protomer = Segment(
    sequence_type=SequenceType.PROTEIN,
)

################
## Constructs ##
################

protomer_construct = Construct([protomer])

################
## Generators ##
################

uniform_gen = UniformMutationGenerator(
    batch_size=1,
    sequence_length=MONOMER_LENGTH,
)

uniform_gen.assign(protomer)

#################
## Constraints ##
#################

esmfold_plddt = Constraint(
    inputs=[protomer],
    scoring_function=esmfold_plddt_constraint,
    scoring_function_config={"n_replications": N_SYMMETRIC_UNITS},
)

esmfold_ptm = Constraint(
    inputs=[protomer],
    scoring_function=esmfold_ptm_constraint,
    scoring_function_config={"n_replications": N_SYMMETRIC_UNITS},
)

symmetry = Constraint(
    inputs=[protomer],
    scoring_function=protein_symmetry_ring_constraint,
    scoring_function_config={
        "n_replications": N_SYMMETRIC_UNITS,
        "all_to_all_protomer_symmetry": True,
    },
)

globularity = Constraint(
    inputs=[protomer],
    scoring_function=protein_globularity_constraint,
    scoring_function_config={"n_replications": N_SYMMETRIC_UNITS},
)

#############
## Program ##
#############


def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
    output_sequence: Sequence = outputs[0].batch_sequences[0]
    print(
        f"Iteration {step} | "
        f"sequence (real): {output_sequence._sequence}, "
        f"sequence: {output_sequence._metadata['esmfolded_sequence']}, "
        f"pLDDT: {output_sequence._metadata['avg_plddt']}, "
        f"pTM: {output_sequence._metadata['ptm']}"
    )


program = Program(
    iterative_generator_type=MCMCGenerator,
    constructs=[protomer_construct],
    generators=[uniform_gen],
    constraints=[
        esmfold_plddt,
        esmfold_ptm,
        symmetry,
        globularity,
    ],
    constraint_weights=[
        20.0,
        20.0,
        1.0,
        1.0,
    ],
    num_steps=N_STEPS,
    track_step_size=1,
    temperature=2.0,
    custom_logging=custom_logging,
)

program.run()

with open("design.pdb", "w") as f:
    # Outputs
    final_construct: Construct = program.constructs[0]
    final_sequence_batch: Tuple[Sequence, ...] = final_construct.batch_sequences
    final_sequence: Sequence = final_sequence_batch[0]
    f.write(final_sequence._metadata["pdb_output"] + "\n")
