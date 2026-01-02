from __future__ import annotations
from typing import List, Tuple

from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    Sequence,
)
from proto_language.language.constraint import (
    structure_plddt_constraint,
    structure_ptm_constraint,
    structure_pae_constraint,
)
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
)
from proto_language.language.constraint.protein_structure.structure_confidence_constraint import (
    StructureConfidenceConfig,
)
from proto_language.language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
)
from proto_language.language.generator import (
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)
from proto_language.language.core import Program


N_STEPS = 8
N_REPLICATIONS = 5
DESIGN_LENGTH = 150
STRUCTURE_TOOL = 'boltz'
OUTPUT_PDB_PATH = 'protein_hunter_output.pdb'

if STRUCTURE_TOOL == 'chai':
    TOOL_CONFIG = { "use_msa_server": False }
elif STRUCTURE_TOOL == 'boltz':
    TOOL_CONFIG = { "use_msa_server": False }
else:
    raise ValueError(f"Invalid structure tool: {STRUCTURE_TOOL}")


#######################
## Segments ##
#######################

protein = Segment(
    sequence='X' * DESIGN_LENGTH,
    sequence_type="protein",
)

################
## Constructs ##
################

protein_construct = Construct([protein])

################
## Generators ##
################

with open(OUTPUT_PDB_PATH, 'w') as f:
    f.write('\n')

proteinmpnn_generator_config = ProteinMPNNGeneratorConfig(
    structure=OUTPUT_PDB_PATH,
    dynamic_structure_path=True,
    chain_ids=['A'],
    temperature=0.1,
    unallowed_amino_acids=['C'],
)
proteinmpnn_generator = ProteinMPNNGenerator(proteinmpnn_generator_config)
proteinmpnn_generator.assign(protein)

#################
## Constraints ##
#################

constraints = []

@ConstraintRegistry.register(
    key="structure-plddt-wrapper",
    label="Structure pLDDT wrapper",
    config=StructureConfidenceConfig,
    description="Wrap structure pLDDT constraint to save output PDB",
    batched=True,
    concatenate=False,
    gpu_required=True,
    tools_called=["esmfold", "alphafold3", "boltz", "chai"],
    category="protein_structure",
)
def _structure_plddt_constraint_wrapper(
    candidates: List[Tuple[Sequence, ...]],
    config: StructureConfidenceConfig,
) -> List[float]:
    """
    Wrap the structure prediction constraint and save the folding result to
    the PDB file for dynamic loading by ProteinMPNN.
    """
    results = structure_plddt_constraint(candidates, config)

    first_sequence = candidates[0][0]
    pdb_output = first_sequence._metadata.get("pdb_output")
    if pdb_output is None:
        raise ValueError("Structure prediction did not produce PDB output")
    with open(OUTPUT_PDB_PATH, 'w') as f:
        f.write(pdb_output)

    return results

constraint_plddt = Constraint(
    inputs=[protein] * N_REPLICATIONS,
    function=_structure_plddt_constraint_wrapper,
    function_config={
        "structure_tool": STRUCTURE_TOOL,
        "tool_config": TOOL_CONFIG,
    },
)
constraints.append(constraint_plddt)

constraint_ptm = Constraint(
    inputs=[protein] * N_REPLICATIONS,
    function=structure_ptm_constraint,
    function_config={
        "structure_tool": STRUCTURE_TOOL,
        "tool_config": TOOL_CONFIG,
    },
)
constraints.append(constraint_ptm)

if STRUCTURE_TOOL == 'boltz':
    constraint_pae = Constraint(
        inputs=[protein] * N_REPLICATIONS,
        function=structure_pae_constraint,
        function_config={
            "structure_tool": STRUCTURE_TOOL,
            "tool_config": TOOL_CONFIG,
        },
    )
    constraints.append(constraint_pae)

#############
## Program ##
#############


def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
    output_sequence: Sequence = outputs[0].selected_sequences[0]
    metakeys = list(output_sequence._metadata.keys())
    avg_plddt, ptm, avg_pae = None, None, None
    for key in metakeys:
        if key.endswith('avg_plddt'):
            avg_plddt = output_sequence._metadata[key]
        if key.endswith('ptm'):
            ptm = output_sequence._metadata[key]
        if key.endswith('avg_pae'):
            avg_pae = output_sequence._metadata[key]
    print(
        f"Iteration {step} | \n"
        f"\tsequence: {output_sequence._sequence} \n"
        f"\tpLDDT: {avg_plddt} \n"
        f"\tpTM: {ptm} \n"
        f"\tpAE: {avg_pae}"
    )

mcmc_optimizer_config = MCMCOptimizerConfig(
    num_selected=1,
    mcmc_width=1,
    num_steps=N_STEPS,
    max_temperature=10000000,
    min_temperature=10000000 - 1,
    track_step_size=1,
    verbose=True,
)

optimizer = MCMCOptimizer(
    config=mcmc_optimizer_config,
    constructs=[protein_construct],
    generators=[proteinmpnn_generator],
    constraints=constraints,
    custom_logging=custom_logging,
)

program = Program(
    optimizers=[optimizer],
)

program.run()
