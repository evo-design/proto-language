"""Metal3D-based metal-site probability constraint."""

from __future__ import annotations

from proto_tools import (
    Metal3DPredictionConfig,
    Metal3DPredictionInput,
    Metal3DStructureInput,
    Structure,
    run_metal3d_prediction,
)
from proto_tools.entities.structures import ResidueSelection

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.structure_preparation import (
    StructurePreparationConfig,
    prepare_structures_for_proposals,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import BaseConfig, ConfigField


class Metal3DProbabilityConfig(BaseConfig):
    """Configuration for the Metal3D probability constraint.

    Attributes:
        target_probability (float): Metal3D probability target where the score reaches zero.
        metal3d_config (Metal3DPredictionConfig): Configuration passed to the Metal3D tool.
        structure_preparation (StructurePreparationConfig): How to prepare proposal structures for scoring.
        candidate_residues (ResidueSelection | None): Optional residues passed to Metal3D as candidates.
    """

    target_probability: float = ConfigField(
        default=0.5,
        ge=0.0,
        le=1.0,
        title="Target Probability",
        description="Desired minimum Metal3D site probability. Scores are zero once this target is met.",
    )
    metal3d_config: Metal3DPredictionConfig = ConfigField(
        default_factory=Metal3DPredictionConfig,
        title="Metal3D Config",
        description="Metal3D tool configuration.",
    )
    structure_preparation: StructurePreparationConfig = ConfigField(
        default_factory=StructurePreparationConfig,
        title="Structure Preparation",
        description="How to obtain proposal-specific structures before Metal3D scoring.",
    )
    candidate_residues: ResidueSelection | None = ConfigField(
        default=None,
        title="Candidate Residues",
        description="Optional candidate residues to pass to Metal3D.",
    )


@constraint(
    key="metal3d-probability",
    label="Metal3D Probability",
    config=Metal3DProbabilityConfig,
    description="Reward protein sequences whose prepared structures contain high-probability Metal3D metal-ion sites.",
    uses_gpu=True,
    tools_called=["fampnn-pack", "metal3d-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def metal3d_probability_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: Metal3DProbabilityConfig,
) -> list[ConstraintOutput]:
    """Score proposals by Metal3D metal-site probability; lower is better."""
    prepared_structures = prepare_structures_for_proposals(input_sequences, config.structure_preparation)
    output = run_metal3d_prediction(
        inputs=Metal3DPredictionInput(
            inputs=[
                Metal3DStructureInput(
                    structure=structure,
                    candidate_residues=_candidate_residues_for_structure(
                        config.candidate_residues,
                        structure,
                        config.structure_preparation,
                    ),
                )
                for structure in prepared_structures
            ]
        ),
        config=config.metal3d_config,
    )

    results: list[ConstraintOutput] = []
    denom = max(config.target_probability, 1e-8)
    for result, seq_tuple in zip(output.results, input_sequences, strict=True):
        pmetal = float(result["pmetal"])
        score = max(0.0, config.target_probability - pmetal) / denom
        metadata = {
            "pmetal": pmetal,
            "target_probability": config.target_probability,
            "found": result.found,
            "num_sites": len(result.sites),
            "sites": [site.model_dump(mode="json") for site in result.sites],
            "residue_probabilities": [rp.model_dump(mode="json") for rp in result.residue_probabilities],
        }
        structures = (result.annotated_structure,) + (None,) * (len(seq_tuple) - 1)
        results.append(ConstraintOutput(score=score, metadata=metadata, structures=structures))
    return results


def _candidate_residues_for_structure(
    candidate_residues: ResidueSelection | None,
    structure: Structure,
    preparation_config: StructurePreparationConfig,
) -> ResidueSelection | None:
    if candidate_residues is None:
        return None

    available_chains = set(structure.get_chain_ids())
    if set(candidate_residues.chains).issubset(available_chains):
        return candidate_residues

    source_chain_ids = preparation_config.chain_ids
    prepared_chain_ids = structure.get_chain_ids()
    if source_chain_ids is None or len(source_chain_ids) != len(prepared_chain_ids):
        return candidate_residues

    chain_map = dict(zip(source_chain_ids, prepared_chain_ids, strict=True))
    if not set(candidate_residues.chains).issubset(chain_map):
        return candidate_residues

    remapped = ResidueSelection(
        chains={
            chain_map[chain_id]: positions
            for chain_id, positions in candidate_residues.chains.items()
        }
    )
    remapped.validate_against(structure, label="candidate_residues")
    return remapped
