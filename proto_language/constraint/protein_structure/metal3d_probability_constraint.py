"""Metal3D-based metal-site probability constraint."""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)


class Metal3DProbabilityConfig(BaseConfig):
    """Configuration for the Metal3D probability constraint.

    Attributes:
        min_probability (float): The single Metal3D probability floor, used for site detection, annotation, and scoring. Sites below it are not reported (score is worst); above it the score improves as the probability approaches 1.0. This overrides ``metal3d_config.probability_threshold``.
        metal3d_config (Metal3DPredictionConfig): Configuration passed to the Metal3D tool. Its ``probability_threshold`` is overridden by ``min_probability``; setting it to a non-default value logs a warning.
        structure_preparation (StructurePreparationConfig): How to prepare proposal structures for scoring.
        candidate_residues (ResidueSelection | None): Optional residues passed to Metal3D as candidates.
    """

    min_probability: float = ConfigField(
        default=0.2,
        ge=0.0,
        le=1.0,
        title="Min Probability",
        description="Single Metal3D probability floor for detection, annotation, and scoring; overrides tool threshold.",
    )
    metal3d_config: Metal3DPredictionConfig = ConfigField(
        default_factory=Metal3DPredictionConfig,
        title="Metal3D Config",
        description="Metal3D tool configuration; its probability_threshold is overridden by min_probability.",
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
    tools_called=["fampnn-pack", "ligandmpnn-sample", "metal3d-prediction"],
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
    metal3d_config = _apply_min_probability_floor(config)
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
        config=metal3d_config,
    )

    results: list[ConstraintOutput] = []
    for result, seq_tuple in zip(output.results, input_sequences, strict=True):
        pmetal = float(result["pmetal"])
        # Reward higher probability; energy is lowest when pmetal reaches 1.0.
        score = 1.0 - pmetal
        metadata = {
            "pmetal": pmetal,
            "min_probability": config.min_probability,
            "found": result.found,
            "num_sites": len(result.sites),
            "sites": [site.model_dump(mode="json") for site in result.sites],
            "residue_probabilities": [rp.model_dump(mode="json") for rp in result.residue_probabilities],
        }
        structures = (result.annotated_structure,) + (None,) * (len(seq_tuple) - 1)
        results.append(ConstraintOutput(score=score, metadata=metadata, structures=structures))
    return results


def _apply_min_probability_floor(config: Metal3DProbabilityConfig) -> Metal3DPredictionConfig:
    """Return a Metal3D tool config whose probability_threshold is forced to min_probability.

    ``min_probability`` is the single Metal3D probability floor; it governs the tool's site
    detection and annotation as well as scoring. Any user-set ``probability_threshold`` is
    overridden, with a warning when it differs from the tool default.
    """
    default_threshold = Metal3DPredictionConfig.model_fields["probability_threshold"].default
    if config.metal3d_config.probability_threshold != default_threshold:
        logger.warning(
            "metal3d-probability: metal3d_config.probability_threshold=%s is ignored; "
            "min_probability=%s is used as the single Metal3D probability floor.",
            config.metal3d_config.probability_threshold,
            config.min_probability,
        )
    return config.metal3d_config.model_copy(update={"probability_threshold": config.min_probability})


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
        chains={chain_map[chain_id]: positions for chain_id, positions in candidate_residues.chains.items()}
    )
    remapped.validate_against(structure, label="candidate_residues")
    return remapped
