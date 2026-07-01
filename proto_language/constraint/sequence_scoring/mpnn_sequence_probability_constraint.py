"""MPNN sequence probability constraint for structure-conditioned objectives."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from proto_tools import (
    InverseFoldingStructureInput,
    LigandMPNNScoringConfig,
    LigandMPNNScoringInput,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    SequenceStructurePair,
    Structure,
    run_ligandmpnn_score,
    run_proteinmpnn_score,
)
from proto_tools.entities.structures import ResidueSelection
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.generator.mpnn_mutation_generator import ProteinMPNNModelChoice
from proto_language.utils.base import BaseConfig, ConfigField

MPNNSequenceProbabilityModel = Literal["ligandmpnn", "proteinmpnn"]
MPNNSequenceProbabilityScoreMode = Literal["probability_loss", "nll", "perplexity"]


class MPNNSequenceProbabilityConfig(BaseConfig):
    """Configuration for structure-conditioned MPNN sequence-probability scoring.

    Attributes:
        model (MPNNSequenceProbabilityModel): Structure-conditioned MPNN
            variant used to score sequence probability.
        structure_inputs (list[InverseFoldingStructureInput]): Backbone
            structure inputs, with optional chains to redesign and fixed
            positions.
        output_chain_id (str | None): Chain whose sequence is supplied by the
            optimizer proposal; inferred when unambiguous.
        score_mode (MPNNSequenceProbabilityScoreMode): Objective returned by
            the constraint: probability loss, mean NLL, or perplexity.
        proteinmpnn_model_choice (ProteinMPNNModelChoice): ProteinMPNN weights
            used when ``model`` is ``"proteinmpnn"``.
        ligand_mpnn_use_side_chain_context (bool): Whether LigandMPNN scoring
            conditions on fixed-residue sidechain atoms.
        ligand_mpnn_cutoff_for_score (float): Ligand-residue distance cutoff
            used by LigandMPNN scoring.
        seed (int | None): Optional random seed for MPNN scoring.
        device (str): Device for MPNN scoring, for example ``"cuda"``.
        verbose (bool): Whether to print MPNN scoring progress.
    """

    model: MPNNSequenceProbabilityModel = ConfigField(
        default="ligandmpnn",
        title="MPNN Model",
        description="Structure-conditioned model used to score the sequence.",
    )
    structure_inputs: list[InverseFoldingStructureInput] = ConfigField(
        title="Structure Inputs",
        description="Structure(s) for MPNN scoring, with optional chains_to_redesign and fixed_positions.",
    )
    output_chain_id: str | None = ConfigField(
        default=None,
        title="Output Chain",
        description="Structure chain corresponding to the scored input sequence.",
    )
    score_mode: MPNNSequenceProbabilityScoreMode = ConfigField(
        default="probability_loss",
        title="Score Mode",
        description="Return 1-exp(avg_log_likelihood), mean NLL, or perplexity.",
    )
    proteinmpnn_model_choice: ProteinMPNNModelChoice = ConfigField(
        default="proteinmpnn",
        title="ProteinMPNN Weights",
        description="ProteinMPNN weights used when model='proteinmpnn'.",
    )
    ligand_mpnn_use_side_chain_context: bool = ConfigField(
        default=False,
        title="LigandMPNN Sidechain Context",
        description="Whether LigandMPNN scoring conditions on fixed-residue sidechain atoms.",
    )
    ligand_mpnn_cutoff_for_score: float = ConfigField(
        default=8.0,
        gt=0.0,
        title="LigandMPNN Ligand Cutoff",
        description="Ligand-residue distance cutoff (Å) used by LigandMPNN scoring.",
    )
    seed: int | None = ConfigField(
        default=None,
        title="Random Seed",
        description="Seed for MPNN scoring. None lets proto-tools choose its default seed behavior.",
        ge=0,
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for MPNN scoring.",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print MPNN scoring progress.",
    )

    @field_validator("structure_inputs", mode="before")
    @classmethod
    def normalize_structure_inputs(cls, value: Any) -> Any:
        """Convert flexible structure inputs to ``InverseFoldingStructureInput`` objects."""
        if not isinstance(value, list):
            value = [value]
        normalized = []
        for item in value:
            if isinstance(item, InverseFoldingStructureInput):
                normalized.append(item)
            elif isinstance(item, (str, Structure)):
                normalized.append(InverseFoldingStructureInput(structure=item))
            elif isinstance(item, dict):
                normalized.append(InverseFoldingStructureInput(**item))
            else:
                raise ValueError(f"Unsupported structure_inputs item type: {type(item)}")
        return normalized


@constraint(
    key="mpnn-sequence-probability",
    label="MPNN Sequence Probability",
    config=MPNNSequenceProbabilityConfig,
    description="Score sequence compatibility with a structure-conditioned MPNN model.",
    uses_gpu=True,
    tools_called=["ligandmpnn-score", "proteinmpnn-score"],
    category="sequence_scoring",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def mpnn_sequence_probability_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: MPNNSequenceProbabilityConfig,
) -> list[ConstraintOutput]:
    """Score proposals by MPNN probability; lower is better."""
    structure_inputs = _paired_structure_inputs(config.structure_inputs, len(input_sequences))
    results: list[ConstraintOutput] = []
    for (sequence,), struct_input in zip(input_sequences, structure_inputs, strict=True):
        output_chain_id = _resolve_output_chain(struct_input, config.output_chain_id)
        full_sequence = _build_scoring_sequence(struct_input, output_chain_id, sequence.sequence)
        fixed_positions = _fixed_non_output_chains(struct_input, output_chain_id)
        score_output = _run_score(full_sequence, struct_input, fixed_positions, config)
        metrics = score_output.model_dump(exclude={"logits", "vocab"}) if hasattr(score_output, "model_dump") else {}
        avg_log_likelihood_value = metrics.get("avg_log_likelihood")
        if avg_log_likelihood_value is not None:
            avg_log_likelihood = float(avg_log_likelihood_value)
        else:
            nll_value = metrics.get("nll")
            if nll_value is None:
                raise ValueError(f"{config.model} scoring did not return avg_log_likelihood or nll.")
            avg_log_likelihood = -float(nll_value)
        nll = -avg_log_likelihood
        pmpnn = float(np.exp(avg_log_likelihood))
        perplexity_value = metrics.get("perplexity")
        perplexity = float(perplexity_value) if perplexity_value is not None else float(np.exp(nll))
        if config.score_mode == "nll":
            score = nll
        elif config.score_mode == "perplexity":
            score = perplexity
        else:
            score = 1.0 - pmpnn
        results.append(
            ConstraintOutput(
                score=float(score),
                metadata={
                    "model": config.model,
                    "output_chain_id": output_chain_id,
                    "pmpnn": pmpnn,
                    "mpnn_avg_log_likelihood": avg_log_likelihood,
                    "mpnn_nll": nll,
                    "mpnn_perplexity": perplexity,
                    "score_mode": config.score_mode,
                },
            )
        )
    return results


def _paired_structure_inputs(
    structure_inputs: list[InverseFoldingStructureInput],
    num_sequences: int,
) -> list[InverseFoldingStructureInput]:
    if len(structure_inputs) == 1:
        return structure_inputs * num_sequences
    if len(structure_inputs) != num_sequences:
        raise ValueError(
            f"Number of structure_inputs ({len(structure_inputs)}) must be 1 or match proposals ({num_sequences})."
        )
    return structure_inputs


def _resolve_output_chain(struct_input: InverseFoldingStructureInput, output_chain_id: str | None) -> str:
    chain_ids = struct_input.structure.get_chain_ids()
    redesign_ids = struct_input.chain_ids_to_redesign
    if output_chain_id is None:
        if len(redesign_ids) == 1:
            output_chain_id = redesign_ids[0]
        elif len(chain_ids) == 1:
            output_chain_id = chain_ids[0]
        else:
            raise ValueError("output_chain_id is required when scoring a multi-chain structure.")
    if output_chain_id not in chain_ids:
        raise ValueError(f"output_chain_id {output_chain_id!r} not found in structure chains {chain_ids}.")
    if output_chain_id not in redesign_ids:
        raise ValueError(f"output_chain_id {output_chain_id!r} must be in chains_to_redesign {redesign_ids}.")
    return output_chain_id


def _build_scoring_sequence(
    struct_input: InverseFoldingStructureInput,
    output_chain_id: str,
    output_sequence: str,
) -> str:
    chain_sequences = {
        chain_id: struct_input.structure.get_chain_sequence(chain_id)
        for chain_id in struct_input.structure.get_chain_ids()
    }
    expected_length = len(chain_sequences[output_chain_id])
    if len(output_sequence) != expected_length:
        raise ValueError(
            f"Sequence length {len(output_sequence)} does not match structure chain "
            f"{output_chain_id!r} length {expected_length}."
        )
    return "".join(
        output_sequence if chain_id == output_chain_id else chain_sequences[chain_id]
        for chain_id in struct_input.structure.get_chain_ids()
    )


def _fixed_non_output_chains(
    struct_input: InverseFoldingStructureInput,
    output_chain_id: str,
) -> ResidueSelection | None:
    fixed: dict[str, set[int]] = {}
    for chain_id in struct_input.structure.get_chain_ids():
        if chain_id != output_chain_id:
            fixed[chain_id] = set(struct_input.structure.get_chain_positions(chain_id))
    if struct_input.fixed_positions is not None:
        for chain_id, positions in struct_input.fixed_positions.chains.items():
            fixed.setdefault(chain_id, set()).update(positions)
    fixed_dict = {chain_id: sorted(positions) for chain_id, positions in fixed.items() if positions}
    return ResidueSelection(chains=fixed_dict) if fixed_dict else None


def _run_score(
    full_sequence: str,
    struct_input: InverseFoldingStructureInput,
    fixed_positions: ResidueSelection | None,
    config: MPNNSequenceProbabilityConfig,
) -> Any:
    pair = SequenceStructurePair(
        sequence=full_sequence,
        structure=struct_input.structure,
        fixed_positions=fixed_positions,
    )
    if config.model == "ligandmpnn":
        scoring_config: dict[str, Any] = {
            "scoring_mode": "single_aa",
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        }
        supported_fields = getattr(LigandMPNNScoringConfig, "model_fields", {})
        if "ligand_mpnn_use_side_chain_context" in supported_fields:
            scoring_config["ligand_mpnn_use_side_chain_context"] = config.ligand_mpnn_use_side_chain_context
        if "ligand_mpnn_cutoff_for_score" in supported_fields:
            scoring_config["ligand_mpnn_cutoff_for_score"] = config.ligand_mpnn_cutoff_for_score
        output = run_ligandmpnn_score(
            inputs=LigandMPNNScoringInput(sequence_structure_pairs=[pair]),
            config=LigandMPNNScoringConfig(**scoring_config),
        )
    else:
        output = run_proteinmpnn_score(
            inputs=ProteinMPNNScoringInput(sequence_structure_pairs=[pair]),
            config=ProteinMPNNScoringConfig(
                model_choice=config.proteinmpnn_model_choice,
                seed=config.seed,
                device=config.device,
                verbose=config.verbose,
            ),
        )
    return output.scores[0]


mpnn_sequence_probability_constraint._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
