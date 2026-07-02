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
MPNNSequenceProbabilityStructureSource = Literal["configured_structure_inputs", "proposal_structure"]
MPNNSequenceProbabilityScoreSource = Literal["model", "proposal_metadata"]
LigandMPNNBackend = Literal["foundry", "reference"]


class MPNNSequenceProbabilityConfig(BaseConfig):
    """Configuration for structure-conditioned MPNN sequence-probability scoring.

    Attributes:
        model (MPNNSequenceProbabilityModel): Structure-conditioned MPNN
            variant used to score sequence probability.
        structure_source (MPNNSequenceProbabilityStructureSource): Whether scoring uses configured structures or the
            structure currently attached to each proposal.
        structure_inputs (list[InverseFoldingStructureInput] | None): Backbone
            structure inputs, with optional chains to redesign and fixed positions. When
            ``structure_source`` is ``"proposal_structure"``, these act as optional selection templates.
        output_chain_id (str | None): Chain whose sequence is supplied by the
            optimizer proposal; inferred when unambiguous.
        score_mode (MPNNSequenceProbabilityScoreMode): Objective returned by
            the constraint: probability loss, mean NLL, or perplexity.
        score_source (MPNNSequenceProbabilityScoreSource): Score with the model or reuse
            proposal generator metadata containing ``pmpnn``.
        metadata_generator_key (str | None): Optional generator metadata namespace to read when
            ``score_source`` is ``"proposal_metadata"``.
        proteinmpnn_model_choice (ProteinMPNNModelChoice): ProteinMPNN weights
            used when ``model`` is ``"proteinmpnn"``.
        ligand_mpnn_use_side_chain_context (bool): Whether LigandMPNN scoring
            conditions on fixed-residue sidechain atoms.
        ligand_mpnn_cutoff_for_score (float): Ligand-residue distance cutoff
            used by LigandMPNN scoring.
        ligand_mpnn_checkpoint_path (str | None): Optional explicit LigandMPNN checkpoint path.
        ligand_mpnn_backend (LigandMPNNBackend): Inference backend for LigandMPNN scoring.
        ligand_mpnn_reference_backend_path (str | None): Local reference LigandMPNN checkout used by backend="reference".
        seed (int | None): Optional random seed for MPNN scoring.
        device (str): Device for MPNN scoring, for example ``"cuda"``.
        verbose (bool): Whether to print MPNN scoring progress.
    """

    model: MPNNSequenceProbabilityModel = ConfigField(
        default="ligandmpnn",
        title="MPNN Model",
        description="Structure-conditioned model used to score the sequence.",
    )
    structure_source: MPNNSequenceProbabilityStructureSource = ConfigField(
        default="configured_structure_inputs",
        title="Structure Source",
        description="Use configured structure_inputs or each proposal's attached structure for MPNN scoring.",
    )
    structure_inputs: list[InverseFoldingStructureInput] | None = ConfigField(
        default=None,
        title="Structure Inputs",
        description="Structures for MPNN scoring; optional when using proposal structures.",
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
    score_source: MPNNSequenceProbabilityScoreSource = ConfigField(
        default="model",
        title="Score Source",
        description="Compute MPNN probability with the model or read pmpnn from proposal generator metadata.",
    )
    metadata_generator_key: str | None = ConfigField(
        default=None,
        title="Metadata Generator Key",
        description="Generator metadata namespace to read for pmpnn when score_source='proposal_metadata'.",
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
    ligand_mpnn_checkpoint_path: str | None = ConfigField(
        default=None,
        title="LigandMPNN Checkpoint Path",
        description="Optional explicit LigandMPNN checkpoint path.",
    )
    ligand_mpnn_backend: LigandMPNNBackend = ConfigField(
        default="foundry",
        title="LigandMPNN Backend",
        description="LigandMPNN inference backend for scoring.",
    )
    ligand_mpnn_reference_backend_path: str | None = ConfigField(
        default=None,
        title="Reference Backend Path",
        description="Path to a local reference LigandMPNN checkout when ligand_mpnn_backend='reference'.",
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
        if value is None:
            return None
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
    if config.score_source == "proposal_metadata":
        return [_metadata_probability_output(seq_tuple, config) for seq_tuple in input_sequences]

    structure_inputs = _paired_structure_inputs(input_sequences, config)
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
                    "structure_source": config.structure_source,
                    "output_chain_id": output_chain_id,
                    "pmpnn": pmpnn,
                    "mpnn_avg_log_likelihood": avg_log_likelihood,
                    "mpnn_nll": nll,
                    "mpnn_perplexity": perplexity,
                    "score_mode": config.score_mode,
                    "score_source": config.score_source,
                },
            )
        )
    return results


def _metadata_probability_output(
    seq_tuple: tuple[Sequence, ...],
    config: MPNNSequenceProbabilityConfig,
) -> ConstraintOutput:
    generator_key, generator_metadata = _metadata_with_pmpnn(seq_tuple, config.metadata_generator_key)
    pmpnn = float(generator_metadata["pmpnn"])
    if not np.isfinite(pmpnn) or pmpnn <= 0.0 or pmpnn > 1.0:
        raise ValueError(f"Generator metadata pmpnn must be a finite probability in (0, 1], got {pmpnn}.")
    nll = -float(np.log(pmpnn))
    perplexity = float(np.exp(nll))
    if config.score_mode == "nll":
        score = nll
    elif config.score_mode == "perplexity":
        score = perplexity
    else:
        score = 1.0 - pmpnn
    return ConstraintOutput(
        score=float(score),
        metadata={
            "model": config.model,
            "structure_source": config.structure_source,
            "output_chain_id": config.output_chain_id,
            "pmpnn": pmpnn,
            "mpnn_nll": nll,
            "mpnn_perplexity": perplexity,
            "score_mode": config.score_mode,
            "score_source": config.score_source,
            "metadata_generator_key": generator_key,
        },
    )


def _metadata_with_pmpnn(seq_tuple: tuple[Sequence, ...], generator_key: str | None) -> tuple[str, dict[str, Any]]:
    if generator_key is not None:
        for sequence in seq_tuple:
            metadata = sequence._generator_metadata.get(generator_key)
            if metadata is not None and metadata.get("pmpnn") is not None:
                return generator_key, metadata
        raise ValueError(f"Generator metadata {generator_key!r} does not contain pmpnn.")

    for sequence in seq_tuple:
        for key, metadata in reversed(sequence._generator_metadata.items()):
            if metadata.get("pmpnn") is not None:
                return key, metadata
    raise ValueError("No proposal generator metadata contains pmpnn.")


def _paired_structure_inputs(
    input_sequences: list[tuple[Sequence, ...]],
    config: MPNNSequenceProbabilityConfig,
) -> list[InverseFoldingStructureInput]:
    structure_inputs = config.structure_inputs
    num_sequences = len(input_sequences)
    if config.structure_source == "proposal_structure":
        templates = _pair_templates(structure_inputs, num_sequences)
        paired_inputs: list[InverseFoldingStructureInput] = []
        for seq_tuple, template in zip(input_sequences, templates, strict=True):
            proposal_structure = _proposal_structure(seq_tuple)
            paired_inputs.append(
                InverseFoldingStructureInput(
                    structure=proposal_structure,
                    chains_to_redesign=(template.chains_to_redesign if template is not None else None),
                    fixed_positions=(template.fixed_positions if template is not None else None),
                )
            )
        return paired_inputs

    if structure_inputs is None:
        raise ValueError(
            "mpnn-sequence-probability requires structure_inputs unless structure_source='proposal_structure'."
        )
    return [template for template in _pair_templates(structure_inputs, num_sequences) if template is not None]


def _pair_templates(
    structure_inputs: list[InverseFoldingStructureInput] | None,
    num_sequences: int,
) -> list[InverseFoldingStructureInput | None]:
    if structure_inputs is None:
        return [None] * num_sequences
    if len(structure_inputs) == 1:
        return structure_inputs * num_sequences
    if len(structure_inputs) != num_sequences:
        raise ValueError(
            f"Number of structure_inputs ({len(structure_inputs)}) must be 1 or match proposals ({num_sequences})."
        )
    return structure_inputs


def _proposal_structure(seq_tuple: tuple[Sequence, ...]) -> Structure:
    for sequence in seq_tuple:
        if sequence.structure is not None:
            return sequence.structure
    raise ValueError("mpnn-sequence-probability structure_source='proposal_structure' requires attached structures.")


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
        if "checkpoint_path" in supported_fields:
            scoring_config["checkpoint_path"] = config.ligand_mpnn_checkpoint_path
        if "backend" in supported_fields:
            scoring_config["backend"] = config.ligand_mpnn_backend
        if "reference_backend_path" in supported_fields:
            scoring_config["reference_backend_path"] = config.ligand_mpnn_reference_backend_path
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
