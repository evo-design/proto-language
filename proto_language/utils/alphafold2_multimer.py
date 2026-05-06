"""AF2 multimer adapter utilities shared by constraints and the compiler.

AF2M currently reaches proto-language through the ColabDesign binder tool API,
not the predictor-style structure API used by ESMFold/AF3/Boltz/Chai. This file
owns the tool-boundary translation: canonical loss names, predictor-like metric
aliases, per-input structure splitting, and forward AF2M calls.

TODO(@brianhie, @dguo): Consider moving some of this adapter logic into
proto-tools if the AF2M binder interface grows a predictor-shaped API.
"""

from typing import Any

from proto_tools import Structure
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2BinderConfig,
    AlphaFold2BinderInput,
    run_alphafold2_binder,
)
from pydantic import BaseModel, ConfigDict

from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    AlphaFold2MultimerStructureConfig,
    StructureBasedConstraintConfig,
)
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.storage import FileType, store_file
from proto_language.utils.helpers import one_hot_protein_matrix

AF2_MULTIMER_LOSS_TERMS: frozenset[str] = frozenset(
    {
        "plddt",
        "iplddt",
        "pae",
        "ipae",
        "con",
        "i_con",
        "rg",
        "iptm",
        "helix",
        "beta_strand",
        "dgram_cce",
        "NC",
    }
)
AF2_MULTIMER_TOOL_LOSS_ALIASES: dict[str, str] = {
    "iplddt": "i_plddt",
    "ipae": "i_pae",
    "iptm": "i_ptm",
}
AF2_MULTIMER_CANONICAL_LOSS_ALIASES: dict[str, str] = {
    tool_key: canonical_key for canonical_key, tool_key in AF2_MULTIMER_TOOL_LOSS_ALIASES.items()
}
# ColabDesign normalizes AF2M PAE/iPAE losses by 31.0, matching its
# predicted-aligned-error head's max_error_bin. Keep this tool-specific scale
# separate from the generic predictor PAE normalization constant of 31.75
# Angstroms used in structure_confidence_constraint.py.
AF2_MULTIMER_PAE_MAXIMUM: float = 31.0
AF2_MULTIMER_TOOL_OBJECTIVE_KEYS: frozenset[str] = frozenset(
    {
        "plddt",
        "i_plddt",
        "pae",
        "i_pae",
        "con",
        "i_con",
        "rg",
        "i_ptm",
        "helix",
        "beta_strand",
        "dgram_cce",
        "NC",
    }
)
AF2_MULTIMER_CONFIDENCE_LOSS_BY_METRIC: dict[str, str | None] = {
    "avg_plddt": "plddt",
    # AF2 Multimer reports global pTM as a metric, but ColabDesign does not expose
    # a separate pTM loss key here. Forward structure-ptm therefore runs a
    # metric-only prediction with empty loss_weights and reads ``ptm`` directly.
    "ptm": None,
    "iptm": "iptm",
    "avg_pae": "pae",
    "iplddt": "iplddt",
    "ipae": "ipae",
}


class AF2MultimerPrediction(BaseModel):
    """Forward AF2 multimer result in proto-language canonical terms."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    loss: float
    metrics: dict[str, Any]
    structure: Structure
    structures: tuple[Structure | None, ...]


def next_af2_multimer_seed(config: AlphaFold2MultimerStructureConfig) -> int | None:
    """Derive deterministic per-evaluation AF2 seeds without replaying one RNG state."""
    if config.seed is None:
        return None
    seed = config.seed + config._evaluation_seed_offset
    config._evaluation_seed_offset += 1
    return seed


def validate_af2_multimer_inputs(
    proposal_tuple: tuple[Sequence, ...],
    config: AlphaFold2MultimerStructureConfig,
) -> None:
    """Validate AF2 multimer role indices and protein-only inputs."""
    n_inputs = len(proposal_tuple)
    all_indices = [config.binder_input_index, *config.target_input_indices]
    out_of_bounds = [idx for idx in all_indices if idx >= n_inputs]
    if out_of_bounds:
        raise ValueError(f"AF2 multimer input indices {out_of_bounds} out of bounds for {n_inputs} input(s).")
    for idx in all_indices:
        if proposal_tuple[idx].sequence_type != "protein":
            raise TypeError(
                f"AF2 multimer structure constraints support protein inputs only; "
                f"input {idx} has type {proposal_tuple[idx].sequence_type!r}."
            )


def af2_multimer_tool_loss_key(loss_key: str) -> str:
    """Return the current tool-layer loss key for a proto-language loss key."""
    return AF2_MULTIMER_TOOL_LOSS_ALIASES.get(loss_key, loss_key)


def af2_multimer_confidence_loss_weights(target_metric: str) -> dict[str, float]:
    """Return the AF2 objective weights needed to expose one confidence metric."""
    if target_metric not in AF2_MULTIMER_CONFIDENCE_LOSS_BY_METRIC:
        return {}
    loss_key = AF2_MULTIMER_CONFIDENCE_LOSS_BY_METRIC[target_metric]
    return {loss_key: 1.0} if loss_key is not None else {}


def canonical_af2_multimer_metrics(output_metrics: dict[str, Any]) -> dict[str, Any]:
    """Return AF2 outputs as predictor-like metrics plus objective terms.

    Predictor-style confidence values use the same names as other structure
    tools (``avg_plddt``, ``iptm``, ``avg_pae``, ``iplddt``, ``ipae``).
    ColabDesign objective terms are kept separately as ``loss_*`` keys, e.g.
    ``loss_ipae`` for the tool's ``i_pae`` objective. When a predictor-style
    value is not reported directly but can be inferred from an AF2 objective,
    this adapter derives it without dropping the original ``loss_*`` value.
    """
    metrics: dict[str, Any] = {}
    for key, value in output_metrics.items():
        if key.startswith("loss_"):
            loss_key = key.removeprefix("loss_")
            canonical_key = "loss_" + AF2_MULTIMER_CANONICAL_LOSS_ALIASES.get(loss_key, loss_key)
        elif key in AF2_MULTIMER_TOOL_OBJECTIVE_KEYS:
            canonical_key = "loss_" + AF2_MULTIMER_CANONICAL_LOSS_ALIASES.get(key, key)
        else:
            canonical_key = AF2_MULTIMER_CANONICAL_LOSS_ALIASES.get(key, key)
        metrics[canonical_key] = value

    loss_plddt = _metric_float(metrics, "loss_plddt")
    if "avg_plddt" not in metrics and loss_plddt is not None:
        metrics["avg_plddt"] = _confidence_from_loss(loss_plddt)

    loss_iptm = _metric_float(metrics, "loss_iptm")
    if "iptm" not in metrics and loss_iptm is not None:
        metrics["iptm"] = _confidence_from_loss(loss_iptm)

    loss_pae = _metric_float(metrics, "loss_pae")
    if "avg_pae" not in metrics and loss_pae is not None:
        metrics["avg_pae"] = loss_pae * AF2_MULTIMER_PAE_MAXIMUM

    loss_iplddt = _metric_float(metrics, "loss_iplddt")
    if "iplddt" not in metrics and loss_iplddt is not None:
        metrics["iplddt"] = _confidence_from_loss(loss_iplddt)

    loss_ipae = _metric_float(metrics, "loss_ipae")
    if "ipae" not in metrics and loss_ipae is not None:
        metrics["ipae"] = loss_ipae * AF2_MULTIMER_PAE_MAXIMUM

    return metrics


def af2_multimer_structures(
    output_structure: Structure, config: AlphaFold2MultimerStructureConfig, n_inputs: int
) -> tuple[Structure | None, ...]:
    """Return per-input structures from an AF2 multimer complex."""
    structures: list[Structure | None] = [None] * n_inputs
    structures[config.binder_input_index] = output_structure.select_chain(config.binder_chain)
    for input_idx, chain_id in zip(config.target_input_indices, config.target_chains, strict=True):
        structures[input_idx] = output_structure.select_chain(chain_id)
    return tuple(structures)


def af2_multimer_constraint_output_metadata(
    output_metrics: dict[str, Any],
    *,
    output_loss: float,
    output_structure: Structure,
    loss_key: str,
    group_loss: float | None = None,
) -> dict[str, Any]:
    """Build metadata for an AF2 multimer-backed structure constraint result."""
    metrics = canonical_af2_multimer_metrics(output_metrics)
    metadata = {
        **metrics,
        "af2_loss_key": loss_key,
        "loss": output_loss,
        "complex_pdb": output_structure.structure_pdb,
        "pdb_output": store_file(output_structure.structure_pdb, FileType.PDB),
        "structure_tool": "alphafold2_multimer",
    }
    if group_loss is not None:
        metadata["af2_group_loss"] = group_loss
    return metadata


def af2_multimer_confidence_output_metadata(
    output_metrics: dict[str, Any],
    *,
    output_loss: float,
    output_structure: Structure,
    target_metric: str,
) -> dict[str, Any]:
    """Build metadata for an AF2 multimer-backed confidence constraint result."""
    loss_key = AF2_MULTIMER_CONFIDENCE_LOSS_BY_METRIC.get(target_metric)
    if loss_key is not None:
        return af2_multimer_constraint_output_metadata(
            output_metrics,
            output_loss=output_loss,
            output_structure=output_structure,
            loss_key=loss_key,
        )

    metrics = canonical_af2_multimer_metrics(output_metrics)
    return {
        **metrics,
        "loss": output_loss,
        "complex_pdb": output_structure.structure_pdb,
        "pdb_output": store_file(output_structure.structure_pdb, FileType.PDB),
        "structure_tool": "alphafold2_multimer",
    }


def evaluate_af2_multimer_predictions(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    *,
    loss_weights: dict[str, float],
) -> list[AF2MultimerPrediction]:
    """Run forward AF2 multimer predictions and return canonical metrics.

    This is the forward-only adapter shared by confidence constraints and
    raw AF2 objective constraints. Public constraints should stay expressed in
    biological terms; this helper owns the current proto-tools binder protocol
    call shape, canonical loss-key translation, and per-chain structure split.
    """
    af2_config = config.alphafold2_multimer_config
    unsupported = sorted(set(loss_weights) - AF2_MULTIMER_LOSS_TERMS)
    if unsupported:
        raise ValueError(f"AF2 multimer loss key(s) {unsupported!r} are not supported.")

    predictions: list[AF2MultimerPrediction] = []
    for proposal_tuple in proposals:
        validate_af2_multimer_inputs(proposal_tuple, af2_config)
        binder_seq = proposal_tuple[af2_config.binder_input_index]
        evaluation_seed = next_af2_multimer_seed(af2_config)
        output = run_alphafold2_binder(
            AlphaFold2BinderInput(
                logits=one_hot_protein_matrix(binder_seq.sequence),
                target_pdb=af2_config.target_pdb,
                target_chain=",".join(af2_config.target_chains),
                target_hotspot=af2_config.target_hotspot,
                binder_chain=af2_config.binder_chain,
                design_positions=af2_config.design_positions,
            ),
            AlphaFold2BinderConfig(
                include_pae_matrix=af2_config.include_pae_matrix,
                bias_redesign=af2_config.bias_redesign,
                omit_aas=af2_config.omit_aas,
                num_recycles=af2_config.num_recycles,
                recycle_mode=af2_config.recycle_mode,
                model_num=af2_config.model_num,
                sample_models=af2_config.sample_models,
                use_multimer=af2_config.use_multimer,
                rm_target_seq=af2_config.rm_target_seq,
                rm_target_sc=af2_config.rm_target_sc,
                rm_template_ic=af2_config.rm_template_ic,
                loss_weights={af2_multimer_tool_loss_key(key): weight for key, weight in loss_weights.items()},
                intra_contact_num=af2_config.intra_contact_num,
                intra_contact_cutoff=af2_config.intra_contact_cutoff,
                inter_contact_num=af2_config.inter_contact_num,
                inter_contact_cutoff=af2_config.inter_contact_cutoff,
                framework_contact_offset=af2_config.framework_contact_offset,
                backend=af2_config.backend,
                device=af2_config.device,
                seed=evaluation_seed,
                soft=0.0,
                hard=1.0,
                compute_gradient=False,
            ),
        )
        metrics = canonical_af2_multimer_metrics(output.metrics)
        predictions.append(
            AF2MultimerPrediction(
                loss=output.loss,
                metrics=metrics,
                structure=output.structure,
                structures=af2_multimer_structures(output.structure, af2_config, len(proposal_tuple)),
            )
        )
    return predictions


def evaluate_af2_multimer_confidence_predictions(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    *,
    target_metric: str,
) -> list[AF2MultimerPrediction]:
    """Run AF2 multimer with the objective needed for a confidence metric."""
    return evaluate_af2_multimer_predictions(
        proposals,
        config,
        loss_weights=af2_multimer_confidence_loss_weights(target_metric),
    )


def evaluate_af2_multimer_loss_constraint(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    loss_key: str,
) -> list[ConstraintOutput]:
    """Forward-score one AF2 multimer loss term for each proposal."""
    if loss_key not in AF2_MULTIMER_LOSS_TERMS:
        raise ValueError(f"AF2 multimer loss key {loss_key!r} is not supported.")

    return [
        ConstraintOutput(
            score=prediction.loss,
            metadata=af2_multimer_constraint_output_metadata(
                prediction.metrics,
                output_loss=prediction.loss,
                output_structure=prediction.structure,
                loss_key=loss_key,
            ),
            structures=prediction.structures,
        )
        for prediction in evaluate_af2_multimer_predictions(proposals, config, loss_weights={loss_key: 1.0})
    ]


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _confidence_from_loss(loss: float) -> float:
    return min(max(1.0 - loss, 0.0), 1.0)
