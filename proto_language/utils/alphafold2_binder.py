"""AF2 binder adapter utilities shared by constraints and the compiler.

The AF2 binder backend currently reaches proto-language through the ColabDesign
binder tool API, not the predictor-style structure API used by the general
AlphaFold2 predictor, ESMFold, AF3, Boltz, and Chai. This file owns the
tool-boundary translation: canonical
loss names, predictor-like metric aliases, per-input structure splitting, and
forward AF2 binder calls.

TODO: Consider moving some of this adapter logic into
proto-tools if the AF2 binder interface grows a predictor-shaped API.
"""

import string
from typing import Any

from proto_tools import Structure
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2GradientConfig,
    AlphaFold2GradientInput,
    run_alphafold2_gradient,
)
from pydantic import BaseModel, ConfigDict

from proto_language.constraint.protein_structure.structure_constraint_config import (
    AlphaFold2BinderStructureConfig,
    StructureBasedConstraintConfig,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.sequence_matrices import one_hot_protein_matrix

AF2_BINDER_LOSS_TERMS: frozenset[str] = frozenset(
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
AF2_BINDER_TOOL_LOSS_ALIASES: dict[str, str] = {
    "iplddt": "i_plddt",
    "ipae": "i_pae",
    "iptm": "i_ptm",
}
AF2_BINDER_CANONICAL_LOSS_ALIASES: dict[str, str] = {
    tool_key: canonical_key for canonical_key, tool_key in AF2_BINDER_TOOL_LOSS_ALIASES.items()
}
# ColabDesign's AF2 binder PAE/iPAE scale (max_error_bin); distinct from the
# generic predictor's 31.75 Angstrom constant in structure_confidence_constraint.py.
AF2_BINDER_PAE_MAXIMUM: float = 31.0
AF2_BINDER_TOOL_OBJECTIVE_KEYS: frozenset[str] = frozenset(
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
AF2_BINDER_CONFIDENCE_LOSS_BY_METRIC: dict[str, str | None] = {
    "avg_plddt": "plddt",
    # No ColabDesign pTM loss key; ptm comes from a metric-only prediction (empty loss_weights).
    "ptm": None,
    "iptm": "iptm",
    "avg_pae": "pae",
    "iplddt": "iplddt",
    "ipae": "ipae",
}


class AF2BinderPrediction(BaseModel):
    """Forward AF2 binder result in proto-language canonical terms."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    loss: float
    metrics: dict[str, Any]
    structure: Structure
    structures: tuple[Structure | None, ...]


def next_af2_binder_seed(config: AlphaFold2BinderStructureConfig) -> int | None:
    """Derive deterministic per-evaluation AF2 seeds without replaying one RNG state."""
    if config.seed is None:
        return None
    seed = config.seed + config._evaluation_seed_offset
    config._evaluation_seed_offset += 1
    return seed


def validate_af2_binder_inputs(
    proposal_tuple: tuple[Sequence, ...],
    config: AlphaFold2BinderStructureConfig,
) -> None:
    """Validate AF2 binder role indices and protein-only inputs."""
    n_inputs = len(proposal_tuple)
    all_indices = [config.binder_input_index, *config.target_input_indices]
    out_of_bounds = [idx for idx in all_indices if idx >= n_inputs]
    if out_of_bounds:
        raise ValueError(f"AF2 binder input indices {out_of_bounds} out of bounds for {n_inputs} input(s).")
    for idx in all_indices:
        if proposal_tuple[idx].sequence_type != "protein":
            raise TypeError(
                f"AF2 binder structure constraints support protein inputs only; "
                f"input {idx} has type {proposal_tuple[idx].sequence_type!r}."
            )


def af2_binder_confidence_loss_weights(target_metric: str) -> dict[str, float]:
    """Return the AF2 objective weights needed to expose one confidence metric."""
    if target_metric not in AF2_BINDER_CONFIDENCE_LOSS_BY_METRIC:
        return {}
    loss_key = AF2_BINDER_CONFIDENCE_LOSS_BY_METRIC[target_metric]
    return {loss_key: 1.0} if loss_key is not None else {}


def canonical_af2_binder_metrics(output_metrics: dict[str, Any]) -> dict[str, Any]:
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
            canonical_key = "loss_" + AF2_BINDER_CANONICAL_LOSS_ALIASES.get(loss_key, loss_key)
        elif key in AF2_BINDER_TOOL_OBJECTIVE_KEYS:
            canonical_key = "loss_" + AF2_BINDER_CANONICAL_LOSS_ALIASES.get(key, key)
        else:
            canonical_key = AF2_BINDER_CANONICAL_LOSS_ALIASES.get(key, key)
        metrics[canonical_key] = value

    loss_plddt = _metric_float(metrics, "loss_plddt")
    if "avg_plddt" not in metrics and loss_plddt is not None:
        metrics["avg_plddt"] = _confidence_from_loss(loss_plddt)

    loss_iptm = _metric_float(metrics, "loss_iptm")
    if "iptm" not in metrics and loss_iptm is not None:
        metrics["iptm"] = _confidence_from_loss(loss_iptm)

    loss_pae = _metric_float(metrics, "loss_pae")
    if "avg_pae" not in metrics and loss_pae is not None:
        metrics["avg_pae"] = loss_pae * AF2_BINDER_PAE_MAXIMUM

    loss_iplddt = _metric_float(metrics, "loss_iplddt")
    if "iplddt" not in metrics and loss_iplddt is not None:
        metrics["iplddt"] = _confidence_from_loss(loss_iplddt)

    loss_ipae = _metric_float(metrics, "loss_ipae")
    if "ipae" not in metrics and loss_ipae is not None:
        metrics["ipae"] = loss_ipae * AF2_BINDER_PAE_MAXIMUM

    return metrics


def af2_binder_structures(
    output_structure: Structure, config: AlphaFold2BinderStructureConfig, n_inputs: int
) -> tuple[Structure | None, ...]:
    """Return per-input structures from an AF2 binder complex.

    Output chains are positional (targets A..N-1; de-novo binder next: N=1 -> "B", N=2 -> "C"),
    not the source-PDB ``target_chains`` ids. An explicit ``binder_chain`` (redesign) is used as-is.
    """
    structures: list[Structure | None] = [None] * n_inputs
    n_targets = len(config.target_chains)
    binder_chain = config.binder_chain if config.binder_chain is not None else string.ascii_uppercase[n_targets]
    structures[config.binder_input_index] = output_structure.select_chain(binder_chain)
    target_letters = [string.ascii_uppercase[position] for position in range(n_targets)]
    if len(config.target_input_indices) == n_targets:
        # One target input slot per chain (Germinal, single-chain de novo).
        for position, input_idx in enumerate(config.target_input_indices):
            structures[input_idx] = output_structure.select_chain(target_letters[position])
    else:
        # One slot holding all target chains (de-novo BindCraft concatenated target).
        structures[config.target_input_indices[0]] = output_structure.select_chains(target_letters)
    return tuple(structures)


def af2_binder_constraint_output_metadata(
    output_metrics: dict[str, Any],
    *,
    output_loss: float,
    output_structure: Structure,
    loss_key: str,
    group_loss: float | None = None,
) -> dict[str, Any]:
    """Build metadata for an AF2 binder-backed structure constraint result."""
    metrics = canonical_af2_binder_metrics(output_metrics)
    metadata = {
        **metrics,
        "af2_loss_key": loss_key,
        "loss": output_loss,
        "pdb_output": output_structure.structure_pdb,
        "structure_tool": "alphafold2_binder",
    }
    if group_loss is not None:
        metadata["af2_group_loss"] = group_loss
    return metadata


def af2_binder_confidence_output_metadata(
    output_metrics: dict[str, Any],
    *,
    output_loss: float,
    output_structure: Structure,
    target_metric: str,
) -> dict[str, Any]:
    """Build metadata for an AF2 binder-backed confidence constraint result."""
    loss_key = AF2_BINDER_CONFIDENCE_LOSS_BY_METRIC.get(target_metric)
    if loss_key is not None:
        return af2_binder_constraint_output_metadata(
            output_metrics,
            output_loss=output_loss,
            output_structure=output_structure,
            loss_key=loss_key,
        )

    metrics = canonical_af2_binder_metrics(output_metrics)
    return {
        **metrics,
        "loss": output_loss,
        "pdb_output": output_structure.structure_pdb,
        "structure_tool": "alphafold2_binder",
    }


def evaluate_af2_binder_predictions(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    *,
    loss_weights: dict[str, float],
) -> list[AF2BinderPrediction]:
    """Run forward AF2 binder predictions and return canonical metrics.

    This is the forward-only adapter shared by confidence constraints and
    raw AF2 objective constraints. Public constraints should stay expressed in
    biological terms; this helper owns the current proto-tools binder protocol
    call shape, canonical loss-key translation, and per-chain structure split.
    """
    af2_config = config.alphafold2_binder_config
    unsupported = sorted(set(loss_weights) - AF2_BINDER_LOSS_TERMS)
    if unsupported:
        raise ValueError(f"AF2 binder loss key(s) {unsupported!r} are not supported.")

    predictions: list[AF2BinderPrediction] = []
    for proposal_tuple in proposals:
        validate_af2_binder_inputs(proposal_tuple, af2_config)
        binder_seq = proposal_tuple[af2_config.binder_input_index]
        evaluation_seed = next_af2_binder_seed(af2_config)
        output = run_alphafold2_gradient(
            AlphaFold2GradientInput(
                logits=one_hot_protein_matrix(binder_seq.sequence),
                target_pdb=af2_config.target_pdb,
                target_chain=",".join(af2_config.target_chains),
                target_hotspot=af2_config.target_hotspot,
                binder_chain=af2_config.binder_chain,
                design_positions=af2_config.design_positions,
            ),
            AlphaFold2GradientConfig(
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
                loss_weights={
                    AF2_BINDER_TOOL_LOSS_ALIASES.get(key, key): weight for key, weight in loss_weights.items()
                },
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
        metrics = canonical_af2_binder_metrics(output.metrics)
        predictions.append(
            AF2BinderPrediction(
                loss=output.loss,
                metrics=metrics,
                structure=output.structure,
                structures=af2_binder_structures(output.structure, af2_config, len(proposal_tuple)),
            )
        )
    return predictions


def evaluate_af2_binder_confidence_predictions(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    *,
    target_metric: str,
) -> list[AF2BinderPrediction]:
    """Run AF2 binder with the objective needed for a confidence metric."""
    return evaluate_af2_binder_predictions(
        proposals,
        config,
        loss_weights=af2_binder_confidence_loss_weights(target_metric),
    )


def evaluate_af2_binder_loss_constraint(
    proposals: list[tuple[Sequence, ...]],
    config: StructureBasedConstraintConfig,
    loss_key: str,
) -> list[ConstraintOutput]:
    """Forward-score one AF2 binder loss term for each proposal."""
    if loss_key not in AF2_BINDER_LOSS_TERMS:
        raise ValueError(f"AF2 binder loss key {loss_key!r} is not supported.")

    return [
        ConstraintOutput(
            score=prediction.loss,
            metadata=af2_binder_constraint_output_metadata(
                prediction.metrics,
                output_loss=prediction.loss,
                output_structure=prediction.structure,
                loss_key=loss_key,
            ),
            structures=prediction.structures,
        )
        for prediction in evaluate_af2_binder_predictions(proposals, config, loss_weights={loss_key: 1.0})
    ]


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _confidence_from_loss(loss: float) -> float:
    return min(max(1.0 - loss, 0.0), 1.0)
