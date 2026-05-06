"""AF2 multimer adapter for the private constraint compiler.

The public protein-structure constraints expose model-agnostic biological
objectives such as pLDDT, iPAE, contacts, and secondary-structure preferences.
The AF2 multimer tool API is different: it accepts one target/binder tool call
with backend-specific loss keys and can return either a grouped scalar score or
one gradient for the weighted sum of requested losses.

This module is the translation layer between those two shapes. It maps public
constraint functions to AF2M loss keys, validates that a constraint can be used
as a differentiable binder objective, groups compatible constraints into a
single AF2M call, and writes the per-public-constraint metadata expected by the
rest of the language layer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2BinderConfig,
    AlphaFold2BinderInput,
    run_alphafold2_binder,
)
from pydantic import ValidationError

from proto_language.language.constraint.protein_structure.structure_confidence_constraint import (
    structure_ipae_constraint,
    structure_iplddt_constraint,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    AlphaFold2MultimerStructureConfig,
    StructureBasedConstraintConfig,
)
from proto_language.language.constraint.protein_structure.structure_geometry_constraint import (
    structure_beta_strand_constraint,
    structure_contact_constraint,
    structure_distogram_cce_constraint,
    structure_helix_constraint,
    structure_interface_contact_constraint,
    structure_radius_gyration_constraint,
    structure_termini_distance_constraint,
)
from proto_language.language.core import Constraint, Segment
from proto_language.language.optimizer.constraint_compiler.base import (
    CompiledConstraint,
    EffectiveWeight,
    GradientProvider,
    GradientProviderOutput,
)
from proto_language.utils import one_hot_protein_matrix
from proto_language.utils.alphafold2_multimer import (
    AF2_MULTIMER_TOOL_LOSS_ALIASES,
    af2_multimer_constraint_output_metadata,
    af2_multimer_structures,
    next_af2_multimer_seed,
    validate_af2_multimer_inputs,
)

logger = logging.getLogger(__name__)

# Compiler-private mapping from public biological constraints to AF2 multimer
# objective keys. Keeping this table in the compiler package avoids adding
# backend execution details to the public constraint functions.
AF2_MULTIMER_STRUCTURE_LOSS_BY_FUNCTION = {
    structure_plddt_constraint: "plddt",
    structure_iptm_constraint: "iptm",
    structure_pae_constraint: "pae",
    structure_iplddt_constraint: "iplddt",
    structure_ipae_constraint: "ipae",
    structure_contact_constraint: "con",
    structure_interface_contact_constraint: "i_con",
    structure_radius_gyration_constraint: "rg",
    structure_helix_constraint: "helix",
    structure_beta_strand_constraint: "beta_strand",
    structure_distogram_cce_constraint: "dgram_cce",
    structure_termini_distance_constraint: "NC",
}


class AF2MultimerGradientProvider(GradientProvider):
    """Grouped AF2 multimer provider for multiple public structure terms.

    AF2M currently illustrates the compiled-backend pattern: constraints that
    share inputs and AF2M config are aggregated into one weighted tool call, and
    the returned grouped gradient is used by the optimizer.
    """

    def __init__(
        self,
        *,
        constraints: list[CompiledConstraint],
        config: AlphaFold2MultimerStructureConfig,
        inputs: list[Segment],
    ):
        """Create a grouped AF2M gradient provider.

        Args:
            constraints (list[CompiledConstraint]): Compiled public constraints to include in the AF2M
                weighted loss. The list may be empty during construction and
                populated by ``add_gradient_constraint``.
            config (AlphaFold2MultimerStructureConfig): AF2M-specific structure config shared by every constraint
                in the group.
            inputs (list[Segment]): Ordered target/binder segments for the grouped tool call.
                Their identities are part of the compiler grouping key.
        """
        self.constraints = constraints
        self.config = config
        self.inputs = inputs
        self.label = _provider_label(constraints)

    def compute(
        self,
        *,
        temperature: float,
        soft: float,
        hard: float,
        step: int,
        effective_weight: EffectiveWeight,
    ) -> GradientProviderOutput:
        """Run AF2M gradients for every proposal in the grouped constraint set.

        AF2M returns one gradient for a weighted sum of loss terms, not one
        gradient per public constraint. This method builds the current
        step-dependent ``loss_weights`` map, executes one AF2M gradient call per
        proposal, and returns the grouped gradient to the optimizer. It also
        writes per-constraint metadata using each term's scalar metric/loss so
        downstream reporting still sees the original public constraints.

        Args:
            temperature (float): Relaxed-sequence sampling temperature passed to AF2M.
            soft (float): Soft sequence coefficient for the AF2M differentiable call.
            hard (float): Hard sequence coefficient for the AF2M differentiable call.
            step (int): Optimizer step used to evaluate scheduled constraint weights.
            effective_weight (EffectiveWeight): Callback returning each public constraint's
                current scalar weight.

        Returns:
            GradientProviderOutput: Proposal-aligned grouped gradients and grouped weighted losses.

        Raises:
            RuntimeError: If a binder proposal has no logits or the AF2M tool
                does not return a gradient for a gradient request.
        """
        loss_weights = {
            compiled.objective_key: effective_weight(compiled.constraint, step) for compiled in self.constraints
        }
        num_proposals = self.inputs[0].num_proposals
        binder_slot = self.config.binder_input_index
        gradients: list[np.ndarray] = []
        losses: list[float] = []

        for proposal_idx in range(num_proposals):
            proposal_tuple = tuple(segment.proposal_sequences[proposal_idx] for segment in self.inputs)
            validate_af2_multimer_inputs(proposal_tuple, self.config)
            binder_seq = proposal_tuple[binder_slot]
            if binder_seq.logits is None:
                raise RuntimeError(f"{self.label} proposal {proposal_idx}: binder input is missing logits.")

            if all(weight == 0.0 for weight in loss_weights.values()):
                gradients.append(np.zeros_like(binder_seq.logits, dtype=np.float64))
                losses.append(0.0)
                continue

            evaluation_seed = next_af2_multimer_seed(self.config)
            output = run_alphafold2_binder(
                AlphaFold2BinderInput(
                    logits=binder_seq.logits.tolist(),
                    temperature=temperature,
                    target_pdb=self.config.target_pdb,
                    target_chain=",".join(self.config.target_chains),
                    target_hotspot=self.config.target_hotspot,
                    binder_chain=self.config.binder_chain,
                    design_positions=self.config.design_positions,
                ),
                AlphaFold2BinderConfig(
                    include_pae_matrix=self.config.include_pae_matrix,
                    bias_redesign=self.config.bias_redesign,
                    omit_aas=self.config.omit_aas,
                    num_recycles=self.config.num_recycles,
                    recycle_mode=self.config.recycle_mode,
                    model_num=self.config.model_num,
                    sample_models=self.config.sample_models,
                    use_multimer=self.config.use_multimer,
                    rm_target_seq=self.config.rm_target_seq,
                    rm_target_sc=self.config.rm_target_sc,
                    rm_template_ic=self.config.rm_template_ic,
                    loss_weights={
                        AF2_MULTIMER_TOOL_LOSS_ALIASES.get(key, key): weight for key, weight in loss_weights.items()
                    },
                    intra_contact_num=self.config.intra_contact_num,
                    intra_contact_cutoff=self.config.intra_contact_cutoff,
                    inter_contact_num=self.config.inter_contact_num,
                    inter_contact_cutoff=self.config.inter_contact_cutoff,
                    framework_contact_offset=self.config.framework_contact_offset,
                    backend=self.config.backend,
                    device=self.config.device,
                    seed=evaluation_seed,
                    soft=soft,
                    hard=hard,
                    compute_gradient=True,
                ),
            )
            if output.gradient is None:
                raise RuntimeError("AF2 multimer compute_gradient=True must populate output.gradient.")

            gradients.append(np.array(output.gradient, dtype=np.float64))
            losses.append(output.loss)
            structures = af2_multimer_structures(output.structure, self.config, len(self.inputs))

            for compiled in self.constraints:
                score = _term_score(output.metrics, compiled.objective_key, output.loss)
                metadata = af2_multimer_constraint_output_metadata(
                    output.metrics,
                    output_loss=score,
                    output_structure=output.structure,
                    loss_key=compiled.objective_key,
                    group_loss=output.loss,
                )
                compiled.constraint._write_constraint_metadata(proposal_idx, score, metadata)

            processed_ids: set[int] = set()
            for seg_idx, segment in enumerate(self.inputs):
                seq = segment.proposal_sequences[proposal_idx]
                if id(seq) in processed_ids:
                    continue
                processed_ids.add(id(seq))
                structure = structures[seg_idx]
                if structure is not None:
                    seq.structure = structure

        return GradientProviderOutput(label=self.label, gradients=gradients, losses=losses)


def objective_key_for_constraint(constraint: Constraint) -> str | None:
    """Return the AF2M loss key that implements ``constraint``.

    The key is derived from the constraint function object rather than from a
    string label. That keeps the registry close to the implementation and avoids
    accepting lookalike user labels that are not actually backed by AF2M.

    Args:
        constraint (Constraint): Public constraint to inspect.

    Returns:
        str | None: AF2M loss key for compiler-backed constraints.
            Returns ``None`` when the constraint is not backed by AF2M.
    """
    if constraint.function is None:
        return None
    return AF2_MULTIMER_STRUCTURE_LOSS_BY_FUNCTION.get(constraint.function)


def unsupported_gradient_reason(constraint: Constraint) -> str | None:
    """Return a targeted differentiability error for known AF2M exclusions.

    Some AF2M-backed forward metrics are valid for scoring but should not be
    presented as differentiable objectives. This helper lets compiler preflight
    report a precise error instead of the generic "no gradient" message.

    Args:
        constraint (Constraint): Constraint that failed objective-key lookup.

    Returns:
        str | None: User-facing explanation for known AF2M-forward-only cases.
            Returns ``None`` when there is no targeted message.
    """
    config = config_for_constraint(constraint) if constraint.function is structure_ptm_constraint else None
    if config is not None and config.structure_tool == "alphafold2_multimer":
        return (
            "structure-ptm with structure_tool='alphafold2_multimer' is not differentiable; "
            "use structure-iptm for AF2 iptm."
        )
    return None


def config_for_constraint(
    constraint: Constraint,
    *,
    strict: bool = False,
) -> StructureBasedConstraintConfig | None:
    """Parse a constraint's structure config into the canonical config model.

    Constraints may carry either an already-instantiated
    ``StructureBasedConstraintConfig`` or a dictionary from serialized program
    construction. This helper normalizes both shapes for the compiler. Exploratory
    support checks keep ``strict=False`` so a non-structure config simply returns
    ``None``; execution paths pass ``strict=True`` to preserve field-level
    validation errors and malformed dictionary errors.

    Args:
        constraint (Constraint): Constraint whose ``function_config`` should describe a
            structure-backed objective.
        strict (bool): If True, re-raise parse errors instead of returning
            ``None``.

    Returns:
        StructureBasedConstraintConfig | None: Parsed structure config.
            Returns ``None`` if the config is absent or cannot be parsed as a
            structure config.
    """
    config = constraint.function_config
    if isinstance(config, StructureBasedConstraintConfig):
        return config
    if isinstance(config, dict):
        try:
            return StructureBasedConstraintConfig(**config)
        except (TypeError, ValidationError):
            if strict:
                raise
            return None
    return None


def missing_config_message(constraint: Constraint) -> str:
    """Return the standard error for constraints without parseable config."""
    return f"Constraint '{constraint.label}' must use StructureBasedConstraintConfig."


def validate_gradient_constraint(
    constraint: Constraint,
    target_segment: Segment,
    config: StructureBasedConstraintConfig,
) -> None:
    """Validate that a constraint can be an AF2M binder-gradient objective.

    Gradient execution is only valid for AF2M constraints where the optimizer's
    target segment is the configured binder input. Target inputs are treated as
    fixed context from ``target_pdb``/chains, while the binder proposal supplies
    logits to optimize. This function also rejects filter constraints because
    thresholded pass/fail logic is not a differentiable objective.

    Args:
        constraint (Constraint): Public constraint being compiled.
        target_segment (Segment): Segment whose logits the gradient optimizer will update.
        config (StructureBasedConstraintConfig): Parsed structure config for the constraint.

    Raises:
        ValueError: If the structure tool is not AF2M, the constraint is a
            filter, configured input indices are invalid, or the optimizer
            target is not the configured binder segment.
        TypeError: If any AF2M target/binder segment is not a protein segment.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError(
            f"Constraint '{constraint.label}' is discrete-only with structure_tool={config.structure_tool!r}; "
            "only structure_tool='alphafold2_multimer' is currently compiler-backed."
        )
    af2_config = config.alphafold2_multimer_config
    if constraint.threshold is not None:
        raise ValueError(f"Constraint '{constraint.label}' is a filter; filters are not differentiable objectives.")
    if af2_config.binder_input_index >= len(constraint.inputs):
        raise ValueError(
            f"Constraint '{constraint.label}' binder_input_index={af2_config.binder_input_index} "
            f"out of bounds for {len(constraint.inputs)} inputs."
        )
    if constraint.inputs[af2_config.binder_input_index] is not target_segment:
        raise ValueError(
            f"Constraint '{constraint.label}' optimizes input {af2_config.binder_input_index}; "
            "GradientOptimizer target_segment must be that binder input."
        )
    for idx in af2_config.target_input_indices:
        if idx >= len(constraint.inputs):
            raise ValueError(
                f"Constraint '{constraint.label}' target_input_index={idx} "
                f"out of bounds for {len(constraint.inputs)} inputs."
            )
    if len(af2_config.target_chains) != len(af2_config.target_input_indices):
        raise ValueError("target_chains must match target_input_indices one-to-one.")
    for idx in [af2_config.binder_input_index, *af2_config.target_input_indices]:
        segment = constraint.inputs[idx]
        if segment.sequence_type != "protein":
            raise TypeError(
                f"Constraint '{constraint.label}' uses AF2 multimer gradients, "
                f"but input {idx} has sequence_type={segment.sequence_type!r}."
            )


def group_key(constraint: Constraint, config: StructureBasedConstraintConfig) -> tuple[Any, ...]:
    """Build the identity key used to decide which AF2M calls can be grouped.

    Two constraints may share one AF2M call only if they reference the same
    segment objects in the same order and have identical AF2M config content.
    The key intentionally uses segment identity, not sequence value, because the
    provider must update metadata and structures on those exact proposal
    objects.

    Args:
        constraint (Constraint): Constraint being considered for grouping.
        config (StructureBasedConstraintConfig): Parsed structure config for the constraint.

    Returns:
        tuple[Any, ...]: Hashable key combining input identities and serialized AF2M config.
    """
    input_ids = tuple(id(segment) for segment in constraint.inputs)
    config_json = config.alphafold2_multimer_config.model_dump_json()
    return (*input_ids, config_json)


def add_gradient_constraint(provider: AF2MultimerGradientProvider, compiled: CompiledConstraint) -> None:
    """Attach one compiled public constraint to an existing AF2M provider.

    Providers are created when the compiler first sees a compatible group key,
    then subsequent compatible constraints are appended. The label is refreshed
    so logs and errors show the full grouped set.

    Args:
        provider (AF2MultimerGradientProvider): Provider to mutate.
        compiled (CompiledConstraint): Compiled public constraint to append.
    """
    provider.constraints.append(compiled)
    provider.label = _provider_label(provider.constraints)


def can_group_scoring_constraint(
    constraint: Constraint,
    objective_key: str | None,
    config: StructureBasedConstraintConfig | None,
) -> bool:
    """Return whether ``constraint`` can join a grouped AF2M forward call.

    Forward grouping is intentionally narrower than ordinary evaluation: the
    constraint must map to an AF2M objective, have parseable AF2M config, use
    ``structure_tool='alphafold2_multimer'``, and be a scoring objective rather
    than a threshold filter. Filters keep their normal direct evaluation path so
    pass/fail semantics remain unchanged.

    Args:
        constraint (Constraint): Public scoring constraint being considered.
        objective_key (str | None): AF2M objective key from ``objective_key_for_constraint``.
        config (StructureBasedConstraintConfig | None): Parsed structure config, if available.

    Returns:
        bool: Whether the constraint is eligible for grouped AF2M scoring.
    """
    return (
        objective_key is not None
        and config is not None
        and config.structure_tool == "alphafold2_multimer"
        and constraint.threshold is None
    )


def evaluate_scoring_group(compiled_constraints: list[CompiledConstraint], mask: list[bool]) -> list[float]:
    """Evaluate a compatible group of AF2M scoring constraints.

    The AF2M tool returns the weighted sum of the requested loss terms for each
    evaluated proposal. That grouped loss is returned to the optimizer's forward
    scoring path, while per-term metadata is written back to every public
    constraint in ``compiled_constraints``. Skipped proposals preserve ``NaN``
    entries, matching the broader scoring-mask convention.

    Args:
        compiled_constraints (list[CompiledConstraint]): Non-empty compatible group produced by the
            compiler. All entries must share inputs and AF2M config.
        mask (list[bool]): Proposal-level evaluation mask. ``False`` entries are skipped.

    Returns:
        list[float]: Proposal-aligned weighted grouped scores.

    Raises:
        ValueError: If the group's first constraint no longer has parseable
            structure config.
    """
    first_constraint = compiled_constraints[0].constraint
    config_model = config_for_constraint(first_constraint, strict=True)
    if config_model is None:
        raise ValueError(f"Constraint '{first_constraint.label}' must use StructureBasedConstraintConfig.")
    config = config_model.alphafold2_multimer_config
    inputs = first_constraint.inputs
    num_proposals = inputs[0].num_proposals
    scores = [float("nan")] * num_proposals
    loss_weights = {compiled.objective_key: compiled.constraint.weight for compiled in compiled_constraints}

    for proposal_idx, should_eval in enumerate(mask):
        if not should_eval:
            continue
        proposal_tuple = tuple(segment.proposal_sequences[proposal_idx] for segment in inputs)
        validate_af2_multimer_inputs(proposal_tuple, config)
        binder_seq = proposal_tuple[config.binder_input_index]
        evaluation_seed = next_af2_multimer_seed(config)
        output = run_alphafold2_binder(
            AlphaFold2BinderInput(
                logits=one_hot_protein_matrix(binder_seq.sequence),
                target_pdb=config.target_pdb,
                target_chain=",".join(config.target_chains),
                target_hotspot=config.target_hotspot,
                binder_chain=config.binder_chain,
                design_positions=config.design_positions,
            ),
            AlphaFold2BinderConfig(
                include_pae_matrix=config.include_pae_matrix,
                bias_redesign=config.bias_redesign,
                omit_aas=config.omit_aas,
                num_recycles=config.num_recycles,
                recycle_mode=config.recycle_mode,
                model_num=config.model_num,
                sample_models=config.sample_models,
                use_multimer=config.use_multimer,
                rm_target_seq=config.rm_target_seq,
                rm_target_sc=config.rm_target_sc,
                rm_template_ic=config.rm_template_ic,
                loss_weights={
                    AF2_MULTIMER_TOOL_LOSS_ALIASES.get(key, key): weight for key, weight in loss_weights.items()
                },
                intra_contact_num=config.intra_contact_num,
                intra_contact_cutoff=config.intra_contact_cutoff,
                inter_contact_num=config.inter_contact_num,
                inter_contact_cutoff=config.inter_contact_cutoff,
                framework_contact_offset=config.framework_contact_offset,
                backend=config.backend,
                device=config.device,
                seed=evaluation_seed,
                soft=0.0,
                hard=1.0,
                compute_gradient=False,
            ),
        )
        scores[proposal_idx] = output.loss
        structures = af2_multimer_structures(output.structure, config, len(inputs))
        for compiled in compiled_constraints:
            term_score = _term_score(output.metrics, compiled.objective_key, output.loss)
            metadata = af2_multimer_constraint_output_metadata(
                output.metrics,
                output_loss=term_score,
                output_structure=output.structure,
                loss_key=compiled.objective_key,
                group_loss=output.loss,
            )
            compiled.constraint._write_constraint_metadata(proposal_idx, term_score, metadata)

        processed_ids: set[int] = set()
        for seg_idx, segment in enumerate(inputs):
            seq = segment.proposal_sequences[proposal_idx]
            if id(seq) in processed_ids:
                continue
            processed_ids.add(id(seq))
            structure = structures[seg_idx]
            if structure is not None:
                seq.structure = structure

    return scores


def _provider_label(constraints: list[CompiledConstraint]) -> str:
    """Return the grouped AF2M provider label shown in optimizer traces."""
    return "af2_multimer[" + ",".join(c.constraint.label for c in constraints) + "]"


def _term_score(metrics: dict[str, Any], objective_key: str, fallback: float) -> float:
    """Extract the scalar score for one AF2M objective from tool metrics.

    AF2M/ColabDesign exposes some terms under several spellings: raw objective
    keys, ``loss_*`` keys, and normalized tool loss keys such as ``i_con``.
    This helper centralizes that lookup so forward scoring and gradient metadata
    use the same per-term scalar. If no specific metric is present, the grouped
    tool loss is used as a fallback and a warning is logged because the
    per-constraint metadata may be less specific than requested.

    Args:
        metrics (dict[str, Any]): Metrics dictionary returned by the AF2M tool.
        objective_key (str): Compiler objective key for the public constraint.
        fallback (float): Grouped loss to use when no per-term metric is found.

    Returns:
        float: Per-term scalar score/loss.
    """
    tool_loss_key = AF2_MULTIMER_TOOL_LOSS_ALIASES.get(objective_key, objective_key)
    candidate_keys = [f"loss_{objective_key}", f"loss_{tool_loss_key}", tool_loss_key, objective_key]
    if tool_loss_key == objective_key:
        candidate_keys = [objective_key, f"loss_{objective_key}"]
    candidate_keys.append(objective_key.replace("_", ""))
    candidate_keys = list(dict.fromkeys(candidate_keys))
    for key in candidate_keys:
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    numeric_keys = sorted(key for key, value in metrics.items() if isinstance(value, int | float))
    logger.warning(
        "AF2 multimer metrics did not include a per-term score for objective %r. "
        "Checked keys: %s. Available numeric metric keys: %s. "
        "Using grouped loss for that constraint's metadata score.",
        objective_key,
        ", ".join(candidate_keys),
        ", ".join(numeric_keys) or "<none>",
    )
    return fallback
