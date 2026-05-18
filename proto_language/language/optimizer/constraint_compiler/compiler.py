"""Private constraint compiler orchestration for optimizer constraints.

Public constraints are intentionally phrased in biological terms: users compose
``structure-plddt``, ``structure-contact``, sequence perplexity terms, and other
objectives without needing to know how a particular model/tool exposes them.
Most constraints can be evaluated directly by calling ``Constraint.evaluate()``
or differentiated directly through a public ``backward`` callable.

Some model backends have a different execution shape. A model may need to
combine several public constraints into one model call, use backend-specific
objective names, or return one gradient for a weighted sum of terms. ESMFold,
AlphaFold2 multimer, and Malinois are compiled backends; their model-specific
code lives in provider modules while this module keeps the optimizer-facing
flow small and explicit.

The important invariant is that compiled providers present the same contract as
direct differentiable constraints: one loss and one target-segment gradient per
proposal. The optimizer therefore does not need to know whether a gradient came
from a public backward function or from a backend-specific grouped model call.
"""

from typing import Any

from pydantic import BaseModel

from proto_language.language.constraint.constraint_registry import ConstraintSpec
from proto_language.language.constraint.sequence_annotation.malinois_activity_constraint import (
    malinois_activity_constraint,
)
from proto_language.language.core import Constraint, Segment
from proto_language.language.core.sequence import SequenceType
from proto_language.language.optimizer.constraint_compiler import alphafold2_multimer_provider as af2m
from proto_language.language.optimizer.constraint_compiler import esmfold_provider as esmfold
from proto_language.language.optimizer.constraint_compiler import malinois_provider as malinois
from proto_language.language.optimizer.constraint_compiler.base import (
    CompiledConstraint,
    EffectiveWeight,
    GradientProvider,
    GradientProviderOutput,
)


class DirectGradientProvider(GradientProvider):
    """Provider for one constraint that already exposes a backward callable.

    Direct providers are the non-compiled path. They call
    ``Constraint.compute_gradient`` and then select the gradient entry for the
    optimizer target segment. This is the reference behavior that compiled
    backend providers must emulate.
    """

    def __init__(self, constraint: Constraint, target_index: int):
        """Create a provider for ``constraint``.

        Args:
            constraint (Constraint): Differentiable public constraint.
            target_index (int): Position of the optimizer target segment inside the
                constraint's input list.
        """
        self.constraint = constraint
        self.target_index = target_index
        self.label = constraint.label

    def compute(
        self,
        *,
        temperature: float,
        soft: float,
        hard: float,
        step: int,
        effective_weight: EffectiveWeight,
    ) -> GradientProviderOutput:
        """Compute gradients by delegating to the constraint backward callable.

        Args:
            temperature (float): Sampling temperature forwarded to the constraint.
            soft (float): Soft sequence interpolation coefficient forwarded to the constraint.
            hard (float): Hard sequence interpolation coefficient forwarded to the constraint.
            step (int): Optimizer step used to evaluate the constraint's effective weight.
            effective_weight (EffectiveWeight): Callback returning the current scalar weight.

        Returns:
            GradientProviderOutput: Proposal-aligned target gradients and weighted losses.
        """
        results = self.constraint.compute_gradient(temperature=temperature, soft=soft, hard=hard)
        weight = effective_weight(self.constraint, step)
        return GradientProviderOutput(
            label=self.label,
            gradients=[result.gradient[self.target_index] for result in results],
            losses=[weight * result.loss for result in results],
            weight=weight,
        )


def compile_gradient_providers(constraints: list[Constraint], target_segment: Segment) -> list[GradientProvider]:
    """Build the gradient providers used by ``GradientOptimizer``.

    The compiler walks the user-requested constraints in order. Direct
    differentiable constraints become ``DirectGradientProvider`` instances.
    Constraints without a public backward function may still be differentiable
    if a backend adapter knows how to compile them into a grouped model call.
    The control flow is intentionally backend-neutral: lookup objective key,
    parse config, validate target segment, group compatible constraints, then
    return providers.

    Grouping is private optimizer infrastructure. Public constraints remain the
    user's API and still receive their own metadata, scores, and weight
    schedules even when they share a backend invocation.

    Args:
        constraints (list[Constraint]): Constraints attached to the optimizer target program.
        target_segment (Segment): Segment whose proposal logits are being optimized.

    Returns:
        list[GradientProvider]: Providers in optimizer execution order.
            Each provider returns one target-segment gradient and one weighted
            loss per proposal.

    Raises:
        ValueError: If a constraint is not differentiable, has no supported
            compiled backend, lacks a parseable config, or targets a segment
            other than ``target_segment``.
    """
    providers: list[GradientProvider] = []
    af2_provider_by_key: dict[tuple[Any, ...], af2m.AF2MultimerGradientProvider] = {}
    esmfold_provider_by_key: dict[tuple[Any, ...], esmfold.ESMFoldGradientProvider] = {}
    malinois_provider_by_key: dict[tuple[Any, ...], malinois.MalinoisGradientProvider] = {}

    for constraint in constraints:
        malinois_objective_key = malinois.objective_key_for_constraint(constraint)
        if malinois_objective_key is not None:
            malinois_config = malinois.config_for_constraint(constraint, strict=True)
            if malinois_config is None:
                raise ValueError(malinois.missing_config_message(constraint))
            malinois.validate_gradient_constraint(constraint, target_segment, malinois_config)
            group_key = malinois.group_key(target_segment, malinois_config)
            malinois_provider = malinois_provider_by_key.get(group_key)
            if malinois_provider is None:
                malinois_provider = malinois.MalinoisGradientProvider(
                    constraints=[],
                    config=malinois_config,
                    target_segment=target_segment,
                )
                malinois_provider_by_key[group_key] = malinois_provider
                providers.append(malinois_provider)
            malinois.add_gradient_constraint(
                malinois_provider,
                CompiledConstraint(constraint=constraint, objective_key=malinois_objective_key),
            )
            continue

        if constraint.supports_gradient:
            providers.append(DirectGradientProvider(constraint, constraint.inputs.index(target_segment)))
            continue

        esmfold_objective_key = esmfold.objective_key_for_constraint(constraint)
        if esmfold_objective_key is not None:
            esmfold_config = esmfold.config_for_constraint(constraint, strict=True)
            if esmfold_config is None:
                raise ValueError(esmfold.missing_config_message(constraint))
            if esmfold_config.structure_tool == "esmfold":
                esmfold.validate_gradient_constraint(constraint, target_segment, esmfold_config)
                group_key = esmfold.group_key(constraint, target_segment, esmfold_config)
                esmfold_provider = esmfold_provider_by_key.get(group_key)
                if esmfold_provider is None:
                    esmfold_provider = esmfold.ESMFoldGradientProvider(
                        constraints=[],
                        config=esmfold_config.esmfold_config,
                        inputs=constraint.inputs,
                        target_segment=target_segment,
                    )
                    esmfold_provider_by_key[group_key] = esmfold_provider
                    providers.append(esmfold_provider)
                esmfold.add_gradient_constraint(
                    esmfold_provider,
                    CompiledConstraint(constraint=constraint, objective_key=esmfold_objective_key),
                )
                continue

        esmfold_reason = esmfold.unsupported_gradient_reason(constraint)
        if esmfold_reason is not None:
            raise ValueError(esmfold_reason)

        objective_key = af2m.objective_key_for_constraint(constraint)
        if objective_key is None:
            reason = af2m.unsupported_gradient_reason(constraint)
            raise ValueError(reason or f"Constraint '{constraint.label}' does not support gradient evaluation.")

        config = af2m.config_for_constraint(constraint, strict=True)
        if config is None:
            raise ValueError(af2m.missing_config_message(constraint))
        af2m.validate_gradient_constraint(constraint, target_segment, config)
        group_key = af2m.group_key(constraint, config)
        af2_provider = af2_provider_by_key.get(group_key)
        if af2_provider is None:
            af2_provider = af2m.AF2MultimerGradientProvider(
                constraints=[],
                config=config.alphafold2_multimer_config,
                inputs=constraint.inputs,
            )
            af2_provider_by_key[group_key] = af2_provider
            providers.append(af2_provider)
        af2m.add_gradient_constraint(
            af2_provider,
            CompiledConstraint(constraint=constraint, objective_key=objective_key),
        )

    return providers


def evaluate_scoring_constraints(
    constraints: list[Constraint],
    *,
    mask: list[bool],
    verbose: bool = False,
) -> list[list[float]]:
    """Evaluate forward scoring constraints, grouping compatible backend calls.

    This is the forward, non-gradient counterpart to
    ``compile_gradient_providers``. Most constraints return one weighted score
    array per public constraint. Backend-compatible constraints may instead
    return one weighted score array per compiled scoring group, because a model
    such as AF2 multimer returns a single scalar for a weighted sum of requested
    terms. Public per-constraint metadata is still written for every constraint
    inside the group.

    Ordering matters. When a non-groupable constraint is encountered, queued
    backend groups are flushed before evaluating that constraint. This keeps
    scoring units ordered by their first public constraint while still avoiding
    redundant backend calls where possible.

    Args:
        constraints (list[Constraint]): Scoring constraints to evaluate.
        mask (list[bool]): Proposal mask passed through to each constraint evaluation.
        verbose (bool): Whether direct constraint evaluations should log per-proposal
            details.

    Returns:
        list[list[float]]: Weighted score arrays, one entry per scoring unit.
            A scoring unit is either one direct public constraint or one
            compiled backend group containing multiple public constraints.
    """
    outputs: list[list[float]] = []
    group_by_key: dict[tuple[str, tuple[Any, ...]], list[CompiledConstraint]] = {}
    group_order: list[tuple[str, tuple[Any, ...]]] = []

    for constraint in constraints:
        objective_key = esmfold.objective_key_for_constraint(constraint)
        if objective_key is not None:
            esmfold_config = esmfold.config_for_constraint(constraint, strict=True)
            if esmfold_config is not None and esmfold.can_group_scoring_constraint(
                constraint, objective_key, esmfold_config
            ):
                group_key = ("esmfold", esmfold.scoring_group_key(constraint, esmfold_config))
                if group_key not in group_by_key:
                    group_by_key[group_key] = []
                    group_order.append(group_key)
                group_by_key[group_key].append(CompiledConstraint(constraint=constraint, objective_key=objective_key))
                continue

        objective_key = malinois.objective_key_for_constraint(constraint)
        if objective_key is not None:
            malinois_config = malinois.config_for_constraint(constraint, strict=True)
            if malinois_config is not None and malinois.can_group_scoring_constraint(
                constraint, objective_key, malinois_config
            ):
                group_key = ("malinois", malinois.scoring_group_key(constraint, malinois_config))
                if group_key not in group_by_key:
                    group_by_key[group_key] = []
                    group_order.append(group_key)
                group_by_key[group_key].append(CompiledConstraint(constraint=constraint, objective_key=objective_key))
                continue

        objective_key = af2m.objective_key_for_constraint(constraint)
        if objective_key is None:
            _flush_scoring_groups(group_order, group_by_key, outputs, mask)
            outputs.append([float(score) for score in constraint.evaluate(mask=mask, verbose=verbose)])
            continue

        af2m_config = af2m.config_for_constraint(constraint, strict=True)
        if af2m_config is None or not af2m.can_group_scoring_constraint(constraint, objective_key, af2m_config):
            _flush_scoring_groups(group_order, group_by_key, outputs, mask)
            outputs.append([float(score) for score in constraint.evaluate(mask=mask, verbose=verbose)])
            continue

        group_key = ("af2", af2m.group_key(constraint, af2m_config))
        if group_key not in group_by_key:
            group_by_key[group_key] = []
            group_order.append(group_key)
        group_by_key[group_key].append(CompiledConstraint(constraint=constraint, objective_key=objective_key))

    _flush_scoring_groups(group_order, group_by_key, outputs, mask)
    return outputs


def constraint_supports_compiled_gradient(
    constraint: Constraint, target_segment: Segment | None = None
) -> tuple[bool, str | None]:
    """Check whether ``constraint`` can be used by ``GradientOptimizer``.

    This helper is the preflight version of ``compile_gradient_providers``. It
    does not create providers or run tools; it only reports whether the
    constraint has either a public backward function or a supported compiled
    backend path. When ``target_segment`` is provided, backend-specific role
    checks are also run so the optimizer can fail before starting proposals.

    As new compiled backends are added, their objective lookup and validation
    should plug in here as well as in ``compile_gradient_providers``. That keeps
    user-facing differentiability errors consistent between validation and
    execution.

    Args:
        constraint (Constraint): Constraint to check.
        target_segment (Segment | None): Optional optimizer target segment. If omitted, only
            backend availability and config parsing are checked.

    Returns:
        tuple[bool, str | None]: Support flag and optional error reason.
            Returns ``(True, None)`` when the constraint is differentiable in
            the current compiler. Otherwise returns ``(False, reason)`` with a
            message suitable for optimizer errors.
    """
    malinois_objective_key = malinois.objective_key_for_constraint(constraint)
    if malinois_objective_key is not None:
        malinois_config = malinois.config_for_constraint(constraint)
        if malinois_config is None:
            return False, malinois.missing_config_message(constraint)
        if target_segment is None:
            return True, None
        try:
            malinois.validate_gradient_constraint(constraint, target_segment, malinois_config)
        except (TypeError, ValueError) as exc:
            return False, str(exc)
        return True, None

    if constraint.supports_gradient:
        if target_segment is not None and target_segment not in constraint.inputs:
            return False, (
                f"Constraint '{constraint.label}' inputs do not include the optimizer target_segment; "
                "GradientOptimizer can only differentiate constraints whose inputs contain the target."
            )
        return True, None

    esmfold_objective_key = esmfold.objective_key_for_constraint(constraint)
    if esmfold_objective_key is not None:
        esmfold_config = esmfold.config_for_constraint(constraint)
        if esmfold_config is None:
            return False, esmfold.missing_config_message(constraint)
        if esmfold_config.structure_tool == "esmfold":
            if target_segment is None:
                return True, None
            try:
                esmfold.validate_gradient_constraint(constraint, target_segment, esmfold_config)
            except (TypeError, ValueError) as exc:
                return False, str(exc)
            return True, None

    esmfold_reason = esmfold.unsupported_gradient_reason(constraint)
    if esmfold_reason is not None:
        return False, esmfold_reason

    objective_key = af2m.objective_key_for_constraint(constraint)
    if objective_key is None:
        reason = af2m.unsupported_gradient_reason(constraint)
        if reason is not None:
            return False, reason
        return False, f"Constraint '{constraint.label}' does not support gradient evaluation."

    af2m_config = af2m.config_for_constraint(constraint)
    if af2m_config is None:
        return False, af2m.missing_config_message(constraint)
    if target_segment is None:
        return True, None
    try:
        af2m.validate_gradient_constraint(constraint, target_segment, af2m_config)
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    return True, None


class GradientInputRequirement(BaseModel):
    """A vocab requirement on a subset of constraint inputs.

    Attributes:
        sequence_types (list[SequenceType]): Allowed sequence types for selected inputs.
        config_path (str | None): Config path selecting inputs; ``None`` means every input.
        config_path_is_list (bool): True if ``config_path`` resolves to a list of indices.
    """

    sequence_types: list[SequenceType]
    config_path: str | None = None
    config_path_is_list: bool = False


class GradientRule(BaseModel):
    """One compiler-backed gradient path for a constraint.

    Attributes:
        label (str): Human-readable backend label.
        structure_tool (str): Backend identifier exposed to clients. The name
            is legacy from structure-prediction constraints; non-structure
            compiled backends such as Malinois also use it.
        target_input_config_path (str | None): Config path of the gradient-receiving input; ``None`` means any input.
        input_requirements (list[GradientInputRequirement]): Vocab requirements per input subset.
    """

    label: str
    structure_tool: str
    target_input_config_path: str | None = None
    input_requirements: list[GradientInputRequirement]


class GradientSupport(BaseModel):
    """Compiler-backed gradient paths discoverable for a constraint.

    Attributes:
        rules (list[GradientRule]): One rule per supporting backend.
    """

    rules: list[GradientRule]


_ESMFOLD_RULE = GradientRule(
    label="ESMFold gradient",
    structure_tool="esmfold",
    target_input_config_path=None,
    input_requirements=[GradientInputRequirement(sequence_types=["protein"])],
)

_AF2_MULTIMER_RULE = GradientRule(
    label="AF2 multimer gradient",
    structure_tool="alphafold2_multimer",
    target_input_config_path="alphafold2_multimer_config.binder_input_index",
    input_requirements=[
        GradientInputRequirement(
            sequence_types=["protein"],
            config_path="alphafold2_multimer_config.binder_input_index",
        ),
        GradientInputRequirement(
            sequence_types=["protein"],
            config_path="alphafold2_multimer_config.target_input_indices",
            config_path_is_list=True,
        ),
    ],
)

_MALINOIS_RULE = GradientRule(
    label="Malinois gradient",
    structure_tool="malinois",
    target_input_config_path=None,
    input_requirements=[GradientInputRequirement(sequence_types=["dna"])],
)


def gradient_support_for_constraint_spec(spec: ConstraintSpec) -> GradientSupport | None:
    """Return compiler-backed gradient paths for a registered constraint.

    Args:
        spec (ConstraintSpec): The constraint registry entry to inspect.

    Returns:
        GradientSupport | None: Compiled gradient paths, or ``None`` when no
            backend supports the constraint.
    """
    if spec.function is None:
        return None
    rules: list[GradientRule] = []
    if spec.function in esmfold.ESMFOLD_STRUCTURE_LOSS_BY_FUNCTION:
        rules.append(_ESMFOLD_RULE)
    if spec.function is malinois_activity_constraint:
        rules.append(_MALINOIS_RULE)
    if spec.function in af2m.AF2_MULTIMER_STRUCTURE_LOSS_BY_FUNCTION:
        rules.append(_AF2_MULTIMER_RULE)
    return GradientSupport(rules=rules) if rules else None


def _flush_scoring_groups(
    group_order: list[tuple[str, tuple[Any, ...]]],
    group_by_key: dict[tuple[str, tuple[Any, ...]], list[CompiledConstraint]],
    outputs: list[list[float]],
    mask: list[bool],
) -> None:
    """Evaluate queued forward scoring groups and clear the queues."""
    for group_key in group_order:
        backend, _key = group_key
        if backend == "af2":
            outputs.append(af2m.evaluate_scoring_group(group_by_key[group_key], mask))
        elif backend == "esmfold":
            outputs.append(esmfold.evaluate_scoring_group(group_by_key[group_key], mask))
        elif backend == "malinois":
            outputs.append(malinois.evaluate_scoring_group(group_by_key[group_key], mask))
        else:
            raise ValueError(f"Unknown scoring compiler backend {backend!r}.")
    group_order.clear()
    group_by_key.clear()
