"""Private constraint compiler orchestration for optimizer constraints.

Public constraints are intentionally phrased in biological terms: users compose
``structure-plddt``, ``structure-contact``, sequence perplexity terms, and other
objectives without needing to know how a particular model/tool exposes them.
Most constraints can be evaluated directly by calling ``Constraint.evaluate()``
or differentiated directly through a public ``backward`` callable.

Some model backends have a different execution shape. A model may need to
combine several public constraints into one model call, use backend-specific
objective names, or return one gradient for a weighted sum of terms. ESMFold,
AlphaFold2 binder, and Malinois are compiled backends; their model-specific
code lives in provider modules while this module keeps the optimizer-facing
flow small and explicit.

The important invariant is that compiled providers present the same contract as
direct differentiable constraints: one loss and one target-segment gradient per
proposal. The optimizer therefore does not need to know whether a gradient came
from a public backward function or from a backend-specific grouped model call.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from proto_language.constraint.constraint_registry import ConstraintSpec
from proto_language.constraint.sequence_annotation.malinois_activity_constraint import (
    malinois_activity_constraint,
)
from proto_language.core import Constraint, Segment
from proto_language.core.sequence import SequenceType
from proto_language.optimizer.constraint_compiler import alphafold2_binder_provider as af2b
from proto_language.optimizer.constraint_compiler import esmfold_provider as esmfold
from proto_language.optimizer.constraint_compiler import malinois_provider as malinois
from proto_language.optimizer.constraint_compiler import protenix_provider as protenix
from proto_language.optimizer.constraint_compiler.base import (
    CompiledConstraint,
    EffectiveWeight,
    GradientProvider,
    GradientProviderOutput,
    clone_execution_config,
)


class DirectGradientProvider(GradientProvider):
    """Provider for one constraint that already exposes a backward callable.

    Direct providers are the non-compiled path. They call
    ``Constraint.compute_gradient`` and then sum the gradient entries for every
    input position occupied by the optimizer target segment. This is the
    reference behavior that compiled backend providers must emulate.
    """

    def __init__(self, constraint: Constraint, target_indices: tuple[int, ...]):
        """Create a provider for ``constraint``.

        Args:
            constraint (Constraint): Differentiable public constraint.
            target_indices (tuple[int, ...]): Positions of the optimizer target
                segment inside the constraint's input list.
        """
        if not target_indices:
            raise ValueError(f"Constraint '{constraint.label}' inputs do not include the optimizer target segment.")
        self.constraint = constraint
        self.target_indices = target_indices
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
            gradients=[
                sum(
                    (result.gradient[index] for index in self.target_indices),
                    start=np.zeros_like(result.gradient[self.target_indices[0]]),
                )
                for result in results
            ],
            losses=[weight * result.loss for result in results],
            weight=weight,
        )


def compile_gradient_providers(
    constraints: list[Constraint], target_segment: Segment, *, seed: int | None = None
) -> list[GradientProvider]:
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
        seed (int | None): Effective optimizer seed used to initialize isolated
            compiler-group seed streams.

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
    providers_by_adapter: dict[str, dict[tuple[Any, ...], GradientProvider]] = {
        adapter.backend_id: {} for adapter in _COMPILER_ADAPTERS
    }
    next_group_seed = _group_seed_factory(seed, namespace=0x47524144)

    for constraint in constraints:
        if constraint.supports_gradient:
            target_indices = tuple(
                index for index, segment in enumerate(constraint.inputs) if segment is target_segment
            )
            providers.append(DirectGradientProvider(constraint, target_indices))
            continue

        for adapter in _COMPILER_ADAPTERS:
            if adapter.compile_gradient is not None and adapter.compile_gradient(
                constraint,
                target_segment,
                providers_by_adapter[adapter.backend_id],
                providers,
                next_group_seed,
            ):
                break
        else:
            capabilities = resolve_constraint_capabilities(constraint, target_segment)
            reason = capabilities.gradient_reason
            raise ValueError(reason or f"Constraint '{constraint.label}' does not support gradient evaluation.")

    return providers


def evaluate_scoring_constraints(
    constraints: list[Constraint],
    *,
    mask: list[bool],
    verbose: bool = False,
    seed: int | None = None,
) -> list[list[float]]:
    """Evaluate forward scoring constraints, grouping compatible backend calls.

    This is the forward, non-gradient counterpart to
    ``compile_gradient_providers``. Most constraints return one weighted score
    array per public constraint. Backend-compatible constraints may instead
    return one weighted score array per compiled scoring group, because a model
    such as AF2 binder returns a single scalar for a weighted sum of requested
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
        seed (int | None): Effective optimizer seed used for isolated compiled
            scoring-group seed streams.

    Returns:
        list[list[float]]: Weighted score arrays, one entry per scoring unit.
            A scoring unit is either one direct public constraint or one
            compiled backend group containing multiple public constraints.
    """
    return compile_scoring_plan(constraints, seed=seed).evaluate(mask=mask, verbose=verbose)


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
    capabilities = resolve_constraint_capabilities(constraint, target_segment)
    return capabilities.supports_gradient, capabilities.gradient_reason


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


@dataclass(frozen=True)
class ConstraintCapabilities:
    """Effective direct and compiler-backed evaluation capabilities.

    Attributes:
        supports_discrete (bool): Whether forward scoring is available.
        supports_gradient (bool): Whether direct or compiled gradients are available.
        gradient_reason (str | None): Reason gradient evaluation is unavailable.
    """

    supports_discrete: bool
    supports_gradient: bool
    gradient_reason: str | None = None

    @property
    def mode(self) -> Literal["discrete", "gradient", "dual"]:
        """Return the registry-compatible effective mode."""
        if self.supports_discrete and self.supports_gradient:
            return "dual"
        if self.supports_gradient:
            return "gradient"
        return "discrete"


GradientCompiler = Callable[
    [
        Constraint,
        Segment,
        dict[tuple[Any, ...], GradientProvider],
        list[GradientProvider],
        Callable[[], int | None],
    ],
    bool,
]
GradientChecker = Callable[[Constraint, Segment | None], tuple[bool, str | None] | None]


@dataclass(frozen=True)
class ScoringMatch:
    """One compiled scoring constraint plus its backend execution config."""

    group_key: tuple[Any, ...]
    compiled: CompiledConstraint
    config: BaseModel


ScoringMatcher = Callable[[Constraint], ScoringMatch | None]


@dataclass(frozen=True)
class CompilerAdapter:
    """One backend's compiler hooks for discovery, gradients, and scoring.

    Attributes:
        backend_id (str): Stable backend identifier used in group keys.
        matches_gradient_function (Callable[[Callable[..., Any] | None], bool]): Potential gradient-support lookup.
        gradient_rule (GradientRule | None): Client-facing gradient metadata.
        compile_gradient (GradientCompiler | None): Gradient provider builder.
        check_gradient (GradientChecker | None): Runtime gradient preflight.
        match_scoring (ScoringMatcher | None): Additive scoring group matcher.
        evaluate_scoring (Callable[[list[CompiledConstraint], list[bool], Any], list[float]] | None): Grouped scoring evaluator.
    """

    backend_id: str
    matches_gradient_function: Callable[[Callable[..., Any] | None], bool]
    gradient_rule: GradientRule | None
    compile_gradient: GradientCompiler | None
    check_gradient: GradientChecker | None
    match_scoring: ScoringMatcher | None
    evaluate_scoring: Callable[[list[CompiledConstraint], list[bool], Any], list[float]] | None


_ESMFOLD_RULE = GradientRule(
    label="ESMFold gradient",
    structure_tool="esmfold",
    target_input_config_path=None,
    input_requirements=[GradientInputRequirement(sequence_types=["protein"])],
)

_AF2_BINDER_RULE = GradientRule(
    label="AF2 binder gradient",
    structure_tool="alphafold2_binder",
    target_input_config_path="alphafold2_binder_config.binder_input_index",
    input_requirements=[
        GradientInputRequirement(
            sequence_types=["protein"],
            config_path="alphafold2_binder_config.binder_input_index",
        ),
        GradientInputRequirement(
            sequence_types=["protein"],
            config_path="alphafold2_binder_config.target_input_indices",
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


def _compile_esmfold_gradient(
    constraint: Constraint,
    target_segment: Segment,
    providers_by_key: dict[tuple[Any, ...], GradientProvider],
    providers: list[GradientProvider],
    next_group_seed: Callable[[], int | None],
) -> bool:
    objective_key = esmfold.objective_key_for_constraint(constraint)
    if objective_key is None:
        return False
    config = esmfold.config_for_constraint(constraint, strict=True)
    if config is None:
        raise ValueError(esmfold.missing_config_message(constraint))
    if config.structure_tool != "esmfold":
        return False
    esmfold.validate_gradient_constraint(constraint, target_segment, config)
    group_key = esmfold.group_key(constraint, target_segment, config)
    provider = providers_by_key.get(group_key)
    if provider is None:
        provider = esmfold.ESMFoldGradientProvider(
            constraints=[],
            config=clone_execution_config(config.esmfold_config, seed=next_group_seed()),
            inputs=constraint.inputs,
            target_segment=target_segment,
        )
        providers_by_key[group_key] = provider
        providers.append(provider)
    assert isinstance(provider, esmfold.ESMFoldGradientProvider)  # noqa: S101 -- type narrowing
    esmfold.add_gradient_constraint(provider, CompiledConstraint(constraint=constraint, objective_key=objective_key))
    return True


def _check_esmfold_gradient(constraint: Constraint, target_segment: Segment | None) -> tuple[bool, str | None] | None:
    objective_key = esmfold.objective_key_for_constraint(constraint)
    config = esmfold.config_for_constraint(constraint)
    if config is None:
        return (False, esmfold.missing_config_message(constraint)) if objective_key is not None else None
    if config.structure_tool != "esmfold":
        return None
    if objective_key is None:
        reason = esmfold.unsupported_gradient_reason(constraint)
        return False, reason or f"Constraint '{constraint.label}' is not differentiable with ESMFold."
    if target_segment is not None:
        try:
            esmfold.validate_gradient_constraint(constraint, target_segment, config)
        except (TypeError, ValueError) as exc:
            return False, str(exc)
    return True, None


def _match_esmfold_scoring(constraint: Constraint) -> ScoringMatch | None:
    objective_key = esmfold.objective_key_for_constraint(constraint)
    if objective_key is None:
        return None
    config = esmfold.config_for_constraint(constraint, strict=True)
    if config is None or not esmfold.can_group_scoring_constraint(constraint, objective_key, config):
        return None
    return ScoringMatch(
        esmfold.scoring_group_key(constraint, config),
        CompiledConstraint(constraint, objective_key),
        config,
    )


def _compile_af2_gradient(
    constraint: Constraint,
    target_segment: Segment,
    providers_by_key: dict[tuple[Any, ...], GradientProvider],
    providers: list[GradientProvider],
    next_group_seed: Callable[[], int | None],
) -> bool:
    objective_key = af2b.objective_key_for_constraint(constraint)
    if objective_key is None:
        return False
    config = af2b.config_for_constraint(constraint, strict=True)
    if config is None:
        raise ValueError(af2b.missing_config_message(constraint))
    if config.structure_tool != "alphafold2_binder":
        return False
    af2b.validate_gradient_constraint(constraint, target_segment, config)
    group_key = af2b.group_key(constraint, config)
    provider = providers_by_key.get(group_key)
    if provider is None:
        provider = af2b.AF2BinderGradientProvider(
            constraints=[],
            config=clone_execution_config(config.alphafold2_binder_config, seed=next_group_seed()),
            inputs=constraint.inputs,
        )
        providers_by_key[group_key] = provider
        providers.append(provider)
    assert isinstance(provider, af2b.AF2BinderGradientProvider)  # noqa: S101 -- type narrowing
    af2b.add_gradient_constraint(provider, CompiledConstraint(constraint=constraint, objective_key=objective_key))
    return True


def _check_af2_gradient(constraint: Constraint, target_segment: Segment | None) -> tuple[bool, str | None] | None:
    objective_key = af2b.objective_key_for_constraint(constraint)
    config = af2b.config_for_constraint(constraint)
    if config is None:
        return (False, af2b.missing_config_message(constraint)) if objective_key is not None else None
    if config.structure_tool != "alphafold2_binder":
        return None
    if objective_key is None:
        reason = af2b.unsupported_gradient_reason(constraint)
        return False, reason or f"Constraint '{constraint.label}' is not differentiable with AF2 binder."
    if target_segment is not None:
        try:
            af2b.validate_gradient_constraint(constraint, target_segment, config)
        except (TypeError, ValueError) as exc:
            return False, str(exc)
    return True, None


def _match_af2_scoring(constraint: Constraint) -> ScoringMatch | None:
    objective_key = af2b.objective_key_for_constraint(constraint)
    if objective_key is None:
        return None
    config = af2b.config_for_constraint(constraint, strict=True)
    if config is None or not af2b.can_group_scoring_constraint(constraint, objective_key, config):
        return None
    return ScoringMatch(
        af2b.group_key(constraint, config),
        CompiledConstraint(constraint, objective_key),
        config,
    )


def _compile_malinois_gradient(
    constraint: Constraint,
    target_segment: Segment,
    providers_by_key: dict[tuple[Any, ...], GradientProvider],
    providers: list[GradientProvider],
    next_group_seed: Callable[[], int | None],
) -> bool:
    objective_key = malinois.objective_key_for_constraint(constraint)
    if objective_key is None:
        return False
    config = malinois.config_for_constraint(constraint, strict=True)
    if config is None:
        raise ValueError(malinois.missing_config_message(constraint))
    malinois.validate_gradient_constraint(constraint, target_segment, config)
    group_key = malinois.group_key(target_segment, config)
    provider = providers_by_key.get(group_key)
    if provider is None:
        provider = malinois.MalinoisGradientProvider(
            constraints=[],
            config=clone_execution_config(config, seed=next_group_seed()),
            target_segment=target_segment,
        )
        providers_by_key[group_key] = provider
        providers.append(provider)
    assert isinstance(provider, malinois.MalinoisGradientProvider)  # noqa: S101 -- type narrowing
    malinois.add_gradient_constraint(provider, CompiledConstraint(constraint=constraint, objective_key=objective_key))
    return True


def _check_malinois_gradient(constraint: Constraint, target_segment: Segment | None) -> tuple[bool, str | None] | None:
    if malinois.objective_key_for_constraint(constraint) is None:
        return None
    config = malinois.config_for_constraint(constraint)
    if config is None:
        return False, malinois.missing_config_message(constraint)
    if target_segment is not None:
        try:
            malinois.validate_gradient_constraint(constraint, target_segment, config)
        except (TypeError, ValueError) as exc:
            return False, str(exc)
    return True, None


def _match_malinois_scoring(constraint: Constraint) -> ScoringMatch | None:
    objective_key = malinois.objective_key_for_constraint(constraint)
    if objective_key is None:
        return None
    config = malinois.config_for_constraint(constraint, strict=True)
    if config is None or not malinois.can_group_scoring_constraint(constraint, objective_key, config):
        return None
    return ScoringMatch(
        malinois.scoring_group_key(constraint, config),
        CompiledConstraint(constraint, objective_key),
        config,
    )


def _match_protenix_scoring(constraint: Constraint) -> ScoringMatch | None:
    objective_key = protenix.objective_key_for_constraint(constraint)
    if objective_key is None:
        return None
    config = protenix.config_for_constraint(constraint, strict=True)
    if config is None or not protenix.can_group_scoring_constraint(constraint, objective_key, config):
        return None
    return ScoringMatch(
        protenix.scoring_group_key(constraint, config),
        CompiledConstraint(constraint, objective_key),
        config,
    )


_COMPILER_ADAPTERS = (
    CompilerAdapter(
        backend_id="esmfold",
        matches_gradient_function=lambda function: function in esmfold.ESMFOLD_STRUCTURE_LOSS_BY_FUNCTION,
        gradient_rule=_ESMFOLD_RULE,
        compile_gradient=_compile_esmfold_gradient,
        check_gradient=_check_esmfold_gradient,
        match_scoring=_match_esmfold_scoring,
        evaluate_scoring=esmfold.evaluate_scoring_group,
    ),
    CompilerAdapter(
        backend_id="protenix",
        matches_gradient_function=lambda _function: False,
        gradient_rule=None,
        compile_gradient=None,
        check_gradient=None,
        match_scoring=_match_protenix_scoring,
        evaluate_scoring=protenix.evaluate_scoring_group,
    ),
    CompilerAdapter(
        backend_id="malinois",
        matches_gradient_function=lambda function: function is malinois_activity_constraint,
        gradient_rule=_MALINOIS_RULE,
        compile_gradient=_compile_malinois_gradient,
        check_gradient=_check_malinois_gradient,
        match_scoring=_match_malinois_scoring,
        evaluate_scoring=malinois.evaluate_scoring_group,
    ),
    CompilerAdapter(
        backend_id="alphafold2_binder",
        matches_gradient_function=lambda function: function in af2b.AF2_BINDER_STRUCTURE_LOSS_BY_FUNCTION,
        gradient_rule=_AF2_BINDER_RULE,
        compile_gradient=_compile_af2_gradient,
        check_gradient=_check_af2_gradient,
        match_scoring=_match_af2_scoring,
        evaluate_scoring=af2b.evaluate_scoring_group,
    ),
)


def resolve_constraint_capabilities(
    constraint: Constraint | ConstraintSpec,
    target_segment: Segment | None = None,
) -> ConstraintCapabilities:
    """Resolve effective direct and compiler-backed evaluation support.

    Args:
        constraint (Constraint | ConstraintSpec): Runtime constraint or registry entry.
        target_segment (Segment | None): Optional gradient target for role validation.

    Returns:
        ConstraintCapabilities: Effective evaluation support and any gradient error.
    """
    if isinstance(constraint, ConstraintSpec):
        supports_discrete = constraint.function is not None
        supports_gradient = constraint.backward is not None or any(
            adapter.matches_gradient_function(constraint.function) for adapter in _COMPILER_ADAPTERS
        )
        return ConstraintCapabilities(supports_discrete, supports_gradient)

    if constraint.supports_gradient:
        if target_segment is not None and target_segment not in constraint.inputs:
            return ConstraintCapabilities(
                constraint.supports_discrete,
                False,
                f"Constraint '{constraint.label}' inputs do not include the optimizer target_segment; "
                "GradientOptimizer can only differentiate constraints whose inputs contain the target.",
            )
        return ConstraintCapabilities(constraint.supports_discrete, True)

    reasons: list[str] = []
    for adapter in _COMPILER_ADAPTERS:
        if adapter.check_gradient is None:
            continue
        result = adapter.check_gradient(constraint, target_segment)
        if result is None:
            continue
        supported, reason = result
        if supported:
            return ConstraintCapabilities(constraint.supports_discrete, True)
        if reason is not None:
            reasons.append(reason)
    reason = reasons[0] if reasons else f"Constraint '{constraint.label}' does not support gradient evaluation."
    return ConstraintCapabilities(constraint.supports_discrete, False, reason)


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
    rules = [
        adapter.gradient_rule
        for adapter in _COMPILER_ADAPTERS
        if adapter.gradient_rule is not None and adapter.matches_gradient_function(spec.function)
    ]
    return GradientSupport(rules=rules) if rules else None


@dataclass(frozen=True)
class _DirectScoringUnit:
    """One ordinary public constraint in a run-scoped scoring plan."""

    constraint: Constraint

    def evaluate(self, *, mask: list[bool], verbose: bool) -> list[float]:
        """Evaluate the direct constraint and normalize numeric outputs."""
        return [float(score) for score in self.constraint.evaluate(mask=mask, verbose=verbose)]


@dataclass
class _CompiledScoringUnit:
    """One compatible backend group with an isolated execution config."""

    adapter: CompilerAdapter
    constraints: list[CompiledConstraint]
    config: BaseModel

    def evaluate(self, *, mask: list[bool], verbose: bool) -> list[float]:
        """Evaluate the group through its compiler adapter."""
        del verbose
        if self.adapter.evaluate_scoring is None:
            raise ValueError(f"Compiler backend {self.adapter.backend_id!r} has no scoring evaluator.")
        return self.adapter.evaluate_scoring(self.constraints, mask, self.config)


@dataclass(frozen=True)
class ScoringPlan:
    """Run-scoped ordered scoring units compiled after optimizer seed reset."""

    units: tuple[_DirectScoringUnit | _CompiledScoringUnit, ...]

    def evaluate(self, *, mask: list[bool], verbose: bool = False) -> list[list[float]]:
        """Evaluate every direct or compiled unit in stable plan order."""
        return [unit.evaluate(mask=mask, verbose=verbose) for unit in self.units]


def compile_scoring_plan(constraints: list[Constraint], *, seed: int | None = None) -> ScoringPlan:
    """Compile additive scoring into a deterministic run-scoped execution plan."""
    units: list[_DirectScoringUnit | _CompiledScoringUnit] = []
    group_by_key: dict[tuple[str, tuple[Any, ...]], _CompiledScoringUnit] = {}
    group_order: list[tuple[str, tuple[Any, ...]]] = []
    next_group_seed = _group_seed_factory(seed, namespace=0x53434F52)

    for constraint in constraints:
        for adapter in _COMPILER_ADAPTERS:
            if adapter.match_scoring is None:
                continue
            match = adapter.match_scoring(constraint)
            if match is None:
                continue
            group_key = (adapter.backend_id, match.group_key)
            unit = group_by_key.get(group_key)
            if unit is None:
                unit = _CompiledScoringUnit(
                    adapter=adapter,
                    constraints=[],
                    config=clone_execution_config(match.config, seed=next_group_seed()),
                )
                group_by_key[group_key] = unit
                group_order.append(group_key)
            unit.constraints.append(match.compiled)
            break
        else:
            _flush_scoring_groups(group_order, group_by_key, units)
            units.append(_DirectScoringUnit(constraint))

    _flush_scoring_groups(group_order, group_by_key, units)
    return ScoringPlan(tuple(units))


def _group_seed_factory(parent_seed: int | None, *, namespace: int) -> Callable[[], int | None]:
    """Return stable first-occurrence seeds for one compiler plan namespace."""
    group_index = 0

    def next_seed() -> int | None:
        nonlocal group_index
        current_index = group_index
        group_index += 1
        if parent_seed is None:
            return None
        sequence = np.random.SeedSequence([parent_seed, namespace, current_index])
        return int(sequence.generate_state(1)[0])

    return next_seed


def _flush_scoring_groups(
    group_order: list[tuple[str, tuple[Any, ...]]],
    group_by_key: dict[tuple[str, tuple[Any, ...]], _CompiledScoringUnit],
    units: list[_DirectScoringUnit | _CompiledScoringUnit],
) -> None:
    """Append queued compiled groups to the plan and clear the queues."""
    units.extend(group_by_key[group_key] for group_key in group_order)
    group_order.clear()
    group_by_key.clear()
