"""Registry-level access to compiler-backed gradient support metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

from proto_language.language.constraint.constraint_registry import ConstraintSpec
from proto_language.language.core.sequence import SequenceType

GradientSupportValue = str | int | float | bool | None


class GradientSupportCondition(TypedDict):
    """A config predicate that must hold for a support rule to apply."""

    config_path: str
    equals: GradientSupportValue


class GradientSupportAnyInputTarget(TypedDict):
    """A rule target that may be any optimizer input included by the constraint."""

    kind: Literal["any_input"]


class GradientSupportInputIndexTarget(TypedDict):
    """A rule target selected by an integer value in constraint config."""

    kind: Literal["input_index_from_config"]
    config_path: str


GradientSupportTargetSegment = GradientSupportAnyInputTarget | GradientSupportInputIndexTarget


class GradientSupportAllInputsRequirement(TypedDict):
    """A segment requirement that applies to every constraint input."""

    kind: Literal["all_inputs"]
    sequence_types: list[SequenceType]


class GradientSupportInputIndexRequirement(TypedDict):
    """A segment requirement selected by one configured input index."""

    kind: Literal["input_index_from_config"]
    config_path: str
    sequence_types: list[SequenceType]


class GradientSupportInputIndicesRequirement(TypedDict):
    """A segment requirement selected by a configured list of input indices."""

    kind: Literal["input_indices_from_config"]
    config_path: str
    sequence_types: list[SequenceType]


GradientSupportRequiredSegment = (
    GradientSupportAllInputsRequirement | GradientSupportInputIndexRequirement | GradientSupportInputIndicesRequirement
)


class GradientSupportRule(TypedDict):
    """One compiler-backed gradient support case for a public constraint."""

    source: Literal["compiled"]
    label: str
    when: list[GradientSupportCondition]
    requires_scoring: bool
    target_segment: GradientSupportTargetSegment
    required_segments: list[GradientSupportRequiredSegment]


class GradientSupport(TypedDict):
    """Compiler-backed gradient support metadata for a public constraint."""

    rules: list[GradientSupportRule]


def config_equals_condition(config_path: str, equals: GradientSupportValue) -> GradientSupportCondition:
    """Return a condition matching a config value by equality."""
    return {"config_path": config_path, "equals": equals}


def any_input_target() -> GradientSupportAnyInputTarget:
    """Return a target descriptor for any constraint input."""
    return {"kind": "any_input"}


def input_index_target_from_config(config_path: str) -> GradientSupportInputIndexTarget:
    """Return a target descriptor selected by a configured input index."""
    return {"kind": "input_index_from_config", "config_path": config_path}


def all_inputs_requirement(sequence_types: Sequence[SequenceType]) -> GradientSupportAllInputsRequirement:
    """Return a requirement applied to every constraint input."""
    return {"kind": "all_inputs", "sequence_types": list(sequence_types)}


def input_index_requirement_from_config(
    config_path: str,
    sequence_types: Sequence[SequenceType],
) -> GradientSupportInputIndexRequirement:
    """Return a requirement for one configured input index."""
    return {
        "kind": "input_index_from_config",
        "config_path": config_path,
        "sequence_types": list(sequence_types),
    }


def input_indices_requirement_from_config(
    config_path: str,
    sequence_types: Sequence[SequenceType],
) -> GradientSupportInputIndicesRequirement:
    """Return a requirement for a configured list of input indices."""
    return {
        "kind": "input_indices_from_config",
        "config_path": config_path,
        "sequence_types": list(sequence_types),
    }


def compiled_gradient_support_rule(
    *,
    label: str,
    when: list[GradientSupportCondition],
    target_segment: GradientSupportTargetSegment,
    required_segments: list[GradientSupportRequiredSegment],
    requires_scoring: bool = True,
) -> GradientSupportRule:
    """Return a rule implemented by a compiler provider."""
    return {
        "source": "compiled",
        "label": label,
        "when": when,
        "requires_scoring": requires_scoring,
        "target_segment": target_segment,
        "required_segments": required_segments,
    }


def gradient_support_for_constraint_spec(spec: ConstraintSpec) -> GradientSupport | None:
    """Return declarative gradient support metadata for compiler-backed constraints.

    ConstraintSpec.mode describes public callable shape: discrete scorer,
    backward gradient, or both. Some forward-only constraints are still
    differentiable through compiler providers when their config matches a
    backend-supported case. This metadata lets clients discover those
    config-dependent cases from compiler-owned rules.
    """
    if spec.function is None:
        return None

    from proto_language.language.optimizer.constraint_compiler import alphafold2_multimer_provider as af2m
    from proto_language.language.optimizer.constraint_compiler import esmfold_provider as esmfold

    candidate_rules = (
        esmfold.gradient_support_rule_for_constraint_spec(spec),
        af2m.gradient_support_rule_for_constraint_spec(spec),
    )
    rules: list[GradientSupportRule] = [rule for rule in candidate_rules if rule is not None]

    return {"rules": rules} if rules else None
