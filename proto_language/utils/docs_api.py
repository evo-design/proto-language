"""Programmatic access to docs for registered constraints, generators, and optimizers.

Each component carries its docs in source: a Google-style docstring on the
registered function or class, and per-field ``title``/``description`` on the
Pydantic config class (enforced by ``ConfigField`` and
``tests/test_docstring_consistency.py``). This module exposes that material as
typed Pydantic models so agents, CLIs, and notebooks can consume it uniformly.

Two surfaces:

1. **Component docs** — ``get_constraint_doc`` / ``get_generator_doc`` /
   ``get_optimizer_doc`` return a ``ComponentDoc`` bundling the spec metadata,
   the function/class docstring, and the config model docs.
2. **Core-type docs** — ``get_core_type_doc`` returns a ``CoreTypeDoc`` with the
   class docstring and ``__init__`` parameter signature for ``Sequence``,
   ``Segment``, ``Construct``, and ``Program`` (none of which are Pydantic
   models, so no JSON schema is emitted).
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from proto_language.constraint.constraint_registry import ConstraintRegistry, ConstraintSpec
from proto_language.core.construct import Construct
from proto_language.core.program import Program
from proto_language.core.segment import Segment
from proto_language.core.sequence import Sequence
from proto_language.generator.generator_registry import GeneratorRegistry, GeneratorSpec
from proto_language.optimizer.optimizer_registry import OptimizerRegistry, OptimizerSpec
from proto_language.utils.base import BaseRegistry, BaseSpec

logger = logging.getLogger(__name__)


ComponentKind = Literal["constraint", "generator", "optimizer"]
CoreTypeName = Literal["Sequence", "Segment", "Construct", "Program"]

_CORE_TYPES: dict[str, type] = {
    "Sequence": Sequence,
    "Segment": Segment,
    "Construct": Construct,
    "Program": Program,
}

_KIND_TO_REGISTRY: dict[ComponentKind, type[BaseRegistry[Any]]] = {
    "constraint": ConstraintRegistry,
    "generator": GeneratorRegistry,
    "optimizer": OptimizerRegistry,
}


# =============================================================================
# Public data models
# =============================================================================


class FieldDoc(BaseModel):
    """One field of a Pydantic config model."""

    name: str = Field(description="Attribute name.")
    type_str: str = Field(description="Stringified type annotation.")
    default: Any | None = Field(
        default=None,
        description="Default value, or None when the field is required.",
    )
    title: str | None = Field(default=None, description="Per-field title from ``Field(title=...)``.")
    description: str | None = Field(default=None, description="Per-field description from ``Field(description=...)``.")
    required: bool = Field(description="True when no default is set.")


class ConfigModelDoc(BaseModel):
    """Normalized view of a config-class docstring and its fields."""

    name: str = Field(description="Config class name.")
    docstring: str = Field(description="Cleaned class docstring.")
    fields: list[FieldDoc] = Field(default_factory=list, description="Per-field docs in declaration order.")


class ConstraintSpecMetadata(BaseModel):
    """Constraint-specific spec fields surfaced on ``ComponentDoc.spec_metadata``."""

    category: str | None = Field(default=None, description="Free-form grouping (e.g. ``protein_structure``).")
    mode: Literal["discrete", "gradient", "dual"] = Field(description="Constraint mode.")
    tools_called: list[str] = Field(default_factory=list, description="proto-tools keys this constraint calls.")
    supported_sequence_types: list[str] = Field(default_factory=list, description="Accepted sequence types.")
    requires_generators: list[str] | None = Field(
        default=None,
        description="Generators required in the same stage; None if no requirement.",
    )
    input_labels: list[str] | None = Field(
        default=None,
        description="Per-slot labels; None when the constraint accepts any number of inputs.",
    )


class GeneratorSpecMetadata(BaseModel):
    """Generator-specific spec fields surfaced on ``ComponentDoc.spec_metadata``."""

    category: Literal["autoregressive", "mutation", "inverse_folding", "gradient"] = Field(
        description="Generator category bucket derived from ``input_type``."
    )
    input_type: Literal["prompt", "starting_sequence", "structure", "logits"] = Field(
        description="Kind of starting input the generator consumes."
    )
    allows_empty_starting_sequence: bool = Field(
        default=False,
        description="Whether the generator can initialize a length-only segment without a starting sequence.",
    )
    tools_called: list[str] = Field(default_factory=list, description="proto-tools keys this generator calls.")
    supported_sequence_types: list[str] = Field(default_factory=list, description="Producible sequence types.")


class OptimizerSpecMetadata(BaseModel):
    """Optimizer-specific spec fields surfaced on ``ComponentDoc.spec_metadata``."""

    targets_single_segment: bool = Field(
        default=False,
        description="Whether the optimizer operates on one segment per run instead of all design segments.",
    )
    compatible_generators: list[str] | None = Field(
        default=None,
        description="Generator keys this optimizer accepts; None means any unclaimed generator.",
    )
    required_constraint_mode: Literal["discrete", "gradient"] | None = Field(
        default=None,
        description="Constraint mode every paired constraint must support; None accepts either.",
    )


SpecMetadata = ConstraintSpecMetadata | GeneratorSpecMetadata | OptimizerSpecMetadata


class ComponentDoc(BaseModel):
    """Per-component docs: spec metadata + function/class docstring + config docs."""

    kind: ComponentKind = Field(description="Which registry the component belongs to.")
    key: str = Field(description="Registry key (e.g. ``gc-content``, ``esm2``, ``mcmc``).")
    label: str = Field(description="Human-readable display label.")
    description: str = Field(description="Short description from the registration decorator.")
    uses_gpu: bool = Field(description="Whether the component requires a GPU at runtime.")
    docstring: str = Field(
        description="Docstring of the registered function (constraint) or class (generator/optimizer)."
    )
    config: ConfigModelDoc = Field(description="Docs for the Pydantic config model.")
    spec_metadata: SpecMetadata = Field(
        description=(
            "Component-specific spec fields not covered by the common header. Concrete type "
            "matches ``kind``: ``ConstraintSpecMetadata`` / ``GeneratorSpecMetadata`` / "
            "``OptimizerSpecMetadata``."
        ),
    )


class ParamDoc(BaseModel):
    """One ``__init__`` parameter of a core type."""

    name: str = Field(description="Parameter name.")
    type_str: str = Field(description="Stringified annotation, or ``Any`` if missing.")
    default: Any | None = Field(default=None, description="Default value, or None when required.")
    required: bool = Field(description="True when the parameter has no default.")


class CoreTypeDoc(BaseModel):
    """Docs for a non-registered core type (``Sequence``, ``Segment``, ``Construct``, ``Program``)."""

    name: CoreTypeName = Field(description="Class name.")
    docstring: str = Field(description="Cleaned class docstring.")
    init_docstring: str = Field(default="", description="Cleaned ``__init__`` docstring.")
    params: list[ParamDoc] = Field(default_factory=list, description="``__init__`` parameters (``self`` omitted).")


# =============================================================================
# Identifier resolution
# =============================================================================


def _normalize_kebab(identifier: str) -> str:
    """Lowercase + kebab-case a free-form identifier."""
    return identifier.strip().lower().replace("_", "-")


def resolve_key(kind: ComponentKind, identifier: str) -> str:
    """Resolve a flexible identifier to its exact registry key.

    Accepts the registry key directly (``gc-content``), the snake-cased form
    (``gc_content``), or the class/function ``__name__`` for the underlying
    callable. Raises ``ValueError`` with the list of registered keys on miss.
    """
    registry = _KIND_TO_REGISTRY[kind]
    keys = {spec.key for spec in registry.list_all()}
    if identifier in keys:
        return identifier

    kebab = _normalize_kebab(identifier)
    if kebab in keys:
        return kebab

    for spec in registry.list_all():
        callable_name = _spec_callable_name(spec)
        if callable_name == identifier or callable_name == kebab.replace("-", "_"):
            return str(spec.key)

    available = ", ".join(sorted(keys))
    raise ValueError(f"Unknown {kind}: '{identifier}'. Available: {available}")


def _spec_callable_name(spec: BaseSpec) -> str:
    """Return the registered function/class ``__name__`` for a spec, or empty string."""
    if isinstance(spec, ConstraintSpec):
        fn = spec.function or spec.backward
        return fn.__name__ if fn is not None else ""
    if isinstance(spec, GeneratorSpec):
        return spec.generator_class.__name__
    if isinstance(spec, OptimizerSpec):
        return spec.optimizer_class.__name__
    return ""


# =============================================================================
# Config-model extraction
# =============================================================================


def _stringify_type(annotation: Any) -> str:
    """Best-effort stringification of a type annotation."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _field_default(field_info: FieldInfo) -> tuple[Any, bool]:
    """Return ``(default_value_or_none, required)`` for a Pydantic ``FieldInfo``."""
    if field_info.is_required():
        return None, True
    default = field_info.get_default(call_default_factory=False)
    if default is None and field_info.default_factory is not None:
        try:
            default = field_info.default_factory()  # type: ignore[call-arg]
        except Exception:
            default = None
    return default, False


def get_config_doc(config_model: type[BaseModel]) -> ConfigModelDoc:
    """Build a ``ConfigModelDoc`` from a Pydantic config model class."""
    fields: list[FieldDoc] = []
    for name, info in config_model.model_fields.items():
        default, required = _field_default(info)
        fields.append(
            FieldDoc(
                name=name,
                type_str=_stringify_type(info.annotation),
                default=default,
                title=info.title,
                description=info.description,
                required=required,
            )
        )
    return ConfigModelDoc(
        name=config_model.__name__,
        docstring=inspect.cleandoc(config_model.__doc__ or ""),
        fields=fields,
    )


# =============================================================================
# Per-kind extractors
# =============================================================================


def _common_doc_kwargs(spec: BaseSpec, kind: ComponentKind) -> dict[str, Any]:
    """Shared fields for every ``ComponentDoc``."""
    return {
        "kind": kind,
        "key": spec.key,
        "label": spec.label,
        "description": spec.description,
        "uses_gpu": spec.uses_gpu,
        "config": get_config_doc(spec.config_model),
    }


def get_constraint_doc(identifier: str) -> ComponentDoc:
    """Build a ``ComponentDoc`` for a registered constraint."""
    key = resolve_key("constraint", identifier)
    spec = ConstraintRegistry.get(key)
    fn = spec.function or spec.backward
    docstring = inspect.cleandoc(fn.__doc__ or "") if fn is not None else ""
    return ComponentDoc(
        **_common_doc_kwargs(spec, "constraint"),
        docstring=docstring,
        spec_metadata=ConstraintSpecMetadata(
            category=spec.category,
            mode=spec.mode,
            tools_called=list(spec.tools_called),
            supported_sequence_types=list(spec.supported_sequence_types),
            requires_generators=list(spec.requires_generators) if spec.requires_generators else None,
            input_labels=[str(lbl) for lbl in spec.input_labels] if spec.input_labels else None,
        ),
    )


def get_generator_doc(identifier: str) -> ComponentDoc:
    """Build a ``ComponentDoc`` for a registered generator."""
    key = resolve_key("generator", identifier)
    spec = GeneratorRegistry.get(key)
    cls = spec.generator_class
    return ComponentDoc(
        **_common_doc_kwargs(spec, "generator"),
        docstring=inspect.cleandoc(cls.__doc__ or ""),
        spec_metadata=GeneratorSpecMetadata(
            category=spec.category,
            input_type=spec.input_type.value,
            allows_empty_starting_sequence=spec.allows_empty_starting_sequence,
            tools_called=list(spec.tools_called),
            supported_sequence_types=list(spec.supported_sequence_types),
        ),
    )


def get_optimizer_doc(identifier: str) -> ComponentDoc:
    """Build a ``ComponentDoc`` for a registered optimizer."""
    key = resolve_key("optimizer", identifier)
    spec = OptimizerRegistry.get(key)
    cls = spec.optimizer_class
    return ComponentDoc(
        **_common_doc_kwargs(spec, "optimizer"),
        docstring=inspect.cleandoc(cls.__doc__ or ""),
        spec_metadata=OptimizerSpecMetadata(
            targets_single_segment=spec.targets_single_segment,
            compatible_generators=list(spec.compatible_generators) if spec.compatible_generators else None,
            required_constraint_mode=spec.required_constraint_mode,
        ),
    )


# =============================================================================
# Core-type extractor
# =============================================================================


def get_core_type_doc(name: str) -> CoreTypeDoc:
    """Return docs for a core type (``Sequence`` / ``Segment`` / ``Construct`` / ``Program``)."""
    if name not in _CORE_TYPES:
        available = ", ".join(_CORE_TYPES)
        raise ValueError(f"Unknown core type: '{name}'. Available: {available}")
    cls = _CORE_TYPES[name]
    init = getattr(cls, "__init__", None)
    params: list[ParamDoc] = []
    try:
        sig: inspect.Signature | None = inspect.signature(cls)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        for p_name, p in sig.parameters.items():
            if p_name == "self":
                continue
            required = p.default is inspect.Parameter.empty
            params.append(
                ParamDoc(
                    name=p_name,
                    type_str=_stringify_type(p.annotation),
                    default=None if required else p.default,
                    required=required,
                )
            )
    return CoreTypeDoc(
        name=name,  # type: ignore[arg-type]
        docstring=inspect.cleandoc(cls.__doc__ or ""),
        init_docstring=inspect.cleandoc((init.__doc__ or "") if init is not None else ""),
        params=params,
    )


# =============================================================================
# Listing helpers
# =============================================================================


def list_specs(kind: ComponentKind) -> list[BaseSpec]:
    """Return all specs for ``kind``, sorted by registry key."""
    registry = _KIND_TO_REGISTRY[kind]
    return sorted(registry.list_all(), key=lambda s: s.key)


def list_categories(kind: ComponentKind) -> list[str]:
    """Return the sorted set of categories present in the registry for ``kind``.

    Optimizers do not carry a category field; returns an empty list for that kind.
    """
    cats: set[str] = set()
    for spec in list_specs(kind):
        cat = getattr(spec, "category", None)
        if cat:
            cats.add(str(cat))
    return sorted(cats)


# =============================================================================
# Compatibility graph
# =============================================================================


class CompatibilityReport(BaseModel):
    """Registry keys compatible with a target component under the spec rules."""

    kind: ComponentKind = Field(description="Target component's kind.")
    key: str = Field(description="Target component's registry key.")
    compatible_constraints: list[str] = Field(
        default_factory=list,
        description="Compatible constraint keys (populated only when the target is an optimizer).",
    )
    compatible_generators: list[str] = Field(
        default_factory=list,
        description="Compatible generator keys (populated only when the target is an optimizer).",
    )
    compatible_optimizers: list[str] = Field(
        default_factory=list,
        description="Compatible optimizer keys (populated only when the target is a constraint or generator).",
    )


def _compatible_for_optimizer(key: str) -> CompatibilityReport:
    """For an optimizer, list constraints (mode-matching) and generators it accepts."""
    spec = OptimizerRegistry.get(key)
    constraints: list[str] = []
    for c in list_specs("constraint"):
        c_constraint: ConstraintSpec = c  # type: ignore[assignment]
        if (
            spec.required_constraint_mode is None
            or (spec.required_constraint_mode == "discrete" and c_constraint.mode in {"discrete", "dual"})
            or (spec.required_constraint_mode == "gradient" and c_constraint.mode in {"gradient", "dual"})
        ):
            constraints.append(c_constraint.key)

    if spec.compatible_generators is None:
        generators = [g.key for g in list_specs("generator")]
    else:
        generators = list(spec.compatible_generators)

    return CompatibilityReport(
        kind="optimizer",
        key=key,
        compatible_constraints=sorted(constraints),
        compatible_generators=sorted(generators),
    )


def _compatible_for_constraint(key: str) -> CompatibilityReport:
    """For a constraint, list optimizers whose required_constraint_mode the constraint satisfies."""
    spec = ConstraintRegistry.get(key)
    optimizers: list[str] = []
    for o in list_specs("optimizer"):
        o_spec: OptimizerSpec = o  # type: ignore[assignment]
        if (
            o_spec.required_constraint_mode is None
            or (o_spec.required_constraint_mode == "discrete" and spec.mode in {"discrete", "dual"})
            or (o_spec.required_constraint_mode == "gradient" and spec.mode in {"gradient", "dual"})
        ):
            optimizers.append(o_spec.key)
    return CompatibilityReport(kind="constraint", key=key, compatible_optimizers=sorted(optimizers))


def _compatible_for_generator(key: str) -> CompatibilityReport:
    """For a generator, list optimizers whose compatible_generators allow it."""
    optimizers: list[str] = []
    for o in list_specs("optimizer"):
        o_spec: OptimizerSpec = o  # type: ignore[assignment]
        if o_spec.compatible_generators is None or key in o_spec.compatible_generators:
            optimizers.append(o_spec.key)
    return CompatibilityReport(kind="generator", key=key, compatible_optimizers=sorted(optimizers))


def get_compatibility(kind: ComponentKind, identifier: str) -> CompatibilityReport:
    """Return the components pairable with the given target."""
    key = resolve_key(kind, identifier)
    if kind == "constraint":
        return _compatible_for_constraint(key)
    if kind == "generator":
        return _compatible_for_generator(key)
    return _compatible_for_optimizer(key)
