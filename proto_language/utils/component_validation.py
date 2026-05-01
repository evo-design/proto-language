"""Developer helpers for validating and smoke-testing language components."""

from __future__ import annotations

import ast
import copy
import math
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from proto_language.language.core import Sequence

ComponentType = Literal["constraint", "generator", "optimizer"]
# Mirrors proto_language.language.core.sequence.SequenceType; kept local to avoid registry import cycles.
SequenceType = Literal["dna", "rna", "protein", "ligand"]
_COMPONENT_TYPES: tuple[ComponentType, ...] = typing.get_args(ComponentType)
_KEY_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_VALIDATION_RULES: dict[ComponentType, dict[str, Any]] = {
    "constraint": {
        "decorator_name": "constraint",
        "node_type": ast.FunctionDef,
        "config_base": "BaseConfig",
        "required_imports": {
            "BaseConfig": "base_config",
            "ConstraintOutput": "language.core",
            "Sequence": "language.core",
            "constraint": "constraint_registry",
        },
        "required_decorator_args": [
            "key",
            "label",
            "config",
            "description",
            "supported_sequence_types",
        ],
        "recommended_decorator_args": [
            "tools_called",
            "category",
        ],
        "base_class": None,
    },
    "generator": {
        "decorator_name": "generator",
        "node_type": ast.ClassDef,
        "config_base": "BaseConfig",
        "required_imports": {
            "BaseConfig": "base_config",
            "Generator": "language.core",
            "generator": "generator_registry",
        },
        "required_decorator_args": [
            "key",
            "label",
            "config",
            "description",
            "category",
            "supported_sequence_types",
        ],
        "recommended_decorator_args": [
            "tools_called",
        ],
        "base_class": "Generator",
    },
    "optimizer": {
        "decorator_name": "optimizer",
        "node_type": ast.ClassDef,
        "config_base": "BaseOptimizerConfig",
        "required_imports": {
            "BaseOptimizerConfig": "base_config",
            "Optimizer": "language.core",
            "optimizer": "optimizer_registry",
        },
        "required_decorator_args": ["key", "label", "config", "description"],
        "recommended_decorator_args": [],
        "base_class": "Optimizer",
    },
}

_PUBLIC_IMPORT_MODULES: dict[str, frozenset[str]] = {
    "BaseConfig": frozenset({"proto_language"}),
    "BaseOptimizerConfig": frozenset({"proto_language.language.optimizer"}),
    "ConstraintOutput": frozenset({"proto_language", "proto_language.language"}),
    "Generator": frozenset({"proto_language", "proto_language.language"}),
    "Optimizer": frozenset({"proto_language", "proto_language.language"}),
    "Sequence": frozenset({"proto_language", "proto_language.language"}),
    "constraint": frozenset({"proto_language", "proto_language.language"}),
    "generator": frozenset({"proto_language", "proto_language.language"}),
    "optimizer": frozenset({"proto_language", "proto_language.language"}),
}


@dataclass
class ValidationResult:
    """Outcome of ``validate_component_file``.

    Attributes:
        success (bool): True when the file parses with no AST-level errors.
        component_type (ComponentType | None): Detected from the decorator.
        registry_key (str | None): Literal ``key=`` value from the decorator.
        errors (list[str]): Hard validation failures.
        warnings (list[str]): Non-blocking style / completeness issues.
        structure (dict[str, Any]): Raw structural facts.
    """

    success: bool
    component_type: ComponentType | None = None
    registry_key: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structure: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Outcome of running a registered component on known inputs.

    Attributes:
        passed (bool): True when actual matches expected within tolerance.
        actual (Any): The component's actual output.
        expected (Any): The expected output passed by the caller.
        diffs (list[str]): Per-element comparison failures.
        message (str): Human-readable summary.
    """

    __test__ = False  # not a pytest test class

    passed: bool
    actual: Any
    expected: Any
    diffs: list[str] = field(default_factory=list)
    message: str = ""


def validate_component_file(path: str | Path) -> ValidationResult:
    """Validate a component file's structure before import.

    Runs AST-level checks: syntax, required imports, decorator presence,
    config base class, decorator args, component shape. Does not import
    the file. This is an advisory authoring check, not a runtime loader
    or executable payload format.

    Args:
        path (str | Path): Filesystem path to the component .py file.

    Returns:
        ValidationResult: Inspect ``result.errors`` to decide whether to import.
    """
    file_path = Path(path)
    try:
        source = file_path.read_text()
    except FileNotFoundError:
        return ValidationResult(success=False, errors=[f"File not found: {file_path}"])
    except OSError as exc:
        return ValidationResult(success=False, errors=[f"Cannot read file: {exc!s}"])

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ValidationResult(success=False, errors=[f"Line {exc.lineno}: {exc.msg}"])

    facts = _analyze_tree(tree)
    component_type = facts["component_type"]
    if component_type is None:
        return ValidationResult(
            success=False,
            errors=["No @constraint, @generator, or @optimizer decorator found."],
        )
    if len(facts["detected_types"]) > 1:
        return ValidationResult(
            success=False,
            errors=[
                "Component files should define exactly one component type; "
                f"found {', '.join(facts['detected_types'])} decorators."
            ],
            structure={"component_types": facts["detected_types"]},
        )

    rules = _VALIDATION_RULES[component_type]
    structure = {
        "component_type": component_type,
        "has_config_class": False,
        "has_decorated_component": False,
        "has_registry_decorator": False,
        "config_class_name": None,
        "component_name": None,
    }
    errors: list[str] = []
    warnings: list[str] = []

    _check_imports(facts["imports"], rules, errors)
    _check_config_class(facts["classes"], component_type, rules, structure, errors)
    _check_component_definition(facts, component_type, rules, structure, errors, warnings)

    return ValidationResult(
        success=not errors,
        component_type=component_type,
        registry_key=facts["registry_key"],
        errors=errors,
        warnings=warnings,
        structure=structure,
    )


def test_constraint(
    key: str,
    sequences: list[tuple[Sequence, ...]] | list[Sequence] | list[str],
    config: dict[str, Any] | None = None,
    expected_scores: list[float] | None = None,
    tolerance: float = 0.01,
    sequence_type: SequenceType = "protein",
) -> TestResult:
    """Run a registered constraint and compare scores to ``expected_scores``.

    ``sequences`` accepts ``list[tuple[Sequence, ...]]``, ``list[Sequence]``,
    or ``list[str]`` (auto-wrapped as ``Sequence(sequence_type=...)``).

    Args:
        key (str): Constraint registry key.
        sequences (list[tuple[Sequence, ...]] | list[Sequence] | list[str]): Inputs to score.
        config (dict[str, Any] | None): Config-field overrides.
        expected_scores (list[float] | None): Per-input expected score, or None to skip comparison.
        tolerance (float): Absolute tolerance for score comparison.
        sequence_type (SequenceType): Used only when wrapping plain strings.

    Returns:
        TestResult: ``passed=True`` when scores match within tolerance.

    Raises:
        KeyError: When ``key`` is not registered.
        ValueError: When the constraint is gradient-only (no scoring function).
    """
    from proto_language.language.constraint import ConstraintRegistry

    spec = ConstraintRegistry.get(key)
    if spec.function is None:
        raise ValueError(f"Constraint '{key}' is gradient-only; test_constraint requires a forward score.")
    typed_inputs = _coerce_constraint_inputs(sequences, sequence_type)
    cfg = spec.config_model(**(config or {}))
    actual = [out.score for out in spec.function(typed_inputs, cfg)]

    if expected_scores is None:
        return TestResult(
            passed=True,
            actual=actual,
            expected=None,
            message=f"{key} ran on {len(typed_inputs)} input(s); no expected_scores given.",
        )

    if len(expected_scores) != len(actual):
        return TestResult(
            passed=False,
            actual=actual,
            expected=expected_scores,
            diffs=[f"length mismatch: actual={len(actual)} expected={len(expected_scores)}"],
            message=f"{key}: input/expected length mismatch",
        )

    diffs = [
        f"[{i}] actual={a:.6g} expected={e:.6g} diff={abs(a - e):.6g} > tol={tolerance}"
        for i, (a, e) in enumerate(zip(actual, expected_scores, strict=True))
        if not _close(a, e, tolerance)
    ]
    return TestResult(
        passed=not diffs,
        actual=actual,
        expected=expected_scores,
        diffs=diffs,
        message=f"{key} matched all {len(actual)} expected score(s) within {tolerance}."
        if not diffs
        else f"{key}: {len(diffs)} of {len(actual)} score(s) outside tolerance {tolerance}.",
    )


def test_generator(
    key: str,
    segment_length: int,
    config: dict[str, Any] | None = None,
    n_samples: int = 5,
    sequence_type: SequenceType = "protein",
    expected_alphabet: str | None = None,
) -> TestResult:
    """Smoke-test a registered generator: instantiate, sample, check shape.

    Builds an empty ``Segment`` of ``segment_length`` with ``n_samples`` proposal
    slots, runs ``assign(segment)`` then ``sample()``, and verifies every slot
    came back populated to the requested length and within ``expected_alphabet``.

    Args:
        key (str): Generator registry key.
        segment_length (int): Length of the target segment.
        config (dict[str, Any] | None): Config overrides.
        n_samples (int): Number of proposal slots to allocate.
        sequence_type (SequenceType): Must match a value in the generator's ``supported_sequence_types``.
        expected_alphabet (str | None): When provided, every char in every proposal must be in this string.

    Returns:
        TestResult: ``passed=True`` when proposals match the requested shape.

    Raises:
        KeyError: When ``key`` is not registered.
    """
    from proto_language.language.core import Segment
    from proto_language.language.generator import GeneratorRegistry

    generator = GeneratorRegistry.create(key, config or {})
    segment = Segment(length=segment_length, sequence_type=sequence_type)
    segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(n_samples)]

    generator.assign(segment)
    generator.sample()
    actual = [seq.sequence for seq in segment.proposal_sequences]

    diffs: list[str] = []
    if len(actual) != n_samples:
        diffs.append(f"expected {n_samples} proposals, got {len(actual)}")
    for i, seq in enumerate(actual):
        if len(seq) != segment_length:
            diffs.append(f"[{i}] length={len(seq)} expected={segment_length}")
        if expected_alphabet is not None:
            bad = sorted(set(seq) - set(expected_alphabet))
            if bad:
                diffs.append(f"[{i}] chars outside alphabet: {''.join(bad)}")

    return TestResult(
        passed=not diffs,
        actual=actual,
        expected={"n_samples": n_samples, "segment_length": segment_length, "alphabet": expected_alphabet},
        diffs=diffs,
        message=f"{key} sampled {len(actual)} proposal(s) cleanly."
        if not diffs
        else f"{key}: {len(diffs)} sampling issue(s).",
    )


def test_optimizer(
    key: str,
    config: dict[str, Any] | None = None,
) -> TestResult:
    """Smoke-test a registered optimizer: validate that its config schema accepts the overrides.

    End-to-end optimizer behavior requires a full ``Program`` with generator,
    constraints, and segments. This helper only confirms the optimizer is
    registered and its config validates.

    Args:
        key (str): Optimizer registry key.
        config (dict[str, Any] | None): Config overrides.

    Returns:
        TestResult: ``passed=True`` when the config is accepted.

    Raises:
        KeyError: When ``key`` is not registered.
    """
    from pydantic import ValidationError

    from proto_language.language.optimizer import OptimizerRegistry

    spec = OptimizerRegistry.get(key)
    try:
        cfg = spec.config_model(**(config or {}))
    except ValidationError as exc:
        return TestResult(
            passed=False,
            actual=None,
            expected=config,
            diffs=[str(exc)],
            message=f"{key}: config validation failed",
        )

    return TestResult(
        passed=True,
        actual=cfg.model_dump(),
        expected=config,
        message=f"{key} config accepted; full Program execution is needed for end-to-end behavior.",
    )


test_constraint.__test__ = False  # type: ignore[attr-defined]
test_generator.__test__ = False  # type: ignore[attr-defined]
test_optimizer.__test__ = False  # type: ignore[attr-defined]


def _close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol


def _coerce_constraint_inputs(
    sequences: list[tuple[Sequence, ...]] | list[Sequence] | list[str],
    sequence_type: SequenceType,
) -> list[tuple[Sequence, ...]]:
    from proto_language.language.core import Sequence

    if not sequences:
        return []
    first = sequences[0]
    if isinstance(first, tuple):
        return list(sequences)  # type: ignore[arg-type]
    if isinstance(first, Sequence):
        return [(seq,) for seq in sequences]  # type: ignore[misc]
    if isinstance(first, str):
        return [(Sequence(sequence=str(s), sequence_type=sequence_type),) for s in sequences]
    raise TypeError(
        f"sequences must be list[tuple[Sequence, ...]], list[Sequence], or list[str]; got {type(first).__name__}"
    )


def _analyze_tree(tree: ast.Module) -> dict[str, Any]:
    """Single-pass AST walk gathering all facts the validator needs."""
    imports: dict[str, str | None] = {}
    classes: list[ast.ClassDef] = []
    decorated: dict[ComponentType, list[ast.FunctionDef | ast.ClassDef]] = {ct: [] for ct in _COMPONENT_TYPES}
    decorator_name_to_type: dict[str, ComponentType] = {
        rules["decorator_name"]: ct for ct, rules in _VALIDATION_RULES.items()
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.name] = node.module
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)):
                    continue
                ct = decorator_name_to_type.get(decorator.func.id)
                if ct is not None:
                    decorated[ct].append(node)

    detected_types = [ct for ct in _COMPONENT_TYPES if decorated[ct]]
    component_type: ComponentType | None = detected_types[0] if detected_types else None

    registry_key: str | None = None
    if component_type is not None:
        decorator_name = _VALIDATION_RULES[component_type]["decorator_name"]
        for node in decorated[component_type]:
            for decorator in node.decorator_list:
                if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)):
                    continue
                if decorator.func.id != decorator_name:
                    continue
                for keyword in decorator.keywords:
                    if (
                        keyword.arg == "key"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        registry_key = keyword.value.value
                        break
                if registry_key is not None:
                    break
            if registry_key is not None:
                break

    return {
        "imports": imports,
        "classes": classes,
        "decorated": decorated,
        "component_type": component_type,
        "detected_types": detected_types,
        "registry_key": registry_key,
    }


def _check_imports(imports: dict[str, str | None], rules: dict[str, Any], errors: list[str]) -> None:
    for name, expected_module in rules["required_imports"].items():
        actual_module = imports.get(name)
        if actual_module is None:
            errors.append(f"Missing required import: {name}")
        elif not _import_matches(name, expected_module, actual_module):
            errors.append(
                f"'{name}' imported from unexpected module '{actual_module}' "
                f"(expected module containing '{expected_module}')"
            )


def _check_config_class(
    classes: list[ast.ClassDef],
    component_type: ComponentType,
    rules: dict[str, Any],
    structure: dict[str, Any],
    errors: list[str],
) -> None:
    config_base = rules["config_base"]
    found_expected_config = False
    found_proto_config = False
    for node in classes:
        for base in node.bases:
            base_name = _base_name(base)
            if base_name == config_base:
                found_expected_config = True
                found_proto_config = True
            elif base_name in ("BaseConfig", "BaseOptimizerConfig"):
                found_proto_config = True
                errors.append(f"{component_type.title()} config should inherit from {config_base}, not {base_name}")

    structure["has_config_class"] = found_expected_config
    if not found_expected_config and not found_proto_config:
        errors.append(f"No config class found inheriting from {config_base}")


def _check_component_definition(
    facts: dict[str, Any],
    component_type: ComponentType,
    rules: dict[str, Any],
    structure: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    decorated_nodes = facts["decorated"][component_type]
    expected_node_type = rules["node_type"]
    decorator_name = rules["decorator_name"]
    if len(decorated_nodes) != 1:
        errors.append(f"Expected exactly one @{decorator_name} component per file; found {len(decorated_nodes)}.")

    for node in decorated_nodes:
        if not isinstance(node, expected_node_type):
            continue
        structure["has_decorated_component"] = True
        structure["has_registry_decorator"] = True
        structure["component_name"] = node.name
        decorator_kwargs = _decorator_kwargs(node, decorator_name)
        _check_decorator_contract(
            decorator_kwargs, facts["classes"], component_type, rules, structure, errors, warnings
        )
        _check_component_shape(node, component_type, rules, errors, warnings)

    if not structure["has_decorated_component"]:
        kind = "function" if component_type == "constraint" else "class"
        errors.append(f"No {component_type} {kind} found with @{decorator_name} decorator")


def _decorator_kwargs(node: ast.FunctionDef | ast.ClassDef, decorator_name: str) -> dict[str, ast.expr]:
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == decorator_name
        ):
            return {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
    return {}


def _check_decorator_contract(
    kwargs: dict[str, ast.expr],
    classes: list[ast.ClassDef],
    component_type: ComponentType,
    rules: dict[str, Any],
    structure: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    errors.extend(
        f"Decorator missing required argument: '{key}'" for key in rules["required_decorator_args"] if key not in kwargs
    )
    warnings.extend(
        f"Decorator missing recommended argument: '{key}'"
        for key in rules["recommended_decorator_args"]
        if key not in kwargs
    )

    key_expr = kwargs.get("key")
    if key_expr is not None:
        key_value = key_expr.value if isinstance(key_expr, ast.Constant) else None
        if not isinstance(key_value, str) or not key_value:
            errors.append("Decorator key must be a non-empty string literal.")
        elif not _KEY_RE.fullmatch(key_value):
            errors.append(f"Decorator key {key_value!r} must be kebab-case.")

    config_expr = kwargs.get("config")
    if config_expr is None:
        return
    config_name = config_expr.id if isinstance(config_expr, ast.Name) else None
    if config_name is None:
        errors.append("Decorator config must reference a config class by name.")
        return

    config_class = next((node for node in classes if node.name == config_name), None)
    if config_class is None:
        errors.append(f"Decorator config references unknown class '{config_name}'.")
        return

    structure["config_class_name"] = config_name
    config_base = rules["config_base"]
    base_names = {_base_name(base) for base in config_class.bases}
    structure["has_config_class"] = config_base in base_names
    if not structure["has_config_class"]:
        errors.append(f"{component_type.title()} config '{config_name}' should inherit from {config_base}.")


def _check_component_shape(
    node: ast.FunctionDef | ast.ClassDef,
    component_type: ComponentType,
    rules: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if component_type == "constraint":
        if not isinstance(node, ast.FunctionDef):
            return
        if len(node.args.args) < 2:
            errors.append(f"Function '{node.name}' must have at least 2 parameters (sequences, config)")
            if not node.args.args:
                return
        if not node.args.args[0].annotation:
            warnings.append(f"Missing type hint for first parameter in '{node.name}'")
        if len(node.args.args) > 1 and not node.args.args[1].annotation:
            warnings.append(f"Missing type hint for config parameter in '{node.name}'")
        if not node.returns:
            warnings.append(f"Missing return type hint in '{node.name}'")
        return

    if not isinstance(node, ast.ClassDef):
        return
    base_class = rules["base_class"]
    if not any(_base_name(base) == base_class for base in node.bases):
        errors.append(f"{component_type.title()} class '{node.name}' must inherit from {base_class}")

    methods = {item.name: item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required_method = "sample" if component_type == "generator" else "run"
    if required_method not in methods:
        errors.append(f"{component_type.title()} '{node.name}' is missing required '{required_method}' method")
    if component_type == "generator":
        init_method = methods.get("__init__")
        if init_method is None:
            errors.append(f"Generator '{node.name}' is missing required '__init__(self, config)' method")
        elif len(init_method.args.args) < 2:
            errors.append(f"Generator '{node.name}' __init__ must accept a config parameter")


def _import_matches(name: str, expected_module: str, actual_module: str | None) -> bool:
    if actual_module is None:
        return False
    return expected_module in actual_module or actual_module in _PUBLIC_IMPORT_MODULES.get(name, frozenset())


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


__all__ = [
    "ComponentType",
    "TestResult",
    "ValidationResult",
    "test_constraint",
    "test_generator",
    "test_optimizer",
    "validate_component_file",
]
