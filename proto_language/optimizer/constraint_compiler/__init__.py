"""Compatibility exports for the private constraint compiler package."""

from proto_language.optimizer.constraint_compiler.base import (
    GradientProvider,
    GradientProviderOutput,
    validate_gradient_provider_output,
)
from proto_language.optimizer.constraint_compiler.compiler import (
    ConstraintCapabilities,
    DirectGradientProvider,
    GradientInputRequirement,
    GradientRule,
    GradientSupport,
    ScoringPlan,
    compile_gradient_providers,
    compile_scoring_plan,
    constraint_supports_compiled_gradient,
    evaluate_scoring_constraints,
    gradient_support_for_constraint_spec,
    resolve_constraint_capabilities,
)

__all__ = [
    "ConstraintCapabilities",
    "DirectGradientProvider",
    "GradientInputRequirement",
    "GradientProvider",
    "GradientProviderOutput",
    "GradientRule",
    "GradientSupport",
    "ScoringPlan",
    "compile_gradient_providers",
    "compile_scoring_plan",
    "constraint_supports_compiled_gradient",
    "evaluate_scoring_constraints",
    "gradient_support_for_constraint_spec",
    "resolve_constraint_capabilities",
    "validate_gradient_provider_output",
]
