"""Compatibility exports for the private constraint compiler package."""

from proto_language.optimizer.constraint_compiler.base import (
    GradientProvider,
    GradientProviderOutput,
    validate_gradient_provider_output,
)
from proto_language.optimizer.constraint_compiler.compiler import (
    DirectGradientProvider,
    GradientInputRequirement,
    GradientRule,
    GradientSupport,
    compile_gradient_providers,
    constraint_supports_compiled_gradient,
    evaluate_scoring_constraints,
    gradient_support_for_constraint_spec,
)

__all__ = [
    "DirectGradientProvider",
    "GradientInputRequirement",
    "GradientProvider",
    "GradientProviderOutput",
    "GradientRule",
    "GradientSupport",
    "compile_gradient_providers",
    "constraint_supports_compiled_gradient",
    "evaluate_scoring_constraints",
    "gradient_support_for_constraint_spec",
    "validate_gradient_provider_output",
]
