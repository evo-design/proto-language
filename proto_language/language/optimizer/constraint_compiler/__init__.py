"""Compatibility exports for the private constraint compiler package."""

from proto_language.language.optimizer.constraint_compiler.base import (
    GradientProvider,
    GradientProviderOutput,
)
from proto_language.language.optimizer.constraint_compiler.capabilities import (
    GradientSupport,
    GradientSupportRule,
    gradient_support_for_constraint_spec,
)
from proto_language.language.optimizer.constraint_compiler.compiler import (
    DirectGradientProvider,
    compile_gradient_providers,
    constraint_supports_compiled_gradient,
    evaluate_scoring_constraints,
)

__all__ = [
    "DirectGradientProvider",
    "GradientSupport",
    "GradientSupportRule",
    "GradientProvider",
    "GradientProviderOutput",
    "compile_gradient_providers",
    "constraint_supports_compiled_gradient",
    "evaluate_scoring_constraints",
    "gradient_support_for_constraint_spec",
]
