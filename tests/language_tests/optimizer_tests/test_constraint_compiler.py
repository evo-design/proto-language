"""Tests for compiler-backed gradient support metadata."""

import pytest

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.optimizer.constraint_compiler import gradient_support_for_constraint_spec


@pytest.mark.parametrize(
    ("constraint_key", "backend_ids"),
    [
        ("structure-plddt", ["esmfold", "alphafold2_multimer"]),
        ("structure-distogram-cce", ["alphafold2_multimer"]),
        ("malinois-activity", ["malinois"]),
    ],
)
def test_compiled_rules_match_supporting_backends(constraint_key: str, backend_ids: list[str]) -> None:
    support = gradient_support_for_constraint_spec(ConstraintRegistry.get(constraint_key))
    assert support is not None
    assert [r.structure_tool for r in support.rules] == backend_ids


def test_discrete_only_constraint_has_no_compiled_metadata() -> None:
    assert gradient_support_for_constraint_spec(ConstraintRegistry.get("gc-content")) is None


def test_af2_multimer_rule_targets_binder_input() -> None:
    support = gradient_support_for_constraint_spec(ConstraintRegistry.get("structure-distogram-cce"))
    assert support is not None
    (rule,) = support.rules
    assert rule.target_input_config_path == "alphafold2_multimer_config.binder_input_index"
    assert sorted(req.config_path for req in rule.input_requirements) == [
        "alphafold2_multimer_config.binder_input_index",
        "alphafold2_multimer_config.target_input_indices",
    ]
