"""Tests for compiler-backed gradient support metadata and direct routing."""

import numpy as np
import pytest
from pydantic import BaseModel

from proto_language import GradientConstraintOutput
from proto_language.constraint import ConstraintRegistry, MalinoisActivityConfig, malinois_activity_constraint
from proto_language.core import Constraint, Segment, Sequence
from proto_language.optimizer.constraint_compiler import (
    DirectGradientProvider,
    compile_gradient_providers,
    constraint_supports_compiled_gradient,
    gradient_support_for_constraint_spec,
)


class _Config(BaseModel):
    """Empty test config."""


@pytest.mark.parametrize(
    ("constraint_key", "backend_ids"),
    [
        ("structure-plddt", ["esmfold", "alphafold2_binder"]),
        ("structure-distogram-cce", ["alphafold2_binder"]),
        ("malinois-activity", ["malinois"]),
    ],
)
def test_compiled_rules_match_supporting_backends(constraint_key: str, backend_ids: list[str]) -> None:
    support = gradient_support_for_constraint_spec(ConstraintRegistry.get(constraint_key))
    assert support is not None
    assert [r.structure_tool for r in support.rules] == backend_ids


def test_discrete_only_constraint_has_no_compiled_metadata() -> None:
    assert gradient_support_for_constraint_spec(ConstraintRegistry.get("gc-content")) is None


def test_af2_binder_rule_targets_binder_input() -> None:
    support = gradient_support_for_constraint_spec(ConstraintRegistry.get("structure-distogram-cce"))
    assert support is not None
    (rule,) = support.rules
    assert rule.target_input_config_path == "alphafold2_binder_config.binder_input_index"
    assert sorted(req.config_path for req in rule.input_requirements) == [
        "alphafold2_binder_config.binder_input_index",
        "alphafold2_binder_config.target_input_indices",
    ]


def test_explicit_backward_takes_precedence_over_malinois_compilation() -> None:
    segment = Segment(sequence="A" * 200, sequence_type="dna")

    def backward(
        input_sequences: list[tuple[Sequence, ...]], *, config: BaseModel, **kwargs: object
    ) -> list[GradientConstraintOutput]:
        return [
            GradientConstraintOutput(gradient=(np.zeros_like(sequence.logits),), loss=0.0)
            for (sequence,) in input_sequences
        ]

    config = MalinoisActivityConfig(device="cpu")
    constraint = Constraint(
        inputs=[segment],
        function=malinois_activity_constraint,
        function_config=config,
        backward=backward,
        backward_config=config,
    )

    providers = compile_gradient_providers([constraint], segment)

    assert len(providers) == 1
    assert isinstance(providers[0], DirectGradientProvider)
    assert constraint_supports_compiled_gradient(constraint, segment) == (True, None)


def test_direct_provider_sums_all_aliased_target_gradients() -> None:
    segment = Segment(sequence="AA", sequence_type="protein")
    segment.proposal_sequences[0].logits = np.zeros((2, 20))

    def backward(
        input_sequences: list[tuple[Sequence, ...]], *, config: BaseModel, **kwargs: object
    ) -> list[GradientConstraintOutput]:
        return [
            GradientConstraintOutput(
                gradient=(np.ones_like(first.logits), np.full_like(second.logits, 2.0)),
                loss=1.0,
            )
            for first, second in input_sequences
        ]

    constraint = Constraint(inputs=[segment, segment], backward=backward, backward_config=_Config())
    (provider,) = compile_gradient_providers([constraint], segment)

    output = provider.compute(
        temperature=1.0,
        soft=1.0,
        hard=0.0,
        step=1,
        effective_weight=lambda constraint, step: 1.0,
    )

    np.testing.assert_array_equal(output.gradients[0], np.full((2, 20), 3.0))
