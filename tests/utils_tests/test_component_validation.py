"""Tests for proto_language.utils.component_validation."""

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from proto_language.utils import (
    TestResult,
    ValidationResult,
    test_constraint,
    test_generator,
    test_optimizer,
    validate_component_file,
)

GOOD_CONSTRAINT_SOURCE = textwrap.dedent(
    """
    from proto_language.base_config import BaseConfig, ConfigField
    from proto_language.language.constraint.constraint_registry import constraint
    from proto_language.language.core import ConstraintOutput, Sequence


    class ToyConfig(BaseConfig):
        threshold: float = ConfigField(default=0.5, description="Threshold.", title="Threshold")


    @constraint(
        key="toy-constraint",
        label="Toy",
        config=ToyConfig,
        description="Toy.",
        supported_sequence_types=["protein"],
        tools_called=[],
        category="testing",
    )
    def toy_constraint(
        input_sequences: list[tuple[Sequence, ...]], config: ToyConfig
    ) -> list[ConstraintOutput]:
        return [ConstraintOutput(score=config.threshold) for _ in input_sequences]
    """
).lstrip()


GOOD_GENERATOR_SOURCE = textwrap.dedent(
    """
    from proto_language import BaseConfig, Generator, generator


    class ToyGeneratorConfig(BaseConfig):
        pass


    @generator(
        key="toy-generator",
        label="Toy",
        config=ToyGeneratorConfig,
        description="Toy.",
        category="mutation",
        supported_sequence_types=["dna"],
    )
    class ToyGenerator(Generator):
        def __init__(self, config: ToyGeneratorConfig):
            super().__init__()
            self.config = config

        def sample(self) -> None:
            pass
    """
).lstrip()


GOOD_OPTIMIZER_SOURCE = textwrap.dedent(
    """
    from proto_language.base_config import BaseOptimizerConfig
    from proto_language.language.core import Optimizer
    from proto_language.language.optimizer.optimizer_registry import optimizer


    class ToyOptimizerConfig(BaseOptimizerConfig):
        pass


    @optimizer(
        key="toy-optimizer",
        label="Toy",
        config=ToyOptimizerConfig,
        description="Toy.",
    )
    class ToyOptimizer(Optimizer):
        def run(self) -> None:
            pass
    """
).lstrip()


def _validate(tmp_path: Path, source: str, filename: str = "component.py") -> ValidationResult:
    path = tmp_path / filename
    path.write_text(source)
    return validate_component_file(path)


def _has_error(result: ValidationResult, substring: str) -> bool:
    return any(substring.lower() in error.lower() for error in result.errors)


@pytest.mark.parametrize(
    ("source", "component_type", "registry_key"),
    [
        (GOOD_CONSTRAINT_SOURCE, "constraint", "toy-constraint"),
        (GOOD_GENERATOR_SOURCE, "generator", "toy-generator"),
        (GOOD_OPTIMIZER_SOURCE, "optimizer", "toy-optimizer"),
    ],
)
def test_validate_accepts_well_formed_components(
    tmp_path: Path, source: str, component_type: str, registry_key: str
) -> None:
    result = _validate(tmp_path, source)

    assert result.success
    assert result.component_type == component_type
    assert result.registry_key == registry_key


def test_validate_accepts_public_proto_language_imports(tmp_path: Path) -> None:
    source = GOOD_CONSTRAINT_SOURCE.replace(
        "from proto_language.base_config import BaseConfig, ConfigField\n"
        "from proto_language.language.constraint.constraint_registry import constraint\n"
        "from proto_language.language.core import ConstraintOutput, Sequence\n",
        "from proto_language import BaseConfig, ConstraintOutput, Sequence, constraint\n",
    ).replace(
        'threshold: float = ConfigField(default=0.5, description="Threshold.", title="Threshold")',
        "threshold: float = 0.5",
    )

    assert _validate(tmp_path, source).success


@pytest.mark.parametrize(
    ("transform", "error_substring"),
    [
        (lambda src: src.replace("from proto_language.base_config import BaseConfig, ConfigField\n", ""), "BaseConfig"),
        (lambda src: src.replace("class ToyConfig(BaseConfig):", "class ToyConfig(SomeOtherBase):"), "config class"),
        (
            lambda src: src.replace(
                "from proto_language.base_config import BaseConfig, ConfigField",
                "from proto_language.base_config import BaseConfig, BaseOptimizerConfig, ConfigField",
            ).replace("class ToyConfig(BaseConfig):", "class ToyConfig(BaseOptimizerConfig):"),
            "should inherit from BaseConfig",
        ),
        (lambda src: src.replace('key="toy-constraint"', 'key="Toy_Constraint"'), "kebab-case"),
        (lambda src: src.replace("config=ToyConfig", "config=MissingConfig"), "unknown class 'MissingConfig'"),
        (lambda src: src.replace('supported_sequence_types=["protein"],\n', ""), "supported_sequence_types"),
        (
            lambda src: src.replace(
                "def toy_constraint(\n"
                "    input_sequences: list[tuple[Sequence, ...]], config: ToyConfig\n"
                ") -> list[ConstraintOutput]:",
                "def toy_constraint() -> list[ConstraintOutput]:",
            ),
            "at least 2 parameters",
        ),
    ],
)
def test_validate_rejects_invalid_constraint_contract(
    tmp_path: Path, transform: Callable[[str], str], error_substring: str
) -> None:
    result = _validate(tmp_path, transform(GOOD_CONSTRAINT_SOURCE))

    assert not result.success
    assert _has_error(result, error_substring)


def test_validate_rejects_multiple_components(tmp_path: Path) -> None:
    second_constraint = textwrap.dedent(
        """

        @constraint(
            key="second-constraint",
            label="Second",
            config=ToyConfig,
            description="Second.",
            supported_sequence_types=["protein"],
        )
        def second_constraint(
            input_sequences: list[tuple[Sequence, ...]], config: ToyConfig
        ) -> list[ConstraintOutput]:
            return []
        """
    )

    result = _validate(tmp_path, GOOD_CONSTRAINT_SOURCE + second_constraint)

    assert not result.success
    assert _has_error(result, "exactly one @constraint")


@pytest.mark.parametrize(
    ("source", "error_substring"),
    [
        ("def broken(:\n    pass\n", "Line "),
        ("def plain():\n    return 1\n", "@constraint"),
        (GOOD_GENERATOR_SOURCE.replace("    def sample(self) -> None:\n        pass\n", ""), "sample"),
    ],
)
def test_validate_rejects_unusable_files(tmp_path: Path, source: str, error_substring: str) -> None:
    result = _validate(tmp_path, source)

    assert not result.success
    assert _has_error(result, error_substring)


def test_validate_missing_path_returns_error(tmp_path: Path) -> None:
    result = validate_component_file(tmp_path / "does_not_exist.py")

    assert not result.success
    assert _has_error(result, "File not found")


@pytest.mark.parametrize(
    ("seq", "expected", "should_pass"),
    [
        ("ATGC", [0.0], True),
        ("AAAA", [0.0], False),
        ("ATGC", None, True),
    ],
)
def test_constraint_helper_gc_content(seq: str, expected: list[float] | None, should_pass: bool) -> None:
    result = test_constraint(
        "gc-content",
        [seq],
        config={"min_gc": 40.0, "max_gc": 60.0},
        expected_scores=expected,
        tolerance=0.05,
        sequence_type="dna",
    )

    assert isinstance(result, TestResult)
    assert result.passed is should_pass
    assert len(result.actual) == 1


def test_generator_helper_checks_length_and_alphabet() -> None:
    result = test_generator(
        "random-nucleotide",
        segment_length=12,
        n_samples=3,
        sequence_type="dna",
        expected_alphabet="ACGT",
    )

    assert result.passed
    assert len(result.actual) == 3
    assert all(len(seq) == 12 and set(seq) <= set("ACGT") for seq in result.actual)


@pytest.mark.parametrize(
    ("config", "should_pass"),
    [({"num_steps": 10}, True), ({}, False)],
)
def test_optimizer_helper_validates_config(config: dict, should_pass: bool) -> None:
    assert test_optimizer("mcmc", config=config).passed is should_pass
