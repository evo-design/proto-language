"""Tests for the Malinois activity constraint."""

import math
from unittest.mock import patch

import pytest
from proto_tools import (
    DEFAULT_MALINOIS_ARTIFACT_MD5,
    DEFAULT_MALINOIS_ARTIFACT_URL,
    MalinoisScoreOutput,
    MalinoisScoreResult,
)
from proto_tools.utils.device import number_of_visible_gpus
from pydantic import ValidationError

from proto_language.language.constraint import ConstraintRegistry, malinois_activity_constraint
from proto_language.language.constraint.sequence_annotation.malinois_activity_constraint import (
    MalinoisActivityConfig,
)
from proto_language.language.core import Constraint, Construct, Segment, Sequence
from proto_language.language.generator import PositionWeightGenerator, PositionWeightGeneratorConfig
from proto_language.language.optimizer import GradientOptimizer, GradientOptimizerConfig

PATCH_TARGET = "proto_language.language.constraint.sequence_annotation.malinois_activity_constraint.run_malinois_score"


def _skip_if_no_malinois_gpu() -> None:
    if number_of_visible_gpus() < 1:
        pytest.skip("Malinois GPU test requires a visible CUDA GPU")


def _output(sequences: list[str], raw_scores: list[float], cell_type: str = "K562") -> MalinoisScoreOutput:
    return MalinoisScoreOutput(
        results=[
            MalinoisScoreResult(
                sequence=sequence,
                sequence_length=len(sequence),
                scores={cell_type: raw_score},
            )
            for sequence, raw_score in zip(sequences, raw_scores, strict=True)
        ],
        cell_types=[cell_type],
        seq_length=200,
    )


def test_malinois_activity_config_rejects_nonpositive_sigmoid_scale() -> None:
    with pytest.raises(ValidationError):
        MalinoisActivityConfig(sigmoid_scale=0.0)


def test_malinois_activity_sigmoid_center_maps_to_half_penalty() -> None:
    sequence = "A" * 200
    with patch(PATCH_TARGET, return_value=_output([sequence], [4.0])):
        results = malinois_activity_constraint(
            [(Sequence(sequence, "dna"),)],
            MalinoisActivityConfig(sigmoid_center=4.0, sigmoid_scale=1.0),
        )

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.5)
    assert results[0].metadata["malinois_raw_score"] == 4.0
    assert results[0].metadata["malinois_activity_score"] == pytest.approx(0.5)


def test_malinois_activity_higher_raw_score_is_better() -> None:
    sequences = ["A" * 200, "C" * 200]
    with patch(PATCH_TARGET, return_value=_output(sequences, [6.0, 2.0])):
        results = malinois_activity_constraint(
            [(Sequence(sequences[0], "dna"),), (Sequence(sequences[1], "dna"),)],
            MalinoisActivityConfig(sigmoid_center=4.0, sigmoid_scale=1.0),
        )

    high_raw_score = 1.0 - (1.0 / (1.0 + math.exp(-2.0)))
    low_raw_score = 1.0 - (1.0 / (1.0 + math.exp(2.0)))
    assert results[0].score == pytest.approx(high_raw_score)
    assert results[1].score == pytest.approx(low_raw_score)
    assert results[0].score < 0.5
    assert results[1].score > 0.5


def test_malinois_activity_minimization_lower_raw_score_is_better() -> None:
    sequences = ["A" * 200, "C" * 200]
    with patch(PATCH_TARGET, return_value=_output(sequences, [6.0, 2.0])):
        results = malinois_activity_constraint(
            [(Sequence(sequences[0], "dna"),), (Sequence(sequences[1], "dna"),)],
            MalinoisActivityConfig(direction="min", sigmoid_center=4.0, sigmoid_scale=1.0),
        )

    high_raw_score = 1.0 / (1.0 + math.exp(-2.0))
    low_raw_score = 1.0 / (1.0 + math.exp(2.0))
    assert results[0].score == pytest.approx(high_raw_score)
    assert results[1].score == pytest.approx(low_raw_score)
    assert results[0].score > 0.5
    assert results[1].score < 0.5


def test_malinois_activity_forwards_config_and_records_metadata() -> None:
    segment = Segment(sequence="ACGT" * 50, sequence_type="dna")
    observed = {}

    def fake_run_malinois_score(inputs, config):
        observed["inputs"] = inputs
        observed["config"] = config
        return _output(inputs.sequences, [5.0], cell_type=config.cell_types[0])

    config = MalinoisActivityConfig(
        cell_type="HepG2",
        sigmoid_center=4.0,
        sigmoid_scale=2.0,
        batch_size=8,
        device="cpu",
    )
    with patch(PATCH_TARGET, side_effect=fake_run_malinois_score):
        scores = Constraint(
            inputs=[segment],
            function=malinois_activity_constraint,
            function_config=config,
        ).evaluate()

    assert scores[0] == pytest.approx(1.0 - (1.0 / (1.0 + math.exp(-0.5))))
    assert observed["inputs"].sequences == ["ACGT" * 50]
    assert observed["config"].cell_types == ["HepG2"]
    assert observed["config"].batch_size == 8
    assert observed["config"].device == "cpu"
    assert observed["config"].artifact_path == ""
    assert observed["config"].artifact_url == DEFAULT_MALINOIS_ARTIFACT_URL
    assert observed["config"].artifact_md5 == DEFAULT_MALINOIS_ARTIFACT_MD5
    assert observed["config"].malinois_dir == ""

    metadata = segment.proposal_sequences[0]._constraints_metadata["malinois_activity_constraint"]["data"]
    assert metadata["malinois_cell_type"] == "HepG2"
    assert metadata["malinois_direction"] == "max"
    assert metadata["malinois_raw_score"] == 5.0
    assert metadata["malinois_scaled_score"] == pytest.approx(0.5)
    assert metadata["malinois_activity_score"] == pytest.approx(scores[0])


def test_malinois_activity_registry_integration() -> None:
    spec = ConstraintRegistry.get("malinois-activity")
    assert spec.label == "Malinois Activity"
    assert spec.uses_gpu is True
    assert spec.tools_called == ["malinois-score", "malinois-gradient"]
    assert "dna" in spec.supported_sequence_types

    segment = Segment(sequence="A" * 200, sequence_type="dna")
    constraint = ConstraintRegistry.create("malinois-activity", [segment], {})
    assert isinstance(constraint, Constraint)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_activity_real_gpu_constraint_evaluation() -> None:
    _skip_if_no_malinois_gpu()

    segment = Segment(sequence="ACGT" * 50, sequence_type="dna")
    scores = Constraint(
        inputs=[segment],
        function=malinois_activity_constraint,
        function_config=MalinoisActivityConfig(
            cell_type="K562",
            sigmoid_center=4.0,
            sigmoid_scale=1.0,
            batch_size=1,
            device="cuda",
        ),
    ).evaluate()

    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
    metadata = segment.proposal_sequences[0]._constraints_metadata["malinois_activity_constraint"]["data"]
    assert metadata["malinois_cell_type"] == "K562"
    assert metadata["malinois_direction"] == "max"
    assert math.isfinite(metadata["malinois_raw_score"])
    assert metadata["malinois_activity_score"] == pytest.approx(scores[0])


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_activity_real_gpu_gradient_optimizer() -> None:
    """Real GPU smoke test for compiler-backed Malinois gradients."""
    _skip_if_no_malinois_gpu()

    segment = Segment(sequence="ACGT" * 50, sequence_type="dna")
    generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
    generator.assign(segment)
    constraints = [
        Constraint(
            inputs=[segment],
            function=malinois_activity_constraint,
            function_config=MalinoisActivityConfig(
                cell_type="K562",
                direction="max",
                sigmoid_center=4.0,
                sigmoid_scale=1.0,
                device="cuda",
            ),
            label="malinois_k562_max",
        ),
        Constraint(
            inputs=[segment],
            function=malinois_activity_constraint,
            function_config=MalinoisActivityConfig(
                cell_type="HepG2",
                direction="min",
                sigmoid_center=4.0,
                sigmoid_scale=1.0,
                device="cuda",
            ),
            label="malinois_hepg2_min",
        ),
        Constraint(
            inputs=[segment],
            function=malinois_activity_constraint,
            function_config=MalinoisActivityConfig(
                cell_type="SKNSH",
                direction="min",
                sigmoid_center=4.0,
                sigmoid_scale=1.0,
                device="cuda",
            ),
            label="malinois_sknsh_min",
        ),
    ]
    optimizer = GradientOptimizer(
        target_segment=segment,
        constructs=[Construct([segment])],
        generators=[generator],
        constraints=constraints,
        config=GradientOptimizerConfig(num_results=2, num_steps=1, lr=0.1),
    )

    optimizer.run()

    assert len(optimizer.energy_scores) == 2
    assert all(math.isfinite(score) for score in optimizer.energy_scores)
    for sequence in segment.result_sequences:
        all_metadata = sequence._constraints_metadata
        assert all_metadata["malinois_k562_max"]["data"]["malinois_cell_type"] == "K562"
        assert all_metadata["malinois_k562_max"]["data"]["malinois_direction"] == "max"
        assert all_metadata["malinois_hepg2_min"]["data"]["malinois_cell_type"] == "HepG2"
        assert all_metadata["malinois_hepg2_min"]["data"]["malinois_direction"] == "min"
        assert all_metadata["malinois_sknsh_min"]["data"]["malinois_cell_type"] == "SKNSH"
        assert all_metadata["malinois_sknsh_min"]["data"]["malinois_direction"] == "min"
        assert all(
            math.isfinite(all_metadata[label]["data"]["malinois_raw_score"])
            for label in ("malinois_k562_max", "malinois_hepg2_min", "malinois_sknsh_min")
        )
