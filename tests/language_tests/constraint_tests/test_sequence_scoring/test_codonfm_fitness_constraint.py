"""Tests for the CodonFM (Encodon) coding-sequence fitness constraint."""

import importlib
import math
from types import SimpleNamespace

import pytest

from proto_language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.constraint.sequence_scoring.codonfm_fitness_constraint import (
    CodonFMFitnessConstraintConfig,
    codonfm_fitness_constraint,
)
from proto_language.core import Constraint, Segment, Sequence
from proto_language.utils import sigmoid_score

_MODULE = importlib.import_module("proto_language.constraint.sequence_scoring.codonfm_fitness_constraint")
_CDS = "ATGGTGAGCAAGGGC"  # 15 nt, 5 codons


def _fake_output(*fitness_values: float, sequences: list[str] | None = None) -> SimpleNamespace:
    """Mimic a CodonFMFitnessOutput with one result per fitness value."""
    if sequences is None:
        return SimpleNamespace(results=[SimpleNamespace(fitness=v) for v in fitness_values])
    return SimpleNamespace(
        results=[
            SimpleNamespace(fitness=value, sequence=sequence)
            for value, sequence in zip(fitness_values, sequences, strict=True)
        ]
    )


class TestForward:
    def test_direction_max_rewards_high_fitness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def fake_run(inputs, config):
            captured["inputs"] = inputs
            captured["config"] = config
            return _fake_output(-1.5)

        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", fake_run)

        config = CodonFMFitnessConstraintConfig(
            model_checkpoint="encodon_80m", direction="max", sigmoid_center=-2.0, sigmoid_scale=1.0, device="cpu"
        )
        outputs = codonfm_fitness_constraint([(Sequence(_CDS, sequence_type="dna"),)], config)

        sigmoid = sigmoid_score(-1.5, -2.0, slope=1.0)
        assert outputs[0].score == pytest.approx(1.0 - sigmoid)  # high fitness -> low energy
        assert outputs[0].metadata["fitness"] == -1.5
        assert outputs[0].metadata["direction"] == "max"
        assert outputs[0].metadata["model_checkpoint"] == "encodon_80m"
        assert outputs[0].metadata["objective"] == "upstream_non_padding_token_mean_log_likelihood"
        # The tool is called with the coding sequence and the selected checkpoint.
        assert captured["inputs"].sequences == [_CDS]
        assert captured["config"].model_checkpoint == "encodon_80m"

    def test_direction_min_reverses_energy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output(-1.5))
        config = CodonFMFitnessConstraintConfig(direction="min", sigmoid_center=-2.0, sigmoid_scale=1.0, device="cpu")
        outputs = codonfm_fitness_constraint([(Sequence(_CDS, sequence_type="dna"),)], config)
        assert outputs[0].score == pytest.approx(sigmoid_score(-1.5, -2.0, slope=1.0))

    def test_batched_and_rna_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output(-1.0, -3.0))
        config = CodonFMFitnessConstraintConfig(device="cpu")
        outputs = codonfm_fitness_constraint(
            [(Sequence("AUGGUG", sequence_type="rna"),), (Sequence("AUGGCC", sequence_type="rna"),)], config
        )
        assert [o.metadata["fitness"] for o in outputs] == [-1.0, -3.0]

    def test_empty_input_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output())
        assert codonfm_fitness_constraint([], CodonFMFitnessConstraintConfig(device="cpu")) == []

    def test_non_finite_fitness_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output(math.nan))
        with pytest.raises(ValueError, match="non-finite fitness"):
            codonfm_fitness_constraint(
                [(Sequence(_CDS, sequence_type="dna"),)], CodonFMFitnessConstraintConfig(device="cpu")
            )

    def test_positive_fitness_log_probability_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output(0.01))
        with pytest.raises(ValueError, match="positive fitness log-probability"):
            codonfm_fitness_constraint(
                [(Sequence(_CDS, sequence_type="dna"),)], CodonFMFitnessConstraintConfig(device="cpu")
            )

    @pytest.mark.parametrize("fitness_values", [(), (-1.0, -2.0)])
    def test_rejects_tool_result_cardinality_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, fitness_values: tuple[float, ...]
    ) -> None:
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output(*fitness_values))
        with pytest.raises(ValueError, match="fitness results for 1 proposals"):
            codonfm_fitness_constraint(
                [(Sequence(_CDS, sequence_type="dna"),)], CodonFMFitnessConstraintConfig(device="cpu")
            )

    def test_rejects_reordered_tool_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        other = "ATGGCCAGCAAGGGC"
        monkeypatch.setattr(
            _MODULE,
            "run_codonfm_fitness",
            lambda inputs, config: _fake_output(-1.0, -2.0, sequences=[other, _CDS]),
        )
        with pytest.raises(ValueError, match="output order is invalid"):
            codonfm_fitness_constraint(
                [(Sequence(_CDS, sequence_type="dna"),), (Sequence(other, sequence_type="dna"),)],
                CodonFMFitnessConstraintConfig(device="cpu"),
            )

    def test_constraint_evaluate_propagates_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output(-1.5))
        segment = Segment(sequence=_CDS, sequence_type="dna")
        constraint = Constraint(
            inputs=[segment],
            function=codonfm_fitness_constraint,
            function_config={"device": "cpu"},
        )
        constraint.evaluate()
        data = segment.proposal_sequences[0].metadata["constraints"]["codonfm_fitness_constraint"]["data"]
        assert data["fitness"] == -1.5
        assert data["objective"] == "upstream_non_padding_token_mean_log_likelihood"

    def test_score_stays_in_unit_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A fitness far from the sigmoid center still clamps to [0, 1].
        monkeypatch.setattr(_MODULE, "run_codonfm_fitness", lambda inputs, config: _fake_output(0.0, -50.0))
        outputs = codonfm_fitness_constraint(
            [(Sequence(_CDS, sequence_type="dna"),), (Sequence(_CDS, sequence_type="dna"),)],
            CodonFMFitnessConstraintConfig(direction="max", device="cpu"),
        )
        for out in outputs:
            assert 0.0 <= out.score <= 1.0


class TestConfigAndRegistry:
    def test_sigmoid_scale_must_be_positive_normal(self) -> None:
        with pytest.raises(ValueError):
            CodonFMFitnessConstraintConfig(sigmoid_scale=0.0)

    def test_batch_size_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError):
            CodonFMFitnessConstraintConfig(batch_size=0)

    def test_registration(self) -> None:
        spec = ConstraintRegistry.get("codonfm-fitness")
        assert spec.mode == "discrete"
        assert spec.function is codonfm_fitness_constraint
        assert spec.backward is None
        assert spec.category == "sequence_scoring"
        assert spec.tools_called == ["codonfm-fitness"]
        assert spec.supported_sequence_types == ["dna", "rna"]
        assert spec.input_labels == [InputSlot(label="Sequence")]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_codonfm_fitness_constraint_real_model() -> None:
    """Real-model smoke test: forward scoring returns a finite [0, 1] energy.

    The ``uses_gpu`` marker auto-skips when no GPU is visible. Encodon checkpoint repositories
    are public; a HuggingFace token is optional.
    """
    outputs = codonfm_fitness_constraint(
        [(Sequence(_CDS, sequence_type="dna"),)],
        CodonFMFitnessConstraintConfig(model_checkpoint="encodon_80m", direction="max", device="cuda"),
    )
    assert len(outputs) == 1
    assert 0.0 <= outputs[0].score <= 1.0
    assert math.isfinite(outputs[0].metadata["fitness"])
