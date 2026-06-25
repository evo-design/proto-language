"""Tests for the generic genetic algorithm optimizer."""

from contextlib import nullcontext

import pytest
from pydantic import BaseModel

from proto_language.core import (
    Constraint,
    ConstraintOutput,
    Construct,
    Generator,
    GeneratorInputType,
    Program,
    Segment,
    Sequence,
)
from proto_language.generator import ESM2Generator, ESM2GeneratorConfig
from proto_language.generator.random_nucleotide_generator import RandomNucleotideGenerator
from proto_language.generator.random_protein_generator import RandomProteinGenerator
from proto_language.optimizer import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig


class TargetAConfig(BaseModel):
    """Dummy config for a deterministic test constraint."""


class MockInverseFoldingGenerator(Generator):
    """Non-mutation generator used to test GA role validation."""

    input_type = GeneratorInputType.STRUCTURE

    def __init__(self) -> None:
        super().__init__()

    def _sample(self) -> None:
        self._validate_generator()


def target_a_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: TargetAConfig,
) -> list[ConstraintOutput]:
    """Score DNA proposals by the fraction of positions that are not A."""
    del config
    results = []
    for (sequence,) in input_sequences:
        score = 1.0 - (sequence.sequence.count("A") / len(sequence.sequence))
        results.append(ConstraintOutput(score=score, metadata={"a_count": sequence.sequence.count("A")}))
    return results


def test_genetic_algorithm_runs_without_generators_and_keeps_best_candidates() -> None:
    segment = Segment(sequence="CCCCCCCCCCCC", sequence_type="dna", label="dna")
    construct = Construct([segment], label="construct")
    constraint = Constraint(
        inputs=[segment],
        function=target_a_constraint,
        function_config=TargetAConfig(),
    )
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[construct],
        generators=[],
        constraints=[constraint],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=3,
            population_size=8,
            offspring_per_generation=8,
            num_results=2,
            initial_mutation_rate=1.0,
            mutation_rate=0.5,
            crossover_rate=0.8,
            seed=123,
            tracking_interval=1,
            track_proposals=True,
        ),
    )

    Program(optimizers=[optimizer], num_results=2, compute=nullcontext()).run()

    assert len(segment.result_sequences) == 2
    assert optimizer.energy_scores == sorted(optimizer.energy_scores)
    assert optimizer.energy_scores[0] < 1.0
    assert optimizer.history[-1]["optimizer"]["type"] == "genetic-algorithm"
    assert optimizer.history[-1]["optimizer"]["generation"] == 3


def test_genetic_algorithm_rejects_more_results_than_population() -> None:
    try:
        GeneticAlgorithmOptimizerConfig(num_generations=1, population_size=2, num_results=3)
    except ValueError as exc:
        assert "num_results cannot exceed population_size" in str(exc)
    else:
        raise AssertionError("Expected validation error")


def test_generational_replacement_backfills_when_offspring_are_few() -> None:
    segment = Segment(sequence="CCCCCCCC", sequence_type="dna", label="dna")
    construct = Construct([segment], label="construct")
    constraint = Constraint(
        inputs=[segment],
        function=target_a_constraint,
        function_config=TargetAConfig(),
    )
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[construct],
        generators=[],
        constraints=[constraint],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=6,
            offspring_per_generation=1,
            num_results=2,
            initial_mutation_rate=0.5,
            mutation_rate=0.5,
            replacement="generational",
            elite_fraction=0.0,
            seed=123,
        ),
    )

    Program(optimizers=[optimizer], num_results=2, compute=nullcontext()).run()

    assert len(segment.proposal_sequences) == 6
    assert len(optimizer._population_energies) == 6


def test_genetic_algorithm_delegates_fallback_mutation_to_uniform_generator(monkeypatch) -> None:
    calls = []

    def fake_sample(self: RandomNucleotideGenerator) -> None:
        calls.append([sequence.sequence for sequence in self.segment.proposal_sequences])
        for sequence in self.segment.proposal_sequences:
            sequence.sequence = "A" * len(sequence.sequence)

    monkeypatch.setattr(RandomNucleotideGenerator, "_sample", fake_sample)

    segment = Segment(sequence="CCCC", sequence_type="dna", label="dna")
    construct = Construct([segment], label="construct")
    constraint = Constraint(
        inputs=[segment],
        function=target_a_constraint,
        function_config=TargetAConfig(),
    )
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[construct],
        generators=[],
        constraints=[constraint],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=4,
            offspring_per_generation=2,
            num_results=1,
            initial_mutation_rate=0.0,
            mutation_rate=1.0,
            seed=123,
        ),
    )

    Program(optimizers=[optimizer], num_results=1, compute=nullcontext()).run()

    assert calls == [["CCCC", "CCCC"]]
    assert segment.result_sequences[0].sequence == "AAAA"


def test_genetic_algorithm_uses_configured_esm2_mutation_generator(monkeypatch) -> None:
    calls = []

    def fake_esm2_sample(self: ESM2Generator) -> None:
        calls.append([sequence.sequence for sequence in self.segment.proposal_sequences])
        for sequence in self.segment.proposal_sequences:
            sequence.sequence = "A" * len(sequence.sequence)

    def fail_random_fallback(self: RandomProteinGenerator) -> None:
        raise AssertionError(f"Unexpected random fallback mutation via {self.__class__.__name__}")

    monkeypatch.setattr(ESM2Generator, "_sample", fake_esm2_sample)
    monkeypatch.setattr(RandomProteinGenerator, "_sample", fail_random_fallback)

    segment = Segment(sequence="CCCC", sequence_type="protein", label="protein")
    generator = ESM2Generator(ESM2GeneratorConfig())
    generator.assign(segment)
    construct = Construct([segment], label="construct")
    constraint = Constraint(
        inputs=[segment],
        function=target_a_constraint,
        function_config=TargetAConfig(),
    )
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=[constraint],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=4,
            offspring_per_generation=2,
            num_results=1,
            initial_mutation_rate=0.0,
            mutation_rate=1.0,
            seed=123,
        ),
    )

    Program(optimizers=[optimizer], num_results=1, compute=nullcontext()).run()

    assert calls == [["CCCC", "CCCC", "CCCC", "CCCC"], ["AAAA", "AAAA"]]
    assert segment.result_sequences[0].sequence == "AAAA"


def test_genetic_algorithm_rejects_hidden_random_fallback_with_non_mutation_generator() -> None:
    segment = Segment(sequence="CCCC", sequence_type="protein", label="protein")
    generator = MockInverseFoldingGenerator()
    generator.assign(segment)
    construct = Construct([segment], label="construct")
    constraint = Constraint(
        inputs=[segment],
        function=target_a_constraint,
        function_config=TargetAConfig(),
    )

    with pytest.raises(ValueError, match="non-mutation generator targets without a mutation generator"):
        GeneticAlgorithmOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=GeneticAlgorithmOptimizerConfig(
                num_generations=1,
                population_size=4,
                offspring_per_generation=2,
                num_results=1,
                mutation_rate=0.1,
            ),
        )
