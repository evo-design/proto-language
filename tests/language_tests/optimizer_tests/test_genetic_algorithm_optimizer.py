"""Tests for the generic genetic algorithm optimizer."""

from contextlib import nullcontext

from pydantic import BaseModel

from proto_language.core import (
    Constraint,
    ConstraintOutput,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.generator import ESM2Generator, ESM2GeneratorConfig
from proto_language.generator.random_protein_generator import RandomProteinGenerator
from proto_language.optimizer import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig
from proto_language.optimizer.genetic_algorithm_optimizer import (
    _crowding_distance,
    _dominates,
    _non_dominated_sort,
    _nsga2_select_survivors,
)


class TargetAConfig(BaseModel):
    """Dummy config for a deterministic test constraint."""


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


def test_genetic_algorithm_without_generators_keeps_sequences_constant() -> None:
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
            crossover_rate=0.8,
            seed=123,
            tracking_interval=1,
            track_proposals=True,
        ),
    )

    Program(optimizers=[optimizer], num_results=2, compute=nullcontext()).run()

    assert len(segment.result_sequences) == 2
    assert {sequence.sequence for sequence in segment.result_sequences} == {"CCCCCCCCCCCC"}
    assert optimizer.energy_scores == sorted(optimizer.energy_scores)
    assert optimizer.energy_scores == [1.0, 1.0]
    assert optimizer.history[-1]["optimizer"]["type"] == "genetic-algorithm"
    assert optimizer.history[-1]["optimizer"]["generation"] == 3


def test_genetic_algorithm_rejects_more_results_than_population() -> None:
    try:
        GeneticAlgorithmOptimizerConfig(num_generations=1, population_size=2, num_results=3)
    except ValueError as exc:
        assert "num_results cannot exceed population_size" in str(exc)
    else:
        raise AssertionError("Expected validation error")


def test_genetic_algorithm_rejects_shared_tournament_with_non_tournament_selection() -> None:
    try:
        GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=2,
            parent_selection="rank",
            parent_pair_selection="shared_tournament",
        )
    except ValueError as exc:
        assert "requires parent_selection='tournament'" in str(exc)
    else:
        raise AssertionError("Expected validation error")


def test_genetic_algorithm_shared_tournament_returns_winner_and_runner_up() -> None:
    segment = Segment(sequence="CCCC", sequence_type="protein", label="protein")
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[Construct([segment], label="construct")],
        generators=[],
        constraints=[
            Constraint(
                inputs=[segment],
                function=target_a_constraint,
                function_config=TargetAConfig(),
            )
        ],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=2,
            parent_pair_selection="shared_tournament",
            survivor_selection="nsga2",
            tournament_size=2,
            tournament_win_probability=1.0,
            seed=0,
        ),
    )
    optimizer._population_energies = [1.0, 2.0]
    optimizer._population_nsga2_ranks = [0, 1]
    optimizer._population_nsga2_crowding = [float("inf"), float("inf")]

    assert optimizer._select_parent_pair([1.0, 2.0]) == (0, 1)


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
            replacement="generational",
            elite_fraction=0.0,
            seed=123,
        ),
    )

    Program(optimizers=[optimizer], num_results=2, compute=nullcontext()).run()

    assert len(segment.proposal_sequences) == 6
    assert len(optimizer._population_energies) == 6


def test_genetic_algorithm_uses_configured_esm2_mutation_generator(monkeypatch) -> None:
    calls = []

    def fake_esm2_sample(self: ESM2Generator) -> None:
        calls.append([sequence.sequence for sequence in self.segment.proposal_sequences])
        for sequence in self.segment.proposal_sequences:
            sequence.sequence = "A" * len(sequence.sequence)

    def fail_random_protein_mutation(self: RandomProteinGenerator) -> None:
        raise AssertionError(f"Unexpected random protein mutation via {self.__class__.__name__}")

    monkeypatch.setattr(ESM2Generator, "_sample", fake_esm2_sample)
    monkeypatch.setattr(RandomProteinGenerator, "_sample", fail_random_protein_mutation)

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
            initialize_with_mutation_generators=True,
            seed=123,
        ),
    )

    Program(optimizers=[optimizer], num_results=1, compute=nullcontext()).run()

    assert calls == [["CCCC", "CCCC", "CCCC", "CCCC"], ["AAAA", "AAAA"]]
    assert segment.result_sequences[0].sequence == "AAAA"


def test_genetic_algorithm_mutates_offspring_only_by_default(monkeypatch) -> None:
    calls = []

    def fake_esm2_sample(self: ESM2Generator) -> None:
        calls.append([sequence.sequence for sequence in self.segment.proposal_sequences])
        for sequence in self.segment.proposal_sequences:
            sequence.sequence = "A" * len(sequence.sequence)

    monkeypatch.setattr(ESM2Generator, "_sample", fake_esm2_sample)

    segment = Segment(sequence="CCCC", sequence_type="protein", label="protein")
    generator = ESM2Generator(ESM2GeneratorConfig())
    generator.assign(segment)
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[Construct([segment], label="construct")],
        generators=[generator],
        constraints=[
            Constraint(
                inputs=[segment],
                function=target_a_constraint,
                function_config=TargetAConfig(),
            )
        ],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=4,
            offspring_per_generation=2,
            num_results=1,
            seed=123,
        ),
    )

    Program(optimizers=[optimizer], num_results=1, compute=nullcontext()).run()

    assert calls == [["CCCC", "CCCC"]]
    assert segment.result_sequences[0].sequence == "AAAA"


def test_genetic_algorithm_does_not_crossover_fixed_segments(monkeypatch) -> None:
    def fake_esm2_sample(self: ESM2Generator) -> None:
        for sequence in self.segment.proposal_sequences:
            sequence.sequence = "AAAA"

    monkeypatch.setattr(ESM2Generator, "_sample", fake_esm2_sample)

    variable = Segment(sequence="CCCC", sequence_type="protein", label="variable")
    fixed = Segment(sequence="CCCC", sequence_type="protein", label="fixed_context")
    generator = ESM2Generator(ESM2GeneratorConfig())
    generator.assign(variable)
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[Construct([variable, fixed], label="construct")],
        generators=[generator],
        constraints=[
            Constraint(
                inputs=[variable],
                function=target_a_constraint,
                function_config=TargetAConfig(),
            )
        ],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=2,
            offspring_per_generation=1,
            num_results=1,
            crossover_rate=1.0,
            seed=123,
        ),
    )
    parent_choices = iter([0, 1])
    optimizer._select_parent = lambda _energies: next(parent_choices)  # type: ignore[method-assign]

    offspring = optimizer._make_offspring(
        parent_sequences=[
            [Sequence("CCCC", "protein"), Sequence("GGGG", "protein")],
            [Sequence("CCCC", "protein"), Sequence("GGGG", "protein")],
        ],
        parent_energies=[0.0, 0.0],
        generation=1,
    )

    assert offspring[0][0].sequence == "AAAA"
    assert offspring[1][0].sequence == "CCCC"


def test_genetic_algorithm_restricts_crossover_to_configured_positions(monkeypatch) -> None:
    def noop_esm2_sample(self: ESM2Generator) -> None:
        return None

    monkeypatch.setattr(ESM2Generator, "_sample", noop_esm2_sample)

    segment = Segment(sequence="AAAAAA", sequence_type="protein", label="protein")
    generator = ESM2Generator(ESM2GeneratorConfig())
    generator.assign(segment)

    optimizer = GeneticAlgorithmOptimizer(
        constructs=[Construct([segment], label="construct")],
        generators=[generator],
        constraints=[
            Constraint(
                inputs=[segment],
                function=target_a_constraint,
                function_config=TargetAConfig(),
            )
        ],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=1,
            population_size=2,
            offspring_per_generation=1,
            num_results=1,
            crossover_rate=1.0,
            crossover_strategy="uniform",
            crossover_positions={"protein": [2, 3]},
            seed=123,
        ),
    )

    assert optimizer._crossover_positions_by_segment()[id(segment)] == {2, 3}

    child = optimizer._crossover_copy(
        Sequence("AAAAAA", "protein"),
        Sequence("CCCCCC", "protein"),
        mutable_indices={2, 3},
    )

    assert child.sequence[:2] == "AA"
    assert child.sequence[4:] == "AA"


def test_genetic_algorithm_rejects_unknown_crossover_position_segment(monkeypatch) -> None:
    def noop_esm2_sample(self: ESM2Generator) -> None:
        return None

    monkeypatch.setattr(ESM2Generator, "_sample", noop_esm2_sample)

    segment = Segment(sequence="AAAAAA", sequence_type="protein", label="protein")
    generator = ESM2Generator(ESM2GeneratorConfig())
    generator.assign(segment)

    try:
        GeneticAlgorithmOptimizer(
            constructs=[Construct([segment], label="construct")],
            generators=[generator],
            constraints=[
                Constraint(
                    inputs=[segment],
                    function=target_a_constraint,
                    function_config=TargetAConfig(),
                )
            ],
            config=GeneticAlgorithmOptimizerConfig(
                num_generations=1,
                population_size=2,
                offspring_per_generation=1,
                num_results=1,
                crossover_positions={"missing": [0]},
                seed=123,
            ),
        )
    except ValueError as exc:
        assert "crossover_positions contains unknown segment labels" in str(exc)
    else:
        raise AssertionError("Expected validation error")


def test_genetic_algorithm_rejects_out_of_bounds_crossover_positions(monkeypatch) -> None:
    def noop_esm2_sample(self: ESM2Generator) -> None:
        return None

    monkeypatch.setattr(ESM2Generator, "_sample", noop_esm2_sample)

    segment = Segment(sequence="AAAAAA", sequence_type="protein", label="protein")
    generator = ESM2Generator(ESM2GeneratorConfig())
    generator.assign(segment)

    try:
        GeneticAlgorithmOptimizer(
            constructs=[Construct([segment], label="construct")],
            generators=[generator],
            constraints=[
                Constraint(
                    inputs=[segment],
                    function=target_a_constraint,
                    function_config=TargetAConfig(),
                )
            ],
            config=GeneticAlgorithmOptimizerConfig(
                num_generations=1,
                population_size=2,
                offspring_per_generation=1,
                num_results=1,
                crossover_positions={"protein": [-1, 6]},
                seed=123,
            ),
        )
    except ValueError as exc:
        assert "crossover_positions for segment 'protein' contains out-of-bounds indices" in str(exc)
    else:
        raise AssertionError("Expected validation error")


def test_nsga2_helper_functions_rank_pareto_fronts() -> None:
    assert _dominates([0.1, 0.2], [0.2, 0.3])
    assert _dominates([0.1, 0.2], [0.1, 0.3])
    assert not _dominates([0.1, 0.4], [0.2, 0.3])
    assert not _dominates([0.1, 0.2], [0.1, 0.2])

    objectives = [
        [0.1, 0.8],
        [0.8, 0.1],
        [0.5, 0.5],
        [0.9, 0.9],
    ]
    fronts = _non_dominated_sort(objectives)
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]

    distances = _crowding_distance(objectives, fronts[0])
    assert distances[0] == float("inf")
    assert distances[1] == float("inf")
    assert distances[2] > 0.0

    survivors = _nsga2_select_survivors(objectives, population_size=2)
    assert set(survivors).issubset({0, 1, 2})
