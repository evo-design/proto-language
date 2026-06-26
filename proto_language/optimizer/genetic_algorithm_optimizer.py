"""Genetic algorithm optimizer for discrete sequence design.

This module implements a general population-based optimizer for discrete
segments. It initializes a population from existing proposals, combines parents
by crossover, applies configured starting-sequence generators as mutation
operators, and keeps low-energy candidates according to the replacement policy.

Examples:
    >>> config = GeneticAlgorithmOptimizerConfig(num_generations=10, population_size=32)
    >>> config.crossover_strategy
    'single_point'
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable
from typing import Any, Literal, final

import numpy as np
from pydantic import model_validator

from proto_language.core import Constraint, Construct, Generator, GeneratorInputType, Optimizer, Sequence
from proto_language.optimizer.optimizer_registry import optimizer
from proto_language.utils.base import BaseOptimizerConfig, ConfigField

logger = logging.getLogger(__name__)


class GeneticAlgorithmOptimizerConfig(BaseOptimizerConfig):
    """Configuration for a general genetic algorithm optimizer.

    Attributes:
        num_generations (int): Number of generations to run after population initialization.
        num_results (int | None): Number of final candidates to return; overrides program count.
        population_size (int): Number of candidates maintained in each population.
        offspring_per_generation (int | None): Number of offspring scored per generation.
        elite_fraction (float): Fraction of top parents copied before child selection.
        crossover_rate (float): Probability that a child recombines two parents.
        crossover_strategy (Literal["single_point", "two_point", "uniform"]): Recombination operator.
        parent_selection (Literal["tournament", "rank", "roulette"]): Strategy for choosing parents.
        tournament_size (int): Number of candidates sampled for tournament parent selection.
        replacement (Literal["elitist", "generational"]): Policy for forming the next population.
        refine_offspring_with_generators (bool): Run non-mutation generators after mutation.
        initialize_with_mutation_generators (bool): Use mutation generators during initialization.
        tracking_interval (int): Number of generations between saved progress snapshots.
        track_proposals (bool): Whether to store proposal sequences alongside accepted results.
    """

    num_generations: int = ConfigField(
        ge=1,
        title="Generations",
        description="Number of genetic algorithm generations.",
    )
    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Number of top-scoring candidates to retain as final results. Overrides program count.",
    )
    population_size: int = ConfigField(
        default=32,
        ge=2,
        title="Population Size",
        description="Number of candidates maintained in the population.",
    )
    offspring_per_generation: int | None = ConfigField(
        default=None,
        ge=1,
        title="Offspring Per Generation",
        description="Number of children scored per generation. Defaults to population_size.",
    )
    elite_fraction: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Elite Fraction",
        description="Fraction of the best parents copied into the next generation before selecting children.",
    )
    crossover_rate: float = ConfigField(
        default=0.8,
        ge=0.0,
        le=1.0,
        title="Crossover Rate",
        description="Probability that an offspring recombines two parents instead of copying one parent.",
    )
    crossover_strategy: Literal["single_point", "two_point", "uniform"] = ConfigField(
        default="single_point",
        title="Crossover Strategy",
        description="Crossover operator used for equal-length parent sequences.",
    )
    parent_selection: Literal["tournament", "rank", "roulette"] = ConfigField(
        default="tournament",
        title="Parent Selection",
        description="Parent selection strategy.",
    )
    tournament_size: int = ConfigField(
        default=3,
        ge=2,
        title="Tournament Size",
        description="Number of candidates sampled for tournament parent selection.",
    )
    replacement: Literal["elitist", "generational"] = ConfigField(
        default="elitist",
        title="Replacement",
        description="Elitist keeps the best parents and children; generational keeps elites plus top children.",
    )
    refine_offspring_with_generators: bool = ConfigField(
        default=False,
        title="Refine With Generators",
        description="Run configured non-mutation generators on offspring after crossover and mutation.",
    )
    initialize_with_mutation_generators: bool = ConfigField(
        default=False,
        title="Mutation Initialization",
        description="If true, run starting-sequence mutation generators when creating the initial population.",
    )

    @model_validator(mode="after")
    def validate_params(self) -> GeneticAlgorithmOptimizerConfig:
        """Fill derived defaults and validate population/result sizes."""
        if self.offspring_per_generation is None:
            self.offspring_per_generation = self.population_size
        if self.num_results is not None and self.num_results > self.population_size:
            raise ValueError("num_results cannot exceed population_size.")
        return self


@optimizer(
    key="genetic-algorithm",
    label="Genetic Algorithm Optimizer",
    config=GeneticAlgorithmOptimizerConfig,
    description="Maintains a population of discrete sequences, generates offspring by crossover and mutation, scores them with constraints, and keeps the lowest-energy candidates.",
    required_constraint_mode="discrete",
)
@final
class GeneticAlgorithmOptimizer(Optimizer):
    """Population-based genetic algorithm optimizer.

    Examples:
        >>> config = GeneticAlgorithmOptimizerConfig(num_generations=5, population_size=16)
        >>> optimizer_config_key = config.parent_selection
        >>> optimizer_config_key
        'tournament'
    """

    config_class = GeneticAlgorithmOptimizerConfig
    _require_non_empty_generators = False
    config: GeneticAlgorithmOptimizerConfig

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: GeneticAlgorithmOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize a population-based discrete optimizer."""
        self.config = config
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_results=config.num_results,
            num_proposals=config.population_size,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
            seed=config.seed,
        )
        self.num_steps = config.num_generations
        self.population_size = config.population_size
        self.offspring_per_generation = config.offspring_per_generation or config.population_size
        self._population_energies: list[float] = []

    def _resolve_num_results(self, num_results: int) -> None:
        if num_results > self.config.population_size:
            raise ValueError(
                f"num_results ({num_results}) cannot exceed population_size ({self.config.population_size})."
            )
        super()._resolve_num_results(num_results)

    def run(self) -> None:
        """Run the genetic algorithm optimization loop."""
        self._prepare_run()
        if self.num_results is None:
            raise RuntimeError("num_results must be resolved before GeneticAlgorithmOptimizer.run().")

        n_filter = sum(1 for c in self.constraints if c.threshold is not None)
        n_score = len(self.constraints) - n_filter
        logger.info(
            "GeneticAlgorithmOptimizer: %d generations, population=%d, offspring=%d, %d constraints (%d filter, %d scoring)",
            self.num_steps,
            self.population_size,
            self.offspring_per_generation,
            len(self.constraints),
            n_filter,
            n_score,
        )

        self._initialize_population()
        self._write_results_from_population()
        self._save_ga_snapshot(0)

        for generation in range(1, self.num_steps + 1):
            parent_sequences = self._copy_current_population()
            parent_energies = list(self._population_energies)
            offspring = self._make_offspring(parent_sequences, parent_energies, generation)
            self._set_proposal_population(offspring)
            if self.config.refine_offspring_with_generators:
                for generator in self._refinement_generators():
                    generator.sample()
            self._score_current_proposals()
            child_sequences = self._copy_current_population()
            child_energies = list(self.energy_scores)
            self._select_next_population(parent_sequences, parent_energies, child_sequences, child_energies)
            self._write_results_from_population()
            if generation % self.tracking_interval == 0 or generation == self.num_steps:
                self._save_ga_snapshot(generation)
                self._log_ga_progress(generation)

    def _initialize_population(self) -> None:
        for segment in self.segments:
            source = segment.result_sequences or [segment.original_sequence]
            segment.proposal_sequences = [copy.deepcopy(source[i % len(source)]) for i in range(self.population_size)]

        for generator in self._initialization_generators():
            generator.sample()

        self._score_current_proposals()
        self._population_energies = list(self.energy_scores)

    def _score_current_proposals(self) -> None:
        proposal_count = len(self.segments[0].proposal_sequences)
        for segment in self.segments:
            if len(segment.proposal_sequences) != proposal_count:
                raise RuntimeError("All segments must have the same proposal population size.")
        if not all(seq.sequence for segment in self.segments for seq in segment.proposal_sequences):
            raise RuntimeError("GeneticAlgorithmOptimizer cannot score empty proposal sequences.")
        self.num_proposals = proposal_count
        self.score_energy()

    def _copy_current_population(self) -> list[list[Sequence]]:
        return [[copy.deepcopy(seq) for seq in segment.proposal_sequences] for segment in self.segments]

    def _set_proposal_population(self, population: list[list[Sequence]]) -> None:
        for segment, sequences in zip(self.segments, population, strict=True):
            segment.proposal_sequences = [copy.deepcopy(seq) for seq in sequences]
        self.num_proposals = len(population[0]) if population else 0

    def _make_offspring(
        self,
        parent_sequences: list[list[Sequence]],
        parent_energies: list[float],
        generation: int,
    ) -> list[list[Sequence]]:
        offspring_by_segment: list[list[Sequence]] = [[] for _ in self.segments]
        variable_segment_ids = self._variable_segment_ids()
        crossover_positions = self._crossover_positions_by_segment()
        for _ in range(self.offspring_per_generation):
            p1 = self._select_parent(parent_energies)
            p2 = self._select_parent(parent_energies)
            for seg_idx, segment in enumerate(self.segments):
                if id(segment) in variable_segment_ids:
                    child = self._crossover_copy(
                        parent_sequences[seg_idx][p1],
                        parent_sequences[seg_idx][p2],
                        mutable_indices=crossover_positions.get(id(segment)),
                    )
                else:
                    child = copy.deepcopy(parent_sequences[seg_idx][p1])
                child._metadata.setdefault("genetic_algorithm", {})
                child._metadata["genetic_algorithm"].update(
                    {"generation": generation, "parents": [p1, p2], "optimizer": "genetic-algorithm"}
                )
                offspring_by_segment[seg_idx].append(child)
        return self._mutate_offspring(offspring_by_segment)

    def _select_parent(self, energies: list[float]) -> int:
        if self.config.parent_selection == "tournament":
            k = min(self.config.tournament_size, len(energies))
            candidates = self._rng.sample(range(len(energies)), k)
            return min(candidates, key=lambda idx: _energy_sort_key(energies[idx]))
        if self.config.parent_selection == "rank":
            ranked = sorted(range(len(energies)), key=lambda idx: _energy_sort_key(energies[idx]))
            weights = list(range(len(ranked), 0, -1))
            return self._rng.choices(ranked, weights=weights, k=1)[0]
        finite = [energy for energy in energies if math.isfinite(energy)]
        if not finite:
            return self._rng.randrange(len(energies))
        worst = max(finite)
        roulette_weights = [(worst - energy + 1e-8) if math.isfinite(energy) else 1e-12 for energy in energies]
        return self._rng.choices(range(len(energies)), weights=roulette_weights, k=1)[0]

    def _crossover_copy(
        self,
        parent_a: Sequence,
        parent_b: Sequence,
        mutable_indices: set[int] | None = None,
    ) -> Sequence:
        child = copy.deepcopy(parent_a)
        seq_a = parent_a.sequence
        seq_b = parent_b.sequence
        if parent_a.sequence_type == "ligand" or len(seq_a) != len(seq_b) or len(seq_a) < 2:
            return child
        if mutable_indices is not None:
            mutable_indices = {idx for idx in mutable_indices if 0 <= idx < len(seq_a)}
            if not mutable_indices:
                return child
        if self._rng.random() >= self.config.crossover_rate:
            return child
        if self.config.crossover_strategy == "uniform":
            child.sequence = "".join(
                self._crossover_residue(i, a, b, mutable_indices)
                for i, (a, b) in enumerate(zip(seq_a, seq_b, strict=True))
            )
        elif self.config.crossover_strategy == "two_point":
            left = self._rng.randint(0, len(seq_a) - 2)
            right = self._rng.randint(left + 1, len(seq_a))
            child.sequence = "".join(
                b if left <= i < right and self._can_crossover_index(i, mutable_indices) else a
                for i, (a, b) in enumerate(zip(seq_a, seq_b, strict=True))
            )
        else:
            point = self._rng.randint(1, len(seq_a) - 1)
            child.sequence = "".join(
                b if i >= point and self._can_crossover_index(i, mutable_indices) else a
                for i, (a, b) in enumerate(zip(seq_a, seq_b, strict=True))
            )
        child.structure = None
        child.logits = None
        return child

    def _crossover_residue(
        self,
        idx: int,
        residue_a: str,
        residue_b: str,
        mutable_indices: set[int] | None,
    ) -> str:
        if self._can_crossover_index(idx, mutable_indices) and self._rng.random() >= 0.5:
            return residue_b
        return residue_a

    @staticmethod
    def _can_crossover_index(idx: int, mutable_indices: set[int] | None) -> bool:
        return mutable_indices is None or idx in mutable_indices

    def _mutate_offspring(self, offspring_by_segment: list[list[Sequence]]) -> list[list[Sequence]]:
        mutation_generators = self._mutation_generators()
        if mutation_generators:
            return self._run_generators_on_population(mutation_generators, offspring_by_segment)
        return offspring_by_segment

    def _run_generators_on_population(
        self,
        generators: list[Generator],
        population: list[list[Sequence]],
    ) -> list[list[Sequence]]:
        original_population = self._copy_current_population()
        original_num_proposals = self.num_proposals
        try:
            self._set_proposal_population(population)
            for generator in generators:
                generator.sample()
            return self._copy_current_population()
        finally:
            self._set_proposal_population(original_population)
            self.num_proposals = original_num_proposals

    def _mutation_generators(self) -> list[Generator]:
        return [generator for generator in self.generators if generator.input_type == GeneratorInputType.STARTING_SEQUENCE]

    def _initialization_generators(self) -> list[Generator]:
        if self.config.initialize_with_mutation_generators:
            return list(self.generators)
        return [generator for generator in self.generators if generator.input_type != GeneratorInputType.STARTING_SEQUENCE]

    def _refinement_generators(self) -> list[Generator]:
        return [generator for generator in self.generators if generator.input_type != GeneratorInputType.STARTING_SEQUENCE]

    def _variable_segment_ids(self) -> set[int]:
        return {id(segment) for generator in self.generators for segment in generator.segments}

    def _crossover_positions_by_segment(self) -> dict[int, set[int] | None]:
        positions_by_segment: dict[int, set[int] | None] = {}
        for generator in self.generators:
            provider = getattr(generator, "crossover_position_indices", None)
            for segment in generator.segments:
                segment_id = id(segment)
                positions_by_segment.setdefault(segment_id, None)
                if not callable(provider):
                    continue
                generator_positions = provider(segment)
                if generator_positions is None:
                    continue
                current = positions_by_segment[segment_id]
                positions_by_segment[segment_id] = (
                    set(generator_positions)
                    if current is None
                    else current.intersection(generator_positions)
                )
        return positions_by_segment

    def _select_next_population(
        self,
        parent_sequences: list[list[Sequence]],
        parent_energies: list[float],
        child_sequences: list[list[Sequence]],
        child_energies: list[float],
    ) -> None:
        if self.config.replacement == "generational":
            elite_count = min(self.population_size, round(self.config.elite_fraction * self.population_size))
            parent_ranked = sorted(range(len(parent_energies)), key=lambda idx: _energy_sort_key(parent_energies[idx]))
            child_ranked = sorted(range(len(child_energies)), key=lambda idx: _energy_sort_key(child_energies[idx]))
            selected = [("parent", idx) for idx in parent_ranked[:elite_count]]
            selected.extend(("child", idx) for idx in child_ranked[: self.population_size - len(selected)])
            if len(selected) < self.population_size:
                selected_parent_ids = {idx for source, idx in selected if source == "parent"}
                selected.extend(
                    ("parent", idx)
                    for idx in parent_ranked
                    if idx not in selected_parent_ids
                )
                selected = selected[: self.population_size]
        else:
            combined = [("parent", idx, energy) for idx, energy in enumerate(parent_energies)]
            combined.extend(("child", idx, energy) for idx, energy in enumerate(child_energies))
            selected = [(source, idx) for source, idx, _ in sorted(combined, key=lambda item: _energy_sort_key(item[2]))[: self.population_size]]

        next_by_segment: list[list[Sequence]] = [[] for _ in self.segments]
        next_energies: list[float] = []
        for source, idx in selected:
            sequences = parent_sequences if source == "parent" else child_sequences
            energies = parent_energies if source == "parent" else child_energies
            for seg_idx in range(len(self.segments)):
                next_by_segment[seg_idx].append(copy.deepcopy(sequences[seg_idx][idx]))
            next_energies.append(float(energies[idx]))

        self._set_proposal_population(next_by_segment)
        self._population_energies = next_energies
        self._proposal_outcomes = ["accepted"] * len(next_energies)
        self._proposal_energy_scores = list(next_energies)

    def _write_results_from_population(self) -> None:
        if self.num_results is None:
            raise RuntimeError("num_results must be resolved before writing GA results.")
        ranked = sorted(range(len(self._population_energies)), key=lambda idx: _energy_sort_key(self._population_energies[idx]))
        selected = ranked[: self.num_results]
        for segment in self.segments:
            segment.result_sequences = [copy.deepcopy(segment.proposal_sequences[idx]) for idx in selected]
        self.energy_scores = [float(self._population_energies[idx]) for idx in selected]

    def _save_ga_snapshot(self, generation: int) -> None:
        self._save_progress_snapshot(
            time_step=generation,
            optimizer_metadata={
                "type": "genetic-algorithm",
                "generation": generation,
                "num_generations": self.num_steps,
                "population_size": self.population_size,
                "offspring_per_generation": self.offspring_per_generation,
                "num_results": self.num_results,
                "best_population_energy": min(self._population_energies) if self._population_energies else None,
                "mean_population_energy": float(np.mean(self._population_energies)) if self._population_energies else None,
            },
        )

    def _log_ga_progress(self, generation: int) -> None:
        logger.info("Generation %d/%d", generation, self.num_steps)
        filter_summary = self._format_filter_summary()
        if filter_summary is not None:
            logger.info("  filters: %s", filter_summary)
        logger.info("  energy:  %s", self._format_energy_summary())
        if self.custom_logging:
            self.custom_logging(generation, self.segments)


def _energy_sort_key(energy: float) -> tuple[int, float]:
    return (0, energy) if math.isfinite(energy) else (1, float("inf"))
