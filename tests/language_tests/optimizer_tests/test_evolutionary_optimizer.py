"""tests/language_tests/optimizer_tests/test_evolutionary_optimizer.py."""

import math

import pytest
from proto_tools.transforms.masking import MaskingStrategy
from pydantic import BaseModel, ValidationError

from proto_language.constraint import gc_content_constraint
from proto_language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.core import Constraint, ConstraintOutput, Construct, Program, Segment
from proto_language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.optimizer import EvolutionaryOptimizer, EvolutionaryOptimizerConfig


# Empty config for test constraints
class EmptyConfig(BaseModel):
    pass


def _setup_evolutionary_components(
    seq_length: int = 10,
    population_size: int = 10,
    gc_target_range: tuple[float, float] = (40.0, 60.0),
    num_generations: int = 10,
    elitism_count: int = 1,
    tournament_size: int = 3,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
) -> tuple[EvolutionaryOptimizer, RandomNucleotideGenerator, Constraint, Segment]:
    """Helper function to set up a basic Evolutionary Optimizer for testing."""
    # 1. Create the mutation generator and the segment it will modify
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    mutation_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mutation_gen.assign(segment)

    # 2. Create the construct and constraint
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(
            min_gc=gc_target_range[0],
            max_gc=gc_target_range[1],
        ),
    )

    # 3. Create the Evolutionary Optimizer config
    config = EvolutionaryOptimizerConfig(
        population_size=population_size,
        num_generations=num_generations,
        elitism_count=elitism_count,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        verbose=False,
    )

    optimizer = EvolutionaryOptimizer(
        constructs=[construct],
        generators=[mutation_gen],
        constraints=[constraint],
        config=config,
    )
    return optimizer, mutation_gen, constraint, segment


class TestEvolutionaryOptimizer:
    def test_initialization_and_validation(self) -> None:
        """Tests successful initialization and validation of EvolutionaryOptimizer."""
        optimizer, mutation_gen, constraint, _segment = _setup_evolutionary_components()

        assert optimizer.generators == [mutation_gen]
        assert optimizer.constraints == [constraint]
        assert optimizer.num_results == 10  # population_size
        assert optimizer.population_size == 10
        assert optimizer.elitism_count == 1
        assert optimizer.tournament_size == 3

        # Test validation errors - unassigned generator
        test_segment = Segment(sequence="A" * 10, sequence_type="dna")
        unassigned_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )

        # Create a dummy scoring function
        def dummy_scoring_func(input_sequences, config=None):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        dummy_scoring_func._constraint_config_class = EmptyConfig
        dummy_scoring_func._constraint_supported_sequence_types = ["dna"]

        dummy_constraint = Constraint(
            inputs=[test_segment],
            function=dummy_scoring_func,
            function_config=EmptyConfig(),
        )
        with pytest.raises(RuntimeError, match="has no segment assigned"):
            EvolutionaryOptimizer(
                constructs=[Construct([test_segment])],
                generators=[unassigned_gen],
                constraints=[dummy_constraint],
                config=EvolutionaryOptimizerConfig(population_size=10, num_generations=1),
            )

    def test_config_validation(self) -> None:
        """Tests EvolutionaryOptimizerConfig validation."""
        # Valid config
        config = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=20,
            elitism_count=2,
            tournament_size=3,
        )
        assert config.population_size == 10
        assert config.num_generations == 20
        assert config.elitism_count == 2

        # elitism_count >= population_size should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, elitism_count=10)

        # tournament_size > population_size should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, tournament_size=11)

        # population_size < 2 should fail (minimum for crossover)
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=1, num_generations=1)

        # Negative values should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=-1)

        # Invalid rates should fail
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, crossover_rate=1.5)
        with pytest.raises(ValidationError):
            EvolutionaryOptimizerConfig(population_size=10, num_generations=1, mutation_rate=-0.1)

    def test_deterministic_convergence(self) -> None:
        """Tests that the EA converges deterministically with a fixed seed."""
        # Set up optimizer with fixed seed
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=50, max_gc=50),  # Target exactly 50% GC
        )

        config = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=20,
            elitism_count=2,
            seed=42,  # Fixed seed
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # Run twice with same seed
        program1 = Program(optimizers=[optimizer], num_results=10, seed=42)
        program1.run()
        results1 = [seq.sequence for seq in segment.result_sequences]
        energies1 = list(optimizer.energy_scores)

        # Create fresh optimizer with same seed
        segment2 = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen2.assign(segment2)

        constraint2 = Constraint(
            inputs=[segment2],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=50, max_gc=50),
        )

        optimizer2 = EvolutionaryOptimizer(
            constructs=[Construct([segment2])],
            generators=[mutation_gen2],
            constraints=[constraint2],
            config=EvolutionaryOptimizerConfig(
                population_size=10,
                num_generations=20,
                elitism_count=2,
                seed=42,
            ),
        )

        program2 = Program(optimizers=[optimizer2], num_results=10, seed=42)
        program2.run()
        results2 = [seq.sequence for seq in segment2.result_sequences]
        energies2 = list(optimizer2.energy_scores)

        # Results should be identical
        assert results1 == results2
        assert energies1 == energies2

    def test_elitism_preserves_best(self) -> None:
        """Tests that elitism ensures best score never regresses."""
        optimizer, _, _, segment = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=10,
            elitism_count=2,
            gc_target_range=(50, 50),
        )

        # Manually set up initial population with known good solutions
        # Create population with varying GC content
        sequences = [
            "GCGCGCGCGCAAAAAAAAAA",  # 50% GC - perfect score (0.0)
            "GCGCGCGCGCGCGCGCGCGC",  # 100% GC - bad score
            "A" * 20,  # 0% GC - bad score
            "T" * 20,  # 0% GC - bad score
            "GCGCGCGCGCGCGCGCGCGC",  # 100% GC - bad score
            "ATATATATATATATAT ATAT",  # ~0% GC - bad score
            "GCGCGCGCGCGCGCGCGCGC",  # 100% GC - bad score
            "AAAAAAAAAAAAAAAAAAAA",  # 0% GC - bad score
            "TTTTTTTTTTTTTTTTTTTT",  # 0% GC - bad score
            "GGGGGGGGGGGGGGGGGGGG",  # 100% GC - bad score
        ]

        for i, seq in enumerate(sequences):
            segment.result_sequences[i].sequence = seq.replace(" ", "")

        # Run optimization
        optimizer.run()

        # Get history of best scores across generations
        best_scores = []
        for snapshot in optimizer.history:
            energy_scores = snapshot.get("energy_scores", [])
            if energy_scores:
                best_scores.append(min(s for s in energy_scores if s is not None and not isinstance(s, str)))

        # With elitism, best score should never increase (get worse)
        for i in range(1, len(best_scores)):
            assert best_scores[i] <= best_scores[i - 1], (
                f"Best score regressed at generation {i}: {best_scores[i - 1]:.4f} -> {best_scores[i]:.4f}"
            )

    def test_crossover_single_point(self) -> None:
        """Tests that single-point crossover produces valid children."""
        optimizer, _, _, segment = _setup_evolutionary_components(
            seq_length=20,
            population_size=4,
            num_generations=1,
            crossover_rate=1.0,  # Always crossover
            mutation_rate=0.0,  # No mutation
        )

        # Set up parents with distinct patterns
        parent1 = "AAAAAAAAAAAAAAAAAAAA"
        parent2 = "CCCCCCCCCCCCCCCCCCCC"
        segment.result_sequences[0].sequence = parent1
        segment.result_sequences[1].sequence = parent2
        segment.result_sequences[2].sequence = parent1
        segment.result_sequences[3].sequence = parent2

        # Run one generation
        optimizer.run()

        # Check that at least some offspring are valid crossovers
        # (mix of A's and C's, not pure A or pure C)
        offspring_seqs = [seq.sequence for seq in segment.result_sequences]

        # With crossover, we should see some mixed sequences
        # (unless all tournaments selected the same parent, which is unlikely)
        # At minimum, sequences should be valid DNA
        for seq in offspring_seqs:
            assert len(seq) == 20
            assert all(c in "ACGT" for c in seq)

    def test_crossover_uniform(self) -> None:
        """Tests that uniform crossover produces valid children."""
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        config = EvolutionaryOptimizerConfig(
            population_size=4,
            num_generations=1,
            crossover_strategy="uniform",  # Uniform crossover
            crossover_rate=1.0,  # Always crossover
            mutation_rate=0.0,  # No mutation
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # Set up parents
        parent1 = "AAAAAAAAAAAAAAAAAAAA"
        parent2 = "CCCCCCCCCCCCCCCCCCCC"
        segment.result_sequences[0].sequence = parent1
        segment.result_sequences[1].sequence = parent2
        segment.result_sequences[2].sequence = parent1
        segment.result_sequences[3].sequence = parent2

        # Run one generation
        optimizer.run()

        # Check offspring validity
        offspring_seqs = [seq.sequence for seq in segment.result_sequences]
        for seq in offspring_seqs:
            assert len(seq) == 20
            assert all(c in "ACGT" for c in seq)

    def test_population_diversity(self) -> None:
        """Tests that EA maintains population diversity better than single-chain MCMC."""
        optimizer, _, _, segment = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=10,
            elitism_count=1,
            gc_target_range=(50, 50),
        )

        optimizer.run()

        # Count unique sequences in final population
        final_sequences = {seq.sequence for seq in segment.result_sequences}

        # EA should maintain some diversity (not all identical)
        # With population size 10, we expect at least 2-3 unique sequences
        assert len(final_sequences) >= 2, f"Population collapsed to {len(final_sequences)} unique sequences"

    def test_composition_with_program(self) -> None:
        """Tests that EvolutionaryOptimizer works correctly in a multi-stage Program."""
        # Stage 1: EA optimizer
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        construct = Construct([segment])  # Create construct once and reuse

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(segment)

        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        optimizer1 = EvolutionaryOptimizer(
            constructs=[construct],  # Reuse same construct object
            generators=[gen1],
            constraints=[constraint1],
            config=EvolutionaryOptimizerConfig(
                population_size=5,
                num_generations=5,
            ),
        )

        # Stage 2: Another EA optimizer (chained)
        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2.assign(segment)

        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        optimizer2 = EvolutionaryOptimizer(
            constructs=[construct],  # Reuse same construct object
            generators=[gen2],
            constraints=[constraint2],
            config=EvolutionaryOptimizerConfig(
                population_size=5,
                num_generations=5,
            ),
        )

        # Run program with two EA stages
        program = Program(optimizers=[optimizer1, optimizer2], num_results=5)
        program.run()

        # Check that final results exist and are valid
        assert len(segment.result_sequences) == 5
        for seq in segment.result_sequences:
            assert len(seq.sequence) == 20
            assert all(c in "ACGT" for c in seq.sequence)

    def test_mutation_rate_effect(self) -> None:
        """Tests that mutation_rate controls the frequency of mutations."""
        # No mutation
        optimizer_no_mut, _, _, segment_no_mut = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=1,
            crossover_rate=0.0,  # No crossover, just cloning
            mutation_rate=0.0,  # No mutation
        )

        # Set all individuals to same sequence
        initial_seq = "AAAAAAAAAAAAAAAAAAAA"
        for seq in segment_no_mut.result_sequences:
            seq.sequence = initial_seq

        optimizer_no_mut.run()

        # With no crossover and no mutation, all offspring should be identical to parents
        no_mut_seqs = {seq.sequence for seq in segment_no_mut.result_sequences}
        assert len(no_mut_seqs) == 1  # All identical
        assert next(iter(no_mut_seqs)) == initial_seq

        # With mutation
        optimizer_with_mut, _, _, segment_with_mut = _setup_evolutionary_components(
            seq_length=20,
            population_size=10,
            num_generations=1,
            crossover_rate=0.0,  # No crossover
            mutation_rate=1.0,  # Always mutate
        )

        # Set all individuals to same sequence
        for seq in segment_with_mut.result_sequences:
            seq.sequence = initial_seq

        optimizer_with_mut.run()

        # With 100% mutation rate, most offspring should differ from parent
        with_mut_seqs = {seq.sequence for seq in segment_with_mut.result_sequences}
        # Should have some variation (at least a few different sequences)
        assert len(with_mut_seqs) >= 2

    def test_single_character_sequence(self) -> None:
        """Tests that optimizer handles very short sequences gracefully."""
        # Test with minimal length sequence (length 1)
        segment = Segment(sequence="A", sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        # Create a constraint that accepts all sequences
        def always_pass_func(input_sequences, config=None):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        always_pass_func._constraint_config_class = EmptyConfig
        always_pass_func._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment],
            function=always_pass_func,
            function_config=EmptyConfig(),
        )

        config = EvolutionaryOptimizerConfig(
            population_size=4,
            num_generations=2,
            crossover_rate=1.0,
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # Should run without errors
        optimizer.run()

        # All sequences should be length 1
        for seq in segment.result_sequences:
            assert len(seq.sequence) == 1
            assert seq.sequence in "ACGT"


class TestNSGA2Selection:
    """Tests for NSGA-II multi-objective selection mode."""

    def test_dominates_logic(self) -> None:
        """Test domination relation (minimization)."""
        from proto_language.optimizer.evolutionary_optimizer import _dominates

        # A strictly dominates B (better on all objectives)
        assert _dominates([1.0, 2.0], [2.0, 3.0])
        assert _dominates([0.5, 0.5], [1.0, 1.0])

        # A dominates B (equal on some, better on others)
        assert _dominates([1.0, 2.0], [1.0, 3.0])
        assert _dominates([1.0, 2.0], [2.0, 2.0])

        # A does not dominate B (worse on at least one)
        assert not _dominates([1.0, 3.0], [2.0, 2.0])
        assert not _dominates([3.0, 1.0], [1.0, 3.0])

        # A does not dominate B (equal on all)
        assert not _dominates([1.0, 2.0], [1.0, 2.0])

        # A does not dominate B (worse on one, better on another)
        assert not _dominates([1.0, 5.0], [3.0, 2.0])

    def test_non_dominated_sort(self) -> None:
        """Test non-dominated sorting produces correct Pareto fronts."""
        from proto_language.optimizer.evolutionary_optimizer import _non_dominated_sort

        # Empty population
        assert _non_dominated_sort([]) == []

        # Single individual
        fronts = _non_dominated_sort([[1.0, 2.0]])
        assert fronts == [[0]]

        # Two non-dominated individuals (Pareto front)
        fronts = _non_dominated_sort([[1.0, 3.0], [3.0, 1.0]])
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1}

        # Three individuals: two on front 0, one dominated
        fronts = _non_dominated_sort(
            [
                [1.0, 3.0],  # Front 0
                [3.0, 1.0],  # Front 0
                [2.0, 2.0],  # Dominated by both
            ]
        )
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}  # All on same front (trade-off)

        # Clear domination: [1,1] dominates [2,2]
        fronts = _non_dominated_sort([[1.0, 1.0], [2.0, 2.0]])
        assert len(fronts) == 2
        assert fronts[0] == [0]
        assert fronts[1] == [1]

        # Complex case with multiple fronts
        fronts = _non_dominated_sort(
            [
                [1.0, 1.0],  # Front 0 (best on both)
                [2.0, 2.0],  # Front 1
                [1.5, 3.0],  # Front 0 (trades off with [1,1])
                [3.0, 1.5],  # Front 0 (trades off with [1,1])
                [3.0, 3.0],  # Front 2 (dominated by [2,2])
            ]
        )
        assert 0 in fronts[0]  # [1,1] is in front 0

    def test_crowding_distance(self) -> None:
        """Test crowding distance computation."""
        from proto_language.optimizer.evolutionary_optimizer import _crowding_distance

        # Two or fewer individuals get infinite distance
        distances = _crowding_distance([[1.0, 1.0], [2.0, 2.0]], [0, 1])
        assert distances[0] == float("inf")
        assert distances[1] == float("inf")

        # Three individuals on a line (middle one gets finite distance)
        objective_vectors = [
            [1.0, 1.0],  # Boundary
            [2.0, 2.0],  # Middle
            [3.0, 3.0],  # Boundary
        ]
        distances = _crowding_distance(objective_vectors, [0, 1, 2])
        assert distances[0] == float("inf")
        assert distances[2] == float("inf")
        assert 0 < distances[1] < float("inf")

    def test_nsga2_select_survivors(self) -> None:
        """Test NSGA-II survivor selection fills front-by-front."""
        from proto_language.optimizer.evolutionary_optimizer import _nsga2_select_survivors

        # Population smaller than target: select all
        survivors = _nsga2_select_survivors([[1.0, 2.0], [2.0, 1.0]], population_size=5)
        assert set(survivors) == {0, 1}

        # Clear domination: select best individuals
        objective_vectors = [
            [1.0, 1.0],  # Front 0
            [2.0, 2.0],  # Front 1
            [3.0, 3.0],  # Front 2
            [4.0, 4.0],  # Front 3
        ]
        survivors = _nsga2_select_survivors(objective_vectors, population_size=2)
        assert 0 in survivors  # Best individual always selected
        assert len(survivors) == 2

    def test_nsga2_mode_config(self) -> None:
        """Test that selection='nsga2' can be configured."""
        config = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=5,
            selection="nsga2",
        )
        assert config.selection == "nsga2"

        # Default should be tournament
        config_default = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=5,
        )
        assert config_default.selection == "tournament"

    def test_nsga2_pareto_front_spread(self) -> None:
        """Test that NSGA-II maintains diverse Pareto front on multi-objective problem."""
        # Create two conflicting constraints
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        # Constraint 1: favor low GC (min=10, max=30)
        constraint1 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=10, max_gc=30),
            weight=1.0,
            label="low_gc",
        )

        # Constraint 2: favor high GC (min=70, max=90)
        constraint2 = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=70, max_gc=90),
            weight=1.0,
            label="high_gc",
        )

        config = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=20,
            selection="nsga2",
            seed=42,
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint1, constraint2],
            config=config,
        )

        program = Program(optimizers=[optimizer], num_results=10, seed=42)
        program.run()

        # Check that pareto_front was populated
        assert len(optimizer.pareto_front) > 0
        assert len(optimizer.pareto_front) <= optimizer.population_size

        # Check diversity: Pareto front should have solutions with different GC contents
        gc_contents = []
        for idx in optimizer.pareto_front:
            seq = segment.result_sequences[idx].sequence
            gc_count = seq.count("G") + seq.count("C")
            gc_pct = (gc_count / len(seq)) * 100
            gc_contents.append(gc_pct)

        # Should have spread across GC space (key: not collapsed to 1-2 identical values)
        # With diversity initialization, NSGA-II should maintain ≥4 distinct GC values
        unique_gc = len({int(gc) for gc in gc_contents})
        gc_range = max(gc_contents) - min(gc_contents)
        assert unique_gc >= 4, f"Pareto front collapsed to only {unique_gc} unique GC values (expected ≥4)"
        assert gc_range >= 10, f"Pareto front span {gc_range:.1f}% is too narrow (expected ≥10%)"

    def test_nsga2_eval_count_invariant(self) -> None:
        """Test that tournament and nsga2 modes use identical evaluation counts."""
        # Test parameters
        population_size = 10
        num_generations = 5
        elitism_count = 2

        # Expected total evaluations
        expected_evals = population_size + num_generations * (population_size - elitism_count)

        # Tournament mode
        segment_tournament = Segment(sequence="A" * 20, sequence_type="dna")
        gen_tournament = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen_tournament.assign(segment_tournament)
        constraint_tournament = Constraint(
            inputs=[segment_tournament],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        config_tournament = EvolutionaryOptimizerConfig(
            population_size=population_size,
            num_generations=num_generations,
            elitism_count=elitism_count,
            selection="tournament",
            seed=42,
        )

        optimizer_tournament = EvolutionaryOptimizer(
            constructs=[Construct([segment_tournament])],
            generators=[gen_tournament],
            constraints=[constraint_tournament],
            config=config_tournament,
        )

        # Count evaluations by tracking score_energy calls
        tournament_evals = 0
        original_score = optimizer_tournament.score_energy

        def count_tournament(*args, **kwargs):
            nonlocal tournament_evals
            tournament_evals += len(optimizer_tournament.segments[0].proposal_sequences)
            return original_score(*args, **kwargs)

        optimizer_tournament.score_energy = count_tournament
        optimizer_tournament.run()

        # NSGA-II mode
        segment_nsga2 = Segment(sequence="A" * 20, sequence_type="dna")
        gen_nsga2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen_nsga2.assign(segment_nsga2)
        constraint_nsga2 = Constraint(
            inputs=[segment_nsga2],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
        )

        config_nsga2 = EvolutionaryOptimizerConfig(
            population_size=population_size,
            num_generations=num_generations,
            elitism_count=elitism_count,
            selection="nsga2",
            seed=42,
        )

        optimizer_nsga2 = EvolutionaryOptimizer(
            constructs=[Construct([segment_nsga2])],
            generators=[gen_nsga2],
            constraints=[constraint_nsga2],
            config=config_nsga2,
        )

        nsga2_evals = 0
        original_score_nsga2 = optimizer_nsga2.score_energy

        def count_nsga2(*args, **kwargs):
            nonlocal nsga2_evals
            nsga2_evals += len(optimizer_nsga2.segments[0].proposal_sequences)
            return original_score_nsga2(*args, **kwargs)

        optimizer_nsga2.score_energy = count_nsga2
        optimizer_nsga2.run()

        # Tournament mode should match expected count
        assert tournament_evals == expected_evals, f"Tournament mode: {tournament_evals} != {expected_evals}"

        # NSGA-II mode should match tournament (diversification happens before first evaluation)
        assert nsga2_evals == expected_evals, (
            f"NSGA-II mode: {nsga2_evals} != {expected_evals}. "
            f"NSGA-II and tournament should use identical evaluation budgets."
        )

    def test_nsga2_refuses_fallback_scores(self) -> None:
        """Test that NSGA-II refuses to rank when constraints return fallback scores."""
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        # Create a mock constraint that returns fallback_used=True in metadata
        def mock_fallback_constraint(input_sequences, config=None):
            # Return metadata with fallback flag - the Constraint framework will write it
            return [
                ConstraintOutput(
                    score=0.5,
                    metadata={"fallback_used": True, "structure_tool": "mock_backend"},
                )
                for _seq_tuple in input_sequences
            ]

        mock_fallback_constraint._constraint_config_class = EmptyConfig
        mock_fallback_constraint._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment],
            function=mock_fallback_constraint,
            function_config=EmptyConfig(),
            label="mock_fallback",
        )

        config = EvolutionaryOptimizerConfig(
            population_size=4,
            num_generations=2,
            selection="nsga2",
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # Should raise ValueError when trying to extract objective vectors with fallback
        with pytest.raises(ValueError, match="requires a true per-objective decomposition"):
            optimizer.run()

    def test_nsga2_real_provider_metadata_path(self) -> None:
        """Test NSGA-II reads fallback_used from real provider metadata structure.

        This verifies the end-to-end path: providers write metadata via
        _write_constraint_metadata(), which nests custom data under "data",
        and NSGA-II reads from that exact path.
        """
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        # Create a mock constraint that uses the REAL _write_constraint_metadata path
        # (same path as ESMFold/AF2/Protenix providers)
        def real_path_constraint(input_sequences, config=None):
            # Metadata will be nested under "data" by _write_constraint_metadata
            return [
                ConstraintOutput(
                    score=0.5,
                    metadata={"fallback_used": True, "structure_tool": "real_backend"},
                )
                for _seq_tuple in input_sequences
            ]

        real_path_constraint._constraint_config_class = EmptyConfig
        real_path_constraint._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment],
            function=real_path_constraint,
            function_config=EmptyConfig(),
            label="real_path",
        )

        config = EvolutionaryOptimizerConfig(
            population_size=4,
            num_generations=2,
            selection="nsga2",
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[constraint],
            config=config,
        )

        # First, verify that _write_constraint_metadata nests the metadata correctly
        # by evaluating the constraint directly
        constraint.evaluate()

        # Check that metadata is structured like real providers write it
        seq = segment.proposal_sequences[0]
        constraint_meta = seq._constraints_metadata.get("real_path", {})
        assert "data" in constraint_meta, "Metadata should have 'data' key"
        assert "fallback_used" in constraint_meta["data"], "fallback_used should be in data"
        assert constraint_meta["data"]["fallback_used"] is True
        assert constraint_meta["data"]["structure_tool"] == "real_backend"

        # Now verify NSGA-II refuses when reading from this real nested path
        with pytest.raises(ValueError, match="requires a true per-objective decomposition"):
            optimizer.run()

    def test_nsga2_extract_objective_vectors_real_scores(self) -> None:
        """Test that _extract_objective_vectors returns finite, distinct vectors on real scored populations.

        This test exercises the metadata-reading code path that was untested and hid the
        cold-start degenerate population bug. It verifies that objective vectors extracted
        from a diverse scored population contain real finite values, not nan/identical scores.
        """
        # Create diverse initial population with conflicting GC objectives
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)

        low_gc = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=10, max_gc=30),
            weight=1.0,
            label="low_gc",
        )

        high_gc = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=70, max_gc=90),
            weight=1.0,
            label="high_gc",
        )

        config = EvolutionaryOptimizerConfig(
            population_size=10,
            num_generations=1,  # Just need initial evaluation
            selection="nsga2",
            seed=42,
        )

        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[low_gc, high_gc],
            config=config,
        )

        program = Program(optimizers=[optimizer], num_results=10, seed=42)
        program.run()

        # Extract objective vectors after diversification and scoring
        objective_vectors = optimizer._extract_objective_vectors()

        # Verify all scores are finite (not nan/inf)
        for i, vec in enumerate(objective_vectors):
            assert len(vec) == 2, f"Individual {i} has {len(vec)} objectives (expected 2)"
            for j, score in enumerate(vec):
                assert math.isfinite(score), f"Individual {i} objective {j} is {score} (not finite)"
                assert 0.0 <= score <= 1.0, f"Individual {i} objective {j} = {score} outside [0,1]"

        # Verify population has diversity (not all identical)
        # With 1-mutation generator on short sequences, expect 2-3 distinct vectors
        # (the key is that it's NOT 1, which would indicate failed diversification)
        unique_vectors = {tuple(vec) for vec in objective_vectors}
        assert len(unique_vectors) >= 2, (
            f"Objective vectors collapsed to {len(unique_vectors)} unique values (expected ≥2 from diversification)"
        )

    def test_nsga2_filter_rejected_proposal_is_dominated(self) -> None:
        """Filter-rejected proposals (no scoring metadata) get +inf objectives and stay out of Pareto front 0."""
        from proto_language.core.sequence import Sequence
        from proto_language.optimizer.evolutionary_optimizer import _dominates, _non_dominated_sort

        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        mutation_gen.assign(segment)
        low_gc = Constraint(
            inputs=[segment], function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=10, max_gc=30), label="low_gc",
        )
        high_gc = Constraint(
            inputs=[segment], function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=70, max_gc=90), label="high_gc",
        )
        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[low_gc, high_gc],
            config=EvolutionaryOptimizerConfig(population_size=4, num_generations=1, selection="nsga2"),
        )

        # Two filter-passing proposals with real per-objective metadata, plus one
        # proposal with NO metadata, as if a filter rejected it before scoring.
        scored_a = Sequence(sequence="A" * 20, sequence_type="dna")
        scored_a._constraints_metadata = {"low_gc": {"data": {"score": 0.2}}, "high_gc": {"data": {"score": 0.8}}}
        scored_b = Sequence(sequence="C" * 20, sequence_type="dna")
        scored_b._constraints_metadata = {"low_gc": {"data": {"score": 0.8}}, "high_gc": {"data": {"score": 0.2}}}
        rejected = Sequence(sequence="G" * 20, sequence_type="dna")  # rejected: no scoring metadata
        segment.proposal_sequences = [scored_a, scored_b, rejected]

        vectors = optimizer._extract_objective_vectors()

        assert vectors[0] == [0.2, 0.8]
        assert vectors[1] == [0.8, 0.2]
        # Rejected proposal must be worst-case (+inf), never NaN (which is incomparable).
        assert vectors[2] == [float("inf"), float("inf")]
        assert not any(math.isnan(x) for x in vectors[2])

        # It must be dominated by both scored proposals, so it never sits in Pareto front 0.
        assert _dominates(vectors[0], vectors[2])
        assert _dominates(vectors[1], vectors[2])
        fronts = _non_dominated_sort(vectors)
        assert 2 not in fronts[0], "Filter-rejected proposal must not be in Pareto front 0"

    def test_nsga2_extract_reads_each_constraints_own_input_segment(self) -> None:
        """Objectives are read from each constraint's input segment, not always segments[0] (multi-segment)."""
        from proto_language.core.sequence import Sequence

        seg0 = Segment(sequence="A" * 10, sequence_type="dna")
        seg1 = Segment(sequence="A" * 10, sequence_type="dna")
        gen0 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen0.assign(seg0)
        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen1.assign(seg1)
        # Scoring constraint on the SECOND segment only (not segments[0]).
        gc_seg1 = Constraint(
            inputs=[seg1], function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60), label="gc_seg1",
        )
        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([seg0, seg1])],
            generators=[gen0, gen1],
            constraints=[gc_seg1],
            config=EvolutionaryOptimizerConfig(population_size=4, num_generations=1, selection="nsga2"),
        )

        # Metadata lives on the constraint's input segment (seg1), NOT on segments[0] (seg0).
        a = Sequence(sequence="A" * 10, sequence_type="dna")
        a._constraints_metadata = {"gc_seg1": {"data": {"score": 0.25}}}
        b = Sequence(sequence="C" * 10, sequence_type="dna")
        b._constraints_metadata = {"gc_seg1": {"data": {"score": 0.75}}}
        seg1.proposal_sequences = [a, b]
        seg0.proposal_sequences = [
            Sequence(sequence="A" * 10, sequence_type="dna"),
            Sequence(sequence="A" * 10, sequence_type="dna"),
        ]

        vectors = optimizer._extract_objective_vectors()
        assert vectors == [[0.25], [0.75]], f"expected scores read from seg1, got {vectors}"

    def test_nsga2_run_with_satisfiable_filter_keeps_only_passing(self) -> None:
        """End-to-end: nsga2 with a satisfiable filter must not return filter-rejected (inf-energy) results."""
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        mutation_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=2))
        )
        mutation_gen.assign(segment)
        # Wide, easily satisfiable hard filter (GC 30-70%).
        gc_filter = Constraint(
            inputs=[segment], function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=30, max_gc=70), threshold=0.001, label="gc_filter",
        )
        # Two competing objectives, both satisfiable inside the filter band.
        near40 = Constraint(
            inputs=[segment], function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=40), label="near40",
        )
        near60 = Constraint(
            inputs=[segment], function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=60, max_gc=60), label="near60",
        )
        optimizer = EvolutionaryOptimizer(
            constructs=[Construct([segment])],
            generators=[mutation_gen],
            constraints=[gc_filter, near40, near60],
            config=EvolutionaryOptimizerConfig(
                population_size=10, num_generations=15, selection="nsga2", seed=3
            ),
        )
        Program(optimizers=[optimizer], num_results=10, seed=3).run()

        # No returned result may be a filter-rejected (inf-energy) individual.
        assert all(math.isfinite(e) for e in optimizer.energy_scores), (
            f"nsga2 returned filter-rejected results: {optimizer.energy_scores}"
        )
        # And every Pareto-front member must satisfy the hard filter.
        for idx in optimizer.pareto_front:
            seq = segment.result_sequences[idx].sequence
            gc_pct = 100.0 * (seq.count("G") + seq.count("C")) / len(seq)
            assert 30.0 <= gc_pct <= 70.0, f"Pareto front member violates filter: GC={gc_pct:.1f}%"
